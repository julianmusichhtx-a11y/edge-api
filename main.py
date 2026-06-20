from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import traceback
import logging

from adapters import get_adapter
from scoring.scorer import score_props
from cache.cache_manager import get, put   # ← Correct names

# Then use them like this:
cached = get(f"props:{sport}:{len(props)}")
if cached:
    return cached

# After processing...
put(f"props:{sport}:{len(props)}", result, ttl=300)

app = FastAPI(title="EdgeLab Prediction API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PropInput(BaseModel):
    player_name: str
    stat_display: str
    line: float
    higher_american_odds: Optional[float] = None
    lower_american_odds: Optional[float] = None
    player_team: Optional[str] = ""
    home_team: Optional[str] = ""
    away_team: Optional[str] = ""
    source: Optional[str] = "underdog"
    category: Optional[str] = ""


class AnalyzeRequest(BaseModel):
    sport: str
    platform: str = "underdog"
    props: List[PropInput]
    min_edge: float = 0.05


@app.get("/health")
async def health():
    return {"status": "ok", "service": "prediction-api"}


@app.post("/predict")
async def predict(request: AnalyzeRequest):
    try:
        adapter = get_adapter(request.sport)
        if not adapter:
            raise HTTPException(status_code=400, detail=f"Unsupported sport: {request.sport}")

        enriched_props = []
        errors = []

        for prop in request.props:
            try:
                enriched = adapter.enrich_prop(prop.dict())
                if enriched:
                    enriched_props.append(enriched)
            except Exception as e:
                errors.append({
                    "player": prop.player_name,
                    "stat": prop.stat_display,
                    "error": str(e)
                })
                logger.warning(f"Failed to enrich {prop.player_name} - {prop.stat_display}: {e}")

        if not enriched_props:
            return {
                "picks": [],
                "passes": [],
                "stats_context": "",
                "errors": errors,
                "summary": {"actionable_picks": 0, "passes": 0}
            }

        # Score the props
        scored = score_props(enriched_props, min_edge=request.min_edge)

        return {
            "picks": scored.get("picks", []),
            "passes": scored.get("passes", []),
            "stats_context": scored.get("stats_context", ""),
            "errors": errors,
            "summary": {
                "actionable_picks": len(scored.get("picks", [])),
                "passes": len(scored.get("passes", [])),
                "enriched": len(enriched_props),
                "total": len(request.props)
            }
        }

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal prediction error",
                "message": str(e),
                "type": type(e).__name__
            }
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)