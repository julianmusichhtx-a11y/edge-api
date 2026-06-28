from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import traceback
import logging
import inspect

from adapters import get_adapter
from scoring.scorer import score_props
from cache.cache_manager import get, put
from config import SPORTRADAR_API_KEY, SPORTSDATAIO_API_KEY

BACKEND_VERSION = "railway-contract-v1"

SUPPORTED_SPORTS = {
    "mlb": {
        "status": "stable",
        "projectionSupported": True,
        "predictionSupported": True,
        "pitcherStatsSupported": True,
        "batterStatsSupported": True,
    },
    "soccer": {
        "status": "beta",
        "projectionSupported": True,
        "predictionSupported": True,
    },
    "wnba": {
        "status": "beta",
        "projectionSupported": True,
        "predictionSupported": True,
        "supportedStats": ["points", "rebounds", "assists", "three_pointers", "pra"],
    },
}


def _sport_support(sport: str) -> dict:
    key = (sport or "").lower()
    return SUPPORTED_SPORTS.get(key, {
        "status": "unsupported",
        "projectionSupported": False,
        "predictionSupported": False,
        "unsupportedReason": f"{sport or 'unknown'} is not supported by this backend contract",
    })


def _default_projection_health(sport: str, props_received: int = 0, attempted: int = 0) -> dict:
    return {
        "enabled": True,
        "sport": (sport or "").lower(),
        "propsReceived": props_received,
        "projectionAttempted": attempted,
        "projectionMatched": 0,
        "projectionUnavailable": attempted,
        "provider": "railway_prediction_api",
        "unavailableReasons": {
            "player_not_matched": 0,
            "stat_not_supported": 0,
            "no_game_logs": 0,
            "no_season_stats": 0,
            "no_stat_history": attempted,
            "projection_exception": 0,
        },
        "sampleFailures": [],
        "errors": [],
    }


def _limited_errors(errors: list) -> list:
    safe = []
    for item in errors[:10]:
        if isinstance(item, dict):
            safe.append({k: v for k, v in item.items() if "key" not in str(k).lower() and "token" not in str(k).lower() and "secret" not in str(k).lower()})
        else:
            safe.append({"error": str(item)})
    return safe

app = FastAPI(title="EdgeLab Prediction API")

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


class PredictRequest(BaseModel):
    sport: str
    platform: str = "underdog"
    props: List[PropInput]
    min_edge: float = 0.05


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "prediction-api",
        "backendVersion": BACKEND_VERSION,
        "supportedSports": list(SUPPORTED_SPORTS.keys()),
        "configuredProviders": {
            "sportradar": bool(SPORTRADAR_API_KEY),
            "sportsDataIo": bool(SPORTSDATAIO_API_KEY),
            "mlbStatsApi": True,
        },
        "projectionEnabled": True,
        "providerReadiness": {
            "sportradar": "ready" if SPORTRADAR_API_KEY else "not_configured",
            "sportsDataIo": "ready" if SPORTSDATAIO_API_KEY else "not_configured",
            "mlbStatsApi": "ready",
        },
        "providers": {
            "sportradarConfigured": bool(SPORTRADAR_API_KEY),
            "sportsDataIoConfigured": bool(SPORTSDATAIO_API_KEY),
        },
        "projections": {
            "enabled": True,
            "supportedSports": list(SUPPORTED_SPORTS.keys()),
        },
    }


@app.post("/analyze")
@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        adapter = get_adapter(request.sport)
        if not adapter:
            projection_health = _default_projection_health(request.sport, len(request.props), 0)
            projection_health["enabled"] = False
            projection_health["unavailableReasons"]["stat_not_supported"] = len(request.props)
            projection_health["backendVersion"] = BACKEND_VERSION
            return {
                "picks": [],
                "passes": [
                    {
                        **p.dict(),
                        "projectionAvailable": False,
                        "projection": None,
                        "projectionEdge": None,
                        "projectionSource": None,
                        "probabilitySource": None,
                        "confidence": None,
                        "sampleSize": None,
                        "unavailableReason": "sport_not_supported",
                        "projectionUnavailableReason": "sport_not_supported",
                        "verdict": "SKIP",
                        "reason": f"Unsupported sport: {request.sport}",
                    }
                    for p in request.props[:50]
                ],
                "projectionHealth": projection_health,
                "projectionAttempted": 0,
                "projectionMatched": 0,
                "projectionUnavailable": len(request.props),
                "unavailableReasons": {"sport_not_supported": len(request.props)},
                "sampleFailures": [
                    {
                        "player": p.player_name,
                        "stat": p.stat_display,
                        "line": p.line,
                        "side": "",
                        "reason": "sport_not_supported",
                    }
                    for p in request.props[:10]
                ],
                "providerErrors": [],
                "sportSupport": _sport_support(request.sport),
                "backendVersion": BACKEND_VERSION,
                "stats_context": f"Unsupported sport: {request.sport}",
                "errors": [],
                "summary": {
                    "actionable_picks": 0,
                    "passes": len(request.props),
                    "enriched": 0,
                    "total": len(request.props)
                }
            }

        raw_props = []
        for p in request.props:
            raw = p.dict()
            raw.setdefault("sport", request.sport.lower())
            raw_props.append(raw)
        enrichment_errors = []

        try:
            # Sportradar-backed adapters (WNBA/NBA/NFL/NHL/Soccer/MMA) define
            # enrich_props as async — it batches roster/gamelog fetches once
            # for the whole slate instead of refetching per prop.
            # MLBAdapter defines it as a plain sync method.
            if inspect.iscoroutinefunction(adapter.enrich_props):
                enriched_props = await adapter.enrich_props(raw_props)
            else:
                enriched_props = adapter.enrich_props(raw_props)
        except Exception as e:
            logger.error(f"Batch enrichment failed for {request.sport}: {e}")
            logger.error(traceback.format_exc())
            enrichment_errors.append({"error": f"Batch enrichment failed: {e}"})
            enriched_props = raw_props  # fall back to unenriched props rather than failing the whole request

        if not enriched_props:
            projection_health = _default_projection_health(request.sport, len(request.props), 0)
            return {
                "picks": [],
                "passes": [],
                "projectionHealth": projection_health,
                "projectionAttempted": 0,
                "projectionMatched": 0,
                "projectionUnavailable": 0,
                "unavailableReasons": projection_health["unavailableReasons"],
                "sampleFailures": [],
                "providerErrors": _limited_errors(enrichment_errors),
                "sportSupport": _sport_support(request.sport),
                "backendVersion": BACKEND_VERSION,
                "stats_context": "",
                "errors": enrichment_errors,
                "summary": {
                    "actionable_picks": 0,
                    "passes": 0,
                    "enriched": 0,
                    "total": len(request.props)
                }
            }

        # Score the enriched props
        scored_result = score_props(enriched_props, min_edge=request.min_edge)
        projection_health = scored_result.get("projectionHealth") or _default_projection_health(
            request.sport,
            len(request.props),
            len(enriched_props),
        )
        projection_health["sport"] = projection_health.get("sport") or request.sport.lower()
        projection_health["backendVersion"] = BACKEND_VERSION

        return {
            "picks": scored_result.get("picks", []),
            "passes": scored_result.get("passes", []),
            "projectionHealth": projection_health,
            "projectionAttempted": projection_health.get("projectionAttempted", 0),
            "projectionMatched": projection_health.get("projectionMatched", 0),
            "projectionUnavailable": projection_health.get("projectionUnavailable", 0),
            "unavailableReasons": projection_health.get("unavailableReasons", {}),
            "sampleFailures": projection_health.get("sampleFailures", [])[:10],
            "providerErrors": _limited_errors(enrichment_errors + projection_health.get("errors", [])),
            "sportSupport": _sport_support(request.sport),
            "backendVersion": BACKEND_VERSION,
            "stats_context": scored_result.get("stats_context", ""),
            "errors": enrichment_errors,
            "summary": {
                "actionable_picks": len(scored_result.get("picks", [])),
                "passes": len(scored_result.get("passes", [])),
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
                "error": "Prediction service error",
                "message": str(e),
                "type": type(e).__name__
            }
        )


@app.get("/soccer-debug")
async def soccer_debug():
    """Show what player names the SoccerAdapter has in its game log."""
    from adapters.soccer_adapter import SoccerAdapter
    adapter = SoccerAdapter()
    await adapter._load_game_logs()
    names = sorted(adapter._player_game_log.keys())
    jimenez = [n for n in names if "jimenez" in n or "raul" in n]
    return {
        "total_players": len(names),
        "jimenez_matches": jimenez,
        "sample_names": names[:20],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
