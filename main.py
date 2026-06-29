from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import traceback
import logging
import inspect
from datetime import datetime, timezone

from adapters import get_adapter
from scoring.scorer import score_props
from cache.cache_manager import get, put
from config import SPORTRADAR_API_KEY, SPORTSDATAIO_API_KEY

BACKEND_VERSION = "railway-contract-v2"

SUPPORTED_SPORTS = {
    "MLB": {
        "status": "stable",
        "projectionSupported": True,
        "predictionSupported": True,
        "pitcherStatsSupported": True,
        "batterStatsSupported": True,
        "supportedStats": [
            "pitcher_strikeouts", "pitching_outs", "hits_allowed", "walks_allowed",
            "earned_runs_allowed", "hits", "runs", "rbis", "home_runs",
            "total_bases", "hits_runs_rbis",
        ],
        "unsupportedStats": [],
        "adapter": "MLBAdapter",
        "notes": ["Stable production path"],
    },
    "WNBA": {
        "status": "beta",
        "projectionSupported": True,
        "predictionSupported": True,
        "supportedStats": [
            "points", "rebounds", "assists", "three_pointers_made",
            "points_rebounds_assists", "points_rebounds", "points_assists",
            "rebounds_assists", "blocks", "steals", "blocks_steals",
            "turnovers", "fantasy_points",
        ],
        "unsupportedStats": ["double_doubles", "period_props"],
        "adapter": "WNBAAdapter",
        "notes": ["Sportradar recent stats when provider data is configured"],
    },
    "SOCCER": {
        "status": "beta",
        "projectionSupported": True,
        "predictionSupported": True,
        "supportedStats": [
            "shots", "shots_on_target", "goals", "assists", "goals_assists",
            "passes", "tackles", "yellow_cards", "corner_kicks", "offsides",
        ],
        "unsupportedStats": [
            "match_result", "double_chance", "both_teams_to_score",
            "team_goals", "unsupported_binary_market",
        ],
        "adapter": "SoccerAdapter",
        "notes": ["World Cup style sample sizes are small; unsupported team/match markets fail closed"],
    },
    "WORLD_CUP": {
        "status": "beta",
        "projectionSupported": True,
        "predictionSupported": True,
        "supportedStats": [
            "shots", "shots_on_target", "goals", "assists", "goals_assists",
            "passes", "tackles", "yellow_cards", "corner_kicks", "offsides",
        ],
        "unsupportedStats": [
            "match_result", "double_chance", "both_teams_to_score",
            "team_goals", "unsupported_binary_market",
        ],
        "adapter": "SoccerAdapter",
        "notes": ["Uses SoccerAdapter and reports World Cup support separately"],
    },
    "TENNIS": {
        "status": "beta",
        "projectionSupported": False,
        "predictionSupported": False,
        "supportedStats": [],
        "unsupportedStats": [
            "aces", "double_faults", "total_games", "games_won", "total_sets",
            "set_winner", "match_winner", "fantasy_points", "break_points_won",
            "break_points_saved", "first_serve_percentage", "first_serve_points_won",
        ],
        "adapter": "TennisAdapter",
        "notes": ["Adapter returns honest unsupported/watchlist contract until a tennis data provider is wired"],
    },
}

SPORT_ALIASES = {
    "mlb": "MLB",
    "baseball_mlb": "MLB",
    "wnba": "WNBA",
    "basketball_wnba": "WNBA",
    "soccer": "SOCCER",
    "world_cup": "WORLD_CUP",
    "worldcup": "WORLD_CUP",
    "fifa": "WORLD_CUP",
    "fifa_world_cup": "WORLD_CUP",
    "soccer_world_cup": "WORLD_CUP",
    "soccer_fifa_world_cup": "WORLD_CUP",
    "football_world_cup": "WORLD_CUP",
    "tennis": "TENNIS",
    "tennis_atp": "TENNIS",
    "tennis_wta": "TENNIS",
    "atp": "TENNIS",
    "wta": "TENNIS",
}


def _normalize_sport_key(sport: str) -> str:
    raw = (sport or "").strip()
    key = raw.lower().replace("-", "_").replace(" ", "_")
    return SPORT_ALIASES.get(key, raw.upper() if raw else "UNKNOWN")


def _sport_support(sport: str) -> dict:
    key = _normalize_sport_key(sport)
    return SUPPORTED_SPORTS.get(key, {
        "status": "unsupported",
        "projectionSupported": False,
        "predictionSupported": False,
        "supportedStats": [],
        "unsupportedStats": [],
        "adapter": None,
        "notes": [],
        "unsupportedReason": f"{sport or 'unknown'} is not supported by this backend contract",
    })


def _default_projection_health(sport: str, props_received: int = 0, attempted: int = 0) -> dict:
    normalized_sport = _normalize_sport_key(sport)
    return {
        "enabled": True,
        "sport": normalized_sport,
        "propsReceived": props_received,
        "attempted": attempted,
        "matched": 0,
        "unavailable": attempted,
        "unsupported": 0,
        "playerNotMatched": 0,
        "statNotSupported": 0,
        "noStatHistory": attempted,
        "providerErrors": [],
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
            "insufficient_sample": 0,
            "provider_unavailable": 0,
            "provider_error": 0,
            "invalid_line": 0,
            "unsupported_market_type": 0,
            "unsupported_team_market": 0,
            "unsupported_match_market": 0,
            "unsupported_binary_market": 0,
            "tennis_adapter_unavailable": 0,
            "sport_not_supported": 0,
            "projection_exception": 0,
        },
        "sampleFailures": [],
        "errors": [],
    }


def _normalize_projection_health(health: dict | None, sport: str, props_received: int = 0, attempted: int = 0) -> dict:
    base = _default_projection_health(sport, props_received, attempted)
    if isinstance(health, dict):
        base.update(health)
        reasons = {**base["unavailableReasons"], **(health.get("unavailableReasons") or {})}
        base["unavailableReasons"] = reasons

    base["attempted"] = base.get("attempted", base.get("projectionAttempted", attempted)) or 0
    base["matched"] = base.get("matched", base.get("projectionMatched", 0)) or 0
    base["unavailable"] = base.get("unavailable", base.get("projectionUnavailable", 0)) or 0
    base["unsupported"] = base.get("unsupported", base["unavailableReasons"].get("stat_not_supported", 0) + base["unavailableReasons"].get("sport_not_supported", 0))
    base["playerNotMatched"] = base.get("playerNotMatched", base["unavailableReasons"].get("player_not_matched", 0))
    base["statNotSupported"] = base.get("statNotSupported", base["unavailableReasons"].get("stat_not_supported", 0))
    base["noStatHistory"] = base.get("noStatHistory", base["unavailableReasons"].get("no_stat_history", 0))
    base["providerErrors"] = base.get("providerErrors", base.get("errors", [])) or []
    base["projectionAttempted"] = base["attempted"]
    base["projectionMatched"] = base["matched"]
    base["projectionUnavailable"] = base["unavailable"]
    base["backendVersion"] = BACKEND_VERSION
    base["sport"] = _normalize_sport_key(sport)
    return base


def _normalize_result_contract(item: dict, sport: str) -> dict:
    if not isinstance(item, dict):
        return item

    stat = item.get("canonicalStat") or item.get("stat") or item.get("stat_type") or item.get("stat_display")
    raw_stat = item.get("rawStat") or item.get("stat_display") or item.get("stat_type")
    reason = item.get("unavailableCode") or item.get("unavailableReason") or item.get("projectionUnavailableReason")

    return {
        **item,
        "player": item.get("player") or item.get("player_name"),
        "team": item.get("team") or item.get("player_team"),
        "opponent": item.get("opponent") or item.get("away_team") or item.get("home_team"),
        "sport": item.get("sport") or _normalize_sport_key(sport),
        "canonicalStat": stat,
        "rawStat": raw_stat,
        "side": item.get("side") or item.get("selectedSide"),
        "modelProbability": item.get("modelProbability", item.get("modelProb")),
        "marketProbability": item.get("marketProbability", item.get("marketProb")),
        "unavailableCode": reason,
        "unavailableReason": item.get("unavailableReason") or reason,
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
    requestedStats: Optional[List[str]] = None
    runContext: Optional[Dict[str, Any]] = None
    slateContext: Optional[Dict[str, Any]] = None


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "proplayr-railway-api",
        "backendVersion": BACKEND_VERSION,
        "predictionEnabled": True,
        "projectionEnabled": True,
        "supportedSports": list(SUPPORTED_SPORTS.keys()),
        "sportSupport": SUPPORTED_SPORTS,
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
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/analyze")
@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        normalized_sport = _normalize_sport_key(request.sport)
        adapter = get_adapter(normalized_sport)
        if not adapter:
            projection_health = _default_projection_health(request.sport, len(request.props), 0)
            projection_health["enabled"] = False
            projection_health["unavailableReasons"]["sport_not_supported"] = len(request.props)
            projection_health["unsupported"] = len(request.props)
            projection_health["backendVersion"] = BACKEND_VERSION
            return {
                "picks": [],
                "passes": [
                    _normalize_result_contract({
                        **p.dict(),
                        "projectionAvailable": False,
                        "projection": None,
                        "projectionEdge": None,
                        "projectionSource": None,
                        "probabilitySource": None,
                        "confidence": None,
                        "sampleSize": None,
                        "unavailableReason": "sport_not_supported",
                        "unavailableCode": "sport_not_supported",
                        "projectionUnavailableReason": "sport_not_supported",
                        "verdict": "SKIP",
                        "reason": f"Unsupported sport: {request.sport}",
                    }, request.sport)
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
            raw.setdefault("sport", normalized_sport.lower())
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
            enrichment_errors.append({"error": f"Batch enrichment failed: {e}", "code": "provider_error"})
            enriched_props = raw_props  # fall back to unenriched props rather than failing the whole request

        if not enriched_props:
            projection_health = _normalize_projection_health(None, request.sport, len(request.props), 0)
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
        projection_health = _normalize_projection_health(scored_result.get("projectionHealth"), request.sport, len(request.props), len(enriched_props))
        projection_health = projection_health or _default_projection_health(
            request.sport,
            len(request.props),
            len(enriched_props),
        )
        projection_health["sport"] = projection_health.get("sport") or request.sport.lower()
        projection_health["backendVersion"] = BACKEND_VERSION

        return {
            "picks": [_normalize_result_contract(item, request.sport) for item in scored_result.get("picks", [])],
            "passes": [_normalize_result_contract(item, request.sport) for item in scored_result.get("passes", [])],
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
