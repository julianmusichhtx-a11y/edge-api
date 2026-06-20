# Edge API - Predictive Prop Analysis Service

Multi-sport backend that powers intelligent player prop analysis for your Base44 frontend.

## Why this exists
Your original Base44 `generateAnalysis.js` was doing too much in the browser (stats fetching, hit-rate scoring, enrichment). This was hitting file-size limits and producing weak "predictions" (backward-looking hit rates).

This service:
- Fetches real player stats server-side (cached aggressively)
- Runs a proper projection model → calibrated probability
- Calculates true edge vs the line
- Returns rich context so your Gemini/DeepSeek narrative is actually grounded in strong math (your real moat)

## Quick Start (Local)
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# add your keys to .env
uvicorn main:app --reload
```

Test:
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "props": [{
      "player": "LeBron James",
      "stat": "Points",
      "line": 24.5,
      "sport": "nba"
    }],
    "platform": "PrizePicks",
    "min_edge": 0.05
  }'
```

## Deployment on Railway (Recommended - Free Tier)
1. Push this folder to a new GitHub repo called `edge-api`
2. Go to railway.app → New Project → Deploy from GitHub
3. Add the environment variables from `.env.example`
4. Railway will auto-deploy. Your URL will be something like `https://edge-api-production-xxxx.up.railway.app`

Health check: `GET /health`

## Quota Safety (You will NOT hit limits quickly)
| API                  | Daily Limit (Trial) | Calls per Analysis (cached) | Safe Daily Runs |
|----------------------|---------------------|-----------------------------|-----------------|
| MLB Stats (public)   | Unlimited           | 2                           | ∞               |
| Sportradar (trial)   | ~1000               | 8-15 (with cache)           | ~60-100         |
| The Odds API (free)  | 500/month           | 0-2 (future bump detection) | Plenty          |
| SportsDataIO         | ~1000               | 4                           | 200+            |

**Caching is aggressive**:
- Completed game boxscores: 7 days (immutable)
- Player recent form: 1 hour
- This means repeat analyses on the same slate cost almost zero quota.

When you scale: Upgrade Sportradar to production (~$200-500/mo removes daily cap) and The Odds API to Scale plan.

## Adding a New Sport
1. Create `adapters/new_sport_adapter.py` extending `BaseAdapter`
2. Implement `get_player_stats` (use your existing API keys)
3. Register in `config.py` SPORT_ADAPTERS and `adapters/__init__.py`
4. Extend `scorer.py` with sport-specific adjustments if needed (e.g. back-to-backs for NBA/NHL)

The architecture is deliberately sport-agnostic after the adapter layer.

## Next Milestones (with you)
1. Full Sportradar integration for NBA/WNBA/NFL (you have the key)
2. Bump / stale-line detector using The Odds API consensus vs DFS line
3. Line movement history storage
4. Switch from rule-based projection to trained XGBoost models per sport (huge accuracy jump)
5. Bet tracking endpoint (store results, compute your actual ROI/CLV over time)

## Connection to your Base44 App
Once deployed, replace the heavy scoring logic in `generateAnalysis.js` with a single `fetch` to this `/analyze` endpoint. The response gives you everything you need for the UI + rich context for the AI narrative step.

This keeps your beautiful frontend exactly as-is while moving the intelligence to a proper backend where it belongs.

Let's build the future of sharp DFS analysis.
