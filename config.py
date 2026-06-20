"""
Configuration for Underdog Edge API.
API keys are loaded from environment variables.
Set them in Railway's dashboard or in a local .env file.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ─── API Keys ───────────────────────────────────────────────────────────────
# Set these in Railway dashboard under Variables, or in a local .env file
SPORTRADAR_API_KEY = os.getenv("SPORTRADAR_API_KEY", "")
THE_ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY", "")
SPORTSDATAIO_API_KEY = os.getenv("SPORTSDATAIO_API_KEY", "")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")

# ─── API Base URLs ──────────────────────────────────────────────────────────
MLB_STATS_BASE = "https://statsapi.mlb.com/api/v1"           # Free, no key needed
NHL_STATS_BASE = "https://api-web.nhle.com/v1"               # Free, no key needed
SPORTRADAR_BASE = "https://api.sportradar.com"
THE_ODDS_BASE = "https://api.the-odds-api.com/v4"
SPORTSDATAIO_BASE = "https://api.sportsdata.io/v3"
OPENWEATHER_BASE = "https://api.openweathermap.org/data/2.5"

# ─── Sportradar Sport Configs ───────────────────────────────────────────────
# Each sport has its own base path, version, and language on Sportradar.
SPORTRADAR_SPORTS = {
    "mlb":  {"base": f"{SPORTRADAR_BASE}/mlb",        "version": "v8", "lang": "en"},
    "wnba": {"base": f"{SPORTRADAR_BASE}/wnba",       "version": "v8", "lang": "en"},
    "nba":  {"base": f"{SPORTRADAR_BASE}/nba",         "version": "v8", "lang": "en"},
    "nfl":  {"base": f"{SPORTRADAR_BASE}/nfl",         "version": "v8", "lang": "en"},
    "nhl":  {"base": f"{SPORTRADAR_BASE}/nhl",         "version": "v8", "lang": "en"},
    "soccer": {"base": f"{SPORTRADAR_BASE}/soccer",    "version": "v4", "lang": "en"},
    "mma":  {"base": f"{SPORTRADAR_BASE}/mma",         "version": "v2", "lang": "en"},
    "ncaafb": {"base": f"{SPORTRADAR_BASE}/ncaafb",    "version": "v8", "lang": "en"},
    "ncaamb": {"base": f"{SPORTRADAR_BASE}/ncaamb",    "version": "v8", "lang": "en"},
}

# ─── Rate Limits ────────────────────────────────────────────────────────────
# Requests per second per API. Exceeding these gets you 429'd or 502'd.
RATE_LIMITS = {
    "sportradar":   {"per_second": 1,   "daily": 1000},   # Trial key limits
    "mlb_stats":    {"per_second": 10,  "daily": 99999},   # Free, generous
    "nhl_stats":    {"per_second": 10,  "daily": 99999},   # Free, generous
    "the_odds_api": {"per_second": 2,   "monthly": 500},   # Free tier: 500/month
    "sportsdataio": {"per_second": 2,   "daily": 1000},
    "openweather":  {"per_second": 1,   "daily": 1000},
}

# ─── Quota Budget Per Analysis Run ──────────────────────────────────────────
# How many API calls each sport uses per analysis run.
# This helps you estimate daily capacity.
#
# Example: Sportradar trial = 1000 calls/day
#   MLB run:  ~40 calls (schedule + 30 rosters + game logs)
#   WNBA run: ~25 calls (7 schedules + 12 summaries + 6 team profiles)
#   NFL run:  ~35 calls (similar pattern)
#   → You can run ~15-20 analyses per day on the trial key
#
# MLB Stats API: unlimited (free, public)
#   → No concern, use freely
#
# The Odds API: 500 calls/month on free tier
#   → Each run uses 1-4 calls = ~125-500 runs/month
#   → Upgrade to $20/month plan for 10,000 calls if needed
#
# To scale beyond trial limits:
#   Sportradar production key: contact sales (~$200-500/month)
#   The Odds API Scale plan: $80/month for 100,000 calls

# ─── Cache TTLs (seconds) ──────────────────────────────────────────────────
CACHE_TTL = {
    "schedule":      3600,      # 1 hour  — games don't change often
    "team_roster":   21600,     # 6 hours — rosters change rarely
    "player_season": 21600,     # 6 hours — season stats update after games
    "game_boxscore": 86400 * 7, # 7 days  — completed games never change
    "player_gamelog": 3600,     # 1 hour  — recent games
}

# ─── Supported Prop Types ──────────────────────────────────────────────────
# Maps stat_display strings to canonical stat keys per sport.
# The scoring engine uses canonical keys to look up the right game log field.
PROP_STAT_MAP = {
    # MLB Pitching
    "strikeouts": "strikeouts", "pitcher strikeouts": "strikeouts",
    "earned runs": "earned_runs", "earned runs allowed": "earned_runs",
    "hits allowed": "hits_allowed", "walks allowed": "walks_allowed",
    "pitching outs": "pitching_outs", "outs recorded": "pitching_outs",
    "innings pitched": "innings_pitched",
    # MLB Batting
    "hits": "hits", "total bases": "total_bases", "home runs": "home_runs",
    "runs": "runs", "rbi": "rbi", "rbis": "rbi",
    "stolen bases": "stolen_bases", "walks": "walks", "singles": "singles",
    "doubles": "doubles", "hits runs rbis": "hits_runs_rbis",
    "hits+runs+rbis": "hits_runs_rbis",
    "batter strikeouts": "batter_strikeouts",
    # Basketball (NBA/WNBA/NCAAB)
    "points": "points", "rebounds": "rebounds", "assists": "assists",
    "threes": "three_pointers", "3-pointers made": "three_pointers",
    "three pointers made": "three_pointers",
    "steals": "steals", "blocks": "blocks", "turnovers": "turnovers",
    "points rebounds assists": "pra", "points+rebounds+assists": "pra",
    "points + rebounds": "points_rebounds",
    "points + assists": "points_assists",
    "rebounds + assists": "rebounds_assists",
    "points + rebounds + assists": "pra",
    "offensive rebounds": "offensive_rebounds",
    "blocks + steals": "blocks_steals",
    # Football (NFL/NCAAF)
    "passing yards": "passing_yards", "pass yards": "passing_yards",
    "rushing yards": "rushing_yards", "rush yards": "rushing_yards",
    "receiving yards": "receiving_yards", "receptions": "receptions",
    "touchdowns": "touchdowns", "passing touchdowns": "passing_tds",
    "rush attempts": "rush_attempts", "interceptions": "interceptions",
    "completions": "completions",
    # Hockey (NHL)
    "goals": "goals", "shots on goal": "shots_on_goal",
    "saves": "saves", "power play points": "pp_points",
    # Soccer
    "shots": "shots", "shots on target": "shots_on_target",
    "shots on goal": "shots_on_target",
    "tackles": "tackles", "passes": "passes",
    "soccer goals": "goals", "goal scored": "goals",
    "soccer assists": "assists",
    # MMA
    "total rounds": "total_rounds", "significant strikes": "sig_strikes",
    # Tennis
    "aces": "aces", "double faults": "double_faults", "games won": "games_won",
    # Esports — CS2 / Valorant / LoL / Dota 2 / Rocket League / CoD
    "kills": "kills", "deaths": "deaths",
    "headshots": "headshots",
    "maps played": "maps_played", "maps": "maps_played",
    "adr": "adr", "avg damage per round": "adr",
    "rating": "rating", "hltv rating": "rating",
    "eliminations": "eliminations",
    # Rocket League esports
    "rl goals": "goals", "rl saves": "saves", "rl score": "score",
    # Combo / alias props
    "kills + assists": "kills_assists",
}
