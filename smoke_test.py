#!/usr/bin/env python3
"""
Smoke test for the EdgeLab prediction API after the MLBAdapter fix.

Usage:
    python3 smoke_test.py
    python3 smoke_test.py https://your-custom-url.up.railway.app

Run this AFTER deploying the new main.py / mlb_adapter.py / base_adapter.py
to Railway. It checks, in order:

  1. /health responds
  2. /predict accepts a single well-known player + stat and returns
     a real scored result (not "Insufficient data")
  3. /predict handles a small batch with a mix of real players,
     a pitcher prop, and a deliberately unsupported stat type
  4. Reports whether _playerStats-driven scoring is actually firing,
     vs everything silently falling into "passes"

If step 2 or 3 still shows 0 picks / all "Insufficient data", the fix
didn't take effect — most likely the Railway deploy is still serving the
old adapter (check the Railway deployment logs for "Enriched X of Y props
with real stats" — that log line only exists in the new mlb_adapter.py).
"""
import json
import sys
import time

import httpx

DEFAULT_BASE_URL = "https://web-production-a5c3b.up.railway.app"


def section(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def check_health(client: httpx.Client, base_url: str) -> bool:
    section("1. Health check")
    try:
        res = client.get(f"{base_url}/health", timeout=10)
        print(f"  GET /health -> {res.status_code}")
        print(f"  Body: {res.text}")
        ok = res.status_code == 200
        print("  PASS" if ok else "  FAIL — server isn't responding correctly")
        return ok
    except Exception as e:
        print(f"  FAIL — request error: {e}")
        return False


def check_single_prop(client: httpx.Client, base_url: str) -> bool:
    section("2. Single well-known player (sanity check)")
    # A high-volume, currently-active hitter — should have plenty of
    # recent MLB Stats API game log data regardless of date.
    payload = {
        "sport": "mlb",
        "platform": "underdog",
        "props": [
            {
                "player_name": "Aaron Judge",
                "stat_display": "Hits",
                "line": 1.5,
                "higher_american_odds": -120,
                "lower_american_odds": 100,
            }
        ],
        "min_edge": 0.05,
    }
    try:
        start = time.time()
        res = client.post(f"{base_url}/predict", json=payload, timeout=30)
        elapsed = time.time() - start
        print(f"  POST /predict -> {res.status_code} ({elapsed:.1f}s)")

        if res.status_code != 200:
            print(f"  FAIL — non-200 response: {res.text[:500]}")
            return False

        data = res.json()
        summary = data.get("summary", {})
        print(f"  summary: {summary}")

        all_results = data.get("picks", []) + data.get("passes", [])
        if not all_results:
            print("  FAIL — no results returned at all")
            return False

        result = all_results[0]
        has_stats = "_playerStats" in result or result.get("seasonAvg") is not None
        reason = result.get("reason", result.get("verdict"))

        print(f"  verdict: {result.get('verdict')}  reason/tier: {reason}")
        print(f"  seasonAvg: {result.get('seasonAvg')}  edge: {result.get('edge')}")

        if reason == "Insufficient data" and not has_stats:
            print("  FAIL — still falling into 'Insufficient data'.")
            print("         This means _playerStats never got populated —")
            print("         the new adapter likely isn't deployed yet.")
            return False

        print("  PASS — real stats flowed into the scorer.")
        return True

    except Exception as e:
        print(f"  FAIL — request error: {e}")
        return False


def check_batch(client: httpx.Client, base_url: str) -> bool:
    section("3. Mixed batch (hitter, pitcher, unsupported stat, unknown player)")
    payload = {
        "sport": "mlb",
        "platform": "underdog",
        "props": [
            {"player_name": "Shohei Ohtani", "stat_display": "Total Bases", "line": 1.5,
             "higher_american_odds": -110, "lower_american_odds": -110},
            {"player_name": "Paul Skenes", "stat_display": "Strikeouts", "line": 6.5,
             "higher_american_odds": -115, "lower_american_odds": -105},
            {"player_name": "Shohei Ohtani", "stat_display": "Fantasy Score", "line": 25.5},
            {"player_name": "Definitely Not A Real Player", "stat_display": "Hits", "line": 0.5},
        ],
        "min_edge": 0.05,
    }
    try:
        start = time.time()
        res = client.post(f"{base_url}/predict", json=payload, timeout=45)
        elapsed = time.time() - start
        print(f"  POST /predict -> {res.status_code} ({elapsed:.1f}s, {len(payload['props'])} props)")

        if res.status_code != 200:
            print(f"  FAIL — non-200 response: {res.text[:500]}")
            return False

        data = res.json()
        summary = data.get("summary", {})
        print(f"  summary: {summary}")

        picks = data.get("picks", [])
        passes = data.get("passes", [])

        for p in picks:
            print(f"  PICK: {p.get('player_name')} / {p.get('stat_display')} "
                  f"-> {p.get('verdict')} (edge {p.get('edge')}%)")
        for p in passes:
            print(f"  PASS: {p.get('player_name')} / {p.get('stat_display')} "
                  f"-> {p.get('reason', p.get('verdict'))}")

        # Sanity expectations:
        # - The unsupported "Fantasy Score" prop and the fake player should
        #   both land in passes with "Insufficient data" / hard_reject style reasons.
        # - At least one of the two real players+stats should score successfully
        #   (not guaranteed to be a PICK above min_edge, but should NOT say
        #   "Insufficient data" if game log data exists for that player/stat).
        real_player_results = [
            p for p in (picks + passes)
            if p.get("player_name") in ("Shohei Ohtani", "Paul Skenes")
            and p.get("stat_display") in ("Total Bases", "Strikeouts")
        ]
        scored_ok = [
            p for p in real_player_results
            if p.get("reason") != "Insufficient data"
        ]

        if not scored_ok:
            print("  FAIL — both real hitter/pitcher props came back as")
            print("         'Insufficient data'. Game log fetching isn't working.")
            return False

        print(f"  PASS — {len(scored_ok)}/{len(real_player_results)} real props "
              f"were actually scored with real stats.")
        return True

    except Exception as e:
        print(f"  FAIL — request error: {e}")
        return False


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASE_URL
    base_url = base_url.rstrip("/")
    print(f"Testing API at: {base_url}")

    results = {}
    with httpx.Client() as client:
        results["health"] = check_health(client, base_url)
        if not results["health"]:
            print("\nStopping early — server isn't even responding to /health.")
            sys.exit(1)

        results["single_prop"] = check_single_prop(client, base_url)
        results["batch"] = check_batch(client, base_url)

    section("SUMMARY")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'} — {name}")

    if all(results.values()):
        print("\nAll checks passed. The fix is live and working.")
        sys.exit(0)
    else:
        print("\nSome checks failed — see details above.")
        print("Most likely cause: Railway hasn't finished redeploying yet,")
        print("or the old mlb_adapter.py is still cached. Check the Railway")
        print("deploy logs for the line: '[MLB] Enriched X of Y props with real stats'")
        sys.exit(1)


if __name__ == "__main__":
    main()
