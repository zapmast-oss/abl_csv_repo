#!/usr/bin/env python
"""
Run the EB regular-season pipeline across a range of seasons.

Example:
  python csv/abl_scripts/_run_eb_regular_season_range.py --start-season 1973 --end-season 1980 --league-id 200
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def log(msg: str) -> None:
    print(msg, flush=True)


def run_season(season: int, league_id: int, root: Path) -> bool:
    cmd = [
        sys.executable,
        "csv/abl_scripts/_run_eb_regular_season_any.py",
        "--season",
        str(season),
        "--league-id",
        str(league_id),
    ]
    log("=" * 30)
    log(f"Running EB pipeline for {season}")
    log("=" * 30)
    proc = subprocess.run(cmd, cwd=root)
    if proc.returncode != 0:
        log(f"[ERROR] Season {season} failed with code {proc.returncode}")
        return False
    log(f"[OK] Season {season} completed")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Run EB pipeline over a range of seasons.")
    parser.add_argument("--start-season", type=int, required=True)
    parser.add_argument("--end-season", type=int, required=True)
    parser.add_argument("--league-id", type=int, default=200)
    args = parser.parse_args()

    start = args.start_season
    end = args.end_season
    league_id = args.league_id
    root = Path(__file__).resolve().parents[2]

    successes: list[int] = []
    failures: list[int] = []

    for season in range(start, end + 1):
        ok = run_season(season, league_id, root)
        if ok:
            successes.append(season)
        else:
            failures.append(season)

    log("[SUMMARY]")
    log(f"  Success: {successes if successes else 'None'}")
    log(f"  Failed: {failures if failures else 'None'}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
