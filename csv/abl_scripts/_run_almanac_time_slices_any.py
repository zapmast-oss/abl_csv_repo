#!/usr/bin/env python
"""
Run almanac scores pipeline + time-slice enrichment for any season/league.

Usage (from repo root):
  python csv/abl_scripts/_run_almanac_time_slices_any.py --season 1980 --league-id 200
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def log(msg: str) -> None:
    print(msg, flush=True)


def run_step(label: str, cmd: list[str]) -> None:
    log(f"[INFO] Running {label}: {' '.join(cmd)}")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        log(f"[ERROR] {label} failed with code {proc.returncode}")
        sys.exit(proc.returncode)
    log(f"[INFO] Completed {label}")


def count_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = sum(1 for _ in reader)
        return max(rows - 1, 0)
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run scores pipeline and enrich time slices for any season/league.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--league-id", type=int, default=200)
    args = parser.parse_args()

    season = args.season
    league_id = args.league_id

    repo_root = Path(__file__).resolve().parents[2]
    scripts_root = repo_root / "csv" / "abl_scripts"

    run_step(
        "Scores pipeline",
        [sys.executable, str(scripts_root / "z_abl_almanac_scores_pipeline.py"), "--season", str(season), "--league-id", str(league_id)],
    )
    run_step(
        "Time slices enriched",
        [sys.executable, str(scripts_root / "z_abl_almanac_time_slices_enriched.py"), "--season", str(season), "--league-id", str(league_id)],
    )

    base_out = repo_root / "csv" / "out" / "almanac" / str(season)
    targets = [
        f"team_monthly_summary_{season}_league{league_id}_enriched.csv",
        f"team_weekly_summary_{season}_league{league_id}_enriched.csv",
        f"series_summary_{season}_league{league_id}_enriched.csv",
    ]

    log("[SUMMARY] Enriched outputs:")
    for fname in targets:
        path = base_out / fname
        rows = count_rows(path)
        if rows is None:
            log(f"  - {fname}: MISSING")
        else:
            log(f"  - {fname}: {rows} rows")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
