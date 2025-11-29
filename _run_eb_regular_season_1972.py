#!/usr/bin/env python
"""
Turnkey runner for the ABL 1972 (league 200) regular-season EB pack.

This executes the canonical scripts in order, checks for failures, and verifies
that all expected outputs exist and are non-empty.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent
SEASON = 1972
LEAGUE_ID = 200


def log(msg: str) -> None:
    print(msg, flush=True)


def run_step(idx: int, label: str, cmd: list[str]) -> None:
    log(f"[STEP {idx}] Running {label}: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    if proc.returncode != 0:
        log(f"[ERROR] Step {idx} failed ({label})")
        log("stdout:")
        log(proc.stdout)
        log("stderr:")
        log(proc.stderr)
        raise SystemExit(proc.returncode)
    if proc.stdout:
        log(proc.stdout.strip())
    if proc.stderr:
        log(proc.stderr.strip())
    log(f"[STEP {idx}] Completed {label}")


def verify_csv(path: Path) -> int:
    if not path.exists():
        raise FileNotFoundError(f"Expected CSV missing: {path}")
    df = pd.read_csv(path)
    rows = len(df)
    if rows <= 0:
        raise ValueError(f"CSV has no data rows: {path}")
    return rows


def verify_md(path: Path) -> int:
    if not path.exists():
        raise FileNotFoundError(f"Expected markdown missing: {path}")
    size = path.stat().st_size
    if size <= 0:
        raise ValueError(f"Markdown file empty: {path}")
    return size


def verify_outputs() -> None:
    base = REPO_ROOT / "csv" / "out" / "almanac" / str(SEASON)
    csv_targets: dict[str, Path] = {
        "league_season_summary": base / f"league_season_summary_{SEASON}_league{LEAGUE_ID}.csv",
        "conference_summary": base / f"conference_summary_{SEASON}_league{LEAGUE_ID}.csv",
        "division_summary": base / f"division_summary_{SEASON}_league{LEAGUE_ID}.csv",
        "team_monthly_momentum": base / f"team_monthly_momentum_{SEASON}_league{LEAGUE_ID}.csv",
        "team_weekly_momentum": base / f"team_weekly_momentum_{SEASON}_league{LEAGUE_ID}.csv",
        "half_summary": base / f"half_summary_{SEASON}_league{LEAGUE_ID}.csv",
        "flashback_story_candidates": base / f"flashback_story_candidates_{SEASON}_league{LEAGUE_ID}.csv",
        "player_batting": base / f"player_batting_{SEASON}_league{LEAGUE_ID}.csv",
        "player_pitching": base / f"player_pitching_{SEASON}_league{LEAGUE_ID}.csv",
        "player_hitting_leaders": base / f"player_hitting_leaders_{SEASON}_league{LEAGUE_ID}.csv",
        "player_pitching_leaders": base / f"player_pitching_leaders_{SEASON}_league{LEAGUE_ID}.csv",
        "player_top_players": base / f"player_top_players_{SEASON}_league{LEAGUE_ID}.csv",
        "player_top_game_performances": base / f"player_top_game_performances_{SEASON}_league{LEAGUE_ID}.csv",
        "preseason_player_predictions": base / f"preseason_player_predictions_{SEASON}_league{LEAGUE_ID}.csv",
        "positional_strength_teams": base / f"positional_strength_teams_{SEASON}_league{LEAGUE_ID}.csv",
        "positional_strength_positions": base / f"positional_strength_positions_{SEASON}_league{LEAGUE_ID}.csv",
        "player_financials": base / f"player_financials_{SEASON}_league{LEAGUE_ID}.csv",
        "transactions": base / f"transactions_{SEASON}_league{LEAGUE_ID}.csv",
        "player_top_prospects": base / f"player_top_prospects_{SEASON}_league{LEAGUE_ID}.csv",
    }
    md_targets: dict[str, Path] = {
        "eb_flashback_brief": base / f"eb_flashback_brief_{SEASON}_league{LEAGUE_ID}.md",
        "eb_player_spotlights": base / f"eb_player_spotlights_{SEASON}_league{LEAGUE_ID}.md",
        "eb_player_leaders": base / f"eb_player_leaders_{SEASON}_league{LEAGUE_ID}.md",
        "eb_player_context": base / f"eb_player_context_{SEASON}_league{LEAGUE_ID}.md",
    }

    log("[VERIFY] Checking CSV outputs...")
    counts = {k: verify_csv(p) for k, p in csv_targets.items()}
    log("[VERIFY] Checking markdown outputs...")
    md_sizes = {k: verify_md(p) for k, p in md_targets.items()}

    log("[SUMMARY] CSV row counts:")
    for k, v in counts.items():
        log(f"  - {k}: {v} rows")
    log("[SUMMARY] Markdown sizes (bytes):")
    for k, v in md_sizes.items():
        log(f"  - {k}: {v}")


def main() -> int:
    py = sys.executable
    scripts: list[tuple[str, list[str]]] = [
        ("Extract core almanac HTML", [py, "csv/abl_scripts/z_abl_almanac_league200_extract_core.py", "--season", str(SEASON), "--league-id", str(LEAGUE_ID)]),
        ("League season summary", [py, "csv/abl_scripts/z_abl_almanac_league_season_summary.py", "--season", str(SEASON), "--league-id", str(LEAGUE_ID)]),
        ("League 4k summary", [py, "csv/abl_scripts/z_abl_almanac_league_4k_summary.py", "--season", str(SEASON), "--league-id", str(LEAGUE_ID)]),
        ("Momentum 3k summary", [py, "csv/abl_scripts/z_abl_almanac_momentum_3k_summary.py", "--season", str(SEASON), "--league-id", str(LEAGUE_ID)]),
        ("Flashback story pack", [py, "csv/abl_scripts/z_abl_almanac_flashback_story_pack.py", "--season", str(SEASON), "--league-id", str(LEAGUE_ID)]),
        ("EB flashback brief", [py, "csv/abl_scripts/z_abl_eb_flashback_brief_1972.py", "--season", str(SEASON), "--league-id", str(LEAGUE_ID)]),
        ("Player stats extract", [py, "csv/abl_scripts/z_abl_almanac_player_stats_extract.py", "--season", str(SEASON), "--league-id", str(LEAGUE_ID)]),
        ("Player leaderboards", [py, "csv/abl_scripts/z_abl_almanac_player_leaderboards.py", "--season", str(SEASON), "--league-id", str(LEAGUE_ID)]),
        ("Player context extract", [py, "csv/abl_scripts/z_abl_almanac_player_context_extract.py", "--season", str(SEASON), "--league-id", str(LEAGUE_ID)]),
        ("EB player spotlights", [py, "csv/abl_scripts/z_abl_eb_player_spotlights_1972.py", "--season", str(SEASON), "--league-id", str(LEAGUE_ID)]),
        ("EB player leaders brief", [py, "csv/abl_scripts/z_abl_eb_player_leaders_1972.py", "--season", str(SEASON), "--league-id", str(LEAGUE_ID)]),
        ("EB player context brief", [py, "csv/abl_scripts/z_abl_eb_player_context_1972.py", "--season", str(SEASON), "--league-id", str(LEAGUE_ID)]),
    ]

    for idx, (label, cmd) in enumerate(scripts, start=1):
        run_step(idx, label, cmd)

    verify_outputs()
    log("[OK] EB 1972 regular-season pack built successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
