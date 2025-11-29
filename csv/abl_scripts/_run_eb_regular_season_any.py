import argparse
import os
import sys
import subprocess
import csv
from pathlib import Path

# Generic EB regular-season pack runner for any ABL season (league_id=200).
# Usage (from repo root):
#   python csv/abl_scripts/_run_eb_regular_season_any.py --season 1973 --league-id 200

def run_step(step_num, title, script_path, extra_args):
    cmd = [sys.executable, script_path] + extra_args
    print(f"[STEP {step_num}] Running {title}: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[ERROR] Step {step_num} FAILED: {title}")
        sys.exit(result.returncode)
    print(f"[STEP {step_num}] Completed {title}")


def ensure_time_slices_enriched(season: int, league_id: int, root: Path) -> None:
    season_dir = root / "csv" / "out" / "almanac" / str(season)
    expected = [
        season_dir / f"team_monthly_summary_{season}_league{league_id}_enriched.csv",
        season_dir / f"team_weekly_summary_{season}_league{league_id}_enriched.csv",
        season_dir / f"series_summary_{season}_league{league_id}_enriched.csv",
    ]
    missing = [p for p in expected if not p.exists()]
    if not missing:
        print("[INFO] Enriched time-slice files already present.")
        return
    print("[STEP TS] Enriched time-slice files missing; running _run_almanac_time_slices_any.py")
    cmd = [
        sys.executable,
        "csv/abl_scripts/_run_almanac_time_slices_any.py",
        "--season",
        str(season),
        "--league-id",
        str(league_id),
    ]
    result = subprocess.run(cmd, cwd=root)
    if result.returncode != 0:
        print("[ERROR] Failed to generate enriched time-slice files.")
        sys.exit(result.returncode)


def count_csv_rows(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            # assume header + data, so subtract one if at least one row
            rows = list(reader)
            if not rows:
                return 0
            return max(len(rows) - 1, 0)
    except Exception as e:
        print(f"[WARN] Could not count rows for {path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Run EB regular-season pack for any ABL season.")
    parser.add_argument("--season", type=int, required=True, help="Season year (e.g. 1972, 1973, ...)")
    parser.add_argument("--league-id", type=int, default=200, help="League ID (default 200)")
    args = parser.parse_args()

    season = args.season
    league_id = args.league_id

    print(f"[INFO] Running EB regular-season pipeline for season {season}, league {league_id}")
    print("[INFO] Current working directory:", os.getcwd())
    repo_root = Path(__file__).resolve().parents[2]

    # Ensure enriched time slices exist before momentum step
    ensure_time_slices_enriched(season, league_id, repo_root)

    # 14 core steps, mirroring the 1972 runner but parameterized by season.
    steps = [
        (1,  "Extract core almanac HTML",
            "csv/abl_scripts/z_abl_almanac_league200_extract_core.py",
            ["--season", str(season), "--league-id", str(league_id)]),
        (2,  "League season summary",
            "csv/abl_scripts/z_abl_almanac_league_season_summary.py",
            ["--season", str(season), "--league-id", str(league_id)]),
        (3,  "League 4k summary",
            "csv/abl_scripts/z_abl_almanac_league_4k_summary.py",
            ["--season", str(season), "--league-id", str(league_id)]),
        (4,  "Momentum 3k summary",
            "csv/abl_scripts/z_abl_almanac_momentum_3k_summary.py",
            ["--season", str(season), "--league-id", str(league_id)]),
        (5,  "Flashback story pack",
            "csv/abl_scripts/z_abl_almanac_flashback_story_pack.py",
            ["--season", str(season), "--league-id", str(league_id)]),
        # Script filename still has "1972" but is parameterized by --season
        (6,  "EB flashback brief",
            "csv/abl_scripts/z_abl_eb_flashback_brief_1972.py",
            ["--season", str(season), "--league-id", str(league_id)]),
        (7,  "Player stats extract",
            "csv/abl_scripts/z_abl_almanac_player_stats_extract.py",
            ["--season", str(season), "--league-id", str(league_id)]),
        (8,  "Player leaderboards",
            "csv/abl_scripts/z_abl_almanac_player_leaderboards.py",
            ["--season", str(season), "--league-id", str(league_id)]),
        (9,  "Player context extract",
            "csv/abl_scripts/z_abl_almanac_player_context_extract.py",
            ["--season", str(season), "--league-id", str(league_id)]),
        # EB player / schedule briefs (filenames still include 1972)
        (10, "EB player spotlights",
            "csv/abl_scripts/z_abl_eb_player_spotlights_1972.py",
            ["--season", str(season), "--league-id", str(league_id)]),
        (11, "Schedule extract",
            "csv/abl_scripts/z_abl_almanac_schedule_extract.py",
            ["--season", str(season), "--league-id", str(league_id)]),
        (12, "EB schedule context",
            "csv/abl_scripts/z_abl_eb_schedule_context_1972.py",
            ["--season", str(season), "--league-id", str(league_id)]),
        (13, "EB player leaders",
            "csv/abl_scripts/z_abl_eb_player_leaders_1972.py",
            ["--season", str(season), "--league-id", str(league_id)]),
        (14, "EB player context",
            "csv/abl_scripts/z_abl_eb_player_context_1972.py",
            ["--season", str(season), "--league-id", str(league_id)]),
    ]

    for num, title, script, extra_args in steps:
        run_step(num, title, script, extra_args)

    # Verification block – same idea as the 1972 runner, but generic.
    base = os.path.join("csv", "out", "almanac", str(season))

    csv_outputs = [
        f"league_season_summary_{season}_league{league_id}.csv",
        f"conference_summary_{season}_league{league_id}.csv",
        f"division_summary_{season}_league{league_id}.csv",
        f"team_monthly_momentum_{season}_league{league_id}.csv",
        f"team_weekly_momentum_{season}_league{league_id}.csv",
        f"half_summary_{season}_league{league_id}.csv",
        f"flashback_story_candidates_{season}_league{league_id}.csv",
        f"player_batting_{season}_league{league_id}.csv",
        f"player_pitching_{season}_league{league_id}.csv",
        f"player_hitting_leaders_{season}_league{league_id}.csv",
        f"player_pitching_leaders_{season}_league{league_id}.csv",
        f"player_top_players_{season}_league{league_id}.csv",
        f"player_top_game_performances_{season}_league{league_id}.csv",
        f"preseason_player_predictions_{season}_league{league_id}.csv",
        f"positional_strength_teams_{season}_league{league_id}.csv",
        f"positional_strength_positions_{season}_league{league_id}.csv",
        f"player_financials_{season}_league{league_id}.csv",
        f"transactions_{season}_league{league_id}.csv",
        f"player_top_prospects_{season}_league{league_id}.csv",
        f"team_schedule_{season}_league{league_id}.csv",
        f"schedule_evaluator_{season}_league{league_id}.csv",
    ]

    md_outputs = [
        f"eb_flashback_brief_{season}_league{league_id}.md",
        f"eb_player_spotlights_{season}_league{league_id}.md",
        f"eb_schedule_context_{season}_league{league_id}.md",
        f"eb_player_leaders_{season}_league{league_id}.md",
        f"eb_player_context_{season}_league{league_id}.md",
    ]

    print("[VERIFY] Checking CSV outputs...")
    for name in csv_outputs:
        path = os.path.join(base, name)
        rows = count_csv_rows(path)
        if rows is None:
            print(f"  - {name}: MISSING")
        else:
            print(f"  - {name}: {rows} data rows")

    print("[VERIFY] Checking markdown outputs...")
    for name in md_outputs:
        path = os.path.join(base, name)
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"  - {name}: {size} bytes")
        else:
            print(f"  - {name}: MISSING")

    print(f"[OK] EB regular-season pack built successfully for season {season}, league {league_id}.")


if __name__ == "__main__":
    main()
