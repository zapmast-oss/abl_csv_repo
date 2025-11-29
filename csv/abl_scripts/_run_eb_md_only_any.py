#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import subprocess
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run EB markdown-generation steps for any season/league.",
        epilog="Example: python csv/abl_scripts/_run_eb_md_only_any.py --season 1980 --league-id 200",
    )
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--league-id", type=int, default=200)
    args = parser.parse_args()

    season = args.season
    league_id = args.league_id

    repo_root = Path(__file__).resolve().parents[2]
    scripts_root = repo_root / "csv" / "abl_scripts"
    almanac_root = repo_root / "csv" / "out" / "almanac" / str(season)

    log(f"[INFO] Running EB MD-only pipeline for season {season}, league {league_id}")

    steps = [
        (
            "EB flashback brief",
            [
                sys.executable,
                str(scripts_root / "z_abl_eb_flashback_brief_1972.py"),
                "--season",
                str(season),
                "--league-id",
                str(league_id),
            ],
        ),
        (
            "EB player spotlights",
            [
                sys.executable,
                str(scripts_root / "z_abl_eb_player_spotlights_1972.py"),
                "--season",
                str(season),
                "--league-id",
                str(league_id),
            ],
        ),
        (
            "EB schedule context",
            [
                sys.executable,
                str(scripts_root / "z_abl_eb_schedule_context_1972.py"),
                "--season",
                str(season),
                "--league-id",
                str(league_id),
            ],
        ),
        (
            "EB player leaders",
            [
                sys.executable,
                str(scripts_root / "z_abl_eb_player_leaders_1972.py"),
                "--season",
                str(season),
                "--league-id",
                str(league_id),
            ],
        ),
        (
            "EB player context",
            [
                sys.executable,
                str(scripts_root / "z_abl_eb_player_context_1972.py"),
                "--season",
                str(season),
                "--league-id",
                str(league_id),
            ],
        ),
    ]

    for label, cmd in steps:
        run_step(label, cmd)

    if not almanac_root.exists():
        log(f"[WARN] Almanac output directory not found: {almanac_root}")
    else:
        md_files = sorted(almanac_root.glob("eb_*.md"))
        if not md_files:
            log(f"[WARN] No EB markdown files found in {almanac_root} (pattern eb_*.md)")
        else:
            log("[SUMMARY] EB markdown outputs:")
            for path in md_files:
                try:
                    size = path.stat().st_size
                    log(f"  - {path.name}: {size} bytes")
                except OSError as exc:
                    log(f"  - {path.name}: error reading size ({exc})")

    log(f"[OK] EB MD-only pipeline completed for season {season}, league {league_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
