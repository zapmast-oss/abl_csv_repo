#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge EB markdown briefs for a season/league into one file."
    )
    parser.add_argument("--season", type=int, required=True, help="Season year, e.g., 1980")
    parser.add_argument("--league-id", type=int, default=200, help="League ID (default 200)")
    args = parser.parse_args()

    season = args.season
    league_id = args.league_id

    repo_root = Path(__file__).resolve().parents[2]
    almanac_root = repo_root / "csv" / "out" / "almanac"
    season_dir = almanac_root / str(season)

    if not season_dir.exists():
        log(f"[ERROR] Almanac directory not found for season {season}, league {league_id}: {season_dir}")
        return 1

    pattern = f"eb_*_{season}_league{league_id}.md"
    md_files = sorted(season_dir.glob(pattern), key=lambda p: p.name.lower())

    if not md_files:
        log(
            f"[ERROR] No EB markdown files found for season {season}, league {league_id} in {season_dir}"
        )
        return 1

    output_path = season_dir / f"eb_regular_season_pack_{season}_league{league_id}.md"

    log(f"[INFO] Merging {len(md_files)} files into {output_path.name}")

    try:
        with output_path.open("w", encoding="utf-8") as out_f:
            out_f.write(f"# ABL {season} Regular Season \u2013 EB Pack (League {league_id})\n\n")
            for md_file in md_files:
                out_f.write("\n---\n\n")
                out_f.write(f"## {md_file.stem}\n\n")
                with md_file.open("r", encoding="utf-8") as in_f:
                    out_f.write(in_f.read())
                out_f.write("\n")

        log(
            f"[OK] Merged {len(md_files)} EB markdown files into eb_regular_season_pack_{season}_league{league_id}.md"
        )
        return 0
    except Exception as exc:  # pragma: no cover
        log(f"[ERROR] Failed to merge EB markdown files: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
