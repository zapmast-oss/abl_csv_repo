#!/usr/bin/env python
"""
Extract core almanac HTML assets for a given season/league, including player stat pages.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Iterable, List


def log(msg: str) -> None:
    print(msg)


def find_zip(season: int) -> Path:
    candidates = [
        Path("data_raw/ootp_html") / f"almanac_{season}.zip",
        Path("csv/in") / f"almanac_{season}.zip",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Almanac zip not found. Tried: {candidates}")


def extract_files(zf: zipfile.ZipFile, members: Iterable[str], dest_dir: Path) -> List[str]:
    extracted = []
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in members:
        target = dest_dir / Path(name).name
        with zf.open(name) as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        extracted.append(target.name)
    return extracted


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract league core HTML (including player pages) from almanac zip.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--league-id", type=int, required=True)
    args = parser.parse_args()
    season = args.season
    league_id = args.league_id

    zip_path = find_zip(season)
    dest_dir = Path("csv/in/almanac_core") / str(season) / "leagues"
    log(f"[INFO] Using almanac zip: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        standings = [n for n in names if f"league_{league_id}_standings.html" in n]
        stats = [n for n in names if f"league_{league_id}_stats.html" in n]
        batting_players = [
            n
            for n in names
            if f"league_{league_id}" in n and "players_batting_reg_by_letter" in n.lower()
        ]
        pitching_players = [
            n
            for n in names
            if f"league_{league_id}" in n and "players_pitching_reg_by_letter" in n.lower()
        ]
        if not batting_players or not pitching_players:
            raise RuntimeError(
                f"No player batting/pitching pages found for league {league_id} in zip {zip_path}"
            )

        copied = []
        copied += extract_files(zf, standings, dest_dir)
        copied += extract_files(zf, stats, dest_dir)
        copied += extract_files(zf, batting_players, dest_dir)
        copied += extract_files(zf, pitching_players, dest_dir)

    log("[INFO] Extracted files:")
    for name in copied:
        log(f"  - {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
