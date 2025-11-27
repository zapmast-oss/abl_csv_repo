"""
z_pack_game_broadcast_1981.py

Build eb_game_pack_1981-05-11_CHI_at_MIA.csv from star_schema + ABL stats.
Target path:
  csv/out/csv_out/season_1981/eb_game_pack_1981-05-11_CHI_at_MIA.csv
"""

import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STAR_SCHEMA_DIR = REPO_ROOT / "csv" / "out" / "star_schema"
OUT_DIR = REPO_ROOT / "csv" / "out" / "csv_out" / "season_1981"

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # TODO: load needed tables from STAR_SCHEMA_DIR
    # TODO: join/compute into broadcast pack
    # TODO: write eb_game_pack_1981-05-11_CHI_at_MIA.csv with locked schema

if __name__ == "__main__":
    main()
