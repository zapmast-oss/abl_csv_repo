import json
from pathlib import Path

import pandas as pd

from abl_config import TEAM_IDS
from z_abl_team_identity import build_team_identity_card

CSV_ROOT = Path(__file__).resolve().parents[1]
STAR_DIR = CSV_ROOT / "out" / "star_schema"
TEXT_OUT_DIR = CSV_ROOT / "out" / "text_out"
DIM_TEAM_PARK_PATH = STAR_DIR / "dim_team_park.csv"
PACK_JSON_PATH = TEXT_OUT_DIR / "eb_team_identity_pack_1981.json"


def load_team_abbrs() -> pd.DataFrame:
    if not DIM_TEAM_PARK_PATH.exists():
        raise SystemExit(f"dim_team_park.csv not found at {DIM_TEAM_PARK_PATH}")
    dim = pd.read_csv(DIM_TEAM_PARK_PATH)

    id_col = None
    for cand in ["ID", "team_id", "Team ID"]:
        if cand in dim.columns:
            id_col = cand
            break
    if id_col is None:
        raise SystemExit("dim_team_park.csv missing required team ID column for team identity pack.")

    abbr_col = None
    if "Abbr" in dim.columns:
        abbr_col = "Abbr"
    else:
        for col in dim.columns:
            if "abbr" in col.lower():
                abbr_col = col
                break
    if abbr_col is None:
        raise SystemExit("dim_team_park.csv missing required Abbr column for team identity pack.")

    filtered = dim[dim[id_col].isin(TEAM_IDS)].copy()
    filtered[id_col] = filtered[id_col].astype(int)
    if filtered.empty or filtered[id_col].nunique() != len(TEAM_IDS):
        raise SystemExit("dim_team_park.csv missing required ID/Abbr rows for team identity pack.")

    filtered = filtered.rename(columns={id_col: "team_id", abbr_col: "team_abbr"})
    filtered["team_abbr"] = filtered["team_abbr"].astype(str).strip()
    return filtered[["team_id", "team_abbr"]]


def main() -> None:
    teams = load_team_abbrs().sort_values("team_abbr")

    cards = []
    for _, row in teams.iterrows():
        tid = int(row["team_id"])
        abbr = str(row["team_abbr"])
        cards.append(
            {
                "team_id": tid,
                "team_abbr": abbr,
                "identity": build_team_identity_card(abbr),
            }
        )

    TEXT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    with PACK_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(cards, f, indent=2, ensure_ascii=False, default=str)
    print(f"EB team identity pack written to: {PACK_JSON_PATH}")


if __name__ == "__main__":
    main()
