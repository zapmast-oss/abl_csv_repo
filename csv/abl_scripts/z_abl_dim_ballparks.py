"""ABL DIM_BALLPARKS builder."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from abl_config import LEAGUE_ID, RAW_CSV_ROOT, TEAM_IDS
from abl_team_helper import load_abl_teams

CSV_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = CSV_ROOT / "out" / "csv_out" / "z_ABL_DIM_Ballparks.csv"
PARKS_PATH = RAW_CSV_ROOT / "parks.csv"
TEAMS_PATH = RAW_CSV_ROOT / "teams.csv"
TEAM_SET = set(TEAM_IDS)


def load_team_metadata() -> pd.DataFrame:
    teams = load_abl_teams().rename(columns={"name": "team_city"})
    teams["team_id"] = teams["team_id"].astype(int)
    teams["team_abbr"] = teams["abbr"].astype(str)
    raw = pd.read_csv(TEAMS_PATH)
    raw = raw[(raw["league_id"] == LEAGUE_ID) & (raw["team_id"].isin(TEAM_SET))]
    raw = raw[["team_id", "name", "nickname", "park_id"]].rename(columns={"name": "city"})
    merged = teams.merge(raw, on="team_id", how="inner")
    if len(merged) != len(TEAM_SET):
        missing = TEAM_SET - set(merged["team_id"])
        raise SystemExit(f"Unable to resolve park/team metadata for team_ids: {sorted(missing)}")
    merged["park_id"] = pd.to_numeric(merged["park_id"], errors="coerce").astype("Int64")
    if merged["park_id"].isna().any():
        raise SystemExit("One or more teams are missing park_id values.")
    merged["team_name"] = merged["city"].astype(str).str.strip() + " " + merged["nickname"].astype(str).str.strip()
    merged["city"] = merged["city"].astype(str).str.strip()
    merged["team_city"] = merged["team_city"].fillna(merged["city"])
    return merged


def load_parks() -> pd.DataFrame:
    if not PARKS_PATH.exists():
        raise FileNotFoundError(f"Parks file not found: {PARKS_PATH}")
    parks = pd.read_csv(PARKS_PATH)
    parks["park_id"] = pd.to_numeric(parks["park_id"], errors="coerce").astype("Int64")
    return parks


def build_dim_ballparks(teams: pd.DataFrame, parks: pd.DataFrame) -> pd.DataFrame:
    merged = teams.merge(parks, on="park_id", how="left", suffixes=("", "_park"))
    if merged["name"].isna().any():
        missing = merged.loc[merged["name"].isna(), "team_id"].tolist()
        raise SystemExit(f"Missing park rows for team_ids: {missing}")

    def pick(col_base: str):
        return merged[col_base] if col_base in merged.columns else pd.Series([pd.NA] * len(merged))

    def fence_height(idx: int):
        col = f"wall_heights{idx}"
        return merged[col] if col in merged.columns else pd.Series([pd.NA] * len(merged))

    dim = pd.DataFrame(
        {
            "team_id": merged["team_id"].astype(int),
            "team_name": merged["team_name"],
            "team_abbr": merged["team_abbr"],
            "park_id": merged["park_id"].astype(int),
            "park_name": merged["name"],
            "city": merged["city"],
            "surface": merged.get("turf", pd.Series([pd.NA] * len(merged))),
            "capacity": merged.get("capacity"),
            "lf_distance": pick("distances0"),
            "cf_distance": pick("distances3"),
            "rf_distance": pick("distances6"),
            "lf_fence_height": fence_height(0),
            "cf_fence_height": fence_height(3),
            "rf_fence_height": fence_height(6),
            "pf_runs": merged.get("avg"),
            "pf_hr": merged.get("hr"),
        }
    )

    if dim["surface"].notna().any():
        dim["surface"] = dim["surface"].map({0: "Grass", 1: "Artificial"}).fillna("Unknown")
    else:
        dim["surface"] = pd.NA

    numeric_cols = [
        "capacity",
        "lf_distance",
        "cf_distance",
        "rf_distance",
        "lf_fence_height",
        "cf_fence_height",
        "rf_fence_height",
    ]
    for col in numeric_cols:
        dim[col] = pd.to_numeric(dim[col], errors="coerce")
    dim["pf_runs"] = pd.to_numeric(dim["pf_runs"], errors="coerce")
    dim["pf_hr"] = pd.to_numeric(dim["pf_hr"], errors="coerce")
    dim["comments"] = ""

    columns = [
        "team_id",
        "team_name",
        "team_abbr",
        "park_id",
        "park_name",
        "city",
        "surface",
        "capacity",
        "lf_distance",
        "cf_distance",
        "rf_distance",
        "lf_fence_height",
        "cf_fence_height",
        "rf_fence_height",
        "pf_runs",
        "pf_hr",
        "comments",
    ]
    return dim[columns].sort_values("team_id").reset_index(drop=True)


def write_output(df: pd.DataFrame) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)


def validate_output() -> None:
    df = pd.read_csv(OUT_PATH)
    if len(df) != len(TEAM_SET):
        raise SystemExit(f"Validation failed: expected {len(TEAM_SET)} rows, found {len(df)}.")
    if df["team_id"].duplicated().any():
        dups = df[df["team_id"].duplicated()]["team_id"].tolist()
        raise SystemExit(f"Validation failed: duplicate team_id values {dups}.")
    invalid = set(df["team_id"]) - TEAM_SET
    if invalid:
        raise SystemExit(f"Validation failed: unexpected team_id values {sorted(invalid)}.")


def main() -> None:
    try:
        teams = load_team_metadata()
        parks = load_parks()
        dim = build_dim_ballparks(teams, parks)
        write_output(dim)
        validate_output()
        print(f"DIM_BALLPARKS: wrote {len(dim)} rows to {OUT_PATH.name}")
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
