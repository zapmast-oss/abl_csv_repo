"""Build the prior-week ABL team snapshot directly from the 1981 game log."""

# Example:
#   py csv/abl_scripts/z_abl_seed_prev_from_games_1981.py --asof 1981-05-03

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

from abl_config import LEAGUE_ID, RAW_CSV_ROOT, TEAM_IDS

CSV_ROOT = Path(__file__).resolve().parents[1]
STAR_DIR = CSV_ROOT / "out" / "star_schema"
CSV_OUT_DIR = CSV_ROOT / "out" / "csv_out"
OUTPUT_PATH = STAR_DIR / "fact_team_reporting_1981_prev.csv"
CURRENT_SNAPSHOT_PATH = STAR_DIR / "fact_team_reporting_1981_current.csv"
GAMES_PATH = RAW_CSV_ROOT / "games.csv"
DIM_TEAM_PARK_PATH = STAR_DIR / "dim_team_park.csv"
DIM_MANAGERS_PATH = CSV_OUT_DIR / "z_ABL_DIM_Managers.csv"
DEFAULT_CUTOFF = "1981-05-03"
TEAM_SET = set(TEAM_IDS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed fact_team_reporting_1981_prev.csv from the regular-season game log."
    )
    parser.add_argument(
        "--asof",
        help="Cutoff date (YYYY-MM-DD) for including games. Defaults to 1981-05-03.",
    )
    return parser.parse_args()


def parse_cutoff(value: str | None) -> pd.Timestamp:
    target = value or DEFAULT_CUTOFF
    try:
        ts = pd.to_datetime(target, format="%Y-%m-%d", errors="raise")
    except ValueError as exc:
        raise SystemExit(f"Invalid --asof date '{target}'; use YYYY-MM-DD.") from exc
    return ts.normalize()


def load_schema_columns() -> List[str]:
    if not CURRENT_SNAPSHOT_PATH.exists():
        print(f"ERROR: Current snapshot not found at {CURRENT_SNAPSHOT_PATH}", file=sys.stderr)
        sys.exit(1)
    existing = pd.read_csv(CURRENT_SNAPSHOT_PATH, nrows=0)
    schema = list(existing.columns)
    print(f"Current snapshot column order ({len(schema)} cols): {schema}")
    return schema


def load_games_through(cutoff: pd.Timestamp) -> pd.DataFrame:
    if not GAMES_PATH.exists():
        print(f"ERROR: games.csv not found at {GAMES_PATH}", file=sys.stderr)
        sys.exit(1)
    games = pd.read_csv(GAMES_PATH)
    print(f"games.csv columns ({len(games.columns)} cols): {list(games.columns)}")
    required = {"date", "league_id", "home_team", "away_team", "runs0", "runs1", "played", "game_type"}
    missing = required.difference(games.columns)
    if missing:
        raise SystemExit(f"games.csv missing required columns: {sorted(missing)}")

    games["date"] = pd.to_datetime(games["date"], errors="coerce").dt.normalize()
    games = games.dropna(subset=["date"])
    games = games[
        (games["league_id"] == LEAGUE_ID)
        & (games["home_team"].isin(TEAM_SET))
        & (games["away_team"].isin(TEAM_SET))
        & (games["played"] == 1)
        & (games["game_type"] == 0)
        & (games["date"] <= cutoff)
    ].copy()
    if games.empty:
        print(
            f"WARNING: No league_id={LEAGUE_ID} regular-season games found on/before {cutoff.date()}",
            file=sys.stderr,
        )
        sys.exit(1)
    return games


def build_ledger_rows(row: pd.Series) -> List[dict]:
    """Return two ledger rows (home/away) using the same logic as z_abl_week_miner."""
    home_id = int(row["home_team"])
    away_id = int(row["away_team"])
    away_runs = int(row["runs0"])
    home_runs = int(row["runs1"])

    home_win = int(home_runs > away_runs)
    away_win = int(away_runs > home_runs)
    tie = int(home_runs == away_runs)

    return [
        {
            "team_id": home_id,
            "runs_scored": home_runs,
            "runs_allowed": away_runs,
            "wins": home_win,
            "losses": away_win,
            "ties": tie,
            "games": 1,
        },
        {
            "team_id": away_id,
            "runs_scored": away_runs,
            "runs_allowed": home_runs,
            "wins": away_win,
            "losses": home_win,
            "ties": tie,
            "games": 1,
        },
    ]


def compute_team_records(games: pd.DataFrame) -> pd.DataFrame:
    ledger_rows: List[dict] = []
    for _, row in games.iterrows():
        ledger_rows.extend(build_ledger_rows(row))
    ledger = pd.DataFrame(ledger_rows)
    grouped = ledger.groupby("team_id", as_index=False).sum()

    records = pd.DataFrame({"team_id": sorted(TEAM_SET)}).merge(
        grouped, on="team_id", how="left"
    )
    fill_cols = ["games", "wins", "losses", "ties", "runs_scored", "runs_allowed"]
    records[fill_cols] = records[fill_cols].fillna(0).astype(int)
    decisions = records["wins"] + records["losses"]
    records["win_pct"] = records["wins"] / decisions.where(decisions != 0, 1)
    records.loc[decisions == 0, "win_pct"] = 0.0
    records["run_diff"] = records["runs_scored"] - records["runs_allowed"]
    return records


def load_team_metadata() -> pd.DataFrame:
    if not DIM_TEAM_PARK_PATH.exists():
        print(f"ERROR: dim_team_park.csv missing at {DIM_TEAM_PARK_PATH}", file=sys.stderr)
        sys.exit(1)
    dim = pd.read_csv(DIM_TEAM_PARK_PATH)

    team_id_col = None
    for candidate in ["ID", "team_id", "Team ID"]:
        if candidate in dim.columns:
            team_id_col = candidate
            break
    if team_id_col is None:
        raise SystemExit("dim_team_park.csv missing a recognizable team ID column")

    dim = dim[dim[team_id_col].isin(TEAM_SET)].copy()
    dim[team_id_col] = dim[team_id_col].astype(int)
    if set(dim[team_id_col]) != TEAM_SET:
        raise SystemExit("dim_team_park.csv does not cover all 24 canonical teams")

    rename_map = {}
    if "Abbr" in dim.columns:
        rename_map["Abbr"] = "team_abbr"
    if "Team Name" in dim.columns:
        rename_map["Team Name"] = "team_name"
    elif "Name" in dim.columns:
        rename_map["Name"] = "team_name"
    if "SL" in dim.columns:
        rename_map["SL"] = "sub_league"
    if "DIV" in dim.columns:
        rename_map["DIV"] = "division"
    if "Park" in dim.columns:
        rename_map["Park"] = "ballpark_name"

    selected_cols = [team_id_col] + list(rename_map.keys())
    teams = dim[selected_cols].rename(columns=rename_map)
    teams = teams.rename(columns={team_id_col: "team_id"})
    for required in ["team_abbr", "team_name"]:
        if required not in teams.columns:
            raise SystemExit(f"dim_team_park.csv missing required column '{required}'")
    if "ballpark_name" not in teams.columns:
        teams["ballpark_name"] = ""
    return teams


def load_manager_metadata() -> pd.DataFrame:
    if not DIM_MANAGERS_PATH.exists():
        print(f"ERROR: Manager dimension missing at {DIM_MANAGERS_PATH}", file=sys.stderr)
        sys.exit(1)
    managers = pd.read_csv(DIM_MANAGERS_PATH)

    team_col = None
    for candidate in ["current_team_id", "team_id", "Team ID"]:
        if candidate in managers.columns:
            team_col = candidate
            break
    if team_col is None:
        raise SystemExit("z_ABL_DIM_Managers.csv missing team assignment column")

    managers = managers[managers[team_col].isin(TEAM_SET)].copy()
    managers[team_col] = managers[team_col].astype(int)
    sort_cols = [col for col in ["last_year", "total_seasons", "career_wins"] if col in managers.columns]
    if sort_cols:
        managers = managers.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    managers = managers.drop_duplicates(team_col, keep="first")
    if len(managers) != len(TEAM_SET):
        raise SystemExit("Manager dimension does not provide exactly one manager per team")

    rename_map = {}
    if "manager_id" in managers.columns:
        rename_map["manager_id"] = "manager_id"
    if "full_name" in managers.columns:
        rename_map["full_name"] = "manager_name"
    elif {"first_name", "last_name"}.issubset(managers.columns):
        managers["manager_name"] = (
            managers["first_name"].astype(str).str.strip()
            + " "
            + managers["last_name"].astype(str).str.strip()
        ).str.strip()
        rename_map["manager_name"] = "manager_name"
    if "career_wins" in managers.columns:
        rename_map["career_wins"] = "manager_career_wins"
    if "career_losses" in managers.columns:
        rename_map["career_losses"] = "manager_career_losses"
    if "career_win_pct" in managers.columns:
        rename_map["career_win_pct"] = "manager_career_win_pct"
    if "total_seasons" in managers.columns:
        rename_map["total_seasons"] = "manager_total_seasons"
    if "titles_won" in managers.columns:
        rename_map["titles_won"] = "manager_titles"

    subset_cols = [team_col] + list(rename_map.keys())
    managers = managers[subset_cols].rename(columns=rename_map)
    managers = managers.rename(columns={team_col: "team_id"})
    missing_cols = {"manager_id", "manager_name"} - set(managers.columns)
    if missing_cols:
        raise SystemExit(f"Manager dimension missing required columns: {sorted(missing_cols)}")
    return managers


def apply_division_context(df: pd.DataFrame) -> pd.DataFrame:
    if not {"division", "wins", "losses", "win_pct"}.issubset(df.columns):
        df["games_back"] = pd.NA
        df["division_rank"] = pd.NA
        return df

    df = df.copy()
    df["games_back"] = 0.0
    df["division_rank"] = 0

    grouping_fields = ["sub_league", "division"] if "sub_league" in df.columns else ["division"]
    for _, group in df.groupby(grouping_fields):
        ordered = group.sort_values(
            ["win_pct", "run_diff", "wins"],
            ascending=[False, False, False],
        )
        leader = ordered.iloc[0]
        leader_wins = leader["wins"]
        leader_losses = leader["losses"]
        for rank, (idx, row) in enumerate(ordered.iterrows(), start=1):
            gb = ((leader_wins - row["wins"]) + (row["losses"] - leader_losses)) / 2.0
            if (row["wins"] + row["losses"]) == 0:
                gb = 0.0
            df.loc[idx, "games_back"] = round(gb, 1)
            df.loc[idx, "division_rank"] = rank

    df["division_rank"] = df["division_rank"].astype(int)
    return df


def align_to_schema(df: pd.DataFrame, schema: List[str]) -> pd.DataFrame:
    aligned = df.copy()
    aligned = aligned.rename(columns={"team_id": "ID"})
    for col in schema:
        if col not in aligned.columns:
            aligned[col] = pd.NA
    aligned = aligned[schema]
    return aligned


def main() -> None:
    args = parse_args()
    cutoff = parse_cutoff(args.asof)
    schema = load_schema_columns()

    games = load_games_through(cutoff)
    records = compute_team_records(games)
    teams = load_team_metadata()
    snapshot = teams.merge(records, on="team_id", how="inner", validate="one_to_one")
    snapshot = apply_division_context(snapshot)
    managers = load_manager_metadata()
    snapshot = snapshot.merge(managers, on="team_id", how="left", validate="one_to_one")

    if len(snapshot) != len(TEAM_SET):
        raise SystemExit("Snapshot merge did not yield 24 teams")
    if snapshot["manager_name"].isna().any():
        raise SystemExit("Manager info missing for one or more teams")

    summary_view = snapshot[
        ["team_abbr", "team_name", "wins", "losses", "win_pct", "run_diff"]
    ].copy()

    ordered = align_to_schema(snapshot, schema)
    ordered.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"Previous snapshot cutoff date: {cutoff.date()}")
    print(f"Wrote {len(ordered)} teams to {OUTPUT_PATH}")
    print("Standings preview:")
    print(summary_view.sort_values("team_abbr").head())


if __name__ == "__main__":
    main()
