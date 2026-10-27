"""ABL DIM_MANAGERS builder."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from abl_config import LEAGUE_ID, RAW_CSV_ROOT, TEAM_IDS
from abl_team_helper import load_abl_teams

CSV_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = CSV_ROOT / "out" / "csv_out" / "z_ABL_DIM_Managers.csv"
TEAM_SET = set(TEAM_IDS)

COACHES_PATH = RAW_CSV_ROOT / "coaches.csv"
TEAM_HISTORY_PATH = RAW_CSV_ROOT / "team_history.csv"
TEAM_RECORD_PATH = RAW_CSV_ROOT / "team_history_record.csv"


def load_coaches() -> pd.DataFrame:
    coaches = pd.read_csv(COACHES_PATH)
    for col in ["first_name", "last_name", "nick_name", "personality"]:
        if col not in coaches.columns:
            coaches[col] = ""
        coaches[col] = coaches[col].fillna("")
    if "age" not in coaches.columns:
        coaches["age"] = pd.NA
    result = pd.DataFrame(
        {
            "manager_id": coaches["coach_id"].astype(int),
            "first_name": coaches["first_name"].astype(str),
            "last_name": coaches["last_name"].astype(str),
            "nick_name": coaches["nick_name"].astype(str),
            "age": pd.to_numeric(coaches["age"], errors="coerce"),
            "personality": coaches["personality"].astype(str),
        }
    )
    result["full_name"] = (
        result["first_name"].str.strip() + " " + result["last_name"].str.strip()
    ).str.strip()
    empties = result["full_name"] == ""
    result.loc[empties, "full_name"] = result.loc[empties, "nick_name"]
    result["full_name"] = result["full_name"].fillna("")
    result["bats_throws"] = ""
    result["reputation"] = ""
    return result


def load_manager_assignments() -> pd.DataFrame:
    history = pd.read_csv(TEAM_HISTORY_PATH)
    history = history[
        (history["league_id"] == LEAGUE_ID)
        & (history["team_id"].isin(TEAM_SET))
        & history["manager_id"].notna()
    ].copy()
    history["manager_id"] = history["manager_id"].astype(int)

    records = pd.read_csv(TEAM_RECORD_PATH)[
        ["team_id", "year", "league_id", "w", "l"]
    ]
    records = records[
        (records["league_id"] == LEAGUE_ID) & (records["team_id"].isin(TEAM_SET))
    ].copy()
    records["team_id"] = records["team_id"].astype(int)

    merged = history.merge(
        records,
        on=["team_id", "year", "league_id"],
        how="left",
        suffixes=("", "_rec"),
    )
    merged["wins"] = pd.to_numeric(merged["w"], errors="coerce").fillna(0)
    merged["losses"] = pd.to_numeric(merged["l"], errors="coerce").fillna(0)
    merged["made_playoffs"] = merged.get("made_playoffs", 0).fillna(0).astype(int)
    merged["won_playoffs"] = merged.get("won_playoffs", 0).fillna(0).astype(int)
    return merged[
        [
            "manager_id",
            "team_id",
            "year",
            "wins",
            "losses",
            "made_playoffs",
            "won_playoffs",
        ]
    ]


def summarize_managers(assignments: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    grouped = assignments.groupby("manager_id")
    summary = grouped.agg(
        total_seasons=("year", lambda s: s.nunique()),
        first_year=("year", "min"),
        last_year=("year", "max"),
        career_wins=("wins", "sum"),
        career_losses=("losses", "sum"),
        playoff_appearances=("made_playoffs", "sum"),
        titles_won=("won_playoffs", "sum"),
    ).reset_index()
    for col in ["total_seasons", "first_year", "last_year"]:
        summary[col] = summary[col].astype(int)
    summary["career_wins"] = summary["career_wins"].astype(int)
    summary["career_losses"] = summary["career_losses"].astype(int)
    summary["playoff_appearances"] = summary["playoff_appearances"].astype(int)
    summary["titles_won"] = summary["titles_won"].astype(int)
    summary["career_win_pct"] = summary.apply(
        lambda row: row["career_wins"] / (row["career_wins"] + row["career_losses"])
        if (row["career_wins"] + row["career_losses"]) > 0
        else 0,
        axis=1,
    )

    latest_year = assignments["year"].max()
    latest_assignments = assignments.sort_values("year").drop_duplicates(
        "manager_id", keep="last"
    )
    summary = summary.merge(latest_assignments[["manager_id", "team_id", "year"]], on="manager_id", how="left")
    summary = summary.rename(columns={"team_id": "current_team_id", "year": "current_year"})
    summary.loc[summary["current_year"] != latest_year, "current_team_id"] = pd.NA
    summary = summary.drop(columns=["current_year"])

    seasons_current = (
        assignments.groupby(["manager_id", "team_id"])["year"].nunique().reset_index(name="seasons_with_team")
    )
    summary = summary.merge(
        seasons_current,
        left_on=["manager_id", "current_team_id"],
        right_on=["manager_id", "team_id"],
        how="left",
    ).rename(columns={"seasons_with_team": "seasons_with_current_team"}).drop(columns=["team_id"])
    summary["seasons_with_current_team"] = summary["seasons_with_current_team"].fillna(0).astype(int)

    summary["playoff_wins"] = 0
    summary["playoff_losses"] = 0

    return summary, latest_year


def attach_team_info(df: pd.DataFrame) -> pd.DataFrame:
    teams = load_abl_teams().rename(columns={"name": "team_name", "abbr": "team_abbr"})
    df["current_team_id"] = pd.to_numeric(df["current_team_id"], errors="coerce")
    df = df.merge(
        teams,
        left_on="current_team_id",
        right_on="team_id",
        how="left",
        suffixes=("", "_lookup"),
    )
    df["current_team_name"] = df["team_name"].fillna("")
    df["current_team_abbr"] = df["team_abbr"].fillna("")
    df = df.drop(columns=["team_id", "team_name", "team_abbr"])
    return df


def build_dimension() -> Tuple[pd.DataFrame, int]:
    coaches = load_coaches()
    assignments = load_manager_assignments()
    summary, latest_year = summarize_managers(assignments)
    df = coaches.merge(summary, on="manager_id", how="inner")
    df = attach_team_info(df)
    df["notes"] = ""
    df = df[
        [
            "manager_id",
            "first_name",
            "last_name",
            "full_name",
            "bats_throws",
            "age",
            "personality",
            "reputation",
            "current_team_id",
            "current_team_name",
            "current_team_abbr",
            "total_seasons",
            "seasons_with_current_team",
            "first_year",
            "last_year",
            "career_wins",
            "career_losses",
            "career_win_pct",
            "playoff_appearances",
            "playoff_wins",
            "playoff_losses",
            "titles_won",
            "notes",
        ]
    ].sort_values("manager_id")
    return df.reset_index(drop=True), latest_year


def write_output(df: pd.DataFrame) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)


def validate_output(latest_year: int) -> None:
    df = pd.read_csv(OUT_PATH)
    if df["manager_id"].duplicated().any():
        raise SystemExit("Validation failed: duplicate manager_id detected.")
    invalid = df["current_team_id"].dropna()
    if not set(invalid.astype(int)).issubset(TEAM_SET):
        raise SystemExit("Validation failed: current_team_id outside ABL canon.")
    print(
        f"DIM_MANAGERS: built {len(df)} managers, "
        f"{df['current_team_id'].notna().sum()} currently assigned in latest season {latest_year}"
    )


def main() -> None:
    try:
        df, latest_year = build_dimension()
        write_output(df)
        validate_output(latest_year)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
