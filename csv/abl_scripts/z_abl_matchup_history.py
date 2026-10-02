"""ABL Matchup History Engine: head-to-head records between ABL clubs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from abl_config import LEAGUE_ID, RAW_CSV_ROOT, TEAM_IDS
from abl_team_helper import load_abl_teams

CSV_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = CSV_ROOT / "out" / "csv_out" / "z_ABL_Matchup_History.csv"
GAMES_PATH = RAW_CSV_ROOT / "games.csv"
TEAM_SET = set(TEAM_IDS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute ABL head-to-head matchup history from games.csv"
    )
    parser.add_argument(
        "--min-date",
        dest="min_date",
        help="Lower bound inclusive for game date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--max-date",
        dest="max_date",
        help="Upper bound inclusive for game date (YYYY-MM-DD)",
    )
    return parser.parse_args()


def coerce_date(value: Optional[str]) -> Optional[pd.Timestamp]:
    if not value:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid date value: {value}")
    return parsed.normalize()


def load_games(min_date: Optional[pd.Timestamp], max_date: Optional[pd.Timestamp]) -> pd.DataFrame:
    if not GAMES_PATH.exists():
        raise FileNotFoundError(f"games.csv not found at {GAMES_PATH}")
    games = pd.read_csv(GAMES_PATH)
    required_cols = {"league_id", "home_team", "away_team", "played", "game_type", "runs0", "runs1"}
    missing = required_cols - set(games.columns)
    if missing:
        raise RuntimeError(f"games.csv missing required columns: {', '.join(sorted(missing))}")

    games = games.copy()
    games["league_id"] = pd.to_numeric(games["league_id"], errors="coerce").astype("Int64")
    games["home_team"] = pd.to_numeric(games["home_team"], errors="coerce").astype("Int64")
    games["away_team"] = pd.to_numeric(games["away_team"], errors="coerce").astype("Int64")
    games["played"] = pd.to_numeric(games["played"], errors="coerce").astype("Int64")
    games["game_type"] = pd.to_numeric(games["game_type"], errors="coerce").astype("Int64")

    games = games[
        (games["league_id"] == LEAGUE_ID)
        & (games["home_team"].isin(TEAM_SET))
        & (games["away_team"].isin(TEAM_SET))
        & (games["played"] == 1)
        & (games["game_type"] == 0)
    ].copy()

    games["game_date"] = pd.to_datetime(games["date"], errors="coerce")
    games = games.dropna(subset=["game_date"])
    games["game_date"] = games["game_date"].dt.normalize()

    if min_date is not None:
        games = games[games["game_date"] >= min_date]
    if max_date is not None:
        games = games[games["game_date"] <= max_date]

    games = games.dropna(subset=["runs0", "runs1"])
    games["runs0"] = pd.to_numeric(games["runs0"], errors="coerce").astype(int)
    games["runs1"] = pd.to_numeric(games["runs1"], errors="coerce").astype(int)
    return games.reset_index(drop=True)


def build_matchup_table(games: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame(
            columns=[
                "team_id",
                "opp_team_id",
                "games",
                "wins",
                "losses",
                "runs_for",
                "runs_against",
            ]
        )

    home = pd.DataFrame(
        {
            "team_id": games["home_team"].astype(int),
            "opp_team_id": games["away_team"].astype(int),
            "runs_for": games["runs0"].astype(int),
            "runs_against": games["runs1"].astype(int),
        }
    )
    away = pd.DataFrame(
        {
            "team_id": games["away_team"].astype(int),
            "opp_team_id": games["home_team"].astype(int),
            "runs_for": games["runs1"].astype(int),
            "runs_against": games["runs0"].astype(int),
        }
    )

    home["wins"] = (home["runs_for"] > home["runs_against"]).astype(int)
    home["losses"] = (home["runs_for"] < home["runs_against"]).astype(int)
    away["wins"] = (away["runs_for"] > away["runs_against"]).astype(int)
    away["losses"] = (away["runs_for"] < away["runs_against"]).astype(int)

    home["games"] = 1
    away["games"] = 1

    oriented = pd.concat([home, away], ignore_index=True)
    summary = (
        oriented.groupby(["team_id", "opp_team_id"], as_index=False)
        .agg(
            games=("games", "sum"),
            wins=("wins", "sum"),
            losses=("losses", "sum"),
            runs_for=("runs_for", "sum"),
            runs_against=("runs_against", "sum"),
        )
        .sort_values(["team_id", "opp_team_id"])
        .reset_index(drop=True)
    )
    summary[["games", "wins", "losses", "runs_for", "runs_against"]] = summary[
        ["games", "wins", "losses", "runs_for", "runs_against"]
    ].astype(int)
    return summary


def attach_team_metadata(rows: pd.DataFrame) -> pd.DataFrame:
    teams = load_abl_teams()
    team_lookup = teams.set_index("team_id")
    rows = rows.copy()
    rows["team_name"] = rows["team_id"].map(team_lookup["name"])
    rows["team_abbr"] = rows["team_id"].map(team_lookup["abbr"])
    rows["opp_name"] = rows["opp_team_id"].map(team_lookup["name"])
    rows["opp_abbr"] = rows["opp_team_id"].map(team_lookup["abbr"])
    return rows[
        [
            "team_id",
            "team_name",
            "team_abbr",
            "opp_team_id",
            "opp_name",
            "opp_abbr",
            "games",
            "wins",
            "losses",
            "runs_for",
            "runs_against",
        ]
    ]


def write_output(df: pd.DataFrame) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)


def validate_output() -> None:
    df = pd.read_csv(OUT_PATH)
    invalid_teams = set(df["team_id"].unique()) - TEAM_SET
    invalid_opps = set(df["opp_team_id"].unique()) - TEAM_SET
    if invalid_teams:
        raise SystemExit(f"Invalid team IDs detected: {sorted(invalid_teams)}")
    if invalid_opps:
        raise SystemExit(f"Invalid opponent IDs detected: {sorted(invalid_opps)}")
    if (df["team_id"] == df["opp_team_id"]).any():
        raise SystemExit("Found rows where team_id == opp_team_id; this should not happen.")


def main() -> None:
    try:
        args = parse_args()
        min_date = coerce_date(args.min_date)
        max_date = coerce_date(args.max_date)
        if min_date and max_date and min_date > max_date:
            raise ValueError("min-date cannot be later than max-date")
        games = load_games(min_date, max_date)
        matchups = build_matchup_table(games)
        enriched = attach_team_metadata(matchups)
        write_output(enriched)
        validate_output()
        print(f"Games processed: {len(games)}")
        print(f"Matchup rows written: {len(enriched)}")
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
