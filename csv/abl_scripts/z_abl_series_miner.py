"""ABL Series Miner: summarize 3-4 game sets between ABL clubs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from abl_config import LEAGUE_ID, RAW_CSV_ROOT, TEAM_IDS
from abl_team_helper import load_abl_teams

CSV_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = CSV_ROOT / "out" / "csv_out" / "z_ABL_Series_Miner.csv"
DECISIVE_RUN_DIFF = 10

GAMES_PATH = RAW_CSV_ROOT / "games.csv"
TEAMS_PATH = RAW_CSV_ROOT / "teams.csv"

TEAM_SET = set(TEAM_IDS)


def parse_date(text: str) -> pd.Timestamp:
    try:
        return pd.to_datetime(text, errors="raise").normalize()
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid date value: {text}") from exc


def load_team_lookup() -> Dict[int, Dict[str, str]]:
    canon = load_abl_teams().rename(columns={"name": "city", "abbr": "team_abbr"})
    canon["team_id"] = canon["team_id"].astype(int)
    raw = pd.read_csv(TEAMS_PATH, usecols=["team_id", "name", "nickname", "league_id"])
    raw = raw[(raw["league_id"] == LEAGUE_ID) & (raw["team_id"].isin(TEAM_SET))]
    merged = canon.merge(raw, on="team_id", how="left")
    if merged["nickname"].isna().any():
        missing = merged.loc[merged["nickname"].isna(), "team_id"].tolist()
        raise SystemExit(f"Missing nickname data for teams: {missing}")
    lookup: Dict[int, Dict[str, str]] = {}
    for _, row in merged.iterrows():
        display = f"{row['name']} {row['nickname']}".strip()
        lookup[int(row["team_id"])] = {"name": display, "abbr": row["team_abbr"]}
    return lookup


def load_games() -> pd.DataFrame:
    if not GAMES_PATH.exists():
        raise FileNotFoundError(f"games.csv missing at {GAMES_PATH}")
    games = pd.read_csv(GAMES_PATH)
    games["date"] = pd.to_datetime(games["date"], errors="coerce")
    games = games.dropna(subset=["date"])
    games["date"] = games["date"].dt.normalize()
    games = games[
        (games["league_id"] == LEAGUE_ID)
        & (games["home_team"].isin(TEAM_SET))
        & (games["away_team"].isin(TEAM_SET))
        & (games["played"] == 1)
        & (games["game_type"] == 0)
    ].copy()
    games["game_id"] = games["game_id"].astype(int)
    games["time"] = games["time"].fillna(0)
    games = games.sort_values(["home_team", "away_team", "date", "time", "game_id"]).reset_index(drop=True)
    return games


def build_series_clusters(games: pd.DataFrame, target_game_ids: Iterable[int]) -> List[pd.DataFrame]:
    """Group games into contiguous home/away date clusters (max 4 games)."""
    target_ids = set(target_game_ids)
    clusters: List[pd.DataFrame] = []
    for (home_id, away_id), group in games.groupby(["home_team", "away_team"]):
        group = group.sort_values(["date", "time", "game_id"]).reset_index(drop=True)
        current: List[pd.Series] = []
        for _, row in group.iterrows():
            if not current:
                current.append(row)
                continue
            prev_date = current[-1]["date"]
            day_gap = (row["date"] - prev_date).days
            if day_gap <= 1 and len(current) < 4:
                current.append(row)
            else:
                if len(current) >= 2:
                    clusters.append(pd.DataFrame(current))
                current = [row]
        if current and len(current) >= 2:
            clusters.append(pd.DataFrame(current))

    if not target_ids:
        return []
    filtered = []
    for cluster in clusters:
        if cluster["game_id"].isin(target_ids).any():
            filtered.append(cluster)
    return filtered


def summarize_cluster(cluster: pd.DataFrame, team_lookup: Dict[int, Dict[str, str]]) -> Dict[str, object]:
    cluster = cluster.sort_values(["date", "time", "game_id"]).reset_index(drop=True)
    home_id = int(cluster["home_team"].iloc[0])
    away_id = int(cluster["away_team"].iloc[0])
    start_date = cluster["date"].min()
    end_date = cluster["date"].max()
    num_games = len(cluster)

    home_scores = cluster["runs0"].astype(int)
    away_scores = cluster["runs1"].astype(int)
    home_wins = (home_scores > away_scores).sum()
    away_wins = num_games - home_wins
    home_runs_scored = home_scores.sum()
    home_runs_allowed = away_scores.sum()

    series_id = f"{home_id}_{away_id}_{start_date.strftime('%Y%m%d')}"

    winners = ["home" if hs > as_ else "away" for hs, as_ in zip(home_scores, away_scores)]
    sweep_flag = int(home_wins == num_games or away_wins == num_games)
    avoided_sweep_flag = 0
    if sweep_flag == 0 and num_games >= 2:
        last_winner = winners[-1]
        losing_team = "home" if home_wins < away_wins else "away"
        wins_for_loser = home_wins if losing_team == "home" else away_wins
        if wins_for_loser == 1 and last_winner == losing_team:
            avoided_sweep_flag = 1
    split_flag = int(home_wins == away_wins)
    decisive_flag = 0
    if split_flag == 0:
        diff = abs(home_runs_scored - home_runs_allowed)
        if diff >= DECISIVE_RUN_DIFF:
            decisive_flag = 1

    return {
        "series_id": series_id,
        "home_team_id": home_id,
        "home_team_name": team_lookup[home_id]["name"],
        "home_team_abbr": team_lookup[home_id]["abbr"],
        "away_team_id": away_id,
        "away_team_name": team_lookup[away_id]["name"],
        "away_team_abbr": team_lookup[away_id]["abbr"],
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "num_games": num_games,
        "home_series_wins": home_wins,
        "home_series_losses": away_wins,
        "home_runs_scored": home_runs_scored,
        "home_runs_allowed": home_runs_allowed,
        "away_series_wins": away_wins,
        "away_series_losses": home_wins,
        "away_runs_scored": home_runs_allowed,
        "away_runs_allowed": home_runs_scored,
        "home_run_diff": home_runs_scored - home_runs_allowed,
        "away_run_diff": -(home_runs_scored - home_runs_allowed),
        "sweep_flag": sweep_flag,
        "avoided_sweep_flag": avoided_sweep_flag,
        "split_flag": split_flag,
        "decisive_flag": decisive_flag,
    }


def write_output(df: pd.DataFrame) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)


def validate_output() -> None:
    df = pd.read_csv(OUT_PATH)
    if df.empty:
        return
    if df["series_id"].isna().any():
        raise SystemExit("Validation failed: series_id missing.")
    if not set(df["home_team_id"]).issubset(TEAM_SET):
        raise SystemExit("Validation failed: invalid home_team_id detected.")
    if not set(df["away_team_id"]).issubset(TEAM_SET):
        raise SystemExit("Validation failed: invalid away_team_id detected.")
    if (df["home_team_id"] == df["away_team_id"]).any():
        raise SystemExit("Validation failed: series contains identical home/away team IDs.")


def main() -> None:
    parser = argparse.ArgumentParser(description="ABL Series Miner")
    parser.add_argument("--start-date", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="End date YYYY-MM-DD")
    args = parser.parse_args()

    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)
    if end_date < start_date:
        raise SystemExit("end-date must be greater than or equal to start-date")

    games = load_games()
    in_range_mask = (games["date"] >= start_date) & (games["date"] <= end_date)
    games_in_window = games.loc[in_range_mask].copy()

    if games_in_window.empty:
        print("No games found in the requested window; writing empty Series Miner output.")
        write_output(pd.DataFrame(columns=[
            "series_id",
            "home_team_id",
            "home_team_name",
            "home_team_abbr",
            "away_team_id",
            "away_team_name",
            "away_team_abbr",
            "start_date",
            "end_date",
            "num_games",
            "home_series_wins",
            "home_series_losses",
            "home_runs_scored",
            "home_runs_allowed",
            "away_series_wins",
            "away_series_losses",
            "away_runs_scored",
            "away_runs_allowed",
            "home_run_diff",
            "away_run_diff",
            "sweep_flag",
            "avoided_sweep_flag",
            "split_flag",
            "decisive_flag",
        ]))
        validate_output()
        return

    print("No explicit series column detected; using heuristic home/away/date clustering.")
    clusters = build_series_clusters(games, games_in_window["game_id"].tolist())
    team_lookup = load_team_lookup()

    summaries = [summarize_cluster(cluster, team_lookup) for cluster in clusters]
    series_df = pd.DataFrame(summaries)
    series_df = series_df.sort_values(["start_date", "home_team_id", "away_team_id"]).reset_index(drop=True)

    write_output(series_df)
    validate_output()
    print(
        f"Series Miner: processed {len(games_in_window)} games, produced {len(series_df)} series rows "
        f"between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}"
    )


if __name__ == "__main__":
    main()
