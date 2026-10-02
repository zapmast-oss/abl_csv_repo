"""ABL Week Miner: day-by-day highlights for Action Baseball League."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from abl_config import LEAGUE_ID, RAW_CSV_ROOT, TEAM_IDS
from abl_team_helper import load_abl_teams

CSV_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = CSV_ROOT / "out" / "csv_out" / "z_ABL_Week_Miner.csv"
TEAM_SET = set(TEAM_IDS)

GAMES_PATH = RAW_CSV_ROOT / "games.csv"
BATTING_PATH = RAW_CSV_ROOT / "players_game_batting.csv"
PITCHING_PATH = RAW_CSV_ROOT / "players_game_pitching_stats.csv"
FIELDING_PATH = RAW_CSV_ROOT / "players_game_fielding.csv"
PLAYERS_PATH = RAW_CSV_ROOT / "players.csv"
TEAMS_PATH = RAW_CSV_ROOT / "teams.csv"

OUTPUT_COLUMNS = [
    "date",
    "game_id",
    "home_team_id",
    "home_team",
    "away_team_id",
    "away_team",
    "home_score",
    "away_score",
    "winner",
    "loser",
    "walkoff_flag",
    "extra_innings_flag",
    "comeback_flag",
    "top_hr_performances",
    "top_rbi_performances",
    "top_sb_performances",
    "top_xbh_games",
    "multi_hit_games",
    "pitching_gems",
    "bullpen_notes",
    "fielding_notes",
    "opening_of_series_flag",
    "closing_of_series_flag",
    "narrative_tags",
]


def parse_date(value: str) -> pd.Timestamp:
    try:
        ts = pd.to_datetime(value, errors="raise")
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid date value: {value}") from exc
    return ts.normalize()


def load_team_lookup() -> Dict[int, Dict[str, str]]:
    canon = load_abl_teams().rename(columns={"name": "team_city", "abbr": "team_abbr"})
    canon["team_id"] = canon["team_id"].astype(int)
    raw = pd.read_csv(TEAMS_PATH, usecols=["team_id", "name", "nickname", "league_id"])
    raw = raw[(raw["league_id"] == LEAGUE_ID) & (raw["team_id"].isin(TEAM_SET))]
    merged = canon.merge(raw, on="team_id", how="left")
    if merged["nickname"].isna().any():
        missing = merged.loc[merged["nickname"].isna(), "team_id"].tolist()
        raise SystemExit(f"Missing nickname/city data for teams: {missing}")
    lookup: Dict[int, Dict[str, str]] = {}
    for _, row in merged.iterrows():
        display = f"{row['name']} {row['nickname']}".strip()
        lookup[int(row["team_id"])] = {"name": display, "abbr": row["team_abbr"]}
    return lookup


def load_players_lookup() -> Dict[int, str]:
    cols = ["player_id", "first_name", "last_name", "nick_name"]
    players = pd.read_csv(PLAYERS_PATH, usecols=cols)
    lookup: Dict[int, str] = {}
    for _, row in players.iterrows():
        first = str(row.get("first_name", "")).strip()
        last = str(row.get("last_name", "")).strip()
        nick = str(row.get("nick_name", "")).strip()
        base = (f"{first} {last}".strip() if first or last else "").strip()
        if nick and base:
            name = f"{base} ({nick})"
        elif base:
            name = base
        else:
            name = nick or "Unknown Player"
        lookup[int(row["player_id"])] = name
    return lookup


def load_games(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
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
        & (games["date"] >= start_date)
        & (games["date"] <= end_date)
    ].copy()
    games["game_id"] = games["game_id"].astype(int)
    return games.sort_values(["date", "time", "game_id"])


def load_batting(game_ids: Iterable[int]) -> pd.DataFrame:
    if not game_ids or not BATTING_PATH.exists():
        return pd.DataFrame()
    batting = pd.read_csv(BATTING_PATH)
    batting = batting[
        (batting["league_id"] == LEAGUE_ID)
        & (batting["team_id"].isin(TEAM_SET))
        & (batting["game_id"].isin(set(game_ids)))
    ].copy()
    if batting.empty:
        return batting
    numeric_cols = ["ab", "h", "hr", "rbi", "sb", "d", "t", "bb", "pa", "cs"]
    present = [c for c in numeric_cols if c in batting.columns]
    agg = batting.groupby(["game_id", "player_id", "team_id"], as_index=False)[present].sum()
    return agg


def load_pitching(game_ids: Iterable[int]) -> pd.DataFrame:
    if not game_ids or not PITCHING_PATH.exists():
        return pd.DataFrame()
    pitching = pd.read_csv(PITCHING_PATH)
    pitching = pitching[
        (pitching["league_id"] == LEAGUE_ID)
        & (pitching["team_id"].isin(TEAM_SET))
        & (pitching["game_id"].isin(set(game_ids)))
    ].copy()
    if pitching.empty:
        return pitching
    numeric_cols = [
        "ip",
        "k",
        "bb",
        "w",
        "l",
        "s",
        "qs",
        "cg",
        "sho",
        "bs",
        "hld",
        "g",
    ]
    present_cols = [c for c in numeric_cols if c in pitching.columns]
    agg = pitching.groupby(["game_id", "player_id", "team_id"], as_index=False)[present_cols].sum()
    return agg


def load_fielding(game_ids: Iterable[int]) -> pd.DataFrame:
    if not game_ids or not FIELDING_PATH.exists():
        return pd.DataFrame()
    fielding = pd.read_csv(FIELDING_PATH)
    cond = pd.Series(True, index=fielding.index)
    if "league_id" in fielding.columns:
        cond &= fielding["league_id"] == LEAGUE_ID
    if "team_id" in fielding.columns:
        cond &= fielding["team_id"].isin(TEAM_SET)
    if "game_id" in fielding.columns:
        cond &= fielding["game_id"].isin(set(game_ids))
    return fielding.loc[cond].copy()


def format_player_name(player_id: int, lookup: Dict[int, str]) -> str:
    return lookup.get(int(player_id), f"Player {player_id}")


def summarize_batting(group: pd.DataFrame, player_lookup: Dict[int, str], tags: List[str]) -> Dict[str, str]:
    if group is None or group.empty:
        return {
            "top_hr_performances": "",
            "top_rbi_performances": "",
            "top_sb_performances": "",
            "top_xbh_games": "",
            "multi_hit_games": "",
        }

    group = group.copy()
    group["xbh"] = group.get("hr", 0) + group.get("d", 0) + group.get("t", 0)

    def build_list(cond_series, sort_col: str, formatter, tag_name: Optional[str] = None) -> str:
        subset = group.loc[cond_series].sort_values(by=sort_col, ascending=False)
        if subset.empty:
            return ""
        if tag_name:
            tags.append(tag_name)
        return ", ".join(formatter(row) for _, row in subset.iterrows())

    result = {}
    result["top_hr_performances"] = build_list(
        group["hr"] >= 2,
        "hr",
        lambda row: f"{format_player_name(row['player_id'], player_lookup)} ({int(row['hr'])} HR)",
        "multi_hr",
    )
    result["top_rbi_performances"] = build_list(
        group["rbi"] >= 4,
        "rbi",
        lambda row: f"{format_player_name(row['player_id'], player_lookup)} ({int(row['rbi'])} RBI)",
        "rbi_barrage",
    )
    result["top_sb_performances"] = build_list(
        group["sb"] >= 2,
        "sb",
        lambda row: f"{format_player_name(row['player_id'], player_lookup)} ({int(row['sb'])} SB)",
        "running_game",
    )
    result["top_xbh_games"] = build_list(
        group["xbh"] >= 3,
        "xbh",
        lambda row: f"{format_player_name(row['player_id'], player_lookup)} ({int(row['xbh'])} XBH)",
        "xbh_showcase",
    )
    result["multi_hit_games"] = build_list(
        group["h"] >= 4,
        "h",
        lambda row: f"{format_player_name(row['player_id'], player_lookup)} ({int(row['h'])} H)",
        "four_hit_game",
    )
    return result


def summarize_pitching(group: pd.DataFrame, player_lookup: Dict[int, str], tags: List[str]) -> Dict[str, str]:
    if group is None or group.empty:
        return {"pitching_gems": "", "bullpen_notes": ""}

    gems: List[str] = []
    bullpen: List[str] = []
    for _, row in group.iterrows():
        name = format_player_name(row["player_id"], player_lookup)
        feats: List[str] = []
        if int(row.get("cg", 0)) >= 1:
            feats.append("CG")
        if int(row.get("sho", 0)) >= 1:
            feats.append("SHO")
        ks = row.get("k", 0)
        if ks >= 10:
            feats.append(f"{int(ks)} K")
        if row.get("bb", 0) == 0:
            feats.append("0 BB")
        if int(row.get("qs", 0)) >= 1:
            feats.append("QS")
        if feats:
            gems.append(f"{name} ({', '.join(feats)})")
            tags.append("pitching_gem")
        saves = int(row.get("s", 0)) if "s" in row else 0
        holds = int(row.get("hld", 0)) if "hld" in row else 0
        blown = int(row.get("bs", 0)) if "bs" in row else 0
        if saves:
            bullpen.append(f"{name} (Save)")
            tags.append("bullpen_lockdown")
        if holds:
            bullpen.append(f"{name} (Hold)")
        if blown:
            bullpen.append(f"{name} (Blown Save)")
            tags.append("bullpen_drama")
    return {"pitching_gems": ", ".join(gems), "bullpen_notes": ", ".join(bullpen)}


def summarize_fielding(group: pd.DataFrame, player_lookup: Dict[int, str], tags: List[str]) -> str:
    if group is None or group.empty:
        return ""
    # No detailed fielding data is available in the exports currently shipped.
    return ""


def summarize_game(row: pd.Series, batting_map, pitching_map, fielding_map, team_lookup, player_lookup) -> Dict[str, object]:
    game_id = int(row["game_id"])
    date = row["date"].strftime("%Y-%m-%d")
    home_id = int(row["home_team"])
    away_id = int(row["away_team"])
    home_score = int(row["runs0"])
    away_score = int(row["runs1"])
    home_team = team_lookup[home_id]["name"]
    away_team = team_lookup[away_id]["name"]
    winner_id = home_id if home_score > away_score else away_id
    loser_id = away_id if winner_id == home_id else home_id
    winner = team_lookup[winner_id]["name"]
    loser = team_lookup[loser_id]["name"]
    walkoff_flag = bool(row.get("innings", 9) > 9 and home_score > away_score)
    extra_flag = bool(row.get("innings", 9) > 9)
    comeback_flag = False
    tags: List[str] = []
    if walkoff_flag:
        tags.append("walkoff")
    if extra_flag:
        tags.append("extra_innings")

    batting = batting_map.get(game_id)
    pitching = pitching_map.get(game_id)
    fielding = fielding_map.get(game_id)

    batting_notes = summarize_batting(batting, player_lookup, tags)
    pitching_notes = summarize_pitching(pitching, player_lookup, tags)
    fielding_notes = summarize_fielding(fielding, player_lookup, tags)

    row_dict = {
        "date": date,
        "game_id": game_id,
        "home_team_id": home_id,
        "home_team": home_team,
        "away_team_id": away_id,
        "away_team": away_team,
        "home_score": home_score,
        "away_score": away_score,
        "winner": winner,
        "loser": loser,
        "walkoff_flag": walkoff_flag,
        "extra_innings_flag": extra_flag,
        "comeback_flag": comeback_flag,
        "top_hr_performances": batting_notes["top_hr_performances"],
        "top_rbi_performances": batting_notes["top_rbi_performances"],
        "top_sb_performances": batting_notes["top_sb_performances"],
        "top_xbh_games": batting_notes["top_xbh_games"],
        "multi_hit_games": batting_notes["multi_hit_games"],
        "pitching_gems": pitching_notes["pitching_gems"],
        "bullpen_notes": pitching_notes["bullpen_notes"],
        "fielding_notes": fielding_notes,
        "opening_of_series_flag": False,
        "closing_of_series_flag": False,
        "narrative_tags": json.dumps(sorted(set(tags))),
    }
    return row_dict


def build_maps(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    if df is None or df.empty:
        return {}
    return {gid: group for gid, group in df.groupby("game_id")}


def validate_output(df: pd.DataFrame) -> None:
    if df.empty:
        return
    if not set(df["home_team_id"]).issubset(TEAM_SET):
        raise SystemExit("Validation failed: home team outside of TEAM_IDS")
    if not set(df["away_team_id"]).issubset(TEAM_SET):
        raise SystemExit("Validation failed: away team outside of TEAM_IDS")
    if df["home_team"].isna().any() or df["away_team"].isna().any():
        raise SystemExit("Validation failed: missing team names in output")


def main() -> None:
    parser = argparse.ArgumentParser(description="ABL Week Miner")
    parser.add_argument("--start-date", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", help="End date YYYY-MM-DD (inclusive)")
    args = parser.parse_args()

    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date) if args.end_date else start_date
    if end_date < start_date:
        raise SystemExit("end-date must be greater than or equal to start-date")

    games = load_games(start_date, end_date)
    team_lookup = load_team_lookup()
    player_lookup = load_players_lookup()

    game_ids = games["game_id"].unique().tolist()
    batting = load_batting(game_ids)
    pitching = load_pitching(game_ids)
    fielding = load_fielding(game_ids)

    batting_map = build_maps(batting)
    pitching_map = build_maps(pitching)
    fielding_map = build_maps(fielding)

    rows: List[Dict[str, object]] = []
    for _, game_row in games.iterrows():
        rows.append(summarize_game(game_row, batting_map, pitching_map, fielding_map, team_lookup, player_lookup))

    output_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUT_PATH, index=False)
    validate_output(output_df)
    print(
        f"Week Miner: processed {len(games)} games from {start_date.strftime('%Y-%m-%d')} to "
        f"{end_date.strftime('%Y-%m-%d')}, wrote {OUT_PATH.name}"
    )


if __name__ == "__main__":
    main()
