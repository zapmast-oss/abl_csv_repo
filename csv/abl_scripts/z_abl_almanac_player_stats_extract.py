#!/usr/bin/env python
"""
Extract player batting/pitching tables from almanac stats HTML and attach team identity.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd


def log(msg: str) -> None:
    print(msg)


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).strip().lower())


def find_cols(tbl: pd.DataFrame, needles: Iterable[str]) -> bool:
    cols = [c.lower() for c in tbl.columns]
    return all(any(n.lower() in c for c in cols) for n in needles)


def pick_name_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        cl = str(c).lower()
        if "player" in cl or "name" in cl:
            return c
    raise KeyError("No player/name column found")


def pick_team_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if "team" in c.lower():
            return c
    raise KeyError("No team column found")


def load_dim_team(dim_path: Path) -> pd.DataFrame:
    dim = pd.read_csv(dim_path)
    team_id_col = "team_id" if "team_id" in dim.columns else "ID"
    abbr_col = "team_abbr" if "team_abbr" in dim.columns else "Abbr"
    name_col = "team_name" if "team_name" in dim.columns else "Team Name"
    conf_col = "conf" if "conf" in dim.columns else ("SL" if "SL" in dim.columns else None)
    div_col = "division" if "division" in dim.columns else ("DIV" if "DIV" in dim.columns else None)
    city_col = "City" if "City" in dim.columns else name_col
    dim_use = dim[[team_id_col, abbr_col, name_col, city_col]].copy()
    dim_use = dim_use.rename(
        columns={
            team_id_col: "team_id",
            abbr_col: "team_abbr",
            name_col: "team_name",
            city_col: "team_city",
        }
    )
    if conf_col:
        dim_use["conf"] = dim[conf_col]
    if div_col:
        dim_use["division"] = dim[div_col]
    dim_use["name_key"] = dim_use["team_city"].str.split("(").str[0].map(norm)
    return dim_use


def join_team(df: pd.DataFrame, dim: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    df["name_key"] = df["team_name"].map(norm)
    merged = df.merge(dim, on="name_key", how="left")
    missing = merged["team_id"].isna().sum()
    if missing:
        log(f"[WARN] {missing} player rows did not match a team")
    return merged, missing


def collect_tables(stats_html: Path) -> Tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    tables = pd.read_html(stats_html)
    batting = []
    pitching = []
    for tbl in tables:
        try:
            name_col = pick_name_col(tbl)
            team_col = pick_team_col(tbl)
        except KeyError:
            continue
        if find_cols(tbl, ["avg"]) and (find_cols(tbl, ["hr"]) or find_cols(tbl, ["rbi"]) or find_cols(tbl, ["obp"])):
            df = tbl.copy()
            df = df.rename(columns={name_col: "player_name", team_col: "team_name"})
            batting.append(df)
        elif find_cols(tbl, ["era"]) and (find_cols(tbl, ["ip"]) or find_cols(tbl, ["so"])):
            df = tbl.copy()
            df = df.rename(columns={name_col: "player_name", team_col: "team_name"})
            pitching.append(df)
    return batting, pitching


def normalize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    id_cols = {"season", "league_id", "team_id", "team_abbr", "team_name", "player_name", "conf", "division"}
    for col in df.columns:
        if col in id_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        try:
            df[col] = pd.to_numeric(df[col], errors="ignore")
        except Exception:
            continue
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract player batting/pitching tables from almanac stats HTML.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--league-id", type=int, required=True)
    args = parser.parse_args()
    season = args.season
    league_id = args.league_id

    stats_html = Path(f"csv/in/almanac_core/{season}/leagues/league_{league_id}_stats.html")
    dim_path = Path("csv/out/star_schema/dim_team_park.csv")
    out_root = Path("csv/out/almanac") / str(season)
    out_root.mkdir(parents=True, exist_ok=True)

    batting_tables, pitching_tables = collect_tables(stats_html)
    if not batting_tables:
        log("[WARN] No batting tables found; creating empty batting dataframe.")
        bat_df = pd.DataFrame(columns=["player_name", "team_name"])
    else:
        bat_df = pd.concat(batting_tables, ignore_index=True)
    if not pitching_tables:
        log("[WARN] No pitching tables found; creating empty pitching dataframe.")
        pit_df = pd.DataFrame(columns=["player_name", "team_name"])
    else:
        pit_df = pd.concat(pitching_tables, ignore_index=True)

    dim = load_dim_team(dim_path)
    bat_df, miss_bat = join_team(bat_df, dim)
    pit_df, miss_pit = join_team(pit_df, dim)

    for df in (bat_df, pit_df):
        df["season"] = season
        df["league_id"] = league_id

    bat_df = normalize_numeric(bat_df)
    pit_df = normalize_numeric(pit_df)

    # Ensure key columns exist
    for df in (bat_df, pit_df):
        for col in ["team_id", "team_abbr", "conf", "division"]:
            if col not in df.columns:
                df[col] = pd.NA

    bat_out = out_root / f"player_batting_{season}_league{league_id}.csv"
    pit_out = out_root / f"player_pitching_{season}_league{league_id}.csv"
    bat_df.to_csv(bat_out, index=False)
    pit_df.to_csv(pit_out, index=False)
    log(
        f"[INFO] Loaded {len(bat_df)} batting players, {len(pit_df)} pitching players after team join for season {season}, league {league_id}"
    )
    log(f"[INFO] Batting sample columns: {list(bat_df.columns)[:5]}")
    log(f"[INFO] Pitching sample columns: {list(pit_df.columns)[:5]}")
    log(f"[OK] Wrote {bat_out} and {pit_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
