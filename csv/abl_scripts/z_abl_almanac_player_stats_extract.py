#!/usr/bin/env python
"""
Extract player batting/pitching tables from almanac player HTML pages and attach team identity.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd


def log(msg: str) -> None:
    print(msg)


def norm_key(val: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(val).strip().lower())


def load_dim_team(dim_path: Path) -> pd.DataFrame:
    dim = pd.read_csv(dim_path)
    team_id_col = "team_id" if "team_id" in dim.columns else "ID"
    abbr_col = "team_abbr" if "team_abbr" in dim.columns else "Abbr"
    name_col = "team_name" if "team_name" in dim.columns else "Team Name"
    city_col = "City" if "City" in dim.columns else name_col
    conf_col = "conf" if "conf" in dim.columns else ("SL" if "SL" in dim.columns else None)
    div_col = "division" if "division" in dim.columns else ("DIV" if "DIV" in dim.columns else None)
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
    dim_use["team_key"] = dim_use["team_city"].str.split("(").str[0].map(norm_key)
    return dim_use


def detect_name_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        cl = str(c).lower()
        if "player" in cl or "name" in cl:
            return c
    raise KeyError("No player/name column found")


def detect_team_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if "team" in str(c).lower():
            return c
    return None


def valid_batting_table(df: pd.DataFrame) -> bool:
    cols = [str(c).lower() for c in df.columns]
    return ("player" in " ".join(cols)) and any(k in cols for k in ["ab", "avg", "hr", "rbi", "obp", "slg", "ops"])


def valid_pitching_table(df: pd.DataFrame) -> bool:
    cols = [str(c).lower() for c in df.columns]
    return ("player" in " ".join(cols)) and any(k in cols for k in ["era", "ip", "so", "bb", "whip", "sv"])


def parse_player_tables(html_path: Path, stat_type: str) -> List[pd.DataFrame]:
    tables = pd.read_html(html_path)
    frames = []
    for tbl in tables:
        if stat_type == "batting" and not valid_batting_table(tbl):
            continue
        if stat_type == "pitching" and not valid_pitching_table(tbl):
            continue
        try:
            name_col = detect_name_col(tbl)
        except KeyError:
            continue
        team_col = detect_team_col(tbl)
        df = tbl.copy()
        df = df.rename(columns={name_col: "player_name"})
        if team_col:
            df = df.rename(columns={team_col: "team_name"})
        frames.append(df)
    return frames


def collect_html_files(base_dir: Path, league_id: int, kind: str) -> List[Path]:
    pattern = f"league_{league_id}_players_{kind}_reg_by_letter_*.html"
    return sorted(base_dir.glob(pattern))


def join_team(df: pd.DataFrame, dim: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    if "team_name" in df.columns:
        df["team_key"] = df["team_name"].map(norm_key)
        merged = df.merge(dim, on="team_key", how="left")
        missing = merged["team_id"].isna().sum()
        if missing:
            log(f"[WARN] {missing} rows missing team match (team_name) – keeping rows")
        return merged, missing
    else:
        for col in ["team_id", "team_abbr", "conf", "division", "team_name"]:
            if col not in df.columns:
                df[col] = pd.NA
        return df, len(df)


def normalize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    skip = {"player_name", "team_name", "team_abbr", "team_city", "team_key", "conf", "division"}
    for col in df.columns:
        if col in skip:
            continue
        try:
            df[col] = pd.to_numeric(df[col], errors="ignore")
        except Exception:
            continue
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract player batting/pitching stats from almanac player pages.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--league-id", type=int, required=True)
    args = parser.parse_args()
    season = args.season
    league_id = args.league_id

    base_dir = Path("csv/in/almanac_core") / str(season) / "leagues"
    dim_path = Path("csv/out/star_schema/dim_team_park.csv")
    out_root = Path("csv/out/almanac") / str(season)
    out_root.mkdir(parents=True, exist_ok=True)

    bat_files = collect_html_files(base_dir, league_id, "batting")
    pit_files = collect_html_files(base_dir, league_id, "pitching")
    if not bat_files or not pit_files:
        log(f"[WARN] No player HTML files found in {base_dir} for league {league_id}")

    bat_frames: List[pd.DataFrame] = []
    for path in bat_files:
        bat_frames.extend(parse_player_tables(path, "batting"))
    pit_frames: List[pd.DataFrame] = []
    for path in pit_files:
        pit_frames.extend(parse_player_tables(path, "pitching"))

    bat_df = pd.concat(bat_frames, ignore_index=True) if bat_frames else pd.DataFrame(columns=["player_name"])
    pit_df = pd.concat(pit_frames, ignore_index=True) if pit_frames else pd.DataFrame(columns=["player_name"])

    dim = load_dim_team(dim_path)
    bat_df, miss_bat = join_team(bat_df, dim) if not bat_df.empty else (bat_df, 0)
    pit_df, miss_pit = join_team(pit_df, dim) if not pit_df.empty else (pit_df, 0)

    for df in (bat_df, pit_df):
        df["season"] = season
        df["league_id"] = league_id
        for col in ["team_id", "team_abbr", "team_name", "conf", "division"]:
            if col not in df.columns:
                df[col] = pd.NA

    bat_df = normalize_numeric(bat_df)
    pit_df = normalize_numeric(pit_df)

    bat_out = out_root / f"player_batting_{season}_league{league_id}.csv"
    pit_out = out_root / f"player_pitching_{season}_league{league_id}.csv"
    bat_df.to_csv(bat_out, index=False)
    pit_df.to_csv(pit_out, index=False)

    log(
        f"[INFO] season {season}, league {league_id}: found {len(bat_df)} batting players, {len(pit_df)} pitching players across {len(bat_files)} batting HTMLs and {len(pit_files)} pitching HTMLs"
    )
    log(f"[OK] Wrote {bat_out} and {pit_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
