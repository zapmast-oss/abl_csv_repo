#!/usr/bin/env python
"""
Build player hitting and pitching leaderboards from extracted almanac player stats.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def log(msg: str) -> None:
    print(msg)


def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "player" or cl == "name":
            rename_map[c] = "player_name"
        if cl == "team":
            rename_map[c] = "team_name"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def detect_col(df: pd.DataFrame, candidates) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    for col in df.columns:
        if any(sub.lower() in col.lower() for sub in candidates):
            return col
    return None


def add_rank(df: pd.DataFrame, ascending: bool) -> pd.DataFrame:
    df = df.copy()
    df["rank_overall"] = range(1, len(df) + 1)
    return df


def build_hitting_leaders(bat: pd.DataFrame) -> pd.DataFrame:
    bat = norm_cols(bat)
    # numeric coercion
    for col in bat.columns:
        if col not in {"player_name", "team_name"}:
            try:
                bat[col] = pd.to_numeric(bat[col], errors="coerce")
            except Exception:
                pass
    bat = bat[bat["player_name"].notna()]
    bat = bat[bat["player_name"] != "Player"]
    if bat.empty:
        return pd.DataFrame(columns=["player_name", "team_id", "team_abbr", "team_name", "conf", "division", "metric_primary", "metric_secondary", "rank_overall"])
    pa_col = detect_col(bat, ["PA"])
    ab_col = detect_col(bat, ["AB"])
    war_col = detect_col(bat, ["WAR"])
    ops_col = detect_col(bat, ["OPS"])
    slg_col = detect_col(bat, ["SLG"])
    avg_col = detect_col(bat, ["AVG"])

    df = bat.copy()
    if pa_col and pd.api.types.is_numeric_dtype(df[pa_col]):
        df = df[df[pa_col] >= 400]
        log(f"[INFO] Hitters with PA>400: {len(df)}")
    elif ab_col and pd.api.types.is_numeric_dtype(df[ab_col]):
        df = df[df[ab_col] >= 350]
        log(f"[INFO] Hitters with AB>350: {len(df)}")
    else:
        log("[WARN] No PA/AB column found; skipping playing-time filter")

    metric_primary = war_col if war_col else ops_col
    metric_secondary = ops_col if ops_col else (slg_col if slg_col else avg_col)
    if metric_primary is None:
        raise RuntimeError("No WAR or OPS column found for hitters")
    if metric_primary not in df.columns:
        raise RuntimeError(f"Primary metric {metric_primary} not in hitter columns")

    df = df.sort_values([metric_primary, metric_secondary] if metric_secondary else [metric_primary], ascending=False)
    df = df.head(40).reset_index(drop=True)
    df["metric_primary"] = df[metric_primary]
    if metric_secondary:
        df["metric_secondary"] = df[metric_secondary]
    else:
        df["metric_secondary"] = pd.NA
    df = add_rank(df, ascending=False)
    return df


def build_pitching_leaders(pit: pd.DataFrame) -> pd.DataFrame:
    pit = norm_cols(pit)
    for col in pit.columns:
        if col not in {"player_name", "team_name"}:
            try:
                pit[col] = pd.to_numeric(pit[col], errors="coerce")
            except Exception:
                pass
    pit = pit[pit["player_name"].notna()]
    pit = pit[pit["player_name"] != "Player"]
    if pit.empty:
        return pd.DataFrame(columns=["player_name", "team_id", "team_abbr", "team_name", "conf", "division", "metric_primary", "metric_secondary", "rank_overall"])
    ip_col = detect_col(pit, ["IP"])
    war_col = detect_col(pit, ["WAR"])
    era_col = detect_col(pit, ["ERA"])
    so_col = detect_col(pit, ["SO"])

    df = pit.copy()
    if ip_col and pd.api.types.is_numeric_dtype(df[ip_col]):
        df_primary = df[df[ip_col] >= 150]
        log(f"[INFO] Pitchers with IP>=150: {len(df_primary)}")
    else:
        df_primary = df
        log("[WARN] No IP column found; skipping IP filter")

    metric_primary = war_col if war_col else era_col
    if metric_primary is None:
        raise RuntimeError("No WAR or ERA column found for pitchers")
    metric_secondary = era_col if war_col else (so_col if so_col else None)

    ascending_primary = False if metric_primary == war_col else True
    df_primary = df_primary.sort_values(
        [metric_primary, metric_secondary] if metric_secondary else [metric_primary],
        ascending=[ascending_primary, False] if metric_secondary else ascending_primary,
    )
    # ensure top relievers by WAR even if IP low
    if war_col and ip_col:
        extra = df[df[ip_col] < 150].sort_values(war_col, ascending=False).head(5)
        df_primary = pd.concat([df_primary, extra], ignore_index=True)
        df_primary = df_primary.drop_duplicates(subset=["player_name", "team_id"], keep="first")

    df_primary = df_primary.head(30).reset_index(drop=True)
    df_primary["metric_primary"] = df_primary[metric_primary]
    if metric_secondary:
        df_primary["metric_secondary"] = df_primary[metric_secondary]
    else:
        df_primary["metric_secondary"] = pd.NA
    df_primary = add_rank(df_primary, ascending=ascending_primary)
    return df_primary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build player leaderboards from extracted player stats.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--league-id", type=int, required=True)
    args = parser.parse_args()
    season = args.season
    league_id = args.league_id

    root = Path("csv/out/almanac") / str(season)
    bat_path = root / f"player_batting_{season}_league{league_id}.csv"
    pit_path = root / f"player_pitching_{season}_league{league_id}.csv"

    bat = pd.read_csv(bat_path)
    pit = pd.read_csv(pit_path)

    hit_leaders = build_hitting_leaders(bat)
    pit_leaders = build_pitching_leaders(pit)

    for df in (hit_leaders, pit_leaders):
        df["season"] = season
        df["league_id"] = league_id
        for col in ["team_id", "team_abbr", "team_name", "conf", "division", "player_name", "metric_primary", "metric_secondary", "rank_overall"]:
            if col not in df.columns:
                df[col] = pd.NA

    hit_out = root / f"player_hitting_leaders_{season}_league{league_id}.csv"
    pit_out = root / f"player_pitching_leaders_{season}_league{league_id}.csv"
    hit_leaders.to_csv(hit_out, index=False)
    pit_leaders.to_csv(pit_out, index=False)

    log(f"[OK] Wrote hitters: {hit_out} ({len(hit_leaders)} rows)")
    if not hit_leaders.empty:
        log(hit_leaders.head()[["player_name", "metric_primary", "metric_secondary", "rank_overall"]])
    log(f"[OK] Wrote pitchers: {pit_out} ({len(pit_leaders)} rows)")
    if not pit_leaders.empty:
        log(pit_leaders.head()[["player_name", "metric_primary", "metric_secondary", "rank_overall"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
