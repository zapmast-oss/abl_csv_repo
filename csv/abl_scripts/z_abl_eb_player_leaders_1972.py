#!/usr/bin/env python
"""EB player leaders brief for a season/league (defaults 1972/200)."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def log(msg: str) -> None:
    print(msg)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    log(f"[INFO] Loaded {len(df)} rows from {path}")
    log(f"[DEBUG] Columns in {path.name}: {df.columns.tolist()}")
    return df


def find_best(df: pd.DataFrame, stat: str, ascending: bool, qualifier: Optional[pd.Series] = None) -> Optional[pd.Series]:
    if stat not in df.columns:
        log(f"[WARN] Missing stat {stat}; skipping leader.")
        return None
    working = df.copy()
    working[stat] = pd.to_numeric(working[stat], errors="coerce")
    if qualifier is not None:
        working = working[qualifier]
    working = working.dropna(subset=[stat])
    if working.empty:
        log(f"[WARN] No rows after filtering for {stat}.")
        return None
    idx = working[stat].idxmin() if ascending else working[stat].idxmax()
    return working.loc[idx]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build EB player leaders brief for a season/league.")
    parser.add_argument("--season", type=int, default=1972)
    parser.add_argument("--league-id", type=int, default=200)
    args = parser.parse_args()
    season = args.season
    league_id = args.league_id

    base = Path("csv/out/almanac") / str(season)
    dim_team = pd.read_csv("csv/out/star_schema/dim_team_park.csv")
    batting_full = load_csv(base / f"player_batting_{season}_league{league_id}.csv")
    pitching_full = load_csv(base / f"player_pitching_{season}_league{league_id}.csv")
    hitting_leaders = load_csv(base / f"player_hitting_leaders_{season}_league{league_id}.csv")
    pitching_leaders = load_csv(base / f"player_pitching_leaders_{season}_league{league_id}.csv")

    dim_team_cols = {c.lower(): c for c in dim_team.columns}
    team_key = dim_team[[dim_team_cols.get("abbr", dim_team_cols.get("team_abbr", "Abbr")), dim_team_cols.get("team name", dim_team_cols.get("team_name", "Team Name"))]].copy()
    team_key.columns = ["team_abbr", "team_name_dim"]

    # Helper to add team name from abbr
    def add_team_info(row: pd.Series) -> str:
        abbr = row.get("team_abbr")
        if pd.isna(abbr):
            return ""
        match = team_key[team_key["team_abbr"] == abbr]
        if not match.empty:
            return match.iloc[0]["team_name_dim"]
        return str(row.get("team_name") or "")

    # Determine qualifiers
    pa_col = "PA" if "PA" in batting_full.columns else None
    ab_col = "AB" if "AB" in batting_full.columns else None
    if pa_col:
        batting_full[pa_col] = pd.to_numeric(batting_full[pa_col], errors="coerce")
        qualifier = batting_full[pa_col] >= 400
    elif ab_col:
        batting_full[ab_col] = pd.to_numeric(batting_full[ab_col], errors="coerce")
        qualifier = batting_full[ab_col] >= 350
    else:
        qualifier = pd.Series([True] * len(batting_full))

    leaders = []
    for stat, asc in [("HR", False), ("SB", False), ("AVG", False), ("OPS", False)]:
        row = find_best(batting_full, stat, ascending=asc, qualifier=qualifier)
        if row is None:
            continue
        team_name = add_team_info(row)
        leaders.append({"stat_name": stat, "stat_value": row[stat], "player_name": row["player_name"], "team_abbr": row.get("team_abbr"), "team_name": team_name})

    # Pitching qualifiers
    ip_col = "IP" if "IP" in pitching_full.columns else None
    if ip_col:
        # convert IP to float cautiously
        def ip_to_float(val):
            try:
                if isinstance(val, str) and "." in val:
                    whole, frac = val.split(".")
                    return float(whole) + float(frac)/10
                return float(val)
            except Exception:
                return pd.NA
        pitching_full[ip_col] = pitching_full[ip_col].map(ip_to_float)
        pitcher_qual = pitching_full[ip_col] >= 162
    else:
        pitcher_qual = pd.Series([True] * len(pitching_full))

    for stat, asc in [("ERA", True), ("SO", False), ("SV", False)]:
        row = find_best(pitching_full, stat, ascending=asc, qualifier=pitcher_qual)
        if row is None:
            continue
        team_name = add_team_info(row)
        leaders.append({"stat_name": stat, "stat_value": row[stat], "player_name": row["player_name"], "team_abbr": row.get("team_abbr"), "team_name": team_name})

    if not leaders:
        raise RuntimeError("No leaders found to report.")

    md_lines = []
    md_lines.append(f"# EB Player Leaders {season} – Data Brief (DO NOT PUBLISH)")
    md_lines.append(f"_League ID {league_id}_")
    md_lines.append("")
    md_lines.append("## League batting leaders")
    for leader in leaders:
        if leader["stat_name"] in {"HR", "SB", "AVG", "OPS"}:
            md_lines.append(f"- {leader['stat_name']} leader: {leader['player_name']} ({leader.get('team_abbr','')}) — {leader['stat_value']}")
    md_lines.append("")
    md_lines.append("## League pitching leaders")
    for leader in leaders:
        if leader["stat_name"] in {"ERA", "SO", "SV"}:
            md_lines.append(f"- {leader['stat_name']} leader: {leader['player_name']} ({leader.get('team_abbr','')}) — {leader['stat_value']}")
    md_lines.append("")

    out_path = base / f"eb_player_leaders_{season}_league{league_id}.md"
    out_path.write_text("\n".join(md_lines), encoding="utf-8")
    log(f"[OK] Wrote player leaders brief to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
