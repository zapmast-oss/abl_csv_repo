#!/usr/bin/env python
"""EB schedule context brief for a season/league (default 1972/200)."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

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


def longest_road_trip(team_df: pd.DataFrame) -> int:
    if team_df.empty or "home_away" not in team_df.columns:
        return 0
    team_df = team_df.sort_values("date")
    best = cur = 0
    for _, r in team_df.iterrows():
        if str(r.get("home_away", "")).upper().startswith("A"):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description="EB schedule context brief.")
    parser.add_argument("--season", type=int, default=1972)
    parser.add_argument("--league-id", type=int, default=200)
    args = parser.parse_args()
    season = args.season
    league_id = args.league_id

    base = Path("csv/out/almanac") / str(season)
    league = load_csv(base / f"league_season_summary_{season}_league{league_id}.csv")
    eval_df = load_csv(base / f"schedule_evaluator_{season}_league{league_id}.csv")
    sched_df = load_csv(base / f"team_schedule_{season}_league{league_id}.csv")

    md: List[str] = []
    md.append(f"# EB Schedule Context {season} – Data Brief (DO NOT PUBLISH)")
    md.append(f"_League ID {league_id}_")
    md.append("")

    # Overview
    md.append("## Schedule Overview")
    if not sched_df.empty:
        games_per_team = sched_df.groupby("team_abbr").size().describe()
        home_counts = sched_df[sched_df["home_away"] == "H"].groupby("team_abbr").size()
        road_counts = sched_df[sched_df["home_away"] == "A"].groupby("team_abbr").size()
        md.append(f"- Games per team (count describe): {games_per_team.to_dict()}")
        if not home_counts.empty and not road_counts.empty:
            max_home = home_counts.idxmax()
            max_road = road_counts.idxmax()
            md.append(f"- Most home games: {max_home} ({home_counts[max_home]})")
            md.append(f"- Most road games: {max_road} ({road_counts[max_road]})")
    else:
        md.append("- Schedule grid missing.")
    md.append("")

    # Brutal stretches: longest road trips
    md.append("## Brutal Stretches")
    road_info = []
    for team, grp in sched_df.groupby("team_abbr"):
        road_info.append((team, longest_road_trip(grp)))
    road_info = sorted(road_info, key=lambda x: x[1], reverse=True)[:5]
    for team, length in road_info:
        md.append(f"- {team}: longest road trip {length} games")
    md.append("")

    # Soft landings from schedule evaluator
    md.append("## Soft Landings")
    sos_cols = [c for c in eval_df.columns if "sos" in c.lower() or "strength" in c.lower()]
    if sos_cols:
        col = sos_cols[0]
        eval_df[col] = pd.to_numeric(eval_df[col], errors="coerce")
        best = eval_df.sort_values(col).head(3)
        worst = eval_df.sort_values(col, ascending=False).head(3)
        md.append(f"- Softest schedules ({col}):")
        for _, r in best.iterrows():
            md.append(f"  - {r.get('Team', r.get('team', ''))}: {col}={r[col]}")
        md.append(f"- Toughest schedules ({col}):")
        for _, r in worst.iterrows():
            md.append(f"  - {r.get('Team', r.get('team', ''))}: {col}={r[col]}")
    else:
        md.append("- [WARN] No SOS/strength column detected in evaluator.")
    md.append("")

    # All-Star break detection
    md.append("## All-Star Break")
    if "is_allstar_break" in sched_df.columns:
        as_dates = sched_df[sched_df["is_allstar_break"] == True]["date"].unique().tolist()
        md.append(f"- Detected all-star break dates: {as_dates}")
    else:
        md.append("- [WARN] All-star flag not found in schedule grid.")
    md.append("")

    out_path = base / f"eb_schedule_context_{season}_league{league_id}.md"
    out_path.write_text("\n".join(md), encoding="utf-8")
    log(f"[OK] Wrote schedule context brief to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
