#!/usr/bin/env python
"""EB player context brief (financials, prospects, preseason) for a season/league."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def log(msg: str) -> None:
    print(msg)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        log(f"[WARN] Missing {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    log(f"[INFO] Loaded {len(df)} rows from {path}")
    log(f"[DEBUG] Columns in {path.name}: {df.columns.tolist()}")
    return df


def numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace({r"[$,]": ""}, regex=True), errors="coerce")


def build_financial_section(fin_df: pd.DataFrame) -> List[str]:
    lines = ["## Highest-paid players"]
    if fin_df.empty:
        lines.append("- Data not available.")
        lines.append("")
        return lines
    salary_col = next((c for c in fin_df.columns if "salary" in c.lower()), None)
    if not salary_col:
        lines.append("- Salary column missing; skipping.")
        lines.append("")
        return lines
    fin_df[salary_col] = numeric_series(fin_df[salary_col])
    top = fin_df.sort_values(salary_col, ascending=False).head(5)
    for _, r in top.iterrows():
        team = r.get("team_abbr") or r.get("team_name_y") or r.get("team_name_x") or ""
        lines.append(f"- {r.get('player_name','?')} ({team}) — salary: {r[salary_col]:,.0f}")
    lines.append("")
    return lines


def build_prospect_section(pro_df: pd.DataFrame) -> List[str]:
    lines = ["## Top prospects on the horizon"]
    if pro_df.empty:
        lines.append("- Data not available.")
        lines.append("")
        return lines
    rank_col = "#" if "#" in pro_df.columns else None
    if rank_col:
        pro_df[rank_col] = numeric_series(pro_df[rank_col])
        pro_df = pro_df.sort_values(rank_col)
    top = pro_df.head(10)
    for _, r in top.iterrows():
        team = r.get("team_abbr") or r.get("team_name_y") or r.get("team_name_x") or ""
        pos = r.get("Pos") or r.get("position") or ""
        lines.append(f"- {r.get('player_name','?')} ({team}) — {pos}")
    lines.append("")
    return lines


def build_preseason_section(pre_df: pd.DataFrame, bat_df: pd.DataFrame, pit_df: pd.DataFrame) -> List[str]:
    lines = ["## Preseason hype – who delivered?"]
    if pre_df.empty:
        lines.append("- Data not available.")
        lines.append("")
        return lines
    if "player_name" not in pre_df.columns:
        lines.append("- No player_name column; skipping.")
        lines.append("")
        return lines
    pre_players = pre_df.dropna(subset=["player_name"]).head(10)
    for _, r in pre_players.iterrows():
        name = r["player_name"]
        stat_line = ""
        found = False
        if not bat_df.empty:
            match = bat_df[bat_df["player_name"] == name]
            if not match.empty:
                m = match.iloc[0]
                slash = "/".join(f"{float(m[c]):.3f}" for c in ["AVG", "OBP", "SLG"] if c in match.columns and pd.notna(m.get(c)))
                stat_line = slash or ""
                found = True
        if not found and not pit_df.empty:
            match = pit_df[pit_df["player_name"] == name]
            if not match.empty:
                m = match.iloc[0]
                if "ERA" in match.columns:
                    stat_line = f"ERA {m['ERA']}"
                found = True
        verdict = "delivered" if found else "no stat found"
        lines.append(f"- {name} — {verdict} ({stat_line})")
    lines.append("")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="EB player context brief (financials, prospects, preseason).")
    parser.add_argument("--season", type=int, default=1972)
    parser.add_argument("--league-id", type=int, default=200)
    args = parser.parse_args()
    season = args.season
    league_id = args.league_id

    base = Path("csv/out/almanac") / str(season)
    financials = load_csv(base / f"player_financials_{season}_league{league_id}.csv")
    prospects = load_csv(base / f"player_top_prospects_{season}_league{league_id}.csv")
    preseason = load_csv(base / f"preseason_player_predictions_{season}_league{league_id}.csv")
    batting = load_csv(base / f"player_batting_{season}_league{league_id}.csv")
    pitching = load_csv(base / f"player_pitching_{season}_league{league_id}.csv")

    md_lines: List[str] = []
    md_lines.append(f"# EB Player Context {season} – Data Brief (DO NOT PUBLISH)")
    md_lines.append(f"_League ID {league_id}_")
    md_lines.append("")
    md_lines.extend(build_financial_section(financials))
    md_lines.extend(build_prospect_section(prospects))
    md_lines.extend(build_preseason_section(preseason, batting, pitching))

    out_path = base / f"eb_player_context_{season}_league{league_id}.md"
    out_path.write_text("\n".join(md_lines), encoding="utf-8")
    log(f"[OK] Wrote player context brief to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
