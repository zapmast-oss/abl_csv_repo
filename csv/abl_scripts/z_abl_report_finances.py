"""Pregame finances report (source-of-truth from known finance files)."""

from __future__ import annotations

import argparse
from typing import List

import pandas as pd

from _abl_pregame_utils import (
    load_team_financials,
    md_table,
    resolve_base,
    write_md,
)


def format_money(val) -> str:
    if pd.isna(val):
        return "N/A"
    num = pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0]
    if pd.notna(num):
        return f"${num:,.0f}"
    return str(val)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ABL finances report.")
    parser.add_argument("--base", help="Repo root (optional).")
    parser.add_argument("--league_id", type=int, default=200)
    parser.add_argument("--season", type=int, required=False)
    args = parser.parse_args()

    base = resolve_base(args.base)
    league_id = args.league_id
    season = args.season

    fin_df, sources_used, load_notes = load_team_financials(base, league_id=league_id, season=season)
    notes: List[str] = list(load_notes)

    rows = []
    for _, row in fin_df.iterrows():
        team_abbr = row.get("team_abbr")
        team_name = row.get("team_name")
        team_label = team_abbr if pd.notna(team_abbr) else (team_name if pd.notna(team_name) else "N/A")
        if pd.notna(team_abbr) and pd.notna(team_name):
            team_label = f"{team_abbr} - {team_name}"
        entry = {
            "Team": team_label,
            "Budget": format_money(row.get("budget")),
            "Payroll": format_money(row.get("payroll")),
            "Cash": format_money(row.get("cash")),
            "Revenue": format_money(row.get("revenue")),
            "Profit": format_money(row.get("profit")),
        }
        rows.append(entry)

    output_df = pd.DataFrame(rows, columns=["Team", "Budget", "Payroll", "Cash", "Revenue", "Profit"])

    budget_values = pd.to_numeric(output_df["Budget"].str.replace(r"[,$]", "", regex=True), errors="coerce")
    output_df["__budget_val"] = budget_values
    budget_nonnull = output_df[budget_values.notna()]
    big_budget = budget_nonnull.sort_values("__budget_val", ascending=False).head(5)
    low_budget = budget_nonnull.sort_values("__budget_val", ascending=True).head(5)

    md_lines = ["# ABL Team Finances", ""]
    md_lines.append(md_table(output_df.drop(columns="__budget_val"), ["Team", "Budget", "Payroll", "Cash", "Revenue", "Profit"]))

    if not big_budget.empty:
        md_lines.append("## Big Budget Teams")
        md_lines.append(md_table(big_budget[["Team", "Budget"]], ["Team", "Budget"]))
    if not low_budget.empty:
        md_lines.append("## Low Budget Teams")
        md_lines.append(md_table(low_budget[["Team", "Budget"]], ["Team", "Budget"]))

    md_lines.append("## Data Sources")
    if sources_used:
        for src in sources_used:
            md_lines.append(f"- {src}")
    else:
        md_lines.append("- None found")
    if season is not None:
        md_lines.append(f"- Selected season: {season}")
    md_lines.append("")
    md_lines.append("## Notes")
    if notes:
        md_lines.extend(f"- {n}" for n in notes)
    else:
        md_lines.append("- None")

    out_path = base / "csv" / "out" / "text_out" / "pregame" / "finances.md"
    write_md(out_path, "\n".join(md_lines).rstrip() + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
