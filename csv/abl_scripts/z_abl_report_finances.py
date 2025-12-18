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
            "Balance": format_money(row.get("profit")),
        }
        rows.append(entry)

    output_df = pd.DataFrame(rows, columns=["Team", "Budget", "Payroll", "Cash", "Revenue", "Balance"])

    payroll_vals = pd.to_numeric(output_df["Payroll"].str.replace(r"[,$]", "", regex=True), errors="coerce")
    budget_vals = pd.to_numeric(output_df["Budget"].str.replace(r"[,$]", "", regex=True), errors="coerce")
    metric = payroll_vals if payroll_vals.notna().any() else budget_vals
    output_df["__metric"] = metric

    def assign_tier(series: pd.Series) -> pd.Series:
        if series.notna().sum() < 4:
            return pd.Series(["N/A"] * len(series))
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        tiers = []
        for val in series:
            if pd.isna(val):
                tiers.append("N/A")
            elif val >= q3:
                tiers.append("High")
            elif val <= q1:
                tiers.append("Low")
            else:
                tiers.append("Mid")
        return pd.Series(tiers)

    output_df["Tier"] = assign_tier(metric)

    budget_nonnull = output_df[metric.notna()]
    top_field = "Payroll" if payroll_vals.notna().any() else "Budget"
    big_budget = budget_nonnull.sort_values("__metric", ascending=False).head(5)
    low_budget = budget_nonnull.sort_values("__metric", ascending=True).head(5)

    md_lines = ["# ABL Team Finances", ""]
    if not big_budget.empty and not low_budget.empty:
        md_lines.append(f"## {top_field} Board")
        md_lines.append("### Top 5")
        md_lines.append(md_table(big_budget[["Team", top_field]], ["Team", top_field]))
        md_lines.append("### Bottom 5")
        md_lines.append(md_table(low_budget[["Team", top_field]], ["Team", top_field]))
    else:
        notes.append("Budget Board omitted (insufficient numeric payroll/budget).")

    md_lines.append(md_table(output_df.drop(columns="__metric"), ["Team", "Budget", "Payroll", "Cash", "Revenue", "Balance", "Tier"]))

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
        md_lines.extend(f"- {n}".replace("profit", "balance") for n in notes)
        md_lines.append("- Balance is sourced from financial_balance (OOTP export field name).")
    else:
        md_lines.append("- None")
        md_lines.append("- Balance is sourced from financial_balance (OOTP export field name).")

    out_path = base / "csv" / "out" / "text_out" / "pregame" / "finances.md"
    write_md(out_path, "\n".join(md_lines).rstrip() + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
