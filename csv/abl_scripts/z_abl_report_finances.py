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


def parse_numeric(val) -> pd.Series:
    """Best-effort numeric parser that handles trailing m (millions)."""
    if pd.isna(val):
        return pd.Series([pd.NA])
    if isinstance(val, str):
        txt = val.strip().lower()
        txt = txt.replace("$", "").replace(",", "")
        multiplier = 1
        if txt.endswith("m"):
            multiplier = 1_000_000
            txt = txt[:-1]
        num = pd.to_numeric(pd.Series([txt]), errors="coerce")
        if num.notna().iloc[0]:
            return num * multiplier
    return pd.to_numeric(pd.Series([val]), errors="coerce")


def format_money(val) -> str:
    if pd.isna(val):
        return "N/A"
    num = parse_numeric(val).iloc[0]
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

    def to_numeric_series(series: pd.Series | None) -> pd.Series:
        if series is None:
            return pd.Series(dtype=float)
        return series.apply(lambda v: parse_numeric(v).iloc[0])

    payroll_raw = to_numeric_series(fin_df.get("payroll") if "payroll" in fin_df.columns else None)
    budget_raw = to_numeric_series(fin_df.get("budget") if "budget" in fin_df.columns else None)
    payroll_count = int(payroll_raw.notna().sum()) if isinstance(payroll_raw, pd.Series) else 0
    budget_count = int(budget_raw.notna().sum()) if isinstance(budget_raw, pd.Series) else 0
    metric_raw = payroll_raw if payroll_count > 0 else budget_raw
    metric_label = "Payroll" if payroll_count > 0 else "Budget"

    def assign_tier(series: pd.Series) -> pd.Series:
        series = pd.to_numeric(series, errors="coerce")
        if series.notna().sum() < 8:
            return pd.Series(["N/A"] * len(series), index=series.index)
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
        return pd.Series(tiers, index=series.index)

    fin_df = fin_df.copy()
    fin_df["tier_calc"] = assign_tier(metric_raw) if not fin_df.empty else pd.Series(dtype=object)

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
            "Tier": row.get("tier_calc") if pd.notna(row.get("tier_calc")) else "N/A",
        }
        rows.append(entry)

    output_df = pd.DataFrame(rows, columns=["Team", "Budget", "Payroll", "Cash", "Revenue", "Balance", "Tier"])
    # Attach raw metric for boards (before formatting)
    if not fin_df.empty:
        output_df["__metric"] = metric_raw.reset_index(drop=True)
    else:
        output_df["__metric"] = pd.Series(dtype=float)

    metric_nonnull = output_df[output_df["__metric"].notna()]
    big_budget = metric_nonnull.sort_values(by="__metric", ascending=False).head(5) if not metric_nonnull.empty else pd.DataFrame()
    low_budget = metric_nonnull.sort_values(by="__metric", ascending=True).head(5) if not metric_nonnull.empty else pd.DataFrame()

    md_lines = ["# ABL Team Finances", ""]
    if not big_budget.empty and not low_budget.empty:
        md_lines.append(f"## {metric_label} Board")
        md_lines.append("### Top 5")
        big_board = big_budget.copy()
        big_board[metric_label] = big_board["__metric"].apply(format_money)
        md_lines.append(md_table(big_board[["Team", metric_label]], ["Team", metric_label]))
        md_lines.append("### Bottom 5")
        low_board = low_budget.copy()
        low_board[metric_label] = low_board["__metric"].apply(format_money)
        md_lines.append(md_table(low_board[["Team", metric_label]], ["Team", metric_label]))
    else:
        notes.append("Budget Board omitted (insufficient numeric payroll/budget).")

    md_lines.append(md_table(output_df, ["Team", "Budget", "Payroll", "Cash", "Revenue", "Balance", "Tier"]))

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
