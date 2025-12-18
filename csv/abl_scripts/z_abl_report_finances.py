"""Pregame finances report."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd

from _abl_pregame_utils import (
    coalesce_team_values,
    list_all_csv_paths,
    load_best_csv,
    load_team_keyed_frame,
    md_table,
    normalize_team_table,
    resolve_base,
    safe_get,
    scan_csv_headers_for_columns,
    write_md,
)

TEAM_PATTERNS = ["dim_team", "teams", "team.csv", "team_list"]
FIN_PATTERNS = ["financial", "finances", "team_financial", "team_budget", "payroll", "revenue"]

BUDGET_COLS = ["budget", "team_budget", "budget_total", "player_budget"]
PAYROLL_COLS = ["payroll", "team_payroll", "salary", "salary_total"]
CASH_COLS = ["cash", "cash_on_hand", "balance"]
REV_COLS = ["revenue", "total_revenue"]
PROFIT_COLS = ["profit", "net", "net_income"]
FIN_NEEDLES = [
    "budget",
    "payroll",
    "revenue",
    "profit",
    "cash",
    "income",
    "expense",
    "expenses",
    "financial",
    "financ",
    "salary",
    "salaries",
    "balance",
]
TEAM_KEY_CANDIDATES = ["team_id", "team_abbr", "team_name", "league_id", "ID", "Abbr"]
VALUE_COLS = ["Budget", "Payroll", "Cash", "Revenue", "Profit"]


def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    series = safe_get(df, candidates)
    return series.name if isinstance(series, pd.Series) else None


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ABL finances report.")
    parser.add_argument("--base", help="Repo root (optional).")
    parser.add_argument("--league_id", type=int, default=200)
    args = parser.parse_args()

    base = resolve_base(args.base)
    league_id = args.league_id

    teams_df_raw, team_path = load_best_csv(base, TEAM_PATTERNS)
    teams = normalize_team_table(teams_df_raw)
    if "league_id" in teams.columns:
        teams = teams[(teams["league_id"].isna()) | (teams["league_id"] == league_id)]
    teams = teams.reset_index(drop=True)

    fin_df, fin_path = load_best_csv(base, FIN_PATTERNS)
    notes: List[str] = []
    header_scan_files: List[Path] = []
    scan_errors: List[Path] = []
    scanned_count = 0

    budget_col = pick_col(fin_df, BUDGET_COLS)
    payroll_col = pick_col(fin_df, PAYROLL_COLS)
    cash_col = pick_col(fin_df, CASH_COLS)
    revenue_col = pick_col(fin_df, REV_COLS)
    profit_col = pick_col(fin_df, PROFIT_COLS)

    def merge_key(df: pd.DataFrame) -> pd.Series:
        if "team_id" in df.columns:
            return df["team_id"].apply(lambda v: f"id:{int(v)}" if pd.notna(v) else "")
        if "team_abbr" in df.columns:
            return df["team_abbr"].fillna("").astype(str).str.upper()
        if "team_name" in df.columns:
            return df["team_name"].fillna("").astype(str).str.lower()
        return pd.Series([""] * len(df))

    teams = teams.copy()
    teams["__merge_key"] = merge_key(teams)

    merged = teams.copy()
    fast_path_used = False
    if not fin_df.empty:
        fin_df = fin_df.copy()
        abbr_series = safe_get(fin_df, ["team_abbr", "abbr", "team_code", "team"])
        if isinstance(abbr_series, pd.Series):
            fin_df["team_abbr"] = abbr_series
        fin_df["__merge_key"] = merge_key(fin_df)
        merged = merged.merge(fin_df, on="__merge_key", how="left", suffixes=("", "_fin"))
        fast_path_used = True

    rows = []
    for _, row in merged.iterrows():
        entry = {
            "Team": row.get("team_abbr") or row.get("team_name") or "N/A",
            "Budget": "N/A",
            "Payroll": "N/A",
            "Cash": "N/A",
            "Revenue": "N/A",
            "Profit": "N/A",
        }
        if budget_col:
            val = numeric(pd.Series([row.get(budget_col)])).iloc[0]
            entry["Budget"] = f"{val:,.0f}" if pd.notna(val) else "N/A"
        if payroll_col:
            val = numeric(pd.Series([row.get(payroll_col)])).iloc[0]
            entry["Payroll"] = f"{val:,.0f}" if pd.notna(val) else "N/A"
        if cash_col:
            val = numeric(pd.Series([row.get(cash_col)])).iloc[0]
            entry["Cash"] = f"{val:,.0f}" if pd.notna(val) else "N/A"
        if revenue_col:
            val = numeric(pd.Series([row.get(revenue_col)])).iloc[0]
            entry["Revenue"] = f"{val:,.0f}" if pd.notna(val) else "N/A"
        if profit_col:
            val = numeric(pd.Series([row.get(profit_col)])).iloc[0]
            entry["Profit"] = f"{val:,.0f}" if pd.notna(val) else "N/A"
        rows.append(entry)

    output_df = pd.DataFrame(rows)

    needs_fallback = output_df[["Budget", "Payroll", "Cash", "Revenue", "Profit"]].replace("N/A", pd.NA).isna().all(axis=None)
    if needs_fallback:
        all_csvs = list_all_csv_paths(base)
        scanned_count = len(all_csvs)
        hits, scan_errors = scan_csv_headers_for_columns(all_csvs, FIN_NEEDLES)
        top_hits = hits[:10]
        frames: List[tuple[pd.DataFrame, Path]] = []
        for path, cols in top_hits:
            keep_cols = TEAM_KEY_CANDIDATES + cols
            try:
                frame, used = load_team_keyed_frame(path, TEAM_KEY_CANDIDATES, keep_cols)
            except Exception:
                continue
            if frame.empty:
                continue
            rename_map = {}
            for col in frame.columns:
                low = col.lower()
                if low in [c.lower() for c in BUDGET_COLS]:
                    rename_map[col] = "Budget"
                elif low in [c.lower() for c in PAYROLL_COLS]:
                    rename_map[col] = "Payroll"
                elif low in [c.lower() for c in CASH_COLS]:
                    rename_map[col] = "Cash"
                elif low in [c.lower() for c in REV_COLS]:
                    rename_map[col] = "Revenue"
                elif low in [c.lower() for c in PROFIT_COLS]:
                    rename_map[col] = "Profit"
            frame = frame.rename(columns=rename_map)
            frames.append((frame, path))
        if frames:
            coalesced, used_files = coalesce_team_values(teams, frames, VALUE_COLS)
            header_scan_files.extend(used_files)
            rows = []
            for _, row in coalesced.iterrows():
                entry = {
                    "Team": row.get("team_abbr") or row.get("team_name") or "N/A",
                    "Budget": "N/A",
                    "Payroll": "N/A",
                    "Cash": "N/A",
                    "Revenue": "N/A",
                    "Profit": "N/A",
                }
                for col in VALUE_COLS:
                    val = row.get(col)
                    if pd.isna(val):
                        continue
                    num_val = pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0]
                    entry[col] = f"{num_val:,.0f}" if pd.notna(num_val) else str(val)
                rows.append(entry)
            output_df = pd.DataFrame(rows)
        else:
            notes.append("Header scan found no usable finance columns.")
            if scanned_count:
                notes.append(f"No usable sources found (scanned {scanned_count} CSV headers).")

    budget_values = pd.to_numeric(output_df["Budget"].str.replace(",", "", regex=False), errors="coerce")
    output_df["__budget_val"] = budget_values
    budget_nonnull = output_df[budget_values.notna()]
    big_budget = budget_nonnull.sort_values("__budget_val", ascending=False).head(5)
    low_budget = budget_nonnull.sort_values("__budget_val", ascending=True).head(5)

    if fin_df.empty:
        notes.append("No finance CSV found; all values set to N/A.")
    if budget_col is None:
        notes.append("Budget column not found.")
    if payroll_col is None:
        notes.append("Payroll column not found.")

    md_lines = ["# ABL Team Finances", ""]
    md_lines.append(md_table(output_df.drop(columns="__budget_val"), ["Team", "Budget", "Payroll", "Cash", "Revenue", "Profit"]))

    if not big_budget.empty:
        md_lines.append("## Big Budget Teams")
        md_lines.append(md_table(big_budget[["Team", "Budget"]], ["Team", "Budget"]))
    if not low_budget.empty:
        md_lines.append("## Low Budget Teams")
        md_lines.append(md_table(low_budget[["Team", "Budget"]], ["Team", "Budget"]))

    md_lines.append("## Data Sources")
    md_lines.append(f"- Teams: {team_path if team_path else 'None found'}")
    md_lines.append(f"- Finances (fast path): {fin_path if fast_path_used else 'None used'}")
    if header_scan_files:
        for f in header_scan_files:
            md_lines.append(f"- Header scan: {f}")
    elif scanned_count:
        md_lines.append("- Header scan: none used")
    md_lines.append(f"- Scanned {scanned_count} CSV headers for finance fields")
    if scan_errors:
        md_lines.append(f"- Header scan errors on {len(scan_errors)} file(s)")
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
