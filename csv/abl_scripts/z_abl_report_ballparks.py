"""Pregame ballparks report."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd

from _abl_pregame_utils import (
    find_csv_files,
    load_best_csv,
    md_table,
    normalize_team_table,
    resolve_base,
    safe_get,
    write_md,
)


TEAM_PATTERNS = ["dim_team", "teams", "team.csv", "team_list"]
BALLPARK_PATTERNS = ["dim_team_park", "parks", "ballpark", "park_factors", "team_park"]


def build_merge_key(df: pd.DataFrame) -> pd.Series:
    """Best-effort merge key: team_id if present, else abbr, else name."""
    key = pd.Series([""] * len(df))
    if "team_id" in df.columns:
        key = df["team_id"].apply(lambda v: f"id:{int(v)}" if pd.notna(v) else "")
    elif "team_abbr" in df.columns:
        key = df["team_abbr"].fillna("").astype(str).str.upper()
    elif "team_name" in df.columns:
        key = df["team_name"].fillna("").astype(str).str.lower()
    return key


def pick_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    series = safe_get(df, candidates)
    return series.name if isinstance(series, pd.Series) else None


def collect_factor_columns(df: pd.DataFrame) -> List[str]:
    targets = []
    for col in df.columns:
        low = col.lower()
        if any(tok in low for tok in ["factor", "pf_", "pf ", "park_", "pf", "hr", "runs", "1b", "2b", "3b", "bb", "so", "l", "r", "avg d", "avg l", "avg r"]):
            targets.append(col)
    # Deduplicate while preserving order
    seen = set()
    result = []
    for col in targets:
        if col.lower() in seen:
            continue
        seen.add(col.lower())
        result.append(col)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ABL ballparks report.")
    parser.add_argument("--base", help="Repo root (optional).")
    parser.add_argument("--league_id", type=int, default=200)
    args = parser.parse_args()

    base = resolve_base(args.base)
    league_id = args.league_id

    teams_df_raw, team_path = load_best_csv(base, TEAM_PATTERNS)
    teams_norm = normalize_team_table(teams_df_raw)
    if "league_id" in teams_norm.columns:
        teams_norm = teams_norm[(teams_norm["league_id"].isna()) | (teams_norm["league_id"] == league_id)]
    teams_norm = teams_norm.reset_index(drop=True)

    parks_df, parks_path = load_best_csv(base, BALLPARK_PATTERNS)

    notes: List[str] = []

    park_name_col = pick_column(parks_df, ["park_name", "ballpark", "stadium", "park"])
    capacity_col = pick_column(parks_df, ["capacity", "cap"])
    factor_cols = collect_factor_columns(parks_df) if not parks_df.empty else []

    team_key = build_merge_key(teams_norm)
    teams_norm = teams_norm.copy()
    teams_norm["__merge_key"] = team_key

    merged = teams_norm.copy()
    if not parks_df.empty:
        parks_df = parks_df.copy()
        parks_df["__merge_key"] = build_merge_key(parks_df.rename(columns={"team": "team_name"}))
        merged = merged.merge(parks_df, on="__merge_key", how="left", suffixes=("", "_park"))

    rows = []
    for _, row in merged.iterrows():
        entry = {
            "Team": row.get("team_abbr") or row.get("team_name") or "N/A",
            "Park Name": row.get(park_name_col, "N/A") if park_name_col else "N/A",
            "Capacity": "N/A",
        }
        if capacity_col:
            cap_val = pd.to_numeric(row.get(capacity_col), errors="coerce")
            entry["Capacity"] = f"{int(cap_val):,}" if pd.notna(cap_val) else "N/A"
        for col in factor_cols:
            val = row.get(col)
            entry[col] = "N/A" if pd.isna(val) else val
        rows.append(entry)

    output_df = pd.DataFrame(rows)
    if output_df.empty:
        notes.append("No ballpark data found; emitted empty table.")
    if park_name_col is None:
        notes.append("park_name not found; filled Park Name with N/A.")
    if capacity_col is None:
        notes.append("capacity not found; filled Capacity with N/A.")
    if not factor_cols:
        notes.append("No park factor columns found.")

    md_lines = ["# ABL Ballparks", ""]
    md_lines.append(md_table(output_df, output_df.columns.tolist()))
    md_lines.append("## Data Sources")
    md_lines.append(f"- Teams: {team_path if team_path else 'None found'}")
    md_lines.append(f"- Ballparks: {parks_path if parks_path else 'None found'}")
    md_lines.append("")
    md_lines.append("## Notes")
    if notes:
        md_lines.extend(f"- {n}" for n in notes)
    else:
        md_lines.append("- None")

    out_path = base / "csv" / "out" / "text_out" / "pregame" / "ballparks.md"
    write_md(out_path, "\n".join(md_lines).rstrip() + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
