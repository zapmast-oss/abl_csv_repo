"""Pregame fans & markets report."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd

from _abl_pregame_utils import (
    coalesce_team_values,
    find_csv_files,
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
FAN_PATTERNS = ["market", "fan", "attendance", "ticket", "media"]
FAN_NEEDLES = ["market", "fan", "interest", "loyalty", "attendance", "attend", "ticket", "media", "tv", "radio", "merch", "demand", "popul"]

MARKET_COLS = ["market", "market_size", "market_value"]
INTEREST_COLS = ["fan_interest", "interest", "fan_interest_current"]
LOYALTY_COLS = ["fan_loyalty", "loyalty"]
ATTENDANCE_COLS = ["attendance", "att", "home_attendance", "attendance_total", "attendance_ytd"]
TICKET_COLS = ["ticket_price", "avg_ticket_price", "ticket"]
MEDIA_COLS = ["media_revenue", "tv_revenue", "radio_revenue", "local_media"]
TEAM_KEY_CANDIDATES = ["team_id", "team_abbr", "team_name", "league_id", "ID", "Abbr"]
VALUE_COLS = ["Market Size", "Fan Interest", "Fan Loyalty", "Attendance", "Ticket Price", "Media Revenue"]


def merge_key(df: pd.DataFrame) -> pd.Series:
    if "team_id" in df.columns:
        return df["team_id"].apply(lambda v: f"id:{int(v)}" if pd.notna(v) else "")
    if "team_abbr" in df.columns:
        return df["team_abbr"].fillna("").astype(str).str.upper()
    if "team_name" in df.columns:
        return df["team_name"].fillna("").astype(str).str.lower()
    return pd.Series([""] * len(df))


def pick_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    series = safe_get(df, candidates)
    return series.name if isinstance(series, pd.Series) else None


def first_value(row: pd.Series, dfs: List[pd.DataFrame], candidates: Sequence[str]) -> Tuple[str, List[str]]:
    """Return first non-null value for candidates across dfs; collect missing notes."""
    key = row.get("__merge_key", "")
    for df in dfs:
        match = df[df["__merge_key"] == key]
        if match.empty:
            continue
        col = pick_col(match, candidates)
        if col:
            val = match.iloc[0].get(col)
            if pd.notna(val):
                return val, []
    missing_names = [c for c in candidates]
    return "N/A", missing_names


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ABL fans & markets report.")
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
    teams["__merge_key"] = merge_key(teams)

    fan_paths = find_csv_files(base, FAN_PATTERNS)
    fan_dfs: List[pd.DataFrame] = []
    used_paths: List[Path] = []
    for path in fan_paths[:5]:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        df = df.copy()
        abbr_series = safe_get(df, ["team_abbr", "abbr", "team_code", "team"])
        if isinstance(abbr_series, pd.Series):
            df["team_abbr"] = abbr_series
        df["__merge_key"] = merge_key(df)
        fan_dfs.append(df)
        used_paths.append(path)

    notes: List[str] = []
    header_scan_files: List[Path] = []
    scan_errors: List[Path] = []
    scanned_count = 0
    if not fan_dfs:
        notes.append("No fans/market CSVs found; attempting header scan fallback.")

    rows = []
    missing_cols: List[str] = []
    for _, team in teams.iterrows():
        market, miss_mkt = first_value(team, fan_dfs, MARKET_COLS)
        interest, miss_int = first_value(team, fan_dfs, INTEREST_COLS)
        loyalty, miss_loy = first_value(team, fan_dfs, LOYALTY_COLS)
        attendance, miss_att = first_value(team, fan_dfs, ATTENDANCE_COLS)
        ticket, miss_tic = first_value(team, fan_dfs, TICKET_COLS)
        media, miss_med = first_value(team, fan_dfs, MEDIA_COLS)
        missing_cols.extend(miss_mkt + miss_int + miss_loy + miss_att + miss_tic + miss_med)
        rows.append(
            {
                "Team": team.get("team_abbr") or team.get("team_name") or "N/A",
                "Market Size": market if pd.notna(market) else "N/A",
                "Fan Interest": interest if pd.notna(interest) else "N/A",
                "Fan Loyalty": loyalty if pd.notna(loyalty) else "N/A",
                "Attendance": attendance if pd.notna(attendance) else "N/A",
                "Ticket Price": ticket if pd.notna(ticket) else "N/A",
                "Media Revenue": media if pd.notna(media) else "N/A",
            }
        )

    output_df = pd.DataFrame(rows)
    def assign_tier(series: pd.Series) -> pd.Series:
        if series.notna().sum() < 4:
            return pd.Series(["N/A"] * len(series))
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        out = []
        for val in series:
            if pd.isna(val):
                out.append("N/A")
            elif val >= q3:
                out.append("High")
            elif val <= q1:
                out.append("Low")
            else:
                out.append("Mid")
        return pd.Series(out)

    market_numeric = pd.to_numeric(output_df["Market Size"], errors="coerce")
    attendance_numeric = pd.to_numeric(output_df["Attendance"], errors="coerce")
    tier_basis = market_numeric if market_numeric.notna().any() else attendance_numeric
    output_df["Tier"] = assign_tier(tier_basis)
    needs_fallback = output_df[VALUE_COLS].replace("N/A", pd.NA).isna().all(axis=None)
    if needs_fallback:
        all_csvs = list_all_csv_paths(base)
        scanned_count = len(all_csvs)
        hits, scan_errors = scan_csv_headers_for_columns(all_csvs, FAN_NEEDLES)
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
                if low in [c.lower() for c in MARKET_COLS]:
                    rename_map[col] = "Market Size"
                elif low in [c.lower() for c in INTEREST_COLS]:
                    rename_map[col] = "Fan Interest"
                elif low in [c.lower() for c in LOYALTY_COLS]:
                    rename_map[col] = "Fan Loyalty"
                elif low in [c.lower() for c in ATTENDANCE_COLS]:
                    rename_map[col] = "Attendance"
                elif low in [c.lower() for c in TICKET_COLS]:
                    rename_map[col] = "Ticket Price"
                elif low in [c.lower() for c in MEDIA_COLS]:
                    rename_map[col] = "Media Revenue"
            frame = frame.rename(columns=rename_map)
            frames.append((frame, path))
        if frames:
            coalesced, used_files = coalesce_team_values(teams, frames, VALUE_COLS)
            header_scan_files.extend(used_files)
            rows = []
            for _, row in coalesced.iterrows():
                entry = {
                    "Team": row.get("team_abbr") or row.get("team_name") or "N/A",
                    "Market Size": "N/A",
                    "Fan Interest": "N/A",
                    "Fan Loyalty": "N/A",
                    "Attendance": "N/A",
                    "Ticket Price": "N/A",
                    "Media Revenue": "N/A",
                }
                for col in VALUE_COLS:
                    val = row.get(col)
                    if pd.isna(val):
                        continue
                    entry[col] = val
                rows.append(entry)
            output_df = pd.DataFrame(rows)
        else:
            notes.append("Header scan found no usable fans/markets columns.")
            if scanned_count:
                notes.append(f"No usable sources found (scanned {scanned_count} CSV headers).")
    md_lines = ["# ABL Fans & Markets", ""]
    md_lines.append(
        md_table(
            output_df,
            ["Team", "Market Size", "Fan Interest", "Fan Loyalty", "Attendance", "Ticket Price", "Media Revenue", "Tier"],
        )
    )

    if tier_basis.notna().any():
        basis_label = "Market Size" if market_numeric.notna().any() else "Attendance"
        sorted_df = output_df.copy()
        sorted_df["__val"] = tier_basis
        top = sorted_df[sorted_df["__val"].notna()].sort_values("__val", ascending=False).head(5)
        bottom = sorted_df[sorted_df["__val"].notna()].sort_values("__val", ascending=True).head(5)
        if not top.empty and not bottom.empty:
            md_lines.append(f"## Market Board ({basis_label})")
            md_lines.append("### Top 5")
            md_lines.append(md_table(top[["Team", basis_label]], ["Team", basis_label]))
            md_lines.append("### Bottom 5")
            md_lines.append(md_table(bottom[["Team", basis_label]], ["Team", basis_label]))

    md_lines.append("## Data Sources")
    md_lines.append(f"- Teams: {team_path if team_path else 'None found'}")
    if used_paths:
        for path in used_paths:
            md_lines.append(f"- Fans/Markets (fast path): {path}")
    else:
        md_lines.append("- Fans/Markets (fast path): None found")
    if header_scan_files:
        for path in header_scan_files:
            md_lines.append(f"- Header scan: {path}")
    elif scanned_count:
        md_lines.append("- Header scan: none used")
    md_lines.append(f"- Scanned {scanned_count} CSV headers for fans/markets fields")
    if scan_errors:
        md_lines.append(f"- Header scan errors on {len(scan_errors)} file(s)")
    md_lines.append("")
    md_lines.append("## Notes")
    if notes:
        md_lines.extend(f"- {n}" for n in notes)
    else:
        unique_missing = sorted(set(missing_cols))
        if unique_missing:
            md_lines.append("- Missing columns (not fatal): " + ", ".join(unique_missing))
        else:
            md_lines.append("- None")

    out_path = base / "csv" / "out" / "text_out" / "pregame" / "fans_markets.md"
    write_md(out_path, "\n".join(md_lines).rstrip() + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
