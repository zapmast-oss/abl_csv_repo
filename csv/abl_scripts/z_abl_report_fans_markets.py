"""Pregame fans & markets report (source-of-truth)."""

from __future__ import annotations

import argparse
from typing import List

import pandas as pd

from _abl_pregame_utils import (
    format_int_commas,
    format_money_short,
    load_team_fans_markets,
    md_table,
    parse_money_to_float,
    resolve_base,
    write_md,
)


def format_val(val):
    if pd.isna(val):
        return "N/A"
    num = pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0]
    if pd.notna(num):
        if abs(num) >= 1000:
            return f"{int(num):,}"
        return f"{num:.2f}".rstrip("0").rstrip(".")
    return str(val)


def board_lines(df: pd.DataFrame, col: str, title: str) -> List[str]:
    lines: List[str] = []
    series = pd.to_numeric(df[col], errors="coerce")
    nonnull = df[series.notna()]
    if nonnull.empty:
        return lines
    top = nonnull.sort_values(col, ascending=False).head(5)
    bot = nonnull.sort_values(col, ascending=True).head(5)
    lines.append(f"## {title} Board")
    lines.append("### Top 5")
    tmp_top = top[["Team", col]].copy()
    tmp_top[col] = tmp_top[col].apply(format_val)
    lines.append(md_table(tmp_top, ["Team", col]))
    lines.append("### Bottom 5")
    tmp_bot = bot[["Team", col]].copy()
    tmp_bot[col] = tmp_bot[col].apply(format_val)
    lines.append(md_table(tmp_bot, ["Team", col]))
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ABL fans/markets report.")
    parser.add_argument("--base", help="Repo root (optional).")
    parser.add_argument("--league_id", type=int, default=200)
    parser.add_argument("--season", type=int, required=False)
    args = parser.parse_args()

    base = resolve_base(args.base)
    league_id = args.league_id
    season = args.season

    df, sources, notes = load_team_fans_markets(base, league_id=league_id, season=season)
    rows = []
    for _, row in df.iterrows():
        team_label = row.get("team_abbr") if pd.notna(row.get("team_abbr")) else "N/A"
        ticket_num = parse_money_to_float(row.get("ticket_price_avg"))
        gate_num = parse_money_to_float(row.get("gate_revenue"))
        att_avg = format_int_commas(row.get("attendance_avg")) if pd.notna(row.get("attendance_avg")) else "N/A"
        att_tot = format_int_commas(row.get("attendance_total")) if pd.notna(row.get("attendance_total")) else "N/A"
        ticket_disp = f"${ticket_num:,.2f}" if pd.notna(ticket_num) and ticket_num <= 500 else ("N/A" if pd.isna(ticket_num) else f"${ticket_num:,.0f}")
        gate_disp = format_money_short(gate_num) if pd.notna(gate_num) else "N/A"
        rows.append(
            {
                "Team": team_label,
                "Market": row.get("market"),
                "Fan Interest": row.get("fan_interest"),
                "Fan Loyalty": row.get("fan_loyalty"),
                "Att (Avg)": att_avg,
                "Att (Total)": att_tot,
                "Ticket Price": ticket_disp,
                "Gate": gate_disp,
                "Local Media": row.get("local_media"),
                "National Media": row.get("national_media"),
                "Season Tickets": row.get("season_tickets"),
            }
        )
    output_df = pd.DataFrame(rows)
    # format display for remaining numeric-ish columns
    for col in ["Market", "Fan Interest", "Fan Loyalty", "Local Media", "National Media", "Season Tickets"]:
        if col in output_df.columns:
            output_df[col] = output_df[col].apply(format_val)

    md_lines = ["# ABL Fans & Markets", ""]
    if not output_df.empty:
        for col, title in [
            ("Market", "Market"),
            ("Fan Interest", "Fan Interest"),
            ("Fan Loyalty", "Fan Loyalty"),
        ]:
            md_lines.extend(board_lines(output_df, col, title))
        # Attendance board prefers avg if present
        att_col = "Att (Avg)" if output_df["Att (Avg)"].ne("N/A").any() else "Att (Total)"
        md_lines.extend(board_lines(output_df, att_col, "Attendance"))
        # Ticket/Gate board
        if output_df["Ticket Price"].ne("N/A").any():
            md_lines.extend(board_lines(output_df, "Ticket Price", "Ticket Price"))
        elif output_df["Gate"].ne("N/A").any():
            md_lines.extend(board_lines(output_df, "Gate", "Gate"))
    else:
        notes.append("No fans/markets data found.")

    md_lines.append(
        md_table(
            output_df,
            ["Team", "Market", "Fan Interest", "Fan Loyalty", "Att (Avg)", "Att (Total)", "Ticket Price", "Gate", "Local Media", "National Media", "Season Tickets"],
        )
    )

    md_lines.append("## Data Sources")
    if sources:
        for src in sources:
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

    out_path = base / "csv" / "out" / "text_out" / "pregame" / "fans_markets.md"
    write_md(out_path, "\n".join(md_lines).rstrip() + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
