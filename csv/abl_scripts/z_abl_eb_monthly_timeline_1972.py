from __future__ import annotations

from pathlib import Path

import pandas as pd

from eb_text_utils import normalize_eb_text

# -------------------------------------------------------------------
# Config – 1972 / league 200 only
# -------------------------------------------------------------------
SEASON = 1972
LEAGUE_ID = 200

ALMANAC_ROOT = Path("csv/out/almanac") / str(SEASON)
MONTHLY_PATH = ALMANAC_ROOT / f"team_monthly_momentum_{SEASON}_league{LEAGUE_ID}.csv"
OUTPUT_PATH = ALMANAC_ROOT / f"eb_monthly_timeline_{SEASON}_league{LEAGUE_ID}.md"


def load_monthly(path: Path) -> tuple[pd.DataFrame, str]:
    if not path.exists():
        raise FileNotFoundError(f"Monthly momentum file not found: {path}")

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    required = [
        "month",
        "team_abbr",
        "team_name",
        "games",
        "wins",
        "losses",
        "runs_for",
        "runs_against",
        "run_diff",
        "month_win_pct",
        "month_win_pct_delta_vs_season",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"team_monthly_momentum is missing required columns: {missing}; "
            f"available={list(df.columns)}"
        )

    # conference column may be 'conf' or 'conference'
    if "conf" in df.columns:
        conf_col = "conf"
    elif "conference" in df.columns:
        conf_col = "conference"
    else:
        raise KeyError("team_monthly_momentum is missing conference/conf column")

    # Parse month into datetime and label
    dt = pd.to_datetime(df["month"], errors="coerce")
    if dt.isna().all():
        raise ValueError("Could not parse any month values to datetime from 'month' column")

    df["month_dt"] = dt
    df["month_label"] = df["month_dt"].dt.strftime("%B %Y")

    # Make sure numeric fields are numeric
    for col in ["games", "wins", "losses", "runs_for", "runs_against", "run_diff",
                "month_win_pct", "month_win_pct_delta_vs_season"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, conf_col


def build_monthly_timeline(df: pd.DataFrame, conf_col: str) -> str:
    if "division" not in df.columns:
        raise KeyError("team_monthly_momentum is missing 'division' column")

    out_lines: list[str] = []
    out_lines.append(f"# EB Monthly Timeline {SEASON} – Data Brief (DO NOT PUBLISH)")
    out_lines.append("")
    out_lines.append(f"_League ID {LEAGUE_ID}_")
    out_lines.append("")
    out_lines.append(
        "This brief summarizes each month of the regular season at a high level for EB’s internal use."
    )
    out_lines.append("")

    # Sort for deterministic grouping
    df_sorted = df.sort_values(
        ["month_dt", conf_col, "division", "team_name"]
    ).reset_index()

    for month_val, month_group in df_sorted.groupby("month_dt"):
        month_label = month_group["month_label"].iloc[0]
        out_lines.append(f"## {month_label}")
        out_lines.append("")

        # For each conference / division within that month
        for (conf, division), g in month_group.groupby([conf_col, "division"]):
            g = g.copy()
            if g.empty:
                continue

            out_lines.append(f"### {conf} / {division}")
            out_lines.append("")

            # Identify key teams:
            # - best month record
            # - worst month record
            # - biggest positive delta vs season
            # - biggest negative delta vs season
            best_idx = g["month_win_pct"].idxmax()
            worst_idx = g["month_win_pct"].idxmin()
            up_idx = g["month_win_pct_delta_vs_season"].idxmax()
            down_idx = g["month_win_pct_delta_vs_season"].idxmin()

            seen: set[int] = set()

            def add_line(idx: int, label: str) -> None:
                if idx in seen:
                    return
                seen.add(idx)
                row = g.loc[idx]

                team_name = row["team_name"]
                team_abbr = row["team_abbr"]
                games = int(row["games"])
                wins = int(row["wins"])
                losses = int(row["losses"])
                rs = int(row["runs_for"])
                ra = int(row["runs_against"])
                rd = int(row["run_diff"])
                month_pct = float(row["month_win_pct"])
                delta = float(row["month_win_pct_delta_vs_season"])

                out_lines.append(
                    f"- {label}: {team_name} ({team_abbr}) went {wins}-{losses} in {games} games "
                    f"({month_pct:.3f}), RS={rs}, RA={ra}, RD={rd:+d}, "
                    f"delta vs season={delta:+.3f}"
                )

            add_line(best_idx, "Best record this month")
            add_line(worst_idx, "Toughest month")
            add_line(up_idx, "Biggest overachiever vs season")
            add_line(down_idx, "Biggest slump vs season")

            out_lines.append("")

        out_lines.append("")

    return "\n".join(out_lines)


def main() -> int:
    print(f"[DEBUG] season={SEASON}, league_id={LEAGUE_ID}")
    print(f"[DEBUG] monthly_path={MONTHLY_PATH}")

    df, conf_col = load_monthly(MONTHLY_PATH)
    md = build_monthly_timeline(df, conf_col)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(normalize_eb_text(md), encoding="utf-8")
    print(f"[OK] Wrote EB monthly timeline brief to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
