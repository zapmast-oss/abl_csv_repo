from __future__ import annotations

from pathlib import Path
import pandas as pd

# -------------------------------------------------------------------
# Config – 1972 / league 200 only
# -------------------------------------------------------------------
SEASON = 1972
LEAGUE_ID = 200

ALMANAC_ROOT = Path("csv/out/almanac") / str(SEASON)
SERIES_PATH = ALMANAC_ROOT / f"series_summary_clean_{SEASON}_league{LEAGUE_ID}.csv"
OUTPUT_PATH = ALMANAC_ROOT / f"eb_series_spotlights_{SEASON}_league{LEAGUE_ID}.md"


def load_series_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Clean series summary file not found: {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def normalize_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take the raw series summary (one row per pair) and produce a
    per-team view: two rows per series (home + away), with wins/losses,
    runs_for, runs_against, run_diff, opponent, and location.
    """
    required = [
        "season",
        "league_id",
        "series_index",
        "start_date",
        "end_date",
        "games",
        "home_team",
        "home_team_wins",
        "away_team_wins",
        "total_home_runs",
        "total_away_runs",
        "away_team",
        "home_team_id",
        "home_team_abbr",
        "home_conference",
        "home_division",
        "away_team_id",
        "away_team_abbr",
        "away_conference",
        "away_division",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"series_summary_clean is missing required columns: {missing}; "
            f"available={list(df.columns)}"
        )

    # Numeric coercions
    for col in ["games", "home_team_wins", "away_team_wins", "total_home_runs", "total_away_runs"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    records: list[dict] = []

    for _, row in df.iterrows():
        season = int(row["season"])
        league_id = int(row["league_id"])
        series_index = int(row["series_index"])
        start_date = row.get("start_date")
        end_date = row.get("end_date")
        games = int(row["games"]) if pd.notna(row["games"]) else 0

        # Home perspective
        home_wins = int(row["home_team_wins"]) if pd.notna(row["home_team_wins"]) else 0
        away_wins = int(row["away_team_wins"]) if pd.notna(row["away_team_wins"]) else 0
        home_runs = int(row["total_home_runs"]) if pd.notna(row["total_home_runs"]) else 0
        away_runs = int(row["total_away_runs"]) if pd.notna(row["total_away_runs"]) else 0

        # Home team row
        records.append(
            {
                "season": season,
                "league_id": league_id,
                "series_index": series_index,
                "start_date": start_date,
                "end_date": end_date,
                "games": games,
                "team_id": int(row["home_team_id"]),
                "team_name": row["home_team"],
                "team_abbr": row["home_team_abbr"],
                "conf": row["home_conference"],
                "division": row["home_division"],
                "opponent_team_id": int(row["away_team_id"]),
                "opponent_abbr": row["away_team_abbr"],
                "location": "home",
                "wins": home_wins,
                "losses": away_wins,
                "runs_for": home_runs,
                "runs_against": away_runs,
                "run_diff": home_runs - away_runs,
            }
        )

        # Away team row
        records.append(
            {
                "season": season,
                "league_id": league_id,
                "series_index": series_index,
                "start_date": start_date,
                "end_date": end_date,
                "games": games,
                "team_id": int(row["away_team_id"]),
                "team_name": row["away_team"],
                "team_abbr": row["away_team_abbr"],
                "conf": row["away_conference"],
                "division": row["away_division"],
                "opponent_team_id": int(row["home_team_id"]),
                "opponent_abbr": row["home_team_abbr"],
                "location": "away",
                "wins": away_wins,
                "losses": home_wins,
                "runs_for": away_runs,
                "runs_against": home_runs,
                "run_diff": away_runs - home_runs,
            }
        )

    df_team = pd.DataFrame.from_records(records)

    # Filter to valid series: games > 0
    df_team = df_team[df_team["games"] > 0].copy()
    if df_team.empty:
        raise ValueError("No valid per-team series rows after normalization (games>0).")

    # Type cleanup
    for col in ["games", "wins", "losses", "runs_for", "runs_against", "run_diff"]:
        df_team[col] = pd.to_numeric(df_team[col], errors="coerce").fillna(0).astype(int)

    return df_team


def build_spotlights(df_team: pd.DataFrame) -> str:
    """
    Build markdown brief from normalized per-team series dataframe.
    """
    lines: list[str] = []
    lines.append(f"# EB Series Spotlights {SEASON} – Data Brief (DO NOT PUBLISH)")
    lines.append("")
    lines.append(f"_League ID {LEAGUE_ID}_")
    lines.append("")
    lines.append(
        "This brief surfaces a few extreme series performances (from one team’s perspective) "
        "to help EB pick narrative turning points."
    )
    lines.append("")

    # Biggest sweeps: team wins all games in a 3+ game set
    sweeps = df_team[(df_team["games"] >= 3) & (df_team["wins"] == df_team["games"])].copy()
    sweeps = sweeps.sort_values(
        ["games", "run_diff"],
        ascending=[False, False],
    ).head(10)

    # Most dominant series by run differential
    dominant = df_team.sort_values("run_diff", ascending=False).head(10)

    # Roughest series by run differential
    rough = df_team.sort_values("run_diff", ascending=True).head(10)

    def series_line(row: pd.Series) -> str:
        team_name = row["team_name"]
        team_abbr = row["team_abbr"]
        opp_abbr = row["opponent_abbr"]
        loc = row["location"]
        games = int(row["games"])
        wins = int(row["wins"])
        losses = int(row["losses"])
        rs = int(row["runs_for"])
        ra = int(row["runs_against"])
        rd = int(row["run_diff"])
        start_date = row.get("start_date") or ""
        end_date = row.get("end_date") or ""

        date_part = ""
        if start_date and end_date:
            if str(start_date) == str(end_date):
                date_part = f" on {start_date}"
            else:
                date_part = f" ({start_date}–{end_date})"
        elif start_date:
            date_part = f" on {start_date}"

        loc_text = "at home" if loc == "home" else "on the road"
        return (
            f"- {team_name} ({team_abbr}) vs {opp_abbr} went {wins}-{losses} in a {games}-game set "
            f"{loc_text}{date_part}, RS={rs}, RA={ra}, RD={rd:+d}"
        )

    if not sweeps.empty:
        lines.append("## Biggest sweeps (3+ game sets, no losses)")
        lines.append("")
        for _, row in sweeps.iterrows():
            lines.append(series_line(row))
        lines.append("")

    lines.append("## Most dominant series by run differential")
    lines.append("")
    for _, row in dominant.iterrows():
        lines.append(series_line(row))
    lines.append("")

    lines.append("## Roughest series by run differential")
    lines.append("")
    for _, row in rough.iterrows():
        lines.append(series_line(row))
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    print(f"[DEBUG] season={SEASON}, league_id={LEAGUE_ID}")
    print(f"[DEBUG] series_path={SERIES_PATH}")

    raw = load_series_raw(SERIES_PATH)
    df_team = normalize_series(raw)
    md = build_spotlights(df_team)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(md, encoding="utf-8")
    print(f"[OK] Wrote EB series spotlight brief to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
