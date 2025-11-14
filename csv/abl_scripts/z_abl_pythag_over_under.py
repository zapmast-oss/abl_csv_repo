"""ABL Pythagorean over/under report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

TEAM_MIN, TEAM_MAX = 1, 24
CANDIDATE_FILES = [
    "team_season.csv",
    "team_totals.csv",
    "team_record.csv",
    "teams_season.csv",
    "standings.csv",
]


def pick_column(df: pd.DataFrame, *names: str) -> Optional[str]:
    lowered = {col.lower(): col for col in df.columns}
    for name in names:
        key = name.lower()
        if key in lowered:
            return lowered[key]
    return None


def autodetect_csv(base: Path, override: Optional[Path]) -> Tuple[pd.DataFrame, Path]:
    if override:
        if not override.exists():
            raise FileNotFoundError(f"Specified input not found: {override}")
        return pd.read_csv(override), override
    for candidate in CANDIDATE_FILES:
        path = base / candidate
        if path.exists():
            return pd.read_csv(path), path
    raise FileNotFoundError(
        f"No season totals file found in {base} (looked for: {', '.join(CANDIDATE_FILES)})"
    )


def load_game_results(base: Path) -> Optional[pd.DataFrame]:
    for candidate in ["games.csv", "games_score.csv"]:
        path = base / candidate
        if path.exists():
            return pd.read_csv(path)
    return None


def expand_games_to_team_runs(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    home_col = pick_column(df, "home_team", "home_team_id", "hometeam", "team1")
    away_col = pick_column(df, "away_team", "away_team_id", "awayteam", "team0")
    home_runs_col = pick_column(df, "home_runs", "runs_home", "score1", "runs1", "r_home")
    away_runs_col = pick_column(df, "away_runs", "runs_away", "score0", "runs0", "r_away")
    if not all([home_col, away_col, home_runs_col, away_runs_col]):
        return None

    records = []
    for _, row in df.iterrows():
        home_id = pd.to_numeric(row[home_col], errors="coerce")
        away_id = pd.to_numeric(row[away_col], errors="coerce")
        home_runs = pd.to_numeric(row[home_runs_col], errors="coerce")
        away_runs = pd.to_numeric(row[away_runs_col], errors="coerce")
        if pd.isna(home_id) or pd.isna(away_id):
            continue
        records.append(
            {
                "team_id": int(home_id),
                "runs_for_fill": home_runs,
                "runs_against_fill": away_runs,
            }
        )
        records.append(
            {
                "team_id": int(away_id),
                "runs_for_fill": away_runs,
                "runs_against_fill": home_runs,
            }
        )
    if not records:
        return None
    return pd.DataFrame.from_records(records)


def build_runs_lookup(base: Path) -> Optional[pd.DataFrame]:
    games_df = load_game_results(base)
    if games_df is None:
        return None
    expanded = expand_games_to_team_runs(games_df)
    if expanded is None or expanded.empty:
        return None
    totals = (
        expanded.groupby("team_id", as_index=False)
        .agg({"runs_for_fill": "sum", "runs_against_fill": "sum"})
        .rename(
            columns={
                "runs_for_fill": "runs_scored_fill",
                "runs_against_fill": "runs_against_fill",
            }
        )
    )
    return totals


def build_report(df: pd.DataFrame, base: Path) -> pd.DataFrame:
    team_col = pick_column(df, "team_id", "teamid", "teamID", "TeamID")
    wins_col = pick_column(df, "wins", "w")
    losses_col = pick_column(df, "losses", "l")
    games_col = pick_column(df, "g", "games")
    rs_col = pick_column(df, "runs_scored", "rs", "r", "Runsscored")
    ra_col = pick_column(df, "runs_against", "ra", "Runsagainst")
    display_col = pick_column(df, "team_display", "team_name", "name", "TeamName")

    required_pairs = [("team_id", team_col), ("wins", wins_col), ("losses", losses_col)]
    missing = [label for label, col in required_pairs if not col]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    work = df.copy()
    work["team_id"] = pd.to_numeric(work[team_col], errors="coerce").astype("Int64")
    work = work[(work["team_id"] >= TEAM_MIN) & (work["team_id"] <= TEAM_MAX)]
    work["wins"] = pd.to_numeric(work[wins_col], errors="coerce")
    work["losses"] = pd.to_numeric(work[losses_col], errors="coerce")
    if rs_col:
        work["runs_scored"] = pd.to_numeric(work[rs_col], errors="coerce")
    else:
        work["runs_scored"] = pd.Series(pd.NA, index=work.index, dtype="Float64")
    if ra_col:
        work["runs_against"] = pd.to_numeric(work[ra_col], errors="coerce")
    else:
        work["runs_against"] = pd.Series(pd.NA, index=work.index, dtype="Float64")
    if games_col:
        work["g"] = pd.to_numeric(work[games_col], errors="coerce")
    else:
        work["g"] = work["wins"] + work["losses"]

    if not rs_col or not ra_col:
        runs_lookup = build_runs_lookup(base)
        if runs_lookup is not None:
            work = work.merge(runs_lookup, on="team_id", how="left")
            if "runs_scored_fill" in work.columns:
                work["runs_scored"] = pd.to_numeric(work["runs_scored"], errors="coerce")
                work["runs_scored_fill"] = pd.to_numeric(
                    work["runs_scored_fill"], errors="coerce"
                )
                mask_scored = work["runs_scored"].isna()
                work.loc[mask_scored, "runs_scored"] = work.loc[
                    mask_scored, "runs_scored_fill"
                ]
            if "runs_against_fill" in work.columns:
                work["runs_against"] = pd.to_numeric(
                    work["runs_against"], errors="coerce"
                )
                work["runs_against_fill"] = pd.to_numeric(
                    work["runs_against_fill"], errors="coerce"
                )
                mask_against = work["runs_against"].isna()
                work.loc[mask_against, "runs_against"] = work.loc[
                    mask_against, "runs_against_fill"
                ]
            work = work.drop(
                columns=["runs_scored_fill", "runs_against_fill"], errors="ignore"
            )
    work["runs_scored"] = pd.to_numeric(work["runs_scored"], errors="coerce")
    work["runs_against"] = pd.to_numeric(work["runs_against"], errors="coerce")

    numeric_cols = ["team_id", "wins", "losses", "runs_scored", "runs_against", "g"]
    work = work.dropna(subset=numeric_cols)
    work = work[work["g"] > 0]

    rs_sq = work["runs_scored"] ** 2
    ra_sq = work["runs_against"] ** 2
    denom = rs_sq + ra_sq

    work["actual_winpct"] = work["wins"] / work["g"]
    work["pythag_winpct"] = rs_sq / denom.replace({0: pd.NA})
    work.loc[denom == 0, "pythag_winpct"] = pd.NA
    work["pythag_expected_wins"] = work["pythag_winpct"] * work["g"]
    work["pythag_diff_winpct"] = work["actual_winpct"] - work["pythag_winpct"]
    work["pythag_diff_wins"] = work["wins"] - work["pythag_expected_wins"]

    if display_col:
        work["team_display"] = work[display_col].fillna("")
    else:
        work["team_display"] = ""

    result = work[
        [
            "team_id",
            "team_display",
            "g",
            "wins",
            "losses",
            "runs_scored",
            "runs_against",
            "actual_winpct",
            "pythag_winpct",
            "pythag_expected_wins",
            "pythag_diff_winpct",
            "pythag_diff_wins",
        ]
    ].copy()

    result["actual_winpct"] = result["actual_winpct"].round(3)
    result["pythag_winpct"] = result["pythag_winpct"].round(3)
    result["pythag_expected_wins"] = result["pythag_expected_wins"].round(1)
    result["pythag_diff_winpct"] = result["pythag_diff_winpct"].round(3)
    result["pythag_diff_wins"] = result["pythag_diff_wins"].round(1)

    result = result.sort_values(
        by=["pythag_diff_winpct", "actual_winpct"],
        ascending=[False, False],
        na_position="last",
    )
    return result


def build_team_meta(base: Path) -> Dict[int, Dict[str, str]]:
    path = base / "teams.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    team_col = pick_column(df, "team_id", "teamid", "teamID", "TeamID")
    if not team_col:
        return {}
    name_col = pick_column(df, "team_display", "team_name", "name", "nickname")
    city_col = pick_column(df, "city", "city_name")
    nickname_col = pick_column(df, "nickname")
    abbr_col = pick_column(df, "abbr")
    sub_col = pick_column(df, "sub_league_id", "sub_league")
    div_col = pick_column(df, "division_id", "division")

    division_map = {0: "E", 1: "C", 2: "W"}
    meta: Dict[int, Dict[str, str]] = {}
    for _, row in df.iterrows():
        tid = row.get(team_col)
        if pd.isna(tid):
            continue
        tid_int = int(tid)
        if tid_int in meta:
            continue
        name_value = ""
        if name_col and pd.notna(row.get(name_col)):
            name_value = str(row.get(name_col))
        elif city_col and nickname_col and pd.notna(row.get(city_col)) and pd.notna(row.get(nickname_col)):
            name_value = f"{row.get(city_col)} {row.get(nickname_col)}"
        elif city_col and pd.notna(row.get(city_col)):
            name_value = str(row.get(city_col))
        elif nickname_col and pd.notna(row.get(nickname_col)):
            name_value = str(row.get(nickname_col))
        elif abbr_col and pd.notna(row.get(abbr_col)):
            name_value = str(row.get(abbr_col))

        conference_letter = ""
        if sub_col and pd.notna(row.get(sub_col)):
            conference_letter = "N" if int(row.get(sub_col)) == 0 else "A"
        division_letter = ""
        if div_col and pd.notna(row.get(div_col)):
            division_letter = division_map.get(int(row.get(div_col)), "")
        conf_div = ""
        if conference_letter and division_letter:
            conf_div = f"{conference_letter}-{division_letter}"
        elif conference_letter:
            conf_div = conference_letter
        elif division_letter:
            conf_div = division_letter

        meta[tid_int] = {"name": name_value, "conf_div": conf_div}
    return meta


def build_text_report(df: pd.DataFrame, limit: int = 24) -> str:
    lines = [
        "ABL Pythagorean Over/Under",
        "=" * 32,
        "Shows which clubs are beating or trailing their runs-based expectation (Pythagorean).",
        "Perfect for spotting lucky hot streaks versus true talent (Actual vs Pythag win%).",
        "",
    ]
    header = f"{'Team':<20} {'CD':<4} {'Profile':<10} {'dWins':>7} {'dPct':>8} {'Actual':>8} {'Pythag':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for _, row in df.head(limit).iterrows():
        name = row["team_display"] or f"Team {int(row['team_id'])}"
        conf_div = row.get("conf_div") or "--"
        diff_wins = row["pythag_diff_wins"]
        diff_pct = row["pythag_diff_winpct"]
        tag = "Lucky" if diff_wins > 0.5 else "Snake-bit" if diff_wins < -0.5 else "On pace"
        actual = row["actual_winpct"]
        py = row["pythag_winpct"]
        lines.append(
            f"{name:<20} {conf_div:<4} {tag:<10} {diff_wins:+7.1f} {diff_pct:+8.3f} "
            f"{actual:>8.3f} {py:>8.3f}"
        )
    lines.append("")
    lines.append("Key:")
    lines.append("  Lucky      -> Actual wins exceed Pythag by >0.5 wins (and +0.015 win pct).")
    lines.append("  On pace    -> Within +/- 0.5 wins of Pythag expectation.")
    lines.append("  Snake-bit  -> Actual wins trail Pythag by >0.5 wins.")
    return "\n".join(lines)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ABL Pythagorean over/under report.")
    parser.add_argument("--base", type=str, default=".", help="Base directory to search.")
    parser.add_argument("--in", dest="input_path", type=str, help="Explicit input CSV.")
    parser.add_argument(
        "--out",
        type=str,
        default="out/csv_out/z_ABL_Pythag_Over_Under.csv",
        help="Output CSV (default: out/csv_out/z_ABL_Pythag_Over_Under.csv).",
    )
    parser.add_argument(
        "--sort",
        choices=["diff", "division"],
        default="diff",
        help="Sort by luck (diff) or by conference/division (division).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    base_dir = Path(args.base).resolve()
    override_path = Path(args.input_path).resolve() if args.input_path else None

    raw_df, source_path = autodetect_csv(base_dir, override_path)
    report_df = build_report(raw_df, base_dir)

    output_path = (base_dir / args.out).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text_filename = output_path.with_suffix(".txt").name
    if output_path.parent.name.lower() in {'csv_out'}:
        text_dir = output_path.parent.parent / "txt_out"
    else:
        text_dir = output_path.parent
    text_dir.mkdir(parents=True, exist_ok=True)
    text_path = text_dir / text_filename

    if report_df.empty:
        report_df["conf_div"] = ""
        text_path.write_text("No qualifying teams found.", encoding="utf-8")
    else:
        meta = build_team_meta(base_dir)
        report_df["conf_div"] = ""
        if meta:
            report_df["team_display"] = report_df.apply(
                lambda row: row["team_display"]
                if row["team_display"]
                else meta.get(int(row["team_id"]), {}).get("name", ""),
                axis=1,
            )
            report_df["conf_div"] = report_df["team_id"].apply(
                lambda tid: meta.get(int(tid), {}).get("conf_div", "")
            )
        if args.sort == "division":
            report_df = report_df.sort_values(
                by=["conf_div", "pythag_diff_wins", "team_display"],
                ascending=[True, False, True],
            )
        text_path.write_text(build_text_report(report_df), encoding="utf-8")
        column_order = [
            "team_id",
            "team_display",
            "conf_div",
            "g",
            "wins",
            "losses",
            "runs_scored",
            "runs_against",
            "actual_winpct",
            "pythag_winpct",
            "pythag_expected_wins",
            "pythag_diff_winpct",
            "pythag_diff_wins",
        ]
        existing = [col for col in column_order if col in report_df.columns]
        report_df = report_df[existing]
    report_df.to_csv(output_path, index=False)

    if report_df.empty:
        print("No qualifying teams found; CSV is empty.")
        return

    preview = report_df.head(12)
    print("Pythagorean over/under (top 12):")
    print(preview.to_string(index=False))
    print(
        f"\nWrote {len(report_df)} rows to {output_path} and summary to {text_path} "
        f"(source: {source_path})."
    )


if __name__ == "__main__":
    main()

