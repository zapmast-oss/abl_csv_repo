from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def log(msg: str) -> None:
    print(msg)


def load_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        print(f"Error: {label} not found at {path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path)
    log(f"[INFO] Loaded {len(df)} rows from {label}")
    return df


def require_columns(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(
            f"Error: {label} missing columns: {missing}. Present: {list(df.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)


def build_baseline(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "season",
        "league_id",
        "team_id",
        "team_abbr",
        "team_name",
        "conf",
        "division",
        "wins",
        "losses",
        "pct",
        "run_diff",
    ]
    require_columns(df, required, "league season summary")
    base = df[required].copy()
    base = base.rename(
        columns={
            "wins": "season_wins",
            "losses": "season_losses",
            "pct": "season_win_pct",
            "run_diff": "season_run_diff",
        }
    )
    return base


def compute_half_summary(weekly: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    required = [
        "season",
        "league_id",
        "team_id",
        "team_abbr",
        "team_name",
        "conference",
        "division",
        "week_index",
        "games",
        "wins",
        "losses",
        "runs_for",
        "runs_against",
    ]
    require_columns(weekly, required, "weekly summary")
    df = weekly.copy()
    if "run_diff" not in df.columns:
        df["run_diff"] = df["runs_for"] - df["runs_against"]

    df["week_index"] = pd.to_numeric(df["week_index"], errors="coerce")
    min_wk = int(df["week_index"].min())
    max_wk = int(df["week_index"].max())
    mid = (min_wk + max_wk) // 2
    df["half_label"] = df["week_index"].apply(lambda x: "first" if x <= mid else "second")

    agg = (
        df.groupby(["team_id", "half_label"])
        .agg(
            half_games=("games", "sum"),
            half_wins=("wins", "sum"),
            half_losses=("losses", "sum"),
            half_runs_for=("runs_for", "sum"),
            half_runs_against=("runs_against", "sum"),
            half_run_diff=("run_diff", "sum"),
        )
        .reset_index()
    )
    agg["half_win_pct"] = agg["half_wins"] / (agg["half_wins"] + agg["half_losses"]).replace(
        {0: pd.NA}
    )

    merged = agg.merge(
        baseline,
        on="team_id",
        how="left",
        validate="many_to_one",
    )
    if merged["team_name"].isna().any():
        print("Error: half summary merge failed for some teams.", file=sys.stderr)
        sys.exit(1)

    merged["half_win_pct_delta_vs_season"] = merged["half_win_pct"] - merged["season_win_pct"]
    merged["half_run_diff_delta_vs_season"] = merged["half_run_diff"] - (
        merged["season_run_diff"] / 2.0
    )

    keep = [
        "season",
        "league_id",
        "team_id",
        "team_abbr",
        "team_name",
        "conf",
        "division",
        "half_label",
        "half_games",
        "half_wins",
        "half_losses",
        "half_win_pct",
        "half_runs_for",
        "half_runs_against",
        "half_run_diff",
        "season_wins",
        "season_losses",
        "season_win_pct",
        "season_run_diff",
        "half_win_pct_delta_vs_season",
        "half_run_diff_delta_vs_season",
    ]
    merged = merged[keep]
    merged = merged.sort_values(
        ["conf", "division", "team_name", "half_label"], ascending=[True, True, True, True]
    )
    return merged


def best_worst_flags(df: pd.DataFrame, value_col: str, group_col: str) -> pd.DataFrame:
    df = df.copy()
    idxmax = df.groupby(group_col)[value_col].idxmax()
    idxmin = df.groupby(group_col)[value_col].idxmin()
    df["is_team_best_month"] = False
    df["is_team_worst_month"] = False
    df.loc[idxmax, "is_team_best_month"] = True
    df.loc[idxmin, "is_team_worst_month"] = True
    return df


def compute_monthly_momentum(monthly: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    required = [
        "season",
        "league_id",
        "team_id",
        "team_abbr",
        "team_name",
        "conference",
        "division",
        "month",
        "games",
        "wins",
        "losses",
        "runs_for",
        "runs_against",
    ]
    require_columns(monthly, required, "monthly summary")
    df = monthly.copy()
    if "run_diff" not in df.columns:
        df["run_diff"] = df["runs_for"] - df["runs_against"]
    df["month_label"] = df["month"]
    df["month_win_pct"] = df["wins"] / (df["wins"] + df["losses"]).replace({0: pd.NA})

    merged = df.merge(baseline, on="team_id", how="left", validate="many_to_one", suffixes=("_slice", ""))
    if merged["team_name"].isna().any():
        print("Error: monthly momentum merge failed for some teams.", file=sys.stderr)
        sys.exit(1)

    merged["month_win_pct_delta_vs_season"] = merged["month_win_pct"] - merged["season_win_pct"]
    merged["month_run_diff_delta_vs_season"] = merged["run_diff"] - merged["season_run_diff"]

    merged = best_worst_flags(merged, "month_win_pct", "team_id")

    keep = [
        "season",
        "league_id",
        "team_id",
        "team_abbr",
        "team_name",
        "conf",
        "division",
        "month",
        "month_label",
        "games",
        "wins",
        "losses",
        "month_win_pct",
        "runs_for",
        "runs_against",
        "run_diff",
        "season_wins",
        "season_losses",
        "season_win_pct",
        "season_run_diff",
        "month_win_pct_delta_vs_season",
        "month_run_diff_delta_vs_season",
        "is_team_best_month",
        "is_team_worst_month",
    ]
    merged = merged[keep]
    merged = merged.sort_values(["conf", "division", "team_name", "month_label"])
    return merged


def compute_weekly_momentum(weekly: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    required = [
        "season",
        "league_id",
        "team_id",
        "team_abbr",
        "team_name",
        "conference",
        "division",
        "week_index",
        "games",
        "wins",
        "losses",
        "runs_for",
        "runs_against",
    ]
    require_columns(weekly, required, "weekly summary")
    df = weekly.copy()
    if "run_diff" not in df.columns:
        df["run_diff"] = df["runs_for"] - df["runs_against"]
    df["week_win_pct"] = df["wins"] / (df["wins"] + df["losses"]).replace({0: pd.NA})

    merged = df.merge(baseline, on="team_id", how="left", validate="many_to_one", suffixes=("_slice", ""))
    if merged["team_name"].isna().any():
        print("Error: weekly momentum merge failed for some teams.", file=sys.stderr)
        sys.exit(1)

    merged["week_win_pct_delta_vs_season"] = merged["week_win_pct"] - merged["season_win_pct"]
    merged["week_run_diff_delta_vs_season"] = merged["run_diff"] - merged["season_run_diff"]

    keep = [
        "season",
        "league_id",
        "team_id",
        "team_abbr",
        "team_name",
        "conf",
        "division",
        "week_index",
        "games",
        "wins",
        "losses",
        "week_win_pct",
        "runs_for",
        "runs_against",
        "run_diff",
        "season_wins",
        "season_losses",
        "season_win_pct",
        "season_run_diff",
        "week_win_pct_delta_vs_season",
        "week_run_diff_delta_vs_season",
    ]
    merged = merged[keep]
    merged = merged.sort_values(["conf", "division", "team_name", "week_index"])
    return merged


def clean_series(series: pd.DataFrame) -> pd.DataFrame:
    banned = {"American Baseball Conference", "National Baseball Conference"}
    if "home_team" in series.columns:
        series = series[~series["home_team"].astype(str).isin(banned)]
    if "away_team" in series.columns:
        series = series[~series["away_team"].astype(str).isin(banned)]
    return series


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 3k momentum summaries (half/month/week).")
    parser.add_argument("--season", type=int, required=True, help="Season year (e.g., 1972)")
    parser.add_argument("--league-id", type=int, required=True, help="League ID (ABL=200)")
    parser.add_argument(
        "--almanac-root",
        default=Path("csv/out/almanac"),
        type=Path,
        help="Root of almanac outputs (default: csv/out/almanac)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    season_dir = args.almanac_root / str(args.season)
    league_summary_path = season_dir / f"league_season_summary_{args.season}_league{args.league_id}.csv"
    monthly_path = season_dir / f"team_monthly_summary_{args.season}_league{args.league_id}_enriched.csv"
    weekly_path = season_dir / f"team_weekly_summary_{args.season}_league{args.league_id}_enriched.csv"
    series_path = season_dir / f"series_summary_{args.season}_league{args.league_id}_enriched.csv"

    log(f"[DEBUG] season={args.season}, league_id={args.league_id}")
    log(f"[DEBUG] league_summary_path={league_summary_path}")
    log(f"[DEBUG] monthly_path={monthly_path}")
    log(f"[DEBUG] weekly_path={weekly_path}")
    log(f"[DEBUG] series_path={series_path}")

    league_df = load_csv(league_summary_path, "league season summary")
    baseline = build_baseline(league_df)

    monthly_df = load_csv(monthly_path, "team monthly summary")
    weekly_df = load_csv(weekly_path, "team weekly summary")
    series_df = load_csv(series_path, "series summary")

    half_summary = compute_half_summary(weekly_df, baseline)
    half_out = season_dir / f"half_summary_{args.season}_league{args.league_id}.csv"
    half_out.parent.mkdir(parents=True, exist_ok=True)
    half_summary.to_csv(half_out, index=False)
    log(f"[OK] Wrote half summary to {half_out} ({len(half_summary)} rows)")

    monthly_momentum = compute_monthly_momentum(monthly_df, baseline)
    monthly_out = season_dir / f"team_monthly_momentum_{args.season}_league{args.league_id}.csv"
    monthly_momentum.to_csv(monthly_out, index=False)
    log(f"[OK] Wrote monthly momentum to {monthly_out} ({len(monthly_momentum)} rows)")

    weekly_momentum = compute_weekly_momentum(weekly_df, baseline)
    weekly_out = season_dir / f"team_weekly_momentum_{args.season}_league{args.league_id}.csv"
    weekly_momentum.to_csv(weekly_out, index=False)
    log(f"[OK] Wrote weekly momentum to {weekly_out} ({len(weekly_momentum)} rows)")

    cleaned_series = clean_series(series_df)
    series_out = season_dir / f"series_summary_clean_{args.season}_league{args.league_id}.csv"
    cleaned_series.to_csv(series_out, index=False)
    log(f"[OK] Wrote cleaned series summary to {series_out} ({len(cleaned_series)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
