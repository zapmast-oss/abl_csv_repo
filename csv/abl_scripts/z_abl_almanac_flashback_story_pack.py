#!/usr/bin/env python
"""
z_abl_almanac_flashback_story_pack.py

Builds Flashback story candidates for a given ABL season from the
almanac-derived summary tables.

Inputs (under csv/out/almanac/<season>/):
- league_season_summary_<season>_league<league_id>.csv
- half_summary_<season>_league<league_id>.csv
- team_monthly_momentum_<season>_league<league_id>.csv
- team_weekly_momentum_<season>_league<league_id>.csv  (not yet used, but wired for future)

Output:
- flashback_story_candidates_<season>_league<league_id>.csv

Story groups (40 rows total):
1) Season Giants – Run Differential (5)
2) Season Giants – Winning Percentage (5)
3) Second-Half Surges (5)
4) Second-Half Collapses (5)
5) Month of Glory – Overachievers (10)
6) Month of Misery – Slumps (10)

This version:
- Explicitly pulls run_diff and pct from league_season_summary
- Sets a metric_value field for every candidate
- Guards against NaNs in the headline metrics (run_diff, pct, deltas)
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Small utility helpers
# ---------------------------------------------------------------------------

def log(msg: str, level: str = "INFO") -> None:
    print(f"[{level}] {msg}")


def load_csv(path: Path, desc: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{desc} not found at {path}")
    log(f"Loading {desc} from {path}", "INFO")
    return pd.read_csv(path)


def ensure_columns(df: pd.DataFrame, required: List[str], desc: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{desc} is missing required columns: {missing}")


def normalize_conf_div(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure we have 'conf' and 'division' columns, even if the source
    used 'conference' or 'division_name', etc.
    """
    df = df.copy()
    if "conf" not in df.columns:
        if "conference" in df.columns:
            df["conf"] = df["conference"]
        elif "Conference" in df.columns:
            df["conf"] = df["Conference"]
    if "division" not in df.columns:
        if "division_name" in df.columns:
            df["division"] = df["division_name"]
        elif "Division" in df.columns:
            df["division"] = df["Division"]
    return df


def backfill_identity(
    target: pd.DataFrame,
    league: pd.DataFrame,
    keys: Tuple[str, ...] = ("team_id", "team_abbr"),
) -> pd.DataFrame:
    """
    Ensure the target table has the canonical identity fields by merging
    from league summary if needed.

    Identity columns we want:
        season, league_id, team_id, team_abbr, team_name, conf, division
    """
    target = target.copy()
    league = normalize_conf_div(league.copy())

    identity_cols = [
        "season",
        "league_id",
        "team_id",
        "team_abbr",
        "team_name",
        "conf",
        "division",
    ]

    # Work out merge key: prefer team_id if present, else team_abbr
    merge_key = None
    for k in keys:
        if k in target.columns and k in league.columns:
            merge_key = k
            break

    if merge_key is None:
        raise KeyError(
            "Cannot backfill identity: none of the keys "
            f"{keys} found in both target and league summary."
        )

    # Only keep identity columns from league once
    league_id_cols = [c for c in identity_cols if c in league.columns] + [merge_key]
    league_id_cols = list(dict.fromkeys(league_id_cols))  # unique, preserve order

    # Merge, but do not lose existing non-null identity fields in target
    merged = target.merge(
        league[league_id_cols].drop_duplicates(merge_key),
        on=merge_key,
        how="left",
        suffixes=("", "_league"),
    )

    # For each identity column, prefer target's own value then league's
    for col in identity_cols:
        if col in target.columns and col in merged.columns:
            # Already there; if we also have col_league, fill nulls from it
            league_col = f"{col}_league"
            if league_col in merged.columns:
                merged[col] = merged[col].fillna(merged[league_col])
        elif col in merged.columns:
            # Came from league
            pass
        else:
            league_col = f"{col}_league"
            if league_col in merged.columns:
                merged[col] = merged[league_col]

    # Drop any *_league helper columns
    drop_cols = [c for c in merged.columns if c.endswith("_league")]
    merged = merged.drop(columns=drop_cols)

    return merged


def check_no_nan(df: pd.DataFrame, cols: List[str], context: str) -> None:
    """
    Ensure there are no NaNs in the given columns. If there are,
    raise a clear error so we can fix upstream.
    """
    bad_cols = []
    for c in cols:
        if c in df.columns and df[c].isna().any():
            bad_cols.append(c)
    if bad_cols:
        raise ValueError(
            f"{context}: found NaN values in headline metric columns {bad_cols}. "
            "Check upstream tables (league summary / momentum)."
        )


# ---------------------------------------------------------------------------
# Story builders
# ---------------------------------------------------------------------------

def build_season_run_diff_giants(league: pd.DataFrame) -> pd.DataFrame:
    df = league.copy()

    # Normalise and ensure run_diff and pct exist
    if "run_diff" not in df.columns:
        if {"runs_for", "runs_against"}.issubset(df.columns):
            df["run_diff"] = df["runs_for"] - df["runs_against"]
        else:
            raise KeyError(
                "league_season_summary is missing run_diff and "
                "runs_for/runs_against; cannot build run-diff stories."
            )

    # Guarantee numeric
    df["run_diff"] = pd.to_numeric(df["run_diff"], errors="coerce")

    # Sort and take top 5
    top = (
        df.sort_values("run_diff", ascending=False)
        .head(5)
        .reset_index(drop=True)
        .copy()
    )

    top["story_group"] = "Season Giants – Run Differential"
    top["story_type"] = "SEASON_GIANTS_RUN_DIFF"
    top["rank_in_group"] = top.index + 1
    top["metric_name"] = "run_diff"
    top["metric_value"] = top["run_diff"]

    # Sanity: no NaNs in metric_value / run_diff
    check_no_nan(top, ["run_diff", "metric_value"], "Season Run Diff Giants")

    return top


def build_season_win_pct_giants(league: pd.DataFrame) -> pd.DataFrame:
    df = league.copy()

    if "pct" not in df.columns:
        if "win_pct" in df.columns:
            df["pct"] = df["win_pct"]
        else:
            raise KeyError(
                "league_season_summary is missing pct/win_pct; "
                "cannot build winning-percentage stories."
            )

    df["pct"] = pd.to_numeric(df["pct"], errors="coerce")

    top = (
        df.sort_values("pct", ascending=False)
        .head(5)
        .reset_index(drop=True)
        .copy()
    )

    top["story_group"] = "Season Giants – Winning Percentage"
    top["story_type"] = "SEASON_GIANTS_WIN_PCT"
    top["rank_in_group"] = top.index + 1
    top["metric_name"] = "pct"
    top["metric_value"] = top["pct"]

    check_no_nan(top, ["pct", "metric_value"], "Season Win% Giants")

    return top


def build_second_half_swing(
    half: pd.DataFrame, positive: bool, label: str, story_type: str
) -> pd.DataFrame:
    df = half.copy()

    # Expect half_label, half_win_pct, half_win_pct_delta_vs_season
    ensure_columns(
        df,
        ["half_label", "half_win_pct", "half_win_pct_delta_vs_season"],
        "half_summary",
    )

    df["half_win_pct"] = pd.to_numeric(df["half_win_pct"], errors="coerce")
    df["half_win_pct_delta_vs_season"] = pd.to_numeric(
        df["half_win_pct_delta_vs_season"], errors="coerce"
    )

    # Only second half; filter strictly for that label
    mask_half = df["half_label"].str.lower().str.contains("2nd")
    df = df[mask_half].copy()

    # Filter for positive or negative deltas
    if positive:
        df = df[df["half_win_pct_delta_vs_season"] > 0].copy()
        df = df.sort_values(
            "half_win_pct_delta_vs_season", ascending=False
        )
    else:
        df = df[df["half_win_pct_delta_vs_season"] < 0].copy()
        df = df.sort_values(
            "half_win_pct_delta_vs_season", ascending=True
        )

    df = df.reset_index(drop=True).head(5).copy()

    df["story_group"] = label
    df["story_type"] = story_type
    df["rank_in_group"] = df.index + 1
    df["metric_name"] = "half_win_pct_delta_vs_season"
    df["metric_value"] = df["half_win_pct_delta_vs_season"]

    check_no_nan(
        df,
        ["half_win_pct", "half_win_pct_delta_vs_season", "metric_value"],
        label,
    )

    return df


def build_month_momentum(
    monthly: pd.DataFrame, positive: bool, label: str, story_type: str, limit: int
) -> pd.DataFrame:
    df = monthly.copy()

    ensure_columns(
        df,
        ["month", "month_win_pct", "month_win_pct_delta_vs_season"],
        "team_monthly_momentum",
    )

    df["month_win_pct"] = pd.to_numeric(df["month_win_pct"], errors="coerce")
    df["month_win_pct_delta_vs_season"] = pd.to_numeric(
        df["month_win_pct_delta_vs_season"], errors="coerce"
    )

    if positive:
        df = df[df["month_win_pct_delta_vs_season"] > 0].copy()
        df = df.sort_values(
            "month_win_pct_delta_vs_season", ascending=False
        )
    else:
        df = df[df["month_win_pct_delta_vs_season"] < 0].copy()
        df = df.sort_values(
            "month_win_pct_delta_vs_season", ascending=True
        )

    df = df.reset_index(drop=True).head(limit).copy()

    df["story_group"] = label
    df["story_type"] = story_type
    df["rank_in_group"] = df.index + 1
    df["metric_name"] = "month_win_pct_delta_vs_season"
    df["metric_value"] = df["month_win_pct_delta_vs_season"]

    check_no_nan(
        df,
        ["month_win_pct", "month_win_pct_delta_vs_season", "metric_value"],
        label,
    )

    return df


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Flashback story candidates from almanac-derived tables."
    )
    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help="Season year (e.g. 1972)",
    )
    parser.add_argument(
        "--league-id",
        type=int,
        default=200,
        help="League ID (default 200 for ABL)",
    )
    parser.add_argument(
        "--almanac-root",
        type=str,
        default="csv/out/almanac",
        help="Root folder for almanac-derived CSV outputs.",
    )

    args = parser.parse_args()
    season = args.season
    league_id = args.league_id

    log(f"season={season}, league_id={league_id}", "DEBUG")

    root = Path(args.almanac_root) / str(season)

    league_path = root / f"league_season_summary_{season}_league{league_id}.csv"
    half_path = root / f"half_summary_{season}_league{league_id}.csv"
    monthly_path = root / f"team_monthly_momentum_{season}_league{league_id}.csv"
    weekly_path = root / f"team_weekly_momentum_{season}_league{league_id}.csv"  # reserved

    out_path = root / f"flashback_story_candidates_{season}_league{league_id}.csv"

    # Load base tables
    league = load_csv(league_path, "league_season_summary")
    half = load_csv(half_path, "half_summary")
    monthly = load_csv(monthly_path, "team_monthly_momentum")

    # Normalise league identity fields
    league = normalize_conf_div(league)

    # Backfill identity into half/monthly from league summary
    half = backfill_identity(half, league)
    monthly = backfill_identity(monthly, league)

    # Build story slices
    season_rd = build_season_run_diff_giants(league)
    season_wp = build_season_win_pct_giants(league)

    second_half_surges = build_second_half_swing(
        half,
        positive=True,
        label="Second-Half Surges",
        story_type="SECOND_HALF_SURGE",
    )
    second_half_collapses = build_second_half_swing(
        half,
        positive=False,
        label="Second-Half Collapses",
        story_type="SECOND_HALF_COLLAPSE",
    )

    month_glory = build_month_momentum(
        monthly,
        positive=True,
        label="Month of Glory – Overachievers",
        story_type="MONTH_GLORY",
        limit=10,
    )
    month_misery = build_month_momentum(
        monthly,
        positive=False,
        label="Month of Misery – Slumps",
        story_type="MONTH_MISERY",
        limit=10,
    )

    # Concatenate all candidates
    candidates = pd.concat(
        [
            season_rd,
            season_wp,
            second_half_surges,
            second_half_collapses,
            month_glory,
            month_misery,
        ],
        ignore_index=True,
    )

    # Final sanity: identity + metric_value should be present
    identity_cols = [
        "season",
        "league_id",
        "team_id",
        "team_abbr",
        "team_name",
        "conf",
        "division",
    ]
    for col in identity_cols:
        if col not in candidates.columns:
            raise KeyError(
                f"Story candidates missing identity column '{col}'. "
                "Check backfill_identity logic."
            )

    check_no_nan(candidates, ["metric_value"], "Flashback story candidates")

    # Sort by story_group, then rank_in_group
    if "story_group" in candidates.columns and "rank_in_group" in candidates.columns:
        candidates = candidates.sort_values(
            ["story_group", "rank_in_group", "team_name"]
        ).reset_index(drop=True)

    # Write output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(out_path, index=False)
    log(f"Wrote Flashback story candidates to {out_path}", "OK")
    log(f"Total rows: {len(candidates)}", "INFO")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
