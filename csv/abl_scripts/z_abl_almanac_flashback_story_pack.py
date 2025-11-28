from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def log(msg: str) -> None:
    print(msg)


def ensure_columns(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{label} is missing required columns: {missing}. Present: {list(df.columns)}")


def check_no_nan(df: pd.DataFrame, cols: list[str], label: str) -> None:
    bad = df[cols].isna().any()
    if bad.any():
        bad_cols = [c for c in cols if bad[c]]
        raise ValueError(f"{label}: found NaN values in columns {bad_cols}. Check upstream tables.")


def ensure_base_cols(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "season",
        "league_id",
        "team_id",
        "team_abbr",
        "team_name",
        "conf",
        "division",
    ]
    ensure_columns(df, required, "identity")
    return df.copy()


def load_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found at {path}")
    df = pd.read_csv(path)
    log(f"[INFO] Loaded {len(df)} rows from {label}")
    return df


def normalize_conf_div(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Harmonize conference/division column names if alternates exist
    if "conference" in df.columns and "conf" not in df.columns:
        df["conf"] = df["conference"]
    if "division" not in df.columns:
        # try DIV, DIVISION
        for cand in ["DIV", "division_name"]:
            if cand in df.columns:
                df["division"] = df[cand]
                break
    return df


def backfill_identity(target: pd.DataFrame, league: pd.DataFrame) -> pd.DataFrame:
    target = target.copy()
    league_map = league.set_index("team_id")
    for col in ["team_abbr", "team_name", "conf", "division", "season", "league_id"]:
        if col not in target.columns and col in league_map.columns:
            target[col] = target["team_id"].map(league_map[col])
        elif col in league_map.columns:
            # fill missing values only
            target[col] = target[col].fillna(target["team_id"].map(league_map[col]))
    return target


def backfill_season_run_diff(league: pd.DataFrame, monthly: pd.DataFrame | None = None) -> pd.DataFrame:
    """Ensure league has numeric runs_for/runs_against/run_diff by deriving from monthly if needed."""
    league = league.copy()
    agg = None
    if monthly is not None and {"runs_for", "runs_against"}.issubset(monthly.columns):
        agg = (
            monthly.groupby("team_id")[["runs_for", "runs_against"]]
            .sum()
            .reset_index()
            .assign(run_diff=lambda d: d["runs_for"] - d["runs_against"])
        )

    if {"runs_for", "runs_against"}.issubset(league.columns):
        league["runs_for"] = pd.to_numeric(league["runs_for"], errors="coerce")
        league["runs_against"] = pd.to_numeric(league["runs_against"], errors="coerce")
    elif agg is not None:
        league = league.merge(agg.rename(columns={"runs_for": "runs_for_src", "runs_against": "runs_against_src"}), on="team_id", how="left")
        existing_rf = pd.to_numeric(league["runs_for"], errors="coerce") if "runs_for" in league else pd.Series([pd.NA] * len(league))
        existing_ra = pd.to_numeric(league["runs_against"], errors="coerce") if "runs_against" in league else pd.Series([pd.NA] * len(league))
        league["runs_for"] = existing_rf.fillna(league["runs_for_src"])
        league["runs_against"] = existing_ra.fillna(league["runs_against_src"])
        league = league.drop(columns=[c for c in ["runs_for_src", "runs_against_src"] if c in league.columns])

    if "run_diff" not in league.columns:
        league["run_diff"] = pd.NA
    league["run_diff"] = pd.to_numeric(league["run_diff"], errors="coerce")

    if {"runs_for", "runs_against"}.issubset(league.columns):
        league["run_diff"] = league["run_diff"].fillna(league["runs_for"] - league["runs_against"])
    elif agg is not None and "run_diff" in agg.columns:
        league["run_diff"] = league["run_diff"].fillna(league["team_id"].map(agg.set_index("team_id")["run_diff"]))

    return league


def build_season_run_diff_giants(league: pd.DataFrame) -> pd.DataFrame:
    df = ensure_base_cols(league)
    if {"runs_for", "runs_against"}.issubset(df.columns):
        df["season_run_diff"] = pd.to_numeric(df["runs_for"], errors="coerce") - pd.to_numeric(
            df["runs_against"], errors="coerce"
        )
    elif "run_diff" in df.columns:
        df["season_run_diff"] = pd.to_numeric(df["run_diff"], errors="coerce")
    else:
        raise KeyError(
            "league_season_summary missing run_diff and runs_for/runs_against; cannot build run-diff stories."
        )
    df = df.dropna(subset=["season_run_diff"])
    top = df.sort_values("season_run_diff", ascending=False).head(5).reset_index(drop=True).copy()
    top["story_group"] = "Season Giants – Run Differential"
    top["story_type"] = "season_run_diff_giant"
    top["rank"] = top.index + 1
    top["metric_name"] = "run_diff"
    top["metric_value"] = top["season_run_diff"]
    top["focus_label"] = top.apply(lambda r: f"{r.team_name} ({r.team_abbr})", axis=1)
    top["comparison_note"] = top.apply(
        lambda r: f"led the league with a run differential of {int(round(r.season_run_diff)):+d}.", axis=1
    )
    check_no_nan(top, ["metric_value"], "Season Run Diff Giants")
    return top[
        [
            "season",
            "league_id",
            "story_group",
            "story_type",
            "rank",
            "team_id",
            "team_abbr",
            "team_name",
            "conf",
            "division",
            "metric_name",
            "metric_value",
            "focus_label",
            "comparison_note",
        ]
    ]


def build_season_win_pct_giants(league: pd.DataFrame) -> pd.DataFrame:
    df = ensure_base_cols(league)
    if "pct" not in df.columns:
        if "win_pct" in df.columns:
            df["pct"] = df["win_pct"]
        else:
            raise KeyError("league_season_summary is missing pct/win_pct; cannot build win-pct stories.")
    df["pct"] = pd.to_numeric(df["pct"], errors="coerce")
    df = df.dropna(subset=["pct"])
    top = df.sort_values("pct", ascending=False).head(5).reset_index(drop=True).copy()
    top["story_group"] = "Season Giants – Winning Percentage"
    top["story_type"] = "season_win_pct_giant"
    top["rank"] = top.index + 1
    top["metric_name"] = "pct"
    top["metric_value"] = top["pct"]
    top["focus_label"] = top.apply(lambda r: f"{r.team_name} ({r.team_abbr})", axis=1)
    top["comparison_note"] = top.apply(
        lambda r: f"stood out on pct: {r.pct:.3f}.",
        axis=1,
    )
    check_no_nan(top, ["metric_value"], "Season Win Pct Giants")
    return top[
        [
            "season",
            "league_id",
            "story_group",
            "story_type",
            "rank",
            "team_id",
            "team_abbr",
            "team_name",
            "conf",
            "division",
            "metric_name",
            "metric_value",
            "focus_label",
            "comparison_note",
        ]
    ]


def build_second_half_swing(half: pd.DataFrame, positive: bool, label: str, story_type: str) -> pd.DataFrame:
    df = ensure_base_cols(half)
    ensure_columns(df, ["half_label", "half_win_pct", "half_win_pct_delta_vs_season"], "half_summary")
    df["half_win_pct"] = pd.to_numeric(df["half_win_pct"], errors="coerce")
    df["half_win_pct_delta_vs_season"] = pd.to_numeric(df["half_win_pct_delta_vs_season"], errors="coerce")
    mask_half = df["half_label"].str.lower().str.contains("2") | df["half_label"].str.lower().str.contains("second")
    df = df[mask_half].copy()
    if positive:
        df = df[df["half_win_pct_delta_vs_season"] > 0].copy()
        df = df.sort_values("half_win_pct_delta_vs_season", ascending=False)
    else:
        df = df[df["half_win_pct_delta_vs_season"] < 0].copy()
        df = df.sort_values("half_win_pct_delta_vs_season", ascending=True)
    df = df.head(5).reset_index(drop=True).copy()
    df["story_group"] = label
    df["story_type"] = story_type
    df["rank"] = df.index + 1
    df["metric_name"] = "half_win_pct_delta_vs_season"
    df["metric_value"] = df["half_win_pct_delta_vs_season"]
    df["focus_label"] = df.apply(lambda r: f"{r.team_name} ({r.team_abbr})", axis=1)
    if positive:
        df["comparison_note"] = df.apply(
            lambda r: f"caught fire after midseason: {r.half_win_pct:.3f} in the second half (delta {r.half_win_pct_delta_vs_season:+.3f} vs season).",
            axis=1,
        )
    else:
        df["comparison_note"] = df.apply(
            lambda r: f"faded after midseason: {r.half_win_pct:.3f} in the second half (delta {r.half_win_pct_delta_vs_season:+.3f} vs season).",
            axis=1,
        )
    check_no_nan(df, ["metric_value"], label)
    return df[
        [
            "season",
            "league_id",
            "story_group",
            "story_type",
            "rank",
            "team_id",
            "team_abbr",
            "team_name",
            "conf",
            "division",
            "metric_name",
            "metric_value",
            "focus_label",
            "comparison_note",
        ]
    ]


def build_month_momentum(monthly: pd.DataFrame, positive: bool, label: str, story_type: str, limit: int) -> pd.DataFrame:
    df = ensure_base_cols(monthly)
    ensure_columns(df, ["month", "month_win_pct", "month_win_pct_delta_vs_season"], "team_monthly_momentum")
    df["month_win_pct"] = pd.to_numeric(df["month_win_pct"], errors="coerce")
    df["month_win_pct_delta_vs_season"] = pd.to_numeric(df["month_win_pct_delta_vs_season"], errors="coerce")
    if positive:
        df = df[df["month_win_pct_delta_vs_season"] > 0].copy()
        df = df.sort_values("month_win_pct_delta_vs_season", ascending=False)
    else:
        df = df[df["month_win_pct_delta_vs_season"] < 0].copy()
        df = df.sort_values("month_win_pct_delta_vs_season", ascending=True)
    df = df.head(limit).reset_index(drop=True).copy()
    df["story_group"] = label
    df["story_type"] = story_type
    df["rank"] = df.index + 1
    df["metric_name"] = "month_win_pct_delta_vs_season"
    df["metric_value"] = df["month_win_pct_delta_vs_season"]
    df["focus_label"] = df["month"].astype(str)
    if positive:
        df["comparison_note"] = df.apply(
            lambda r: f"In {r.month}, played above their season pace: {r.month_win_pct:.3f} (delta {r.month_win_pct_delta_vs_season:+.3f}).",
            axis=1,
        )
    else:
        df["comparison_note"] = df.apply(
            lambda r: f"In {r.month}, slumped below their season pace: {r.month_win_pct:.3f} (delta {r.month_win_pct_delta_vs_season:+.3f}).",
            axis=1,
        )
    check_no_nan(df, ["metric_value"], label)
    return df[
        [
            "season",
            "league_id",
            "story_group",
            "story_type",
            "rank",
            "team_id",
            "team_abbr",
            "team_name",
            "conf",
            "division",
            "metric_name",
            "metric_value",
            "focus_label",
            "comparison_note",
        ]
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Flashback story candidates from almanac tables.")
    parser.add_argument("--season", type=int, required=True, help="Season year (e.g. 1972)")
    parser.add_argument("--league-id", type=int, default=200, help="League ID (default 200)")
    parser.add_argument(
        "--almanac-root",
        type=str,
        default="csv/out/almanac",
        help="Root folder for almanac-derived CSV outputs.",
    )
    args = parser.parse_args()
    season = args.season
    league_id = args.league_id

    root = Path(args.almanac_root) / str(season)
    league_path = root / f"league_season_summary_{season}_league{league_id}.csv"
    half_path = root / f"half_summary_{season}_league{league_id}.csv"
    monthly_path = root / f"team_monthly_momentum_{season}_league{league_id}.csv"
    out_path = root / f"flashback_story_candidates_{season}_league{league_id}.csv"

    log(f"[DEBUG] season={season}, league_id={league_id}")
    log(f"[DEBUG] league_path={league_path}")
    log(f"[DEBUG] half_path={half_path}")
    log(f"[DEBUG] monthly_path={monthly_path}")

    league = load_csv(league_path, "league_season_summary")
    half = load_csv(half_path, "half_summary")
    monthly = load_csv(monthly_path, "team_monthly_momentum")

    league = normalize_conf_div(league)
    half = normalize_conf_div(half)
    monthly = normalize_conf_div(monthly)

    # Backfill run differential if the league summary lacks it
    league = backfill_season_run_diff(league, monthly)

    # Backfill identity into half/monthly from league summary
    half = backfill_identity(half, league)
    monthly = backfill_identity(monthly, league)

    season_rd = build_season_run_diff_giants(league)
    season_wp = build_season_win_pct_giants(league)
    second_half_surges = build_second_half_swing(
        half, positive=True, label="Second-Half Surges", story_type="second_half_surge"
    )
    second_half_collapses = build_second_half_swing(
        half, positive=False, label="Second-Half Collapses", story_type="second_half_collapse"
    )
    month_glory = build_month_momentum(
        monthly,
        positive=True,
        label="Month of Glory – Overachievers",
        story_type="month_glory_overachiever",
        limit=10,
    )
    month_misery = build_month_momentum(
        monthly,
        positive=False,
        label="Month of Misery – Slumps",
        story_type="month_misery_slump",
        limit=10,
    )

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

    check_no_nan(candidates, ["metric_value"], "Flashback story candidates")

    candidates = candidates.sort_values(["story_group", "rank", "team_name"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(out_path, index=False)
    log(f"[OK] Wrote Flashback story candidates to {out_path}")
    log(f"[INFO] Total rows: {len(candidates)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
