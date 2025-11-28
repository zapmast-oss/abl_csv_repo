from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


REQUIRED = [
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


def load_league_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"Error: league summary not found at {path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        print(f"Error: league summary missing columns: {missing}", file=sys.stderr)
        print(f"Columns present: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)
    return df


def summarize_conference(df: pd.DataFrame) -> pd.DataFrame:
    groups = []
    for conf, grp in df.groupby("conf"):
        grp = grp.copy()
        grp["pct_num"] = pd.to_numeric(grp["pct"], errors="coerce")
        grp["run_diff_num"] = pd.to_numeric(grp["run_diff"], errors="coerce")
        num_teams = grp["team_id"].nunique()
        total_wins = pd.to_numeric(grp["wins"], errors="coerce").sum()
        total_losses = pd.to_numeric(grp["losses"], errors="coerce").sum()
        avg_win_pct = grp["pct_num"].mean()
        total_run_diff = grp["run_diff_num"].sum()
        avg_run_diff = grp["run_diff_num"].mean()

        best = grp.sort_values("pct_num", ascending=False).iloc[0]
        worst = grp.sort_values("pct_num", ascending=True).iloc[0]

        groups.append(
            {
                "season": grp["season"].iloc[0],
                "league_id": grp["league_id"].iloc[0],
                "conf": conf,
                "num_teams": num_teams,
                "total_wins": total_wins,
                "total_losses": total_losses,
                "avg_win_pct": avg_win_pct,
                "total_run_diff": total_run_diff,
                "avg_run_diff": avg_run_diff,
                "best_team_id": best["team_id"],
                "best_team_name": best["team_name"],
                "best_team_win_pct": best["pct_num"],
                "worst_team_id": worst["team_id"],
                "worst_team_name": worst["team_name"],
                "worst_team_win_pct": worst["pct_num"],
            }
        )
    out = pd.DataFrame(groups)
    out = out.sort_values("conf")
    return out


def summarize_division(df: pd.DataFrame) -> pd.DataFrame:
    groups = []
    for (conf, division), grp in df.groupby(["conf", "division"]):
        grp = grp.copy()
        grp["pct_num"] = pd.to_numeric(grp["pct"], errors="coerce")
        grp["run_diff_num"] = pd.to_numeric(grp["run_diff"], errors="coerce")
        num_teams = grp["team_id"].nunique()
        total_wins = pd.to_numeric(grp["wins"], errors="coerce").sum()
        total_losses = pd.to_numeric(grp["losses"], errors="coerce").sum()
        avg_win_pct = grp["pct_num"].mean()
        total_run_diff = grp["run_diff_num"].sum()
        avg_run_diff = grp["run_diff_num"].mean()

        best = grp.sort_values("pct_num", ascending=False).iloc[0]
        worst = grp.sort_values("pct_num", ascending=True).iloc[0]

        groups.append(
            {
                "season": grp["season"].iloc[0],
                "league_id": grp["league_id"].iloc[0],
                "conf": conf,
                "division": division,
                "num_teams": num_teams,
                "total_wins": total_wins,
                "total_losses": total_losses,
                "avg_win_pct": avg_win_pct,
                "total_run_diff": total_run_diff,
                "avg_run_diff": avg_run_diff,
                "best_team_id": best["team_id"],
                "best_team_name": best["team_name"],
                "best_team_win_pct": best["pct_num"],
                "worst_team_id": worst["team_id"],
                "worst_team_name": worst["team_name"],
                "worst_team_win_pct": worst["pct_num"],
            }
        )
    out = pd.DataFrame(groups)
    out = out.sort_values(["conf", "division"])
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build 4k conference/division summaries from league season summary."
    )
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
    conf_out = season_dir / f"conference_summary_{args.season}_league{args.league_id}.csv"
    div_out = season_dir / f"division_summary_{args.season}_league{args.league_id}.csv"

    print(f"[DEBUG] season={args.season}, league_id={args.league_id}")
    print(f"[DEBUG] league_summary_path={league_summary_path}")

    df = load_league_summary(league_summary_path)
    print(f"[INFO] Loaded {len(df)} team rows from league-season summary")

    conf_df = summarize_conference(df)
    conf_out.parent.mkdir(parents=True, exist_ok=True)
    conf_df.to_csv(conf_out, index=False)
    print(f"[INFO] Wrote conference summary to {conf_out}")

    div_df = summarize_division(df)
    div_df.to_csv(div_out, index=False)
    print(f"[INFO] Wrote division summary to {div_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
