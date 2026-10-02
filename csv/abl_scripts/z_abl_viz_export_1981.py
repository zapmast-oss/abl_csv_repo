from pathlib import Path
import argparse
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
CSV_ROOT = SCRIPT_PATH.parents[1]
STAR_DIR = CSV_ROOT / "out" / "star_schema"


def main(dry_run: bool = False):
    pythag_path = STAR_DIR / "abl_1981_30for30_pythag_report.csv"
    power_path = STAR_DIR / "monday_1981_power_ranking.csv"
    change_path = STAR_DIR / "fact_team_reporting_1981_weekly_change.csv"

    missing = [p for p in [pythag_path, power_path] if not p.exists()]
    if missing:
        print("ERROR: Missing required input(s) for viz export:")
        for path in missing:
            print("  -", path)
        print("Run the 1981 pipeline + 30for30 Pythag script first.")
        return

    pythag_df = pd.read_csv(pythag_path)
    power_df = pd.read_csv(power_path)
    change_df = pd.read_csv(change_path) if change_path.exists() else None

    print("PYTHAG columns:", list(pythag_df.columns))
    print("POWER columns:", list(power_df.columns))
    if change_df is not None:
        print("WEEKLY CHANGE columns:", list(change_df.columns))
    else:
        print("No weekly change file found; weekly viz export will be skipped.")

    required_pythag_cols = {"team_abbr", "team_name", "pythag_diff"}
    if not required_pythag_cols.issubset(pythag_df.columns):
        print("ERROR: Pythag report missing expected columns:", required_pythag_cols)
        return

    pythag_bar = pythag_df[["team_abbr", "team_name", "pythag_diff"]].copy()
    pythag_bar = pythag_bar.sort_values("pythag_diff", ascending=False).reset_index(drop=True)

    if not dry_run:
        out_pythag = STAR_DIR / "viz_1981_pythag_bar.csv"
        pythag_bar.to_csv(out_pythag, index=False)
        print("VIZ: wrote Pythag bar data to", out_pythag)

    required_power_cols = {"team_abbr", "team_name", "sub_league", "division", "run_diff", "power_rank"}
    if not required_power_cols.issubset(power_df.columns):
        print("ERROR: Power ranking file missing expected columns:", required_power_cols)
        return

    power_viz = power_df[
        ["team_abbr", "team_name", "sub_league", "division", "power_rank", "run_diff"]
    ].copy()
    power_viz = power_viz.sort_values("power_rank", ascending=True).reset_index(drop=True)

    if not dry_run:
        out_power = STAR_DIR / "viz_1981_power_vs_run_diff.csv"
        power_viz.to_csv(out_power, index=False)
        print("VIZ: wrote Power vs RunDiff data to", out_power)

    if change_df is not None:
        if "team_abbr" not in change_df.columns:
            print("VIZ: Weekly change file lacks team_abbr; skipping weekly viz export.")
        else:
            keep_cols = ["team_abbr"]
            for col in ["delta_wins", "delta_run_diff", "delta_win_pct", "games_this_week"]:
                if col in change_df.columns:
                    keep_cols.append(col)

            weekly_bar = change_df[keep_cols].copy()
            sort_cols = [c for c in ["delta_wins", "delta_run_diff"] if c in weekly_bar.columns]
            if sort_cols:
                weekly_bar = weekly_bar.sort_values(
                    by=sort_cols, ascending=[False] * len(sort_cols)
                ).reset_index(drop=True)

            if not dry_run:
                out_weekly = STAR_DIR / "viz_1981_weekly_change_bar.csv"
                weekly_bar.to_csv(out_weekly, index=False)
                print("VIZ: wrote Weekly Change bar data to", out_weekly)
    else:
        print("VIZ: Weekly change file not present; viz_1981_weekly_change_bar.csv will not be created.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run without writing output CSVs")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
