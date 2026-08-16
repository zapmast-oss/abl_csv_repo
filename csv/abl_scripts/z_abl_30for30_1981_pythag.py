from pathlib import Path
import argparse
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
CSV_ROOT = SCRIPT_PATH.parents[1]
STAR_DIR = CSV_ROOT / "out" / "star_schema"


def classify_angle(diff: float) -> str:
    if diff >= 3.0:
        return "big_over_achiever"
    if diff >= 1.0:
        return "mild_over_achiever"
    if diff <= -3.0:
        return "big_under_achiever"
    if diff <= -1.0:
        return "mild_under_achiever"
    return "true_to_form"


def main(dry_run: bool = False):
    backbone_path = STAR_DIR / "fact_team_season_1981_backbone.csv"
    if not backbone_path.exists():
        print(f"ERROR: 1981 backbone not found at: {backbone_path}")
        print("Run z_abl_team_season_backbone_1981.py first.")
        return

    df = pd.read_csv(backbone_path)
    print("BACKBONE columns:", list(df.columns))

    required_cols = [
        "team_abbr",
        "team_name",
        "wins",
        "losses",
        "games_played",
        "run_diff",
        "pythag_win_pct",
        "pythag_expected_wins",
        "pythag_expected_losses",
        "pythag_diff",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"ERROR: Missing columns in backbone: {missing}")
        return

    if df["team_abbr"].nunique() != 24:
        print("WARNING: backbone does not show 24 unique teams; check exports.")

    slim = df[
        [
            "team_abbr",
            "team_name",
            "games_played",
            "wins",
            "losses",
            "run_diff",
            "pythag_win_pct",
            "pythag_expected_wins",
            "pythag_expected_losses",
            "pythag_diff",
        ]
    ].copy()

    slim["angle_category"] = slim["pythag_diff"].apply(classify_angle)
    slim["angle_rank_over"] = slim["pythag_diff"].rank(ascending=False, method="min").astype(int)
    slim["angle_rank_under"] = slim["pythag_diff"].rank(ascending=True, method="min").astype(int)
    slim["angle_rank_abs"] = slim["pythag_diff"].abs().rank(ascending=False, method="min").astype(int)

    out_cols = [
        "team_abbr",
        "team_name",
        "games_played",
        "wins",
        "losses",
        "run_diff",
        "pythag_win_pct",
        "pythag_expected_wins",
        "pythag_expected_losses",
        "pythag_diff",
        "angle_category",
        "angle_rank_over",
        "angle_rank_under",
        "angle_rank_abs",
    ]
    report = slim[out_cols].copy()

    out_path = STAR_DIR / "abl_1981_30for30_pythag_report.csv"
    if not dry_run:
        report.to_csv(out_path, index=False)
        print("ABL 1981 30FOR30 PYTHAG REPORT: wrote", out_path)

    print(report.sort_values("angle_rank_abs").head(10))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run without writing output CSV")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
