from pathlib import Path
import argparse
import pandas as pd

from abl_config import LEAGUE_ID, RAW_CSV_ROOT, TEAM_IDS
from abl_team_helper import load_abl_teams

SCRIPT_PATH = Path(__file__).resolve()
CSV_ROOT = SCRIPT_PATH.parents[1]
STAR_DIR = CSV_ROOT / "out" / "star_schema"
TEAM_SET = set(TEAM_IDS)


def load_team_records(season: int) -> pd.DataFrame:
    path = RAW_CSV_ROOT / "team_history_record.csv"
    df = pd.read_csv(path)
    df = df[
        (df["league_id"] == LEAGUE_ID)
        & (df["year"] == season)
        & (df["team_id"].isin(TEAM_SET))
    ].copy()
    df["team_id"] = df["team_id"].astype(int)
    df["wins"] = pd.to_numeric(df["w"], errors="coerce").fillna(0).astype(int)
    df["losses"] = pd.to_numeric(df["l"], errors="coerce").fillna(0).astype(int)
    df["sub_league_id"] = df["sub_league_id"].astype(int)
    return df[["team_id", "wins", "losses", "sub_league_id"]]


def load_runs_scored(season: int) -> pd.DataFrame:
    path = RAW_CSV_ROOT / "team_history_batting_stats.csv"
    df = pd.read_csv(path)
    df = df[
        (df["league_id"] == LEAGUE_ID)
        & (df["year"] == season)
        & (df["team_id"].isin(TEAM_SET))
    ].copy()
    df["team_id"] = df["team_id"].astype(int)
    df["runs_scored"] = pd.to_numeric(df["r"], errors="coerce")
    return df[["team_id", "runs_scored"]]


def load_runs_allowed(season: int) -> pd.DataFrame:
    path = RAW_CSV_ROOT / "team_history_pitching_stats.csv"
    df = pd.read_csv(path)
    df = df[
        (df["league_id"] == LEAGUE_ID)
        & (df["year"] == season)
        & (df["team_id"].isin(TEAM_SET))
    ].copy()
    df["team_id"] = df["team_id"].astype(int)
    df["runs_allowed"] = pd.to_numeric(df["r"], errors="coerce")
    return df[["team_id", "runs_allowed"]]


def load_playoff_flags(season: int) -> tuple[pd.DataFrame, int | None]:
    path = RAW_CSV_ROOT / "team_history.csv"
    df = pd.read_csv(path)
    df = df[
        (df["league_id"] == LEAGUE_ID)
        & (df["year"] == season)
        & (df["team_id"].isin(TEAM_SET))
    ][["team_id", "made_playoffs", "won_playoffs"]].copy()
    df["team_id"] = df["team_id"].astype(int)
    df["made_playoffs"] = df["made_playoffs"].fillna(0).astype(int)
    df["won_playoffs"] = df["won_playoffs"].fillna(0).astype(int)
    champs = df.loc[df["won_playoffs"] == 1, "team_id"].tolist()
    champion_id = champs[0] if champs else None
    df = df.rename(columns={"won_playoffs": "won_title"})
    return df, champion_id


def compute_pythag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["games"] = df["wins"] + df["losses"]
    df["win_pct"] = df["wins"] / df["games"].where(df["games"] != 0, other=pd.NA)
    rs_sq = df["runs_scored"] ** 2
    ra_sq = df["runs_allowed"] ** 2
    denom = rs_sq + ra_sq
    df["pythag_win_pct"] = rs_sq / denom
    df.loc[denom == 0, "pythag_win_pct"] = pd.NA
    df["pythag_win_pct"] = df["pythag_win_pct"].fillna(0.5)
    df["pythag_wins"] = (df["pythag_win_pct"] * df["games"]).round(1)
    df["pythag_losses"] = (df["games"] - df["pythag_wins"]).round(1)
    df["run_diff"] = df["runs_scored"] - df["runs_allowed"]
    df["pythag_diff"] = (df["pythag_wins"] - df["wins"]).round(1)
    return df


def determine_conference_winners(df: pd.DataFrame) -> dict[int, int]:
    ranked = df.sort_values(["sub_league_id", "wins", "run_diff"], ascending=[True, False, False])
    winners = ranked.groupby("sub_league_id").head(1)
    return {row["team_id"]: 1 for _, row in winners.iterrows()}


def build_season_backbone(season: int, dry_run: bool = False) -> pd.DataFrame:
    records = load_team_records(season)
    if records.empty:
        raise SystemExit(f"No team records found for season {season}")

    batting = load_runs_scored(season)
    pitching = load_runs_allowed(season)
    playoff_flags, champion_id = load_playoff_flags(season)

    df = (
        records.merge(batting, on="team_id", how="left")
        .merge(pitching, on="team_id", how="left")
        .merge(playoff_flags, on="team_id", how="left")
    )

    df["runs_scored"] = df["runs_scored"].fillna(0)
    df["runs_allowed"] = df["runs_allowed"].fillna(0)
    df["made_playoffs"] = df["made_playoffs"].fillna(0).astype(int)
    df["won_title"] = df["won_title"].fillna(0).astype(int)
    df["season"] = season

    df = compute_pythag(df)
    conference_flags = determine_conference_winners(df)
    df["won_conference"] = df["team_id"].map(conference_flags).fillna(0).astype(int)

    teams_lookup_df = load_abl_teams().rename(columns={"name": "team_name", "abbr": "team_abbr"})
    df["team_name"] = df["team_id"].map(teams_lookup_df.set_index("team_id")["team_name"])
    df["team_abbr"] = df["team_id"].map(teams_lookup_df.set_index("team_id")["team_abbr"])

    df = df[
        [
            "team_id",
            "team_name",
            "team_abbr",
            "season",
            "wins",
            "losses",
            "win_pct",
            "runs_scored",
            "runs_allowed",
            "run_diff",
            "pythag_wins",
            "pythag_losses",
            "pythag_diff",
            "made_playoffs",
            "won_conference",
            "won_title",
        ]
    ].sort_values("team_id").reset_index(drop=True)

    if (df["team_id"] < 1).any() or (df["team_id"] > 24).any():
        raise SystemExit("Detected team_id outside ABL canon")

    if not dry_run:
        output_path = STAR_DIR / f"fact_schedule_{season}.csv"
        STAR_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Wrote {output_path}")

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True, help="Season year (e.g. 1980)")
    parser.add_argument("--dry-run", action="store_true", help="Run without writing output CSV")
    args = parser.parse_args()

    df = build_season_backbone(season=args.season, dry_run=args.dry_run)
    print(f"SEASON_BACKBONE {args.season}: built {len(df)} teams")
    print("Sample:")
    print(df.head(10))


if __name__ == "__main__":
    main()
