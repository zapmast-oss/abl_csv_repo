"""ABL 1980 Season Backbone: one-row summary per club for broadcast context."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from abl_config import LEAGUE_ID, RAW_CSV_ROOT, TEAM_IDS
from abl_team_helper import load_abl_teams

CSV_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = CSV_ROOT / "out" / "csv_out" / "z_ABL_1980_Season_Backbone.csv"
SEASON = 1980
TEAM_SET = set(TEAM_IDS)


def load_team_records() -> pd.DataFrame:
    path = RAW_CSV_ROOT / "team_history_record.csv"
    df = pd.read_csv(path)
    df = df[
        (df["league_id"] == LEAGUE_ID)
        & (df["year"] == SEASON)
        & (df["team_id"].isin(TEAM_SET))
    ].copy()
    df["team_id"] = df["team_id"].astype(int)
    df["wins"] = pd.to_numeric(df["w"], errors="coerce").fillna(0).astype(int)
    df["losses"] = pd.to_numeric(df["l"], errors="coerce").fillna(0).astype(int)
    df["sub_league_id"] = df["sub_league_id"].astype(int)
    return df[["team_id", "wins", "losses", "sub_league_id"]]


def load_runs_scored() -> pd.DataFrame:
    path = RAW_CSV_ROOT / "team_history_batting_stats.csv"
    df = pd.read_csv(path)
    df = df[
        (df["league_id"] == LEAGUE_ID)
        & (df["year"] == SEASON)
        & (df["team_id"].isin(TEAM_SET))
    ].copy()
    df["team_id"] = df["team_id"].astype(int)
    df["runs_scored"] = pd.to_numeric(df["r"], errors="coerce")
    return df[["team_id", "runs_scored"]]


def load_runs_allowed() -> pd.DataFrame:
    path = RAW_CSV_ROOT / "team_history_pitching_stats.csv"
    df = pd.read_csv(path)
    df = df[
        (df["league_id"] == LEAGUE_ID)
        & (df["year"] == SEASON)
        & (df["team_id"].isin(TEAM_SET))
    ].copy()
    df["team_id"] = df["team_id"].astype(int)
    df["runs_allowed"] = pd.to_numeric(df["r"], errors="coerce")
    return df[["team_id", "runs_allowed"]]


def load_playoff_flags() -> Tuple[pd.DataFrame, int]:
    path = RAW_CSV_ROOT / "team_history.csv"
    df = pd.read_csv(path)
    df = df[
        (df["league_id"] == LEAGUE_ID)
        & (df["year"] == SEASON)
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


def determine_conference_winners(df: pd.DataFrame) -> Dict[int, int]:
    """Approximate conference champions by best regular-season record per sub-league."""
    ranked = df.sort_values(
        ["sub_league_id", "wins", "run_diff"], ascending=[True, False, False]
    )
    winners = ranked.groupby("sub_league_id").head(1)
    return {row["team_id"]: 1 for _, row in winners.iterrows()}


def build_backbone() -> Tuple[pd.DataFrame, Dict[int, Dict[str, str]], int]:
    teams_lookup_df = load_abl_teams().rename(
        columns={"name": "team_name", "abbr": "team_abbr"}
    )
    team_lookup = {
        row["team_id"]: {"name": row["team_name"], "abbr": row["team_abbr"]}
        for _, row in teams_lookup_df.iterrows()
    }

    records = load_team_records()
    batting = load_runs_scored()
    pitching = load_runs_allowed()
    playoff_flags, champion_id = load_playoff_flags()

    df = (
        records.merge(batting, on="team_id", how="left")
        .merge(pitching, on="team_id", how="left")
        .merge(playoff_flags, on="team_id", how="left")
    )

    df["runs_scored"] = df["runs_scored"].fillna(0)
    df["runs_allowed"] = df["runs_allowed"].fillna(0)
    df["made_playoffs"] = df["made_playoffs"].fillna(0).astype(int)
    df["won_title"] = df["won_title"].fillna(0).astype(int)
    df["season"] = SEASON

    df = compute_pythag(df)

    conference_flags = determine_conference_winners(df)
    df["won_conference"] = df["team_id"].map(conference_flags).fillna(0).astype(int)

    df["team_name"] = df["team_id"].map(lambda tid: team_lookup[tid]["name"])
    df["team_abbr"] = df["team_id"].map(lambda tid: team_lookup[tid]["abbr"])

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
            "sub_league_id",
        ]
    ].sort_values("team_id")
    return df.reset_index(drop=True), team_lookup, champion_id


def write_output(df: pd.DataFrame) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_output = df.drop(columns=["sub_league_id"])
    df_output.to_csv(OUT_PATH, index=False)


def validate_output() -> None:
    df = pd.read_csv(OUT_PATH)
    if len(df) != len(TEAM_SET):
        raise SystemExit(f"Validation failed: expected {len(TEAM_SET)} rows, found {len(df)}.")
    if not set(df["team_id"]).issubset(TEAM_SET):
        raise SystemExit("Validation failed: team_id outside canonical set.")
    if (df["season"] != SEASON).any():
        raise SystemExit("Validation failed: season column contains non-1980 values.")


def main() -> None:
    try:
        backbone, team_lookup, champion_id = build_backbone()
        write_output(backbone)
        validate_output()

        playoff_teams = int(backbone["made_playoffs"].sum())
        conference_winners = backbone.loc[backbone["won_conference"] == 1, "team_id"].tolist()
        conference_labels = []
        for idx, team_id in enumerate(conference_winners):
            info = team_lookup.get(team_id, {"name": f"Team {team_id}"})
            conference_labels.append(f"Conf {idx + 1}: {info['name']} (team_id={team_id})")

        champ_label = (
            f"{team_lookup.get(champion_id, {'name': 'Unknown'})['name']} (team_id={champion_id})"
            if champion_id is not None
            else "Unknown"
        )
        print(
            f"1980 Playoffs: {playoff_teams} playoff teams, "
            + ", ".join(conference_labels)
            + f", Grand Champion = {champ_label}"
        )
        print(
            f"1980 Season Backbone: {len(backbone)} teams, champion team_id={champion_id}, "
            f"wrote {OUT_PATH.name}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
