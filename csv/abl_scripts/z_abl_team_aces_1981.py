from pathlib import Path
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
CSV_ROOT = SCRIPT_PATH.parents[1]
STAR_DIR = CSV_ROOT / "out" / "star_schema"

OUT_PATH = STAR_DIR / "team_aces_1981.csv"


def main() -> None:
    profile_path = STAR_DIR / "dim_player_profile.csv"
    bat_path = STAR_DIR / "fact_player_batting.csv"
    pit_path = STAR_DIR / "fact_player_pitching.csv"
    teamrep_path = STAR_DIR / "fact_team_reporting_1981_current.csv"

    missing = [p for p in [profile_path, bat_path, pit_path, teamrep_path] if not p.exists()]
    if missing:
        print("ERROR: Missing required input(s) for team_aces_1981:")
        for p in missing:
            print("  -", p)
        return

    profile = pd.read_csv(profile_path)
    bat = pd.read_csv(bat_path)
    pit = pd.read_csv(pit_path)
    teamrep = pd.read_csv(teamrep_path)

    team_abbrs = set(teamrep["team_abbr"].unique())

    # Batting WAR per (team, player)
    bat_use = (
        bat[["ID", "TM", "WAR"]]
        .rename(columns={"TM": "team_abbr", "WAR": "bat_war"})
    )
    bat_use = bat_use[bat_use["team_abbr"].isin(team_abbrs)]
    bat_agg = bat_use.groupby(["team_abbr", "ID"], as_index=False)["bat_war"].sum()

    # Pitching WAR per (team, player)
    pit_use = (
        pit[["ID", "TM_p1", "WAR"]]
        .rename(columns={"TM_p1": "team_abbr", "WAR": "pit_war"})
    )
    pit_use = pit_use[pit_use["team_abbr"].isin(team_abbrs)]
    pit_agg = pit_use.groupby(["team_abbr", "ID"], as_index=False)["pit_war"].sum()

    profile_small = profile[["ID", "Name", "POS"]].copy()

    # Batting aces: max WAR per team
    if not bat_agg.empty:
        bat_idx = bat_agg.groupby("team_abbr")["bat_war"].idxmax()
        bat_top = bat_agg.loc[bat_idx].copy()
        bat_top = bat_top.merge(profile_small, on="ID", how="left")
        bat_top = bat_top.rename(
            columns={
                "ID": "batting_ace_player_id",
                "Name": "batting_ace_name",
                "POS": "batting_ace_pos",
                "bat_war": "batting_ace_war",
            }
        )
    else:
        bat_top = pd.DataFrame(
            columns=[
                "team_abbr",
                "batting_ace_player_id",
                "batting_ace_name",
                "batting_ace_pos",
                "batting_ace_war",
            ]
        )

    # Pitching aces: max WAR per team
    if not pit_agg.empty:
        pit_idx = pit_agg.groupby("team_abbr")["pit_war"].idxmax()
        pit_top = pit_agg.loc[pit_idx].copy()
        pit_top = pit_top.merge(profile_small, on="ID", how="left")
        pit_top = pit_top.rename(
            columns={
                "ID": "pitching_ace_player_id",
                "Name": "pitching_ace_name",
                "POS": "pitching_ace_pos",
                "pit_war": "pitching_ace_war",
            }
        )
    else:
        pit_top = pd.DataFrame(
            columns=[
                "team_abbr",
                "pitching_ace_player_id",
                "pitching_ace_name",
                "pitching_ace_pos",
                "pitching_ace_war",
            ]
        )

    # Merge batting + pitching aces into one row per team
    aces = pd.merge(bat_top, pit_top, on="team_abbr", how="outer")

    # Ensure we only keep canon ABL teams
    aces = aces[aces["team_abbr"].isin(team_abbrs)].copy()

    # Type clean-up
    for col in ["batting_ace_player_id", "pitching_ace_player_id"]:
        if col in aces.columns:
            aces[col] = aces[col].astype("Int64")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    aces.to_csv(OUT_PATH, index=False)
    print(f"Wrote team aces table to {OUT_PATH}")


if __name__ == "__main__":
    main()
