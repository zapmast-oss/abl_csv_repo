from pathlib import Path
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
CSV_ROOT = SCRIPT_PATH.parents[1]
STAR_DIR = CSV_ROOT / "out" / "star_schema"

OUT_PATH = STAR_DIR / "face_of_franchise_1981.csv"


POP_RANK = {
    "Unkn.": 0,
    "Insig.": 1,
    "Fair": 2,
    "Known": 3,
    "Pop.": 4,
    "V.Pop": 5,
    "Ex.Pop.": 6,
}


def main() -> None:
    # Required inputs
    profile_path = STAR_DIR / "dim_player_profile.csv"
    bat_path = STAR_DIR / "fact_player_batting.csv"
    pit_path = STAR_DIR / "fact_player_pitching.csv"
    teamrep_path = STAR_DIR / "fact_team_reporting_1981_current.csv"

    missing = [p for p in [profile_path, bat_path, pit_path, teamrep_path] if not p.exists()]
    if missing:
        print("ERROR: Missing required input(s) for face_of_franchise_1981:")
        for p in missing:
            print("  -", p)
        return

    profile = pd.read_csv(profile_path)
    bat = pd.read_csv(bat_path)
    pit = pd.read_csv(pit_path)
    teamrep = pd.read_csv(teamrep_path)

    # Only consider the 24 canon ABL teams
    team_abbrs = set(teamrep["team_abbr"].unique())

    # Build WAR per (player, team)
    bat_use = (
        bat[["ID", "TM", "WAR"]]
        .rename(columns={"TM": "team_abbr", "WAR": "bat_war"})
    )
    pit_use = (
        pit[["ID", "TM_p1", "WAR"]]
        .rename(columns={"TM_p1": "team_abbr", "WAR": "pit_war"})
    )

    bat_use = bat_use[bat_use["team_abbr"].isin(team_abbrs)]
    pit_use = pit_use[pit_use["team_abbr"].isin(team_abbrs)]

    bat_agg = bat_use.groupby(["ID", "team_abbr"], as_index=False)["bat_war"].sum()
    pit_agg = pit_use.groupby(["ID", "team_abbr"], as_index=False)["pit_war"].sum()

    war = pd.merge(bat_agg, pit_agg, on=["ID", "team_abbr"], how="outer")
    war["bat_war"] = war["bat_war"].fillna(0.0)
    war["pit_war"] = war["pit_war"].fillna(0.0)
    war["war_total"] = war["bat_war"] + war["pit_war"]

    # Attach identity + popularity
    profile_small = profile[["ID", "Name", "POS", "Nat. Pop.", "Loc. Pop."]].copy()
    df = war.merge(profile_small, on="ID", how="left")

    # Popularity scores (for deterministic sorting)
    df["loc_pop_score"] = df["Loc. Pop."].map(POP_RANK).fillna(0).astype(int)
    df["nat_pop_score"] = df["Nat. Pop."].map(POP_RANK).fillna(0).astype(int)

    # Sort within each team: local pop, national pop, WAR, then name
    df_sorted = df.sort_values(
        ["team_abbr", "loc_pop_score", "nat_pop_score", "war_total", "Name"],
        ascending=[True, False, False, False, True],
    )

    # Take the top row per team_abbr
    face = df_sorted.groupby("team_abbr", as_index=False).first()

    face["is_two_way"] = (face["bat_war"].abs() > 0.0) & (face["pit_war"].abs() > 0.0)

    out = pd.DataFrame(
        {
            "team_abbr": face["team_abbr"],
            "player_id": face["ID"].astype(int),
            "player_name": face["Name"],
            "position": face["POS"],
            "bat_war": face["bat_war"],
            "pit_war": face["pit_war"],
            "war_total": face["war_total"],
            "nat_pop": face["Nat. Pop."],
            "loc_pop": face["Loc. Pop."],
            "loc_pop_score": face["loc_pop_score"],
            "nat_pop_score": face["nat_pop_score"],
            "is_two_way": face["is_two_way"],
        }
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote face-of-franchise table to {OUT_PATH}")


if __name__ == "__main__":
    main()
