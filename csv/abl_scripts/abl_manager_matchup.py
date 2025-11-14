import pandas as pd

LEAGUE_ID = 200
SEASON = 1981

# Featured matchup: CHI at MIA
AWAY_TEAM_ID = 12  # Chicago Fire
HOME_TEAM_ID = 1   # Miami Hurricanes

# Canonical manager names – these are the ones that matter for ABL
MANAGER_OVERRIDES = {
    12: "Matt Mead",  # Chicago Fire
    1: "Seth Coe",    # Miami Hurricanes
}

# Canonical team display names for ABL clubs
TEAM_DISPLAY_OVERRIDES = {
    12: "Chicago Fire",
    1: "Miami Hurricanes",
}


def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(name)


def get_team_display(teams: pd.DataFrame, team_id: int) -> str:
    """
    Return a nice display name for the team.

    Priority:
      1) Hard-coded TEAM_DISPLAY_OVERRIDES (for ABL majors)
      2) name + nickname from teams.csv
      3) team_name from teams.csv
      4) nickname from teams.csv
      5) 'Team {team_id}' fallback
    """
    # 1) Hard override for our two teams (and any others you add later)
    if team_id in TEAM_DISPLAY_OVERRIDES:
        return TEAM_DISPLAY_OVERRIDES[team_id]

    df = teams.copy()

    if "league_id" in df.columns:
        df = df[df["league_id"] == LEAGUE_ID]
    if "year" in df.columns:
        df = df[df["year"] == SEASON]

    if "team_id" not in df.columns:
        return f"Team {team_id}"

    row = df.loc[df["team_id"] == team_id]
    if row.empty:
        return f"Team {team_id}"

    row = row.iloc[0]

    # 2) name + nickname, e.g. "Chicago Fire"
    name = None
    nickname = None

    if "name" in row.index:
        name = str(row["name"]).strip()
    if "nickname" in row.index:
        nickname = str(row["nickname"]).strip()

    if name and nickname:
        return f"{name} {nickname}"
    if name:
        return name
    if "team_name" in row.index:
        tn = str(row["team_name"]).strip()
        if tn:
            return tn
    if nickname:
        return nickname

    return f"Team {team_id}"


def get_manager_name(team_id: int) -> str:
    """
    Always use the canonical name from MANAGER_OVERRIDES for ABL clubs.
    """
    return MANAGER_OVERRIDES.get(team_id, "Unknown manager")


def build_manager_block(manager_name: str, team_name: str, is_home: bool) -> str:
    """
    Build the descriptive block for one manager, using your lore:
      - Both in 10th season with the club
      - Coe has 2 ABL titles
      - Shared broad tactical tendencies (big fly, long leash for starters, etc.)
    """
    if is_home:
        header = (
            f"HOME manager – {manager_name} ({team_name}), "
            "in his 10th season with the club. He’s already brought home 2 ABL championships."
        )
    else:
        header = (
            f"AWAY manager – {manager_name} ({team_name}), "
            "in his 10th season with the club."
        )

    lines = []
    lines.append(header)
    lines.append(
        "He tends to let the clubhouse police itself rather than being overly hands-on."
    )
    lines.append(
        "Prefers to sit back and wait for the big hit rather than play much small ball."
    )
    lines.append(
        "He likes to give his starters a long leash, and is patient with his bullpen once he goes to it. "
        "He is flexible at the back end and not married to a strict closer role, and tends not to chase "
        "left/right matchups too much."
    )
    return "\n".join(lines)


def main() -> None:
    # We only need teams.csv to get city + nickname if overrides don't exist
    teams = load_csv("teams.csv")

    away_team_name = get_team_display(teams, AWAY_TEAM_ID)  # 'Chicago Fire'
    home_team_name = get_team_display(teams, HOME_TEAM_ID)  # 'Miami Hurricanes'

    away_mgr_name = get_manager_name(AWAY_TEAM_ID)          # 'Matt Mead'
    home_mgr_name = get_manager_name(HOME_TEAM_ID)          # 'Seth Coe'

    print("=== ABL Manager Matchup – CHI Fire at MIA Hurricanes ===\n")

    print(build_manager_block(away_mgr_name, away_team_name, is_home=False))
    print()
    print(build_manager_block(home_mgr_name, home_team_name, is_home=True))
    print()

    print("--- Manager Matchup (Broadcast lines) ---")
    print(
        f"Manager matchup tonight: {away_mgr_name} runs the {away_team_name}, "
        f"while {home_mgr_name} leads the {home_team_name}."
    )
    print(
        f"Between them, these two skippers have 11 playoff appearances and "
        f"2 ABL titles on {home_mgr_name}'s resume with the {home_team_name}."
    )
    print(
        f"{away_team_name} under {away_mgr_name} has a very defined identity, and "
        f"{home_team_name} under {home_mgr_name} has become one of the league’s benchmark franchises."
    )


if __name__ == "__main__":
    main()
