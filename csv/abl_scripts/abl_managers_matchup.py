import pandas as pd

# Constants
TEAMS_CSV = "teams.csv"
COACHES_CSV = "coaches.csv"
STAFF_CSV = "team_roster_staff.csv"

# Featured matchup: CHI at MIA
AWAY_TEAM_ID = 12  # Chicago Fire
HOME_TEAM_ID = 1   # Miami Hurricanes


def load_data():
    teams = pd.read_csv(TEAMS_CSV)
    coaches = pd.read_csv(COACHES_CSV)
    staff = pd.read_csv(STAFF_CSV)
    return teams, coaches, staff


def build_team_display(teams: pd.DataFrame) -> pd.DataFrame:
    """Add a 'team_display' column like 'CHI Fire' or fall back to name."""
    teams = teams.copy()
    abbr = teams["abbr"].fillna("").astype(str).str.strip()
    nickname = teams["nickname"].fillna("").astype(str).str.strip()
    name = teams["name"].fillna("").astype(str).str.strip()

    teams["team_display"] = (
        (abbr + " " + nickname).str.strip()
    )
    # Fallbacks if abbr/nickname are missing
    teams.loc[teams["team_display"] == "", "team_display"] = name
    teams.loc[teams["team_display"] == "", "team_display"] = (
        "Team " + teams["team_id"].astype(str)
    )
    return teams


def get_manager_name(team_id: int,
                     staff: pd.DataFrame,
                     coaches: pd.DataFrame) -> str:
    """Return the manager's full name for a given team_id, or 'Unknown manager'."""
    srow = staff.loc[staff["team_id"] == team_id]
    if srow.empty:
        return "Unknown manager"

    manager_id = srow["manager"].iloc[0]
    crow = coaches.loc[coaches["coach_id"] == manager_id]
    if crow.empty:
        return "Unknown manager"

    c = crow.iloc[0]
    first = str(c.get("first_name", "")).strip()
    last = str(c.get("last_name", "")).strip()
    full = (first + " " + last).strip()
    return full if full else "Unknown manager"


def describe_side(label: str,
                  team_id: int,
                  teams: pd.DataFrame,
                  staff: pd.DataFrame,
                  coaches: pd.DataFrame):
    trow = teams.loc[teams["team_id"] == team_id]
    if trow.empty:
        team_display = f"Team {team_id}"
    else:
        team_display = trow.iloc[0]["team_display"]

    mgr_name = get_manager_name(team_id, staff, coaches)

    print(f"{label} manager – {mgr_name} ({team_display}).")


def main():
    teams, coaches, staff = load_data()
    teams = build_team_display(teams)

    print("=== ABL Manager Matchup – CHI Fire at MIA Hurricanes ===\n")
    describe_side("AWAY", AWAY_TEAM_ID, teams, staff, coaches)
    describe_side("HOME", HOME_TEAM_ID, teams, staff, coaches)


if __name__ == "__main__":
    main()
