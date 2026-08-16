import pandas as pd
from pathlib import Path

from abl_team_helper import allowed_team_ids

CSV_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = CSV_ROOT / "ootp_csv"

LEAGUE_ID = 200
SEASON = 1981

# Featured matchup: CHI at MIA
AWAY_TEAM_ID = 12  # Chicago Fire
HOME_TEAM_ID = 1   # Miami Hurricanes

# Canonical manager names for our featured clubs
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
    path = DATA_ROOT / name
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    return pd.read_csv(path)


def get_team_display(teams: pd.DataFrame, team_id: int) -> str:
    """Return a nice display name for the team."""
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

    name = str(row.get("name", "")).strip()
    nickname = str(row.get("nickname", "")).strip()

    if name and nickname:
        return f"{name} {nickname}"
    if name:
        return name
    team_name = str(row.get("team_name", "")).strip()
    if team_name:
        return team_name
    if nickname:
        return nickname

    return f"Team {team_id}"


def get_manager_name(team_id: int) -> str:
    """Always use the canonical name from MANAGER_OVERRIDES for ABL clubs."""
    return MANAGER_OVERRIDES.get(team_id, "Unknown manager")


def build_manager_block(manager_name: str, team_name: str, is_home: bool) -> str:
    """Build a descriptive summary for one manager."""
    if is_home:
        header = (
            f"HOME manager - {manager_name} ({team_name}), in his 10th season with the club. "
            "He has already brought home 2 ABL championships."
        )
    else:
        header = (
            f"AWAY manager - {manager_name} ({team_name}), in his 10th season with the club."
        )

    lines = [header]
    lines.append("He tends to let the clubhouse police itself rather than being overly hands-on.")
    lines.append("Prefers to sit back and wait for the big hit rather than play much small ball.")
    lines.append(
        "He likes to give his starters a long leash, and is patient with his bullpen once he goes to it. "
        "He is flexible at the back end and not married to a strict closer role, and tends not to chase "
        "left/right matchups too much."
    )
    return "\n".join(lines)


def main() -> None:
    teams = load_csv("teams.csv")
    allowed_ids = set(allowed_team_ids())
    teams = teams[teams["team_id"].isin(allowed_ids)].copy()
    if "league_id" in teams.columns:
        league_ids = {int(x) for x in teams["league_id"].unique()}
        print(f"[check] Manager matchup league_ids after filter: {league_ids}")

    away_team_name = get_team_display(teams, AWAY_TEAM_ID)
    home_team_name = get_team_display(teams, HOME_TEAM_ID)

    away_mgr_name = get_manager_name(AWAY_TEAM_ID)
    home_mgr_name = get_manager_name(HOME_TEAM_ID)

    print("=== ABL Manager Matchup - CHI Fire at MIA Hurricanes ===\n")

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
        f"{home_team_name} under {home_mgr_name} has become one of the league's benchmark franchises."
    )


if __name__ == "__main__":
    main()
