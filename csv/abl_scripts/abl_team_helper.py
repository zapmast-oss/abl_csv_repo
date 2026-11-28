import pandas as pd
from pathlib import Path

# Path to OOTP teams.csv
TEAMS_CSV = Path("csv/ootp_csv/teams.csv")

def load_abl_teams():
    """
    Load only the 24 official ABL teams (league_id = 200, team_id 1Ã¢â‚¬â€œ24)
    and return a clean DataFrame with standardized names.
    """
    df = pd.read_csv(TEAMS_CSV)

    # Core ABL filter
    df = df[(df["league_id"] == 200) & (df["team_id"].between(1, 24))]

    # Clean team names (strip anything like ' (PIT)')
    df["name"] = df["name"].str.replace(r"\s*\(.*\)$", "", regex=True).str.strip()

    return df[["team_id", "name", "abbr"]].copy()

def allowed_team_ids():
    """Return a Python set of allowed team_ids."""
    return set(load_abl_teams()["team_id"].tolist())

def allowed_team_names():
    """Return a Python set of clean team names."""
    return set(load_abl_teams()["name"].tolist())

def is_abl_team(team_id: int) -> bool:
    """Check if a team_id is one of the 24 core ABL teams."""
    return team_id in allowed_team_ids()

def get_team_by_name(name: str):
    """Return a row (Series) for a team by name, or None if not found."""
    df = load_abl_teams()
    df_match = df[df["name"].str.lower() == name.lower()]
    return df_match.iloc[0].to_dict() if not df_match.empty else None

def get_team_by_abbr(abbr: str):
    """Return a row (Series) for a team by abbriation (CHI, MIA, etc.)."""
    df = load_abl_teams()
    df_match = df[df["abbr"].str.lower() == abbr.lower()]
    return df_match.iloc[0].to_dict() if not df_match.empty else None
