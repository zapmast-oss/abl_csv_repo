"""Temporary mapping of 24 ABL team codes to IDs/names; hand-edit later with real values."""

from abl_config import TEAM_IDS

TEAM_CODES = {
    f"T{team_id:02d}": {"id": team_id, "name": f"Team {team_id}"}
    for team_id in TEAM_IDS
}


def get_team_by_code(code: str) -> dict:
    """Return a dict with keys 'id' and 'name' for the requested 3-letter team code."""
    normalized = (code or "").strip().upper()
    if not normalized:
        raise KeyError("Team code is empty")
    if normalized not in TEAM_CODES:
        raise KeyError(f"Unknown team code: {code}")
    return TEAM_CODES[normalized]


def list_team_codes() -> list[str]:
    """Return a sorted list of all known 3-letter team codes."""
    return sorted(TEAM_CODES.keys())
