"""Print ABL standings for the 24 real teams (IDs 1–24) from the OOTP exports."""

import sys

import pandas as pd

from abl_config import TEAM_IDS
from abl_loader import read_league_csv


def pick_column(df: pd.DataFrame, candidates: list[str]) -> str:
    """Return the first existing column name from candidates (case-insensitive)."""
    lowered = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        key = candidate.lower()
        if key in lowered:
            return lowered[key]
    raise KeyError(f"No columns found among: {candidates}")


def load_team_records() -> pd.DataFrame:
    """Load the ABL team records and keep only team IDs 1–24."""
    candidates = ["team_record.csv", "team_records.csv"]
    last_error = None
    for name in candidates:
        try:
            df = read_league_csv(name)
            break
        except FileNotFoundError as exc:
            last_error = exc
    else:
        attempted = ", ".join(candidates)
        raise FileNotFoundError(f"Could not find any standings CSV ({attempted})") from last_error

    team_id_col = pick_column(df, ["team_id", "teamid", "team"])
    filtered = df[df[team_id_col].isin(TEAM_IDS)].copy()

    if "team_display" not in filtered.columns:
        lookup = build_team_display_lookup()
        filtered["team_display"] = filtered[team_id_col].map(lookup)
        filtered["team_display"] = filtered["team_display"].fillna(
            filtered[team_id_col].apply(lambda tid: f"Team {tid}")
        )

    return filtered


def build_team_display_lookup() -> dict[int, str]:
    """Return a {team_id: 'City Nickname'} mapping using teams.csv."""
    teams = read_league_csv("teams.csv")
    id_col = pick_column(teams, ["team_id", "teamid", "id", "team"])

    def optional_column(candidates: list[str]) -> str | None:
        try:
            return pick_column(teams, candidates)
        except KeyError:
            return None

    city_col = optional_column(["team_display", "name", "city_name", "city"])
    nickname_col = optional_column(["nickname", "team_nickname", "nick"])

    lookup: dict[int, str] = {}
    for _, row in teams.iterrows():
        try:
            tid = int(row[id_col])
        except (TypeError, ValueError):
            continue
        city = str(row[city_col]).strip() if city_col else ""
        nickname = str(row[nickname_col]).strip() if nickname_col else ""
        if city and nickname and nickname.lower() not in city.lower():
            display = f"{city} {nickname}".strip()
        else:
            display = city or nickname or f"Team {tid}"
        lookup[tid] = display
    return lookup


def build_standings(df: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy standings DataFrame for the 24 ABL teams."""
    team_col = pick_column(df, ["team_display", "team_name", "name"])
    wins_col = pick_column(df, ["w", "wins"])
    losses_col = pick_column(df, ["l", "losses"])
    pct_col = pick_column(df, ["pct", "win_pct", "winning_pct"])
    gb_col = None
    try:
        gb_col = pick_column(df, ["gb", "games_back"])
    except KeyError:
        gb_col = None

    def optional_column(candidates: list[str]) -> str | None:
        try:
            return pick_column(df, candidates)
        except KeyError:
            return None

    rs_col = optional_column(["r", "rs", "runs_scored"])
    ra_col = optional_column(["ra", "runs_allowed", "r_opp"])
    last10_col = optional_column(["last10", "last_10", "l10"])

    data = {
        "Team": df[team_col],
        "W": df[wins_col],
        "L": df[losses_col],
        "PCT": df[pct_col],
        "GB": df[gb_col] if gb_col else 0.0,
    }

    if rs_col is not None:
        data["RS"] = df[rs_col]
    if ra_col is not None:
        data["RA"] = df[ra_col]
    if rs_col is not None and ra_col is not None:
        data["Diff"] = df[rs_col] - df[ra_col]
    if last10_col is not None:
        data["Last10"] = df[last10_col]

    out = pd.DataFrame(data)
    out = out.sort_values(["PCT", "Team"], ascending=[False, True])
    return out


def main() -> None:
    try:
        df = load_team_records()
        standings = build_standings(df)
        print("=== Action Baseball League Standings ===")
        print(standings.to_string(index=False))
    except (FileNotFoundError, KeyError) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
