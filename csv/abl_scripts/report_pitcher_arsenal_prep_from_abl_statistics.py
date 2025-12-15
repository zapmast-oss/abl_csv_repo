"""Build a Pitcher Arsenal Prep report from ABL Statistics pitch ratings."""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from abl_config import LEAGUE_ID, TEAM_IDS, TXT_OUT_ROOT, csv_path, stamp_text_block

STATS_FILE = "../abl_statistics/abl_statistics_player_statistics_-_sortable_stats_player_pitch_ratings.csv"
TEAM_MIN, TEAM_MAX = min(TEAM_IDS), max(TEAM_IDS)

ROLE_MAP = {11: "SP", 12: "RP", 13: "CL"}
ROLE_ORDER = {"SP": 0, "RP": 1, "CL": 2}

PITCH_COLUMNS: Sequence[str] = (
    "FB",
    "CH",
    "CB",
    "SL",
    "SI",
    "SP",
    "CT",
    "FO",
    "CC",
    "SC",
    "KC",
    "KN",
)

PITCH_NAMES: Dict[str, str] = {
    "FB": "Fastball",
    "CH": "Changeup",
    "CB": "Curveball",
    "SL": "Slider",
    "SI": "Sinker",
    "SP": "Splitter",
    "CT": "Cutter",
    "FO": "Forkball",
    "CC": "Circle Change",
    "SC": "Screwball",
    "KC": "Knuckle Curve",
    "KN": "Knuckleball",
}


def to_ascii(value: object) -> str:
    """Coerce any value to ASCII-only string."""
    if value is None:
        return ""
    text = str(value)
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("ascii", "ignore").decode("ascii")


def fail(msg: str) -> None:
    print(msg, file=sys.stderr)
    sys.exit(1)


def comment_prefix_count(path: Path) -> int:
    """Count leading comment lines that start with '#' so we can skip them."""
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                count += 1
                continue
            break
    return count


def require_columns(df: pd.DataFrame, cols: Iterable[str], label: str) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        fail(f"{label} missing columns: {', '.join(missing)}")


def load_stats() -> pd.DataFrame:
    stats_path = csv_path(STATS_FILE).resolve()
    if not stats_path.exists():
        fail(f"Missing stats file: {stats_path}")
    skiprows = comment_prefix_count(stats_path)
    stats = pd.read_csv(stats_path, skiprows=skiprows)
    required = [
        "ID",
        "TM",
        "STU",
        "MOV",
        "CON",
        "STM",
        "HLD",
        "VELO",
        "VT",
        "PIT",
        *PITCH_COLUMNS,
    ]
    require_columns(stats, required, "abl_statistics pitch ratings")
    stats = stats.rename(columns={"ID": "player_id"})
    stats["player_id"] = pd.to_numeric(stats["player_id"], errors="coerce").astype("Int64")
    stats = stats.dropna(subset=["player_id"])
    stats = stats.drop_duplicates(subset=["player_id"], keep="first")
    return stats


def load_roster() -> pd.DataFrame:
    roster_path = csv_path("players_roster_status.csv")
    if not roster_path.exists():
        fail(f"Missing roster_status file: {roster_path}")
    roster = pd.read_csv(roster_path)
    required = ["player_id", "team_id", "league_id", "is_active", "playing_level", "position", "role"]
    require_columns(roster, required, "players_roster_status")
    roster["player_id"] = pd.to_numeric(roster["player_id"], errors="coerce").astype("Int64")
    roster["team_id"] = pd.to_numeric(roster["team_id"], errors="coerce").astype("Int64")
    roster["league_id"] = pd.to_numeric(roster["league_id"], errors="coerce").astype("Int64")
    roster["is_active"] = pd.to_numeric(roster["is_active"], errors="coerce").fillna(0).astype(int)
    roster["playing_level"] = pd.to_numeric(roster["playing_level"], errors="coerce").fillna(-1).astype(int)
    roster["position"] = pd.to_numeric(roster["position"], errors="coerce").fillna(-1).astype(int)
    roster["role"] = pd.to_numeric(roster["role"], errors="coerce").astype("Int64")
    active = roster[
        (roster["league_id"] == LEAGUE_ID)
        & (roster["team_id"] >= TEAM_MIN)
        & (roster["team_id"] <= TEAM_MAX)
        & (roster["is_active"] == 1)
        & (roster["playing_level"] == 0)
        & (roster["position"] == 1)
    ].copy()
    if not active.empty:
        bad_league = active[active["league_id"] != LEAGUE_ID]
        bad_team = active[~active["team_id"].between(TEAM_MIN, TEAM_MAX)]
        if not bad_league.empty or not bad_team.empty:
            fail("Filtered roster_status rows violate league_id/team_id guardrails.")
    return active


def load_players() -> pd.DataFrame:
    players_path = csv_path("players.csv")
    if not players_path.exists():
        fail(f"Missing players file: {players_path}")
    players = pd.read_csv(players_path)
    required = ["player_id", "first_name", "last_name", "throws"]
    require_columns(players, required, "players")
    players["player_id"] = pd.to_numeric(players["player_id"], errors="coerce").astype("Int64")
    throws_map = {1: "R", 2: "L", 3: "S"}
    players["throws_hand"] = pd.to_numeric(players["throws"], errors="coerce").map(throws_map).fillna("")
    return players[["player_id", "first_name", "last_name", "throws_hand"]]


def load_teams() -> pd.DataFrame:
    teams_path = csv_path("teams.csv")
    if not teams_path.exists():
        fail(f"Missing teams file: {teams_path}")
    teams = pd.read_csv(teams_path)
    required = ["team_id", "abbr", "league_id"]
    require_columns(teams, required, "teams")
    teams["team_id"] = pd.to_numeric(teams["team_id"], errors="coerce").astype("Int64")
    teams["league_id"] = pd.to_numeric(teams["league_id"], errors="coerce").astype("Int64")
    return teams


def normalize_abbr(abbr: str) -> str:
    return to_ascii(abbr).strip().upper()


def build_team_maps(teams: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, str]]:
    eligible = teams[
        (teams["league_id"] == LEAGUE_ID)
        & (teams["team_id"] >= TEAM_MIN)
        & (teams["team_id"] <= TEAM_MAX)
        & teams["abbr"].notna()
    ].copy()
    eligible["abbr_norm"] = eligible["abbr"].apply(normalize_abbr)
    abbr_to_id = {row.abbr_norm: int(row.team_id) for row in eligible.itertuples()}
    id_to_abbr = {int(row.team_id): row.abbr_norm for row in eligible.itertuples()}
    return abbr_to_id, id_to_abbr


def role_label(value: object) -> str:
    try:
        key = int(value)
    except (TypeError, ValueError):
        return "ROLE_?"
    return ROLE_MAP.get(key, f"ROLE_{key}")


def rating_str(value: object) -> str:
    num = pd.to_numeric(value, errors="coerce")
    if pd.isna(num):
        return "-"
    return f"{int(num)}"


def format_pitch_block(row: pd.Series) -> Tuple[int, str, str]:
    pitch_pairs: List[Tuple[str, int]] = []
    for code in PITCH_COLUMNS:
        raw = row.get(code)
        val = pd.to_numeric(raw, errors="coerce")
        if pd.isna(val):
            continue
        rating = int(val)
        if rating <= 0:
            continue
        pitch_pairs.append((code, rating))
    pitch_pairs.sort(key=lambda item: (-item[1], item[0]))
    pit_val = pd.to_numeric(row.get("PIT"), errors="coerce")
    pitch_count = int(pit_val) if pd.notna(pit_val) else len(pitch_pairs)
    if not pitch_pairs:
        return pitch_count, "Best: none", "Arsenal: none listed"
    best_code, best_rating = pitch_pairs[0]
    best = f"Best {PITCH_NAMES[best_code]} {best_rating}"
    arsenal = ", ".join(f"{PITCH_NAMES[code]} {rating}" for code, rating in pitch_pairs)
    return pitch_count, best, f"Arsenal: {arsenal}"


def sanitize_name(first: str, last: str) -> str:
    first_ascii = to_ascii(first) if first else ""
    last_ascii = to_ascii(last) if last else ""
    return f"{first_ascii} {last_ascii}".strip()


def format_pitcher_lines(row: pd.Series) -> List[str]:
    name = sanitize_name(row.get("first_name", ""), row.get("last_name", ""))
    throws = to_ascii(row.get("throws_hand", "")).upper() or "?"
    pitch_count, best_pitch, arsenal = format_pitch_block(row)
    line_one = (
        f"{name} ({throws}) "
        f"STU {rating_str(row.get('STU'))} "
        f"MOV {rating_str(row.get('MOV'))} "
        f"CON {rating_str(row.get('CON'))} "
        f"STM {rating_str(row.get('STM'))} "
        f"HLD {rating_str(row.get('HLD'))} "
        f"VELO {to_ascii(row.get('VELO')) or '-'} "
        f"VT {to_ascii(row.get('VT')) or '-'} "
        f"| Pitches {pitch_count} | {best_pitch}"
    )
    return [line_one, arsenal]


def format_missing_pitcher(row: pd.Series) -> str:
    name = sanitize_name(row.get("first_name", ""), row.get("last_name", ""))
    hand = to_ascii(row.get("throws_hand", "")).upper() or "?"
    return f"{name} ({hand}) player_id={int(row.player_id)} [missing stats]"


def build_team_section(
    team_id: int,
    team_abbr: str,
    merged: pd.DataFrame,
    stats_ids: set[int],
    limit: int | None,
) -> str:
    team_rows = merged[merged["team_id"] == team_id].copy()
    missing = team_rows[~team_rows["player_id"].isin(stats_ids)].copy()
    available = team_rows[team_rows["player_id"].isin(stats_ids)].copy()
    available["role_label"] = available["role"].apply(role_label)
    available["role_order"] = available["role_label"].apply(lambda r: ROLE_ORDER.get(r, 3))
    available["STU_sort"] = pd.to_numeric(available["STU"], errors="coerce").fillna(-1)
    available["CON_sort"] = pd.to_numeric(available["CON"], errors="coerce").fillna(-1)
    available["last_sort"] = available["last_name"].fillna("").apply(to_ascii)
    available["first_sort"] = available["first_name"].fillna("").apply(to_ascii)
    available = available.sort_values(
        by=["role_order", "STU_sort", "CON_sort", "last_sort", "first_sort"],
        ascending=[True, False, False, True, True],
    )
    if limit and limit > 0:
        available = available.head(limit)

    if not missing.empty:
        missing["role_label"] = missing["role"].apply(role_label)

    lines: List[str] = [f"TEAM {team_abbr} - Active Pitchers"]
    grouped_roles = ["SP", "RP", "CL"]
    seen_roles = set(grouped_roles)
    for role in grouped_roles:
        lines.append(f"{role}:")
        role_df = available[available["role_label"] == role]
        role_missing = missing[missing["role_label"] == role] if not missing.empty else pd.DataFrame()
        if role_df.empty and role_missing.empty:
            lines.append("(none)")
            lines.append("")
            continue
        for row in role_df.itertuples(index=False):
            lines.extend(format_pitcher_lines(pd.Series(row._asdict())))
            lines.append("")
        for row in role_missing.itertuples(index=False):
            lines.append(format_missing_pitcher(pd.Series(row._asdict())))
            lines.append("")
    other_df = available[~available["role_label"].isin(seen_roles)]
    other_missing = missing[~missing["role_label"].isin(seen_roles)] if not missing.empty else pd.DataFrame()
    if not other_df.empty:
        for role_value, grp in other_df.groupby("role_label", sort=True):
            lines.append(f"{role_value}:")
            for row in grp.itertuples(index=False):
                lines.extend(format_pitcher_lines(pd.Series(row._asdict())))
                lines.append("")
            missing_group = other_missing[other_missing["role_label"] == role_value]
            for row in missing_group.itertuples(index=False):
                lines.append(format_missing_pitcher(pd.Series(row._asdict())))
                lines.append("")
    elif not other_missing.empty:
        for role_value, grp in other_missing.groupby("role_label", sort=True):
            lines.append(f"{role_value}:")
            for row in grp.itertuples(index=False):
                lines.append(format_missing_pitcher(pd.Series(row._asdict())))
                lines.append("")
    if not missing.empty:
        missing_block = missing[missing["role_label"] != "SP"]
    else:
        missing_block = missing
    if not missing_block.empty:
        lines.append("MISSING STATS ROW (active but not found in abl_statistics):")
        for row in missing_block.itertuples(index=False):
            lines.append(format_missing_pitcher(pd.Series(row._asdict())))
    return "\n".join(lines).rstrip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Pitcher Arsenal Prep report from ABL Statistics.")
    parser.add_argument("--team", help="Single team abbreviation (e.g., CHI)")
    parser.add_argument("--away", help="Away team abbreviation for matchup mode")
    parser.add_argument("--home", help="Home team abbreviation for matchup mode")
    parser.add_argument("--limit", type=int, help="Limit pitchers per team")
    parser.add_argument("--out", dest="out_path", help="Override output path")
    return parser.parse_args()


def resolve_selection(args: argparse.Namespace, abbr_to_id: Dict[str, int]) -> Tuple[List[int], List[str]]:
    has_team = bool(args.team)
    has_matchup = bool(args.away or args.home)
    if has_team and has_matchup:
        fail("Use either --team or --away/--home, not both.")
    if not has_team and not has_matchup:
        fail("Provide --team XYZ or --away XYZ --home ABC.")
    if has_team:
        abbr = normalize_abbr(args.team)
        if abbr not in abbr_to_id:
            fail(f"Unknown team abbr: {abbr}")
        return [abbr_to_id[abbr]], [abbr]
    away = normalize_abbr(args.away or "")
    home = normalize_abbr(args.home or "")
    if not away or not home:
        fail("Both --away and --home are required for matchup mode.")
    if away == home:
        fail("Away and home teams must be different.")
    missing: List[str] = [abbr for abbr in (away, home) if abbr not in abbr_to_id]
    if missing:
        fail(f"Unknown team abbr(s): {', '.join(missing)}")
    return [abbr_to_id[away], abbr_to_id[home]], [away, home]


def main() -> None:
    args = parse_args()
    roster = load_roster()
    players = load_players()
    teams = load_teams()
    stats = load_stats()

    abbr_to_id, id_to_abbr = build_team_maps(teams)
    team_ids, abbrs = resolve_selection(args, abbr_to_id)

    merged = roster.merge(players, on="player_id", how="left")
    merged = merged.merge(stats, on="player_id", how="left", suffixes=("", "_stats"))

    stats_ids = set(int(pid) for pid in stats["player_id"].dropna().astype(int).tolist())

    sections = []
    for team_id in team_ids:
        team_abbr = id_to_abbr.get(team_id, f"TEAM_{team_id}")
        sections.append(build_team_section(team_id, team_abbr, merged, stats_ids, args.limit))

    if len(team_ids) == 1:
        title = f"Pitcher Arsenal Prep - {abbrs[0]}"
        default_out = TXT_OUT_ROOT / "prep" / f"pitcher_arsenal_{abbrs[0]}.txt"
    else:
        title = f"Pitcher Arsenal Prep - {abbrs[0]} at {abbrs[1]}"
        default_out = TXT_OUT_ROOT / "prep" / f"pitcher_arsenal_{abbrs[0]}_at_{abbrs[1]}.txt"

    full_text = title + "\n\n" + "\n\n\n".join(sections)
    stamped = stamp_text_block(full_text)

    out_path = Path(args.out_path) if args.out_path else default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(stamped, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

# Self-test commands:
# python csv/abl_scripts/report_pitcher_arsenal_prep_from_abl_statistics.py --team CHI
# python csv/abl_scripts/report_pitcher_arsenal_prep_from_abl_statistics.py --away CHI --home MIA
