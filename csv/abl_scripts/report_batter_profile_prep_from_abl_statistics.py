"""Build a Batter Profile Prep report from ABL Statistics bat ratings."""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from abl_config import LEAGUE_ID, TEAM_IDS, TXT_OUT_ROOT, csv_path, stamp_text_block

BAT_STATS_FILE = "../abl_statistics/abl_statistics_player_statistics_-_sortable_stats_player_bat_ratings.csv"
TEAM_MIN, TEAM_MAX = min(TEAM_IDS), max(TEAM_IDS)

REQUIRED_BAT_COLS: Sequence[str] = (
    "ID",
    "POS",
    "Name",
    "TM",
    "CON",
    "GAP",
    "POW",
    "EYE",
    "SPE",
    "BA vL",
    "BA vR",
)

OPTIONAL_BAT_COLS: Sequence[str] = ("STE", "RUN", "BUN")


def to_ascii(value: object) -> str:
    """Return ASCII-safe string for any value."""
    if value is None:
        return ""
    text = str(value)
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("ascii", "ignore").decode("ascii")


def fail(msg: str) -> None:
    print(msg, file=sys.stderr)
    sys.exit(1)


def comment_prefix_count(path: Path) -> int:
    """Count leading comment lines that start with '#'."""
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


def load_bat_stats() -> pd.DataFrame:
    stats_path = csv_path(BAT_STATS_FILE).resolve()
    if not stats_path.exists():
        fail(f"Missing bat ratings file: {stats_path}")
    skiprows = comment_prefix_count(stats_path)
    stats = pd.read_csv(stats_path, skiprows=skiprows)
    require_columns(stats, REQUIRED_BAT_COLS, "bat ratings")
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
    required = ["player_id", "team_id", "league_id", "position", "playing_level", "is_active"]
    require_columns(roster, required, "players_roster_status")
    roster["player_id"] = pd.to_numeric(roster["player_id"], errors="coerce").astype("Int64")
    roster["team_id"] = pd.to_numeric(roster["team_id"], errors="coerce").astype("Int64")
    roster["league_id"] = pd.to_numeric(roster["league_id"], errors="coerce").astype("Int64")
    roster["position"] = pd.to_numeric(roster["position"], errors="coerce").fillna(-1).astype(int)
    roster["playing_level"] = pd.to_numeric(roster["playing_level"], errors="coerce").fillna(-1).astype(int)
    roster["is_active"] = pd.to_numeric(roster["is_active"], errors="coerce").fillna(0).astype(int)
    active = roster[
        (roster["league_id"] == LEAGUE_ID)
        & (roster["team_id"] >= TEAM_MIN)
        & (roster["team_id"] <= TEAM_MAX)
        & (roster["is_active"] == 1)
        & (roster["playing_level"] == 0)
        & (roster["position"] != 1)
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
    required = ["player_id", "first_name", "last_name", "bats"]
    require_columns(players, required, "players")
    players["player_id"] = pd.to_numeric(players["player_id"], errors="coerce").astype("Int64")
    bats_map = {1: "R", 2: "L", 3: "S"}
    players["bats_hand"] = pd.to_numeric(players["bats"], errors="coerce").map(bats_map).fillna("")
    return players[["player_id", "first_name", "last_name", "bats_hand"]]


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


def rating_str(value: object) -> str:
    num = pd.to_numeric(value, errors="coerce")
    if pd.isna(num):
        return "-"
    if float(num).is_integer():
        return f"{int(num)}"
    return f"{float(num):.1f}"


def compute_overall(row: pd.Series) -> float:
    vals = []
    for col in ("CON", "GAP", "POW", "EYE"):
        num = pd.to_numeric(row.get(col), errors="coerce")
        if pd.notna(num):
            vals.append(float(num))
    if not vals:
        return float("nan")
    return round(sum(vals) / len(vals), 1)


def best_tools(row: pd.Series) -> Tuple[str, int, str, int]:
    tools = []
    for col in ("CON", "GAP", "POW", "EYE", "SPE"):
        val = pd.to_numeric(row.get(col), errors="coerce")
        if pd.isna(val):
            continue
        tools.append((col, int(val)))
    if not tools:
        return ("", 0, "", 0)
    tools.sort(key=lambda item: (-item[1], item[0]))
    best_name, best_val = tools[0]
    second_name, second_val = ("", 0)
    if len(tools) > 1:
        second_name, second_val = tools[1]
    return best_name, best_val, second_name, second_val


def sanitize_name(first: str, last: str) -> str:
    first_ascii = to_ascii(first) if first else ""
    last_ascii = to_ascii(last) if last else ""
    return f"{first_ascii} {last_ascii}".strip()


def format_batter_line(row: pd.Series, optional_cols: Sequence[str]) -> str:
    name = sanitize_name(row.get("first_name", ""), row.get("last_name", ""))
    bats = to_ascii(row.get("bats_hand", "")).upper() or "?"
    pos = to_ascii(row.get("POS", ""))
    overall = compute_overall(row)
    best_name, best_val, second_name, second_val = best_tools(row)
    extras = []
    for col in optional_cols:
        if col in row.index:
            extras.append(f"{col} {rating_str(row[col])}")
    ba_l = rating_str(row.get("BA vL"))
    ba_r = rating_str(row.get("BA vR"))
    best_part = f"Best {best_name} {best_val}" if best_name else "Best n/a"
    second_part = f"Second {second_name} {second_val}" if second_name else "Second n/a"
    extra_part = " ".join(extras)
    return (
        f"{name} ({bats}) POS {pos} "
        f"CON {rating_str(row.get('CON'))} GAP {rating_str(row.get('GAP'))} "
        f"POW {rating_str(row.get('POW'))} EYE {rating_str(row.get('EYE'))} "
        f"SPE {rating_str(row.get('SPE'))} "
        f"OverallBat {rating_str(overall)} | {best_part} | {second_part} | "
        f"BA vL {ba_l} BA vR {ba_r}"
        + (f" | {extra_part}" if extra_part else "")
    )


def format_missing_batter(row: pd.Series) -> str:
    name = sanitize_name(row.get("first_name", ""), row.get("last_name", ""))
    return f"{name} player_id={int(row.player_id)}"


def build_team_section(
    team_id: int,
    team_abbr: str,
    merged: pd.DataFrame,
    stats_ids: set[int],
    limit: int | None,
    optional_cols: Sequence[str],
) -> str:
    team_rows = merged[merged["team_id"] == team_id].copy()
    missing = team_rows[~team_rows["player_id"].isin(stats_ids)]
    available = team_rows[team_rows["player_id"].isin(stats_ids)].copy()
    available["OverallBat"] = available.apply(compute_overall, axis=1)
    available["POW_sort"] = pd.to_numeric(available["POW"], errors="coerce").fillna(-1)
    available["name_sort"] = available["last_name"].fillna("").apply(to_ascii) + " " + available["first_name"].fillna("").apply(to_ascii)
    available = available.sort_values(
        by=["POS", "OverallBat", "POW_sort", "name_sort"],
        ascending=[True, False, False, True],
    )
    pos_order: List[str] = []
    if not available.empty:
        pos_order = sorted(set(available["POS"].apply(to_ascii)))
    lines: List[str] = [f"TEAM {team_abbr} - Active Batters"]
    # Iterate positions present; if none, still handle missing block.
    emitted = 0
    for pos in pos_order:
        if limit and limit > 0 and emitted >= limit:
            break
        pos_df = available[available["POS"].apply(to_ascii) == pos]
        if pos_df.empty:
            continue
        lines.append(f"{pos}:")
        for row in pos_df.itertuples(index=False):
            if limit and limit > 0 and emitted >= limit:
                break
            lines.append(format_batter_line(pd.Series(row._asdict()), optional_cols))
            emitted += 1
        lines.append("")
    if not missing.empty:
        lines.append("MISSING STATS ROW (active but not found in abl_statistics):")
        for row in missing.itertuples(index=False):
            lines.append(format_missing_batter(pd.Series(row._asdict())))
    return "\n".join(lines).rstrip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Batter Profile Prep report from ABL Statistics.")
    parser.add_argument("--team", help="Single team abbreviation (e.g., CHI)")
    parser.add_argument("--away", help="Away team abbreviation for matchup mode")
    parser.add_argument("--home", help="Home team abbreviation for matchup mode")
    parser.add_argument("--all", action="store_true", help="Output league-wide master list of all active ABL batters")
    parser.add_argument("--limit", type=int, help="Limit batters per team")
    parser.add_argument("--out", dest="out_path", help="Override output path")
    return parser.parse_args()


def resolve_selection(
    args: argparse.Namespace,
    abbr_to_id: Dict[str, int],
    team_city_map: Dict[int, str] | None = None,
) -> Tuple[List[int], List[str]]:
    has_team = bool(args.team)
    has_matchup = bool(args.away or args.home)
    has_all = bool(args.all)
    if sum([has_team, has_matchup, has_all]) > 1:
        fail("Use only one selection mode: --team OR --away/--home OR --all.")
    if not (has_team or has_matchup or has_all):
        fail("Provide --team XYZ, --away XYZ --home ABC, or --all.")
    if has_all:
        def sort_key(item: tuple[str, int]) -> tuple[str, str]:
            abbr, team_id = item
            city = team_city_map.get(team_id, "") if team_city_map else ""
            return (to_ascii(city).upper(), abbr)

        sorted_items = sorted(abbr_to_id.items(), key=sort_key)
        ids = [tid for _, tid in sorted_items]
        abbrs = [abbr for abbr, _ in sorted_items]
        return ids, abbrs
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
    stats = load_bat_stats()

    team_city_map: Dict[int, str] = {}
    for row in teams.itertuples():
        if pd.isna(row.team_id):
            continue
        city_val = getattr(row, "name", "")
        team_city_map[int(row.team_id)] = to_ascii(city_val).strip()

    abbr_to_id, id_to_abbr = build_team_maps(teams)
    team_ids, abbrs = resolve_selection(args, abbr_to_id, team_city_map)

    merged = roster.merge(players, on="player_id", how="left")
    merged = merged.merge(stats, on="player_id", how="left", suffixes=("", "_stats"))

    stats_ids = set(int(pid) for pid in stats["player_id"].dropna().astype(int).tolist())

    optional_cols_present = [col for col in OPTIONAL_BAT_COLS if col in stats.columns]

    sections = []
    for team_id in team_ids:
        team_abbr = id_to_abbr.get(team_id, f"TEAM_{team_id}")
        sections.append(
            build_team_section(team_id, team_abbr, merged, stats_ids, args.limit, optional_cols_present)
        )

    if args.all:
        title = "Batter Profile Prep - ALL TEAMS"
        default_out = TXT_OUT_ROOT / "prep" / "batter_profile_all.txt"
    elif len(team_ids) == 1:
        title = f"Batter Profile Prep - {abbrs[0]}"
        default_out = TXT_OUT_ROOT / "prep" / f"batter_profile_{abbrs[0]}.txt"
    else:
        title = f"Batter Profile Prep - {abbrs[0]} at {abbrs[1]}"
        default_out = TXT_OUT_ROOT / "prep" / f"batter_profile_{abbrs[0]}_at_{abbrs[1]}.txt"

    full_text = title + "\n\n" + "\n\n\n".join(sections)
    stamped = stamp_text_block(full_text)

    out_path = Path(args.out_path) if args.out_path else default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(stamped, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

# Self-test commands:
# python csv/abl_scripts/report_batter_profile_prep_from_abl_statistics.py --team CHI
# python csv/abl_scripts/report_batter_profile_prep_from_abl_statistics.py --away CHI --home MIA
