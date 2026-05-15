from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


LEAGUE_ID = "200"
SEASON = "1981"

SCRIPT_PATH = Path(__file__).resolve()
CSV_ROOT = SCRIPT_PATH.parents[1]
RAW_DIR = CSV_ROOT / "ootp_csv"
OUT_DIR = CSV_ROOT / "out"
OUT_PATH = OUT_DIR / "abl_1981_asb_player_pack.md"

REQUIRED_FILES = [
    "players.csv",
    "teams.csv",
    "team_roster.csv",
    "cities.csv",
    "states.csv",
    "nations.csv",
    "players_career_batting_stats.csv",
    "players_career_pitching_stats.csv",
    "players_batting.csv",
    "players_pitching.csv",
]

OPTIONAL_FILES = [
    "players_value.csv",
    "players_roster_status.csv",
    "players_injury_history.csv",
    "players_league_leader.csv",
]

CONTENDER_ORDER = [
    "CHA",
    "DAL",
    "SF",
    "CHI",
    "DET",
    "MIA",
    "PHO",
    "LA",
    "LV",
    "HOU",
    "BOS",
    "SEA",
    "DEN",
    "PHI",
    "NY",
    "CIN",
]


@dataclass
class Team:
    team_id: str
    abbr: str
    name: str


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def present_file(name: str) -> bool:
    return (RAW_DIR / name).exists()


def to_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, "", "NULL"):
            return default
        return int(float(str(value)))
    except ValueError:
        return default


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, "", "NULL"):
            return default
        return float(str(value))
    except ValueError:
        return default


def safe_div(num: float, den: float) -> float | None:
    if den == 0:
        return None
    return num / den


def fmt_num(value: float | None, digits: int = 1) -> str:
    if value is None or math.isnan(value):
        return "Not available"
    return f"{value:.{digits}f}"


def fmt_pct(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "Not available"
    return f"{value:.3f}".replace("0.", ".")


def fmt_ip(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "Not available"
    outs = int(round(value * 3))
    return f"{outs // 3}.{outs % 3}"


def handedness(value: str) -> str:
    return {"1": "R", "2": "L", "3": "S"}.get(str(value), "Not available")


def player_name(player: dict[str, str]) -> str:
    first = player.get("first_name", "").strip()
    last = player.get("last_name", "").strip()
    return f"{first} {last}".strip() or f"Player {player.get('player_id', '')}".strip()


def total_bases(row: dict[str, Any]) -> int:
    hits = to_int(row.get("h"))
    doubles = to_int(row.get("d"))
    triples = to_int(row.get("t"))
    homers = to_int(row.get("hr"))
    singles = hits - doubles - triples - homers
    return singles + (2 * doubles) + (3 * triples) + (4 * homers)


def batting_rates(row: dict[str, Any]) -> dict[str, float | None]:
    ab = to_int(row.get("ab"))
    hits = to_int(row.get("h"))
    bb = to_int(row.get("bb"))
    hp = to_int(row.get("hp"))
    sf = to_int(row.get("sf"))
    obp_den = ab + bb + hp + sf
    avg = safe_div(hits, ab)
    obp = safe_div(hits + bb + hp, obp_den)
    slg = safe_div(total_bases(row), ab)
    ops = (obp + slg) if obp is not None and slg is not None else None
    return {"avg": avg, "obp": obp, "slg": slg, "ops": ops}


def pitcher_ip(row: dict[str, Any]) -> float:
    outs = to_int(row.get("outs"))
    if outs:
        return outs / 3
    return to_float(row.get("ip")) + (to_float(row.get("ipf")) / 3)


def pitching_rates(row: dict[str, Any]) -> dict[str, float | None]:
    ip = pitcher_ip(row)
    er = to_int(row.get("er"))
    hits = to_int(row.get("ha"))
    bb = to_int(row.get("bb"))
    era = safe_div(er * 9, ip)
    whip = safe_div(hits + bb, ip)
    return {"ip": ip, "era": era, "whip": whip}


def load_teams(missing: list[str]) -> tuple[dict[str, Team], dict[str, Team]]:
    if not present_file("teams.csv"):
        missing.append("teams.csv")
        return {}, {}
    teams_by_id: dict[str, Team] = {}
    teams_by_abbr: dict[str, Team] = {}
    for row in read_csv(RAW_DIR / "teams.csv"):
        if row.get("league_id") != LEAGUE_ID or row.get("allstar_team") == "1":
            continue
        team = Team(row["team_id"], row["abbr"], f"{row['name']} {row['nickname']}".strip())
        teams_by_id[team.team_id] = team
        teams_by_abbr[team.abbr] = team
    return teams_by_id, teams_by_abbr


def load_geo() -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    cities = {row["city_id"]: row for row in read_csv(RAW_DIR / "cities.csv")} if present_file("cities.csv") else {}
    states = {row["state_id"]: row for row in read_csv(RAW_DIR / "states.csv")} if present_file("states.csv") else {}
    nations = {row["nation_id"]: row for row in read_csv(RAW_DIR / "nations.csv")} if present_file("nations.csv") else {}
    return cities, states, nations


def birthplace(player: dict[str, str], cities: dict[str, dict[str, str]], states: dict[str, dict[str, str]], nations: dict[str, dict[str, str]]) -> str:
    city = cities.get(player.get("city_of_birth_id", ""))
    nation = nations.get(player.get("nation_id", ""))
    if not city and not nation:
        return "Not available"
    parts: list[str] = []
    if city:
        parts.append(city.get("name", ""))
        state = states.get(city.get("state_id", ""))
        if state and state.get("abbreviation"):
            parts.append(state["abbreviation"])
    if nation:
        parts.append(nation.get("name", ""))
    return ", ".join(part for part in parts if part) or "Not available"


def school_context(player: dict[str, str]) -> str:
    college = player.get("college", "")
    school = player.get("school", "")
    values = []
    if college and college != "0":
        values.append(f"college id {college}")
    if school and school != "0":
        values.append(f"school id {school}")
    return ", ".join(values) if values else "Not available"


def current_export_timestamp(existing_files: list[str]) -> str:
    times = [(RAW_DIR / name).stat().st_mtime for name in existing_files if (RAW_DIR / name).exists()]
    if not times:
        return "Not available"
    return datetime.fromtimestamp(max(times)).strftime("%Y-%m-%d %H:%M:%S")


def filter_stat_rows(filename: str, missing: list[str]) -> list[dict[str, str]]:
    if not present_file(filename):
        missing.append(filename)
        return []
    rows = []
    for row in read_csv(RAW_DIR / filename):
        if (
            row.get("year") == SEASON
            and row.get("league_id") == LEAGUE_ID
            and row.get("level_id") == "1"
            and row.get("split_id") == "1"
            and row.get("game_id") == "0"
        ):
            rows.append(row)
    return rows


def load_rating_map(filename: str, missing: list[str]) -> dict[str, dict[str, str]]:
    if not present_file(filename):
        missing.append(filename)
        return {}
    return {
        row["player_id"]: row
        for row in read_csv(RAW_DIR / filename)
        if row.get("league_id") == LEAGUE_ID
    }


def enrich_batters(rows: list[dict[str, str]], players: dict[str, dict[str, str]], teams: dict[str, Team]) -> list[dict[str, Any]]:
    enriched = []
    for row in rows:
        player = players.get(row["player_id"])
        team = teams.get(row["team_id"])
        if not player or not team:
            continue
        rates = batting_rates(row)
        merged: dict[str, Any] = dict(row)
        merged.update(rates)
        merged["name"] = player_name(player)
        merged["team_abbr"] = team.abbr
        merged["team_name"] = team.name
        merged["war"] = to_float(row.get("war"), math.nan)
        enriched.append(merged)
    return enriched


def enrich_pitchers(rows: list[dict[str, str]], players: dict[str, dict[str, str]], teams: dict[str, Team]) -> list[dict[str, Any]]:
    enriched = []
    for row in rows:
        player = players.get(row["player_id"])
        team = teams.get(row["team_id"])
        if not player or not team:
            continue
        rates = pitching_rates(row)
        merged: dict[str, Any] = dict(row)
        merged.update(rates)
        merged["name"] = player_name(player)
        merged["team_abbr"] = team.abbr
        merged["team_name"] = team.name
        merged["war"] = to_float(row.get("war"), math.nan)
        enriched.append(merged)
    return enriched


def top_rows(rows: list[dict[str, Any]], key: str, reverse: bool = True, minimum: tuple[str, float] | None = None, limit: int = 10) -> list[dict[str, Any]]:
    filtered = rows
    if minimum:
        min_key, min_value = minimum
        filtered = [row for row in rows if to_float(row.get(min_key), 0) >= min_value]
    filtered = [row for row in filtered if row.get(key) is not None and not (isinstance(row.get(key), float) and math.isnan(row.get(key)))]
    return sorted(filtered, key=lambda row: to_float(row.get(key), 0), reverse=reverse)[:limit]


def hitter_line(row: dict[str, Any], metric: str) -> str:
    return (
        f"- {row['name']} ({row['team_abbr']}) - {metric}; "
        f"{to_int(row.get('hr'))} HR, {to_int(row.get('rbi'))} RBI, OPS {fmt_pct(row.get('ops'))}, WAR {fmt_num(row.get('war'))}"
    )


def pitcher_line(row: dict[str, Any], metric: str) -> str:
    return (
        f"- {row['name']} ({row['team_abbr']}) - {metric}; "
        f"{fmt_ip(row.get('ip'))} IP, ERA {fmt_num(row.get('era'), 2)}, "
        f"{to_int(row.get('k'))} K, {to_int(row.get('s'))} SV, WAR {fmt_num(row.get('war'))}"
    )


def bio_line(player_id: str, players: dict[str, dict[str, str]], teams: dict[str, Team], cities: dict[str, dict[str, str]], states: dict[str, dict[str, str]], nations: dict[str, dict[str, str]]) -> str:
    player = players.get(player_id, {})
    team = teams.get(player.get("team_id", ""))
    return (
        f"Age {player.get('age', 'Not available')}; B/T {handedness(player.get('bats', ''))}/{handedness(player.get('throws', ''))}; "
        f"Birthplace: {birthplace(player, cities, states, nations)}; School: {school_context(player)}; "
        f"Current team: {team.name if team else 'Not available'}"
    )


def batter_rating_summary(player_id: str, ratings: dict[str, dict[str, str]]) -> str:
    row = ratings.get(player_id)
    if not row:
        return "Batter ratings: Not available"
    return (
        "Batter ratings: "
        f"CON {row.get('batting_ratings_overall_contact', 'NA')}, "
        f"GAP {row.get('batting_ratings_overall_gap', 'NA')}, "
        f"POW {row.get('batting_ratings_overall_power', 'NA')}, "
        f"EYE {row.get('batting_ratings_overall_eye', 'NA')}, "
        f"Speed {row.get('running_ratings_speed', 'NA')}"
    )


def pitcher_arsenal_summary(player_id: str, ratings: dict[str, dict[str, str]]) -> str:
    row = ratings.get(player_id)
    if not row:
        return "Pitcher arsenal: Not available"
    pitch_keys = [
        ("FB", "pitching_ratings_pitches_fastball"),
        ("SL", "pitching_ratings_pitches_slider"),
        ("CB", "pitching_ratings_pitches_curveball"),
        ("CH", "pitching_ratings_pitches_changeup"),
        ("SI", "pitching_ratings_pitches_sinker"),
        ("SP", "pitching_ratings_pitches_splitter"),
        ("CT", "pitching_ratings_pitches_cutter"),
        ("KC", "pitching_ratings_pitches_knucklecurve"),
        ("KN", "pitching_ratings_pitches_knuckleball"),
    ]
    pitches = []
    for label, key in pitch_keys:
        value = to_int(row.get(key))
        if value > 0:
            pitches.append(f"{label} {value}")
    if not pitches:
        pitches.append("Not available")
    return (
        "Pitcher arsenal: "
        f"STU {row.get('pitching_ratings_overall_stuff', 'NA')}, "
        f"MOV {row.get('pitching_ratings_overall_movement', 'NA')}, "
        f"CON {row.get('pitching_ratings_overall_control', 'NA')}, "
        f"VELO {row.get('pitching_ratings_misc_velocity', 'NA')}, "
        f"pitches {'; '.join(pitches[:5])}"
    )


def write_leader_section(lines: list[str], title: str, rows: list[dict[str, Any]], formatter) -> None:
    lines.extend(["", f"### {title}"])
    if not rows:
        lines.append("- Not available from raw source.")
        return
    for row in rows:
        lines.append(formatter(row))


def main() -> int:
    missing: list[str] = []
    existing = [name for name in REQUIRED_FILES + OPTIONAL_FILES if present_file(name)]
    for name in REQUIRED_FILES:
        if not present_file(name):
            missing.append(name)

    teams_by_id, teams_by_abbr = load_teams(missing)
    cities, states, nations = load_geo()
    players = {
        row["player_id"]: row
        for row in read_csv(RAW_DIR / "players.csv")
        if row.get("league_id") == LEAGUE_ID and row.get("retired") == "0"
    } if present_file("players.csv") else {}

    batting_rows = filter_stat_rows("players_career_batting_stats.csv", missing)
    pitching_rows = filter_stat_rows("players_career_pitching_stats.csv", missing)
    bat_ratings = load_rating_map("players_batting.csv", missing)
    pitch_ratings = load_rating_map("players_pitching.csv", missing)

    hitters = enrich_batters(batting_rows, players, teams_by_id)
    pitchers = enrich_pitchers(pitching_rows, players, teams_by_id)

    max_team_games = max([to_int(row.get("g")) for row in hitters] or [0])
    min_pa = max(1, int(max_team_games * 2.7)) if max_team_games else 1
    min_ip = max(1, int(max_team_games * 0.9)) if max_team_games else 1

    war_available_hit = any(not math.isnan(row.get("war", math.nan)) for row in hitters)
    war_available_pitch = any(not math.isnan(row.get("war", math.nan)) for row in pitchers)

    lines: list[str] = [
        "# ABL 1981 All-Star Break Player Pack",
        "",
        "## Source And Verification",
        f"- Latest raw export timestamp among input files: {current_export_timestamp(existing)}",
        "- Source mode: raw OOTP CSV files from csv/ootp_csv.",
        "- Excluded sources: csv/abl_statistics and stale star_schema player files.",
        f"- Batting filter: year={SEASON}, league_id={LEAGUE_ID}, level_id=1, split_id=1, game_id=0.",
        f"- Pitching filter: year={SEASON}, league_id={LEAGUE_ID}, level_id=1, split_id=1, game_id=0.",
        f"- OPS/AVG qualifier: at least {min_pa} PA.",
        f"- ERA/WHIP qualifier: at least {min_ip} IP.",
        f"- Missing files: {', '.join(sorted(set(missing))) if missing else 'None'}",
        "- Note: school/college fields are emitted only when raw IDs are populated; names are not present in these raw files.",
    ]

    lines.extend(["", "## League Hitter Leaders"])
    if war_available_hit:
        write_leader_section(
            lines,
            "Top 10 WAR",
            top_rows(hitters, "war"),
            lambda row: hitter_line(row, f"WAR {fmt_num(row.get('war'))}"),
        )
    else:
        lines.extend(["", "### Top 10 WAR", "- WAR not available in raw source."])
    write_leader_section(lines, "Top 10 OPS", top_rows(hitters, "ops", minimum=("pa", min_pa)), lambda row: hitter_line(row, f"OPS {fmt_pct(row.get('ops'))}"))
    write_leader_section(lines, "Top 10 HR", top_rows(hitters, "hr"), lambda row: hitter_line(row, f"{to_int(row.get('hr'))} HR"))
    write_leader_section(lines, "Top 10 RBI", top_rows(hitters, "rbi"), lambda row: hitter_line(row, f"{to_int(row.get('rbi'))} RBI"))
    write_leader_section(lines, "Top 10 AVG", top_rows(hitters, "avg", minimum=("pa", min_pa)), lambda row: hitter_line(row, f"AVG {fmt_pct(row.get('avg'))}"))
    write_leader_section(lines, "Top 10 SB", top_rows(hitters, "sb"), lambda row: hitter_line(row, f"{to_int(row.get('sb'))} SB"))

    lines.extend(["", "## League Pitcher Leaders"])
    if war_available_pitch:
        write_leader_section(
            lines,
            "Top 10 WAR",
            top_rows(pitchers, "war"),
            lambda row: pitcher_line(row, f"WAR {fmt_num(row.get('war'))}"),
        )
    else:
        lines.extend(["", "### Top 10 WAR", "- WAR not available in raw source."])
    write_leader_section(lines, "Top 10 ERA", top_rows(pitchers, "era", reverse=False, minimum=("ip", min_ip)), lambda row: pitcher_line(row, f"ERA {fmt_num(row.get('era'), 2)}"))
    write_leader_section(lines, "Top 10 Strikeouts", top_rows(pitchers, "k"), lambda row: pitcher_line(row, f"{to_int(row.get('k'))} K"))
    write_leader_section(lines, "Top 10 Saves", top_rows(pitchers, "s"), lambda row: pitcher_line(row, f"{to_int(row.get('s'))} SV"))
    write_leader_section(lines, "Top 10 WHIP", top_rows(pitchers, "whip", reverse=False, minimum=("ip", min_ip)), lambda row: pitcher_line(row, f"WHIP {fmt_num(row.get('whip'), 2)}"))

    lines.extend(["", "## 16 Contender Team Player Capsules"])
    missing_contenders = [abbr for abbr in CONTENDER_ORDER if abbr not in teams_by_abbr]
    for abbr in CONTENDER_ORDER:
        team = teams_by_abbr.get(abbr)
        if not team:
            lines.extend(["", f"### {abbr}", "- Team not found in raw teams.csv."])
            continue
        team_hitters = [row for row in hitters if row["team_abbr"] == abbr]
        team_pitchers = [row for row in pitchers if row["team_abbr"] == abbr]
        hitter_key = "war" if war_available_hit else "ops"
        pitcher_key = "war" if war_available_pitch else "era"
        top_hitters = top_rows(team_hitters, hitter_key, reverse=True, limit=3)
        top_pitchers = top_rows(team_pitchers, pitcher_key, reverse=(pitcher_key != "era"), limit=3)
        watch_pool = top_hitters + top_pitchers
        watch = max(watch_pool, key=lambda row: to_float(row.get("war"), 0), default=None)

        lines.extend(["", f"### {team.name} ({team.abbr})", "#### Top Hitters"])
        if not top_hitters:
            lines.append("- Not available from raw source.")
        for row in top_hitters:
            pid = row["player_id"]
            lines.append(hitter_line(row, f"WAR {fmt_num(row.get('war'))}" if war_available_hit else f"OPS {fmt_pct(row.get('ops'))}"))
            lines.append(f"  - Bio: {bio_line(pid, players, teams_by_id, cities, states, nations)}")
            lines.append(f"  - {batter_rating_summary(pid, bat_ratings)}")

        lines.append("#### Top Pitchers")
        if not top_pitchers:
            lines.append("- Not available from raw source.")
        for row in top_pitchers:
            pid = row["player_id"]
            lines.append(pitcher_line(row, f"WAR {fmt_num(row.get('war'))}" if war_available_pitch else f"ERA {fmt_num(row.get('era'), 2)}"))
            lines.append(f"  - Bio: {bio_line(pid, players, teams_by_id, cities, states, nations)}")
            lines.append(f"  - {pitcher_arsenal_summary(pid, pitch_ratings)}")

        lines.append("#### Player To Watch")
        if watch:
            lines.append(f"- {watch['name']} ({team.abbr})")
            lines.append(f"  - Bio: {bio_line(watch['player_id'], players, teams_by_id, cities, states, nations)}")
        else:
            lines.append("- Not available from raw source.")
        lines.append("- EB player angle placeholder: TODO_VERIFY")

    lines.extend(
        [
            "",
            "## Final Verification",
            f"- Hitters loaded: {len(hitters)}",
            f"- Pitchers loaded: {len(pitchers)}",
            f"- Teams matched: {len(teams_by_id)}",
            f"- All 16 contender teams found: {'Yes' if not missing_contenders else 'No'}",
            f"- Missing contender teams: {', '.join(missing_contenders) if missing_contenders else 'None'}",
            f"- Hitter WAR available in raw source: {'Yes' if war_available_hit else 'No'}",
            f"- Pitcher WAR available in raw source: {'Yes' if war_available_pitch else 'No'}",
        ]
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("ALL_STAR_BREAK_1981_PLAYER_PACK_FROM_RAW: wrote output")
    print(f"- {OUT_PATH}")
    print("Verification:")
    print(f"number of hitters loaded: {len(hitters)}")
    print(f"number of pitchers loaded: {len(pitchers)}")
    print(f"number of teams matched: {len(teams_by_id)}")
    print(f"16 contender teams all found: {'Yes' if not missing_contenders else 'No'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
