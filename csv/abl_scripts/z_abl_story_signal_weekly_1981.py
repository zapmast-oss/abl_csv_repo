from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "csv" / "out" / "docs" / "abl_data_catalog.csv"
ACTIVE_SEASON = 1981
ACTIVE_AS_OF_DATE = "1981-07-12"
ACTIVE_COVERAGE_LABEL = "1981_week_15"
EXPECTED_GAMES_PER_TEAM = 89
CANDIDATE_COLUMNS = [
    "candidate_id", "run_id", "week_label", "coverage_label", "as_of_date",
    "season", "expected_games_per_team",
    "hierarchy_level", "signal_type", "subject_type", "subject_id",
    "subject_name", "related_subjects", "headline_factual", "stakes",
    "evidence_summary", "source_files", "signal_score", "confidence", "status",
]
EVIDENCE_COLUMNS = [
    "evidence_id", "candidate_id", "metric", "value", "comparison",
    "source_file", "source_row_key", "season", "coverage_label", "as_of_date",
    "expected_games_per_team", "notes",
]
MANIFEST_COLUMNS = [
    "source_file", "used_by_engine", "row_count", "min_date", "max_date",
    "detected_games_per_team", "source_status", "reason",
]
HIERARCHY_ORDER = {
    "tournament": 0, "standings": 1, "race": 2, "fans": 3,
    "management": 4, "players": 5, "team": 6, "game": 7,
}
SOURCES = {
    "games": "csv/ootp_csv/games.csv",
    "teams": "csv/ootp_csv/teams.csv",
    "divisions": "csv/ootp_csv/divisions.csv",
    "players": "csv/ootp_csv/players.csv",
    "career_batting": "csv/ootp_csv/players_career_batting_stats.csv",
    "career_pitching": "csv/ootp_csv/players_career_pitching_stats.csv",
    "champions": "csv/out/almanac/1980/league_champions_1980_league200.csv",
}
SOURCE_POLICIES = [
    (SOURCES["games"], "used", "active_current", "Authoritative date-filterable game source; completed ABL games are filtered through the active cutoff."),
    (SOURCES["teams"], "used", "compatible_static", "Team identity and league/division keys are dimensions, not performance snapshots."),
    (SOURCES["divisions"], "used", "compatible_static", "Division names and keys are static structural dimensions."),
    (SOURCES["players"], "used", "compatible_static", "Player names and team identifiers support current raw statistical rows."),
    (SOURCES["career_batting"], "used", "active_current", "Raw 1981 cumulative batting totals belong to the coordinated current OOTP export."),
    (SOURCES["career_pitching"], "used", "active_current", "Raw 1981 cumulative pitching totals belong to the coordinated current OOTP export."),
    (SOURCES["champions"], "used", "compatible_historical", "1980 context is intentionally historical and cannot replace current-season evidence."),
    ("csv/out/star_schema/monday_1981_standings_by_division.csv", "excluded", "excluded_wrong_games_count", "Preserved standings snapshot contains 32 games per team, not the active 89-game state."),
    ("csv/out/star_schema/fact_team_reporting_1981_weekly_change.csv", "excluded", "excluded_wrong_games_count", "Preserved weekly-change rows end at 32 games, not the active 89-game state."),
    ("csv/out/star_schema/fact_player_batting.csv", "excluded", "excluded_stale_snapshot", "Preserved early-season player fact is not used for active Week 15 leaders."),
    ("csv/out/star_schema/fact_player_pitching.csv", "excluded", "excluded_stale_snapshot", "Preserved early-season player fact is not used for active Week 15 leaders."),
    ("csv/story_candidates_1981_week_05.csv", "excluded", "excluded_stale_snapshot", "Preserved Week 5 story candidates are historical artifacts, not active inputs."),
    ("csv/story_menu_1981_week_05.csv", "excluded", "excluded_stale_snapshot", "Preserved Week 5 story menu is not an active input."),
    ("csv/story_menu_1981_week_07.csv", "excluded", "excluded_stale_snapshot", "Preserved early-season story menu is not an active input."),
    ("csv/out/star_schema/fact_manager_scorecard_1981_current.csv", "disabled", "disabled_incompatible", "Manager scorecard is tied to a 32-game standings snapshot; management signals are disabled."),
    ("csv/out/csv_out/z_ABL_Manager_Tendencies.csv", "disabled", "disabled_incompatible", "Manager tendencies stop at 62 games; management signals require the 89-game cutoff."),
    ("csv/out/csv_out/z_ABL_Division_Leverage.csv", "disabled", "disabled_incompatible", "Division leverage stops at 62 games and cannot support active race signals."),
    ("csv/out/csv_out/z_ABL_Rotation_Stability.csv", "disabled", "disabled_incompatible", "Rotation stability stops at 62 games and cannot support active pitcher/team signals."),
]


def number(value: object, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def read_csv(rel_path: str) -> list[dict[str, str]]:
    path = ROOT / rel_path
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        lines = (line for line in handle if not line.lstrip().startswith("#"))
        return list(csv.DictReader(lines))


def write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_id(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


class SignalBuilder:
    def __init__(self, season: int, week_label: str, as_of: str, run_id: str,
                 expected_games_per_team: int, catalog_paths: set[str]):
        self.season = season
        self.week_label = week_label
        self.as_of = as_of
        self.run_id = run_id
        self.expected_games_per_team = expected_games_per_team
        self.catalog_paths = catalog_paths
        self.candidates: list[dict[str, object]] = []
        self.evidence: list[dict[str, object]] = []

    def add(self, *, hierarchy: str, signal: str, subject_type: str,
            subject_id: str, subject_name: str, related: list[str], headline: str,
            stakes: str, summary: str, sources: list[str], score: float,
            confidence: str, evidence: list[dict[str, str]]) -> None:
        missing = [source for source in sources if source not in self.catalog_paths]
        if missing:
            raise ValueError(f"Sources missing from authoritative catalog: {missing}")
        candidate_id = f"{self.week_label}__{signal}__{safe_id(subject_id)}"
        self.candidates.append({
            "candidate_id": candidate_id, "run_id": self.run_id,
            "week_label": self.week_label, "coverage_label": self.week_label,
            "as_of_date": self.as_of, "season": self.season,
            "expected_games_per_team": self.expected_games_per_team,
            "hierarchy_level": hierarchy, "signal_type": signal,
            "subject_type": subject_type, "subject_id": subject_id,
            "subject_name": subject_name, "related_subjects": "|".join(related),
            "headline_factual": headline, "stakes": stakes,
            "evidence_summary": summary, "source_files": "|".join(sources),
            "signal_score": round(score, 1), "confidence": confidence,
            "status": "candidate",
        })
        for index, item in enumerate(evidence, start=1):
            source = item["source_file"]
            if source not in sources:
                raise ValueError(f"Evidence source {source} not declared by {candidate_id}")
            self.evidence.append({
                "evidence_id": f"{candidate_id}__e{index:02d}",
                "candidate_id": candidate_id, "season": self.season,
                "coverage_label": self.week_label, "as_of_date": self.as_of,
                "expected_games_per_team": self.expected_games_per_team, **item,
            })


def load_catalog_paths() -> set[str]:
    if not CATALOG.exists():
        raise FileNotFoundError(f"Authoritative catalog not found: {CATALOG}")
    with CATALOG.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["file_path"] for row in csv.DictReader(handle)}


def detect_date_range(rows: list[dict[str, str]], source_file: str,
                      as_of: date) -> tuple[str, str]:
    values: list[date] = []
    date_columns = ("date", "game_date", "start_date", "end_date", "transaction_date")
    for row in rows:
        if source_file == SOURCES["games"]:
            if row.get("league_id") != "200" or row.get("game_type") != "0" or row.get("played") != "1":
                continue
        for column in date_columns:
            raw = row.get(column, "").strip()
            if not raw:
                continue
            try:
                parsed = parse_date(raw)
            except ValueError:
                continue
            if source_file != SOURCES["games"] or parsed <= as_of:
                values.append(parsed)
    if not values:
        return "", ""
    return min(values).isoformat(), max(values).isoformat()


def detect_games_per_team(source_file: str, rows: list[dict[str, str]],
                          as_of: date) -> str:
    values: set[int] = set()
    if source_file == SOURCES["games"]:
        counts: dict[str, int] = defaultdict(int)
        for row in rows:
            if (row.get("league_id") == "200" and row.get("game_type") == "0"
                    and row.get("played") == "1" and parse_date(row["date"]) <= as_of):
                counts[row["home_team"]] += 1
                counts[row["away_team"]] += 1
        values = set(counts.values())
    elif source_file.endswith("monday_1981_standings_by_division.csv"):
        values = {int(number(row.get("games"))) for row in rows}
    elif source_file.endswith("fact_team_reporting_1981_weekly_change.csv"):
        values = {int(number(row.get("wins_curr")) + number(row.get("losses_curr"))) for row in rows}
    elif source_file.endswith("fact_manager_scorecard_1981_current.csv"):
        values = {int(number(row.get("wins")) + number(row.get("losses"))) for row in rows}
    elif source_file.endswith("z_ABL_Manager_Tendencies.csv"):
        values = {int(number(row.get("g_est"))) for row in rows}
    elif source_file.endswith("z_ABL_Division_Leverage.csv"):
        values = {int(number(row.get("overall_g"))) for row in rows}
    elif source_file.endswith("z_ABL_Rotation_Stability.csv"):
        values = {int(number(row.get("total_starts"))) for row in rows}
    elif source_file.endswith("fact_player_batting.csv"):
        values = {int(max((number(row.get("G")) for row in rows), default=0))}
    if not values:
        return ""
    if len(values) == 1:
        return str(next(iter(values)))
    return f"{min(values)}-{max(values)}"


def build_source_manifest(season: int, as_of_text: str, coverage_label: str,
                          expected_games: int) -> list[dict[str, object]]:
    as_of = date.fromisoformat(as_of_text)
    manifest: list[dict[str, object]] = []
    for source_file, use, status, reason in SOURCE_POLICIES:
        path = ROOT / source_file
        if not path.exists():
            manifest.append({
                "source_file": source_file, "used_by_engine": "no", "row_count": "",
                "min_date": "", "max_date": "", "detected_games_per_team": "",
                "source_status": "disabled_incompatible",
                "reason": f"Declared source is missing. {reason}",
            })
            continue
        rows = read_csv(source_file)
        min_date, max_date = detect_date_range(rows, source_file, as_of)
        detected = detect_games_per_team(source_file, rows, as_of)
        manifest.append({
            "source_file": source_file, "used_by_engine": "yes" if use == "used" else "no",
            "row_count": len(rows), "min_date": min_date, "max_date": max_date,
            "detected_games_per_team": detected, "source_status": status, "reason": reason,
        })
    games_row = next(row for row in manifest if row["source_file"] == SOURCES["games"])
    if games_row["detected_games_per_team"] != str(expected_games):
        raise ValueError(
            f"Active games source detected {games_row['detected_games_per_team']} games per team; "
            f"expected {expected_games}. Refusing to generate signals."
        )
    return manifest


def write_source_manifest(rows: list[dict[str, object]], season: int,
                          as_of_text: str, coverage_label: str,
                          expected_games: int) -> dict[str, str]:
    out_dir = ROOT / "csv" / "out" / "story" / "manifests"
    stem = f"story_engine_source_manifest_{coverage_label}"
    csv_path = out_dir / f"{stem}.csv"
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    write_csv(csv_path, MANIFEST_COLUMNS, rows)
    payload = {
        "season": season, "as_of_date": as_of_text, "coverage_label": coverage_label,
        "expected_games_per_team": expected_games,
        "sources": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        f"# Story Engine Source Manifest — {coverage_label}", "",
        f"- Season: {season}", f"- As of: {as_of_text}",
        f"- Expected coverage: {expected_games} games per team", "",
        "| Used | Status | Source | Rows | Date range | Games/team | Reason |",
        "|---|---|---|---:|---|---:|---|",
    ]
    for row in rows:
        date_range = f"{row['min_date']} to {row['max_date']}" if row["min_date"] else "n/a"
        lines.append(
            f"| {row['used_by_engine']} | `{row['source_status']}` | `{row['source_file']}` | "
            f"{row['row_count'] or 'n/a'} | {date_range} | {row['detected_games_per_team'] or 'n/a'} | {row['reason']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"csv": str(csv_path), "json": str(json_path), "markdown": str(md_path)}


def build_latest_signals(builder: SignalBuilder) -> None:
    """Build signals from the latest raw, date-filterable 1981 exports.

    This path intentionally avoids older derivative snapshots that coexist in
    the repository with the current 89-game raw export.
    """
    as_of = date.fromisoformat(builder.as_of)
    games_all = read_csv(SOURCES["games"])
    teams_all = read_csv(SOURCES["teams"])
    divisions_all = read_csv(SOURCES["divisions"])
    players_all = read_csv(SOURCES["players"])
    batting_all = read_csv(SOURCES["career_batting"])
    pitching_all = read_csv(SOURCES["career_pitching"])
    champions = read_csv(SOURCES["champions"])

    teams = {
        row["team_id"]: row for row in teams_all
        if row["league_id"] == "200" and row["level"] == "1"
    }
    division_names = {
        (row["sub_league_id"], row["division_id"]): row["name"]
        for row in divisions_all if row["league_id"] == "200"
    }
    subleague_names = {"0": "National Baseball Conference", "1": "American Baseball Conference"}

    played = []
    scheduled = []
    for row in games_all:
        if row["league_id"] != "200" or row["game_type"] != "0":
            continue
        game_date = parse_date(row["date"])
        if row["played"] == "1" and game_date <= as_of:
            played.append(row)
        elif row["played"] == "0" and as_of < game_date <= as_of + timedelta(days=7):
            scheduled.append(row)

    records = {
        team_id: {"w": 0, "l": 0, "rs": 0, "ra": 0, "week_w": 0, "week_l": 0}
        for team_id in teams
    }
    week_start = as_of - timedelta(days=6)
    for row in played:
        home, away = row["home_team"], row["away_team"]
        if home not in records or away not in records:
            continue
        # OOTP games.csv stores away runs in runs0 and home runs in runs1.
        away_runs, home_runs = int(row["runs0"]), int(row["runs1"])
        records[home]["rs"] += home_runs; records[home]["ra"] += away_runs
        records[away]["rs"] += away_runs; records[away]["ra"] += home_runs
        home_won = home_runs > away_runs
        records[home]["w" if home_won else "l"] += 1
        records[away]["l" if home_won else "w"] += 1
        if parse_date(row["date"]) >= week_start:
            records[home]["week_w" if home_won else "week_l"] += 1
            records[away]["week_l" if home_won else "week_w"] += 1

    standings = []
    for team_id, team in teams.items():
        rec = records[team_id]
        games = rec["w"] + rec["l"]
        standings.append({
            "team_id": team_id, "team_abbr": team["abbr"],
            "team_name": f"{team['name']} {team['nickname']}",
            "division": f"{subleague_names.get(team['sub_league_id'], team['sub_league_id'])} {division_names.get((team['sub_league_id'], team['division_id']), 'Division')}",
            "w": rec["w"], "l": rec["l"], "games": games,
            "win_pct": rec["w"] / games if games else 0,
            "rs": rec["rs"], "ra": rec["ra"], "week_w": rec["week_w"], "week_l": rec["week_l"],
        })
    by_team = {row["team_id"]: row for row in standings}
    by_abbr = {row["team_abbr"]: row for row in standings}

    divisions: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in standings:
        divisions[str(row["division"])].append(row)
    for division, rows in divisions.items():
        ranked = sorted(rows, key=lambda r: (-number(r["win_pct"]), -number(r["w"])))
        leader, second = ranked[:2]
        gap = number(leader["w"]) - number(second["w"])
        if gap <= 4:
            sources = [SOURCES["games"], SOURCES["teams"], SOURCES["divisions"]]
            builder.add(
                hierarchy="standings", signal="division_race", subject_type="division",
                subject_id=division, subject_name=division,
                related=[str(leader["team_name"]), str(second["team_name"])],
                headline=f"{division}: {leader['team_name']} leads {second['team_name']} by {int(gap)} game(s)",
                stakes="With the season past its midpoint, each head-to-head result and lost game has less schedule left to absorb it.",
                summary=f"Through {builder.as_of}, {leader['team_name']} is {leader['w']}-{leader['l']} and {second['team_name']} is {second['w']}-{second['l']}.",
                sources=sources, score=91 - gap * 3, confidence="high",
                evidence=[
                    {"metric": "division_leader_record", "value": f"{leader['w']}-{leader['l']}",
                     "comparison": f"second={second['w']}-{second['l']}", "source_file": SOURCES["games"],
                     "source_row_key": f"through={builder.as_of};team_id={leader['team_id']}",
                     "notes": "Recomputed from completed league-200 regular-season games."},
                    {"metric": "division_membership", "value": division, "comparison": "top two clubs",
                     "source_file": SOURCES["teams"], "source_row_key": f"team_id={leader['team_id']}|{second['team_id']}",
                     "notes": "Division IDs resolved through divisions.csv."},
                ],
            )

    weekly_ranked = sorted(standings, key=lambda r: (number(r["week_w"]) - number(r["week_l"]), number(r["week_w"])), reverse=True)
    for row, direction in ((weekly_ranked[0], "rise"), (weekly_ranked[-1], "fall")):
        builder.add(
            hierarchy="race", signal=f"weekly_{direction}", subject_type="team",
            subject_id=str(row["team_id"]), subject_name=str(row["team_name"]), related=[],
            headline=f"{row['team_name']} posted the week's sharpest {direction}",
            stakes="The week changed immediate pressure, but the next games determine whether it changes the race.",
            summary=f"From {week_start.isoformat()} through {builder.as_of}: {row['week_w']}-{row['week_l']}; season record {row['w']}-{row['l']}.",
            sources=[SOURCES["games"], SOURCES["teams"]], score=78, confidence="high",
            evidence=[{"metric": "last_seven_day_record", "value": f"{row['week_w']}-{row['week_l']}",
                       "comparison": f"season={row['w']}-{row['l']}", "source_file": SOURCES["games"],
                       "source_row_key": f"team_id={row['team_id']};{week_start.isoformat()}..{builder.as_of}",
                       "notes": "Calendar-window record recomputed from completed games."}],
        )

    pythag_rows = []
    for row in standings:
        rs, ra, games = number(row["rs"]), number(row["ra"]), number(row["games"])
        expected = games * (rs * rs / (rs * rs + ra * ra)) if rs + ra else 0
        pythag_rows.append((number(row["w"]) - expected, expected, row))
    for diff, expected, row in (max(pythag_rows, key=lambda x: x[0]), min(pythag_rows, key=lambda x: x[0])):
        label = "over_record" if diff > 0 else "under_record"
        builder.add(
            hierarchy="team", signal=f"pythag_{label}", subject_type="team",
            subject_id=str(row["team_id"]), subject_name=str(row["team_name"]), related=[],
            headline=f"{row['team_name']}'s record and run balance remain far apart",
            stakes="The gap is pressure to explain and monitor, not proof that reversal is due.",
            summary=f"Record {row['w']}-{row['l']}; runs {row['rs']}-{row['ra']}; Pythagorean expectation {expected:.1f} wins; gap {diff:+.1f}.",
            sources=[SOURCES["games"], SOURCES["teams"]], score=70 + min(abs(diff), 10), confidence="high",
            evidence=[{"metric": "pythag_diff_wins", "value": f"{diff:+.1f}",
                       "comparison": f"actual_wins={row['w']};expected_wins={expected:.1f}", "source_file": SOURCES["games"],
                       "source_row_key": f"team_id={row['team_id']};through={builder.as_of}",
                       "notes": "Expected winning percentage uses RS^2/(RS^2+RA^2); it is not a forecast."}],
        )

    player_names = {r["player_id"]: f"{r['first_name']} {r['last_name']}".strip() for r in players_all}
    batting = [r for r in batting_all if r["year"] == "1981" and r["league_id"] == "200" and r["level_id"] == "1" and r["split_id"] == "1" and r["game_id"] == "0" and number(r["pa"]) >= 150]
    hitter = max(batting, key=lambda r: number(r["war"]))
    hitter_team = by_team.get(hitter["team_id"])
    hitter_name = player_names.get(hitter["player_id"], f"Player {hitter['player_id']}")
    avg = number(hitter["h"]) / number(hitter["ab"]) if number(hitter["ab"]) else 0
    builder.add(
        hierarchy="players", signal="player_value_leader", subject_type="player",
        subject_id=hitter["player_id"], subject_name=hitter_name,
        related=[str(hitter_team["team_name"])] if hitter_team else [],
        headline=f"{hitter_name} owns the strongest current position-player value line",
        stakes="The second half tests whether that production continues to carry league-leading weight.",
        summary=f"{number(hitter['war']):.1f} WAR, {hitter['hr']} HR and a {avg:.3f} average in {hitter['pa']} PA.",
        sources=[SOURCES["career_batting"], SOURCES["players"]], score=78, confidence="high",
        evidence=[{"metric": "position_player_war", "value": f"{number(hitter['war']):.1f}",
                   "comparison": "highest league-200 total among hitters with >=150 PA", "source_file": SOURCES["career_batting"],
                   "source_row_key": f"player_id={hitter['player_id']};year=1981;split_id=1", "notes": "Current raw season total."}],
    )

    pitching = [r for r in pitching_all if r["year"] == "1981" and r["league_id"] == "200" and r["level_id"] == "1" and r["split_id"] == "1" and r["game_id"] == "0" and number(r["gs"]) >= 10 and number(r["outs"]) >= 240]
    starter = max(pitching, key=lambda r: number(r["war"]))
    starter_team = by_team.get(starter["team_id"])
    starter_name = player_names.get(starter["player_id"], f"Player {starter['player_id']}")
    innings = number(starter["outs"]) / 3
    era = number(starter["er"]) * 9 / innings if innings else 0
    builder.add(
        hierarchy="players", signal="ace_performance", subject_type="player",
        subject_id=starter["player_id"], subject_name=starter_name,
        related=[str(starter_team["team_name"])] if starter_team else [],
        headline=f"{starter_name} has built the strongest current starter value case",
        stakes="Each remaining start now carries both team-race weight and the burden of sustaining an ace-level season.",
        summary=f"{number(starter['war']):.1f} WAR, {era:.2f} ERA, {innings:.1f} innings and a {starter['w']}-{starter['l']} record.",
        sources=[SOURCES["career_pitching"], SOURCES["players"]], score=80, confidence="high",
        evidence=[{"metric": "starter_war", "value": f"{number(starter['war']):.1f}",
                   "comparison": "highest among league-200 pitchers with >=10 GS and >=80 IP", "source_file": SOURCES["career_pitching"],
                   "source_row_key": f"player_id={starter['player_id']};year=1981;split_id=1", "notes": "Current raw season total."}],
    )

    if scheduled:
        def matchup_score(game: dict[str, str]) -> tuple[float, float]:
            home, away = by_team.get(game["home_team"]), by_team.get(game["away_team"])
            if not home or not away:
                return (-1, -1)
            same_div = home["division"] == away["division"]
            combined = number(home["win_pct"]) + number(away["win_pct"])
            closeness = 1 - abs(number(home["win_pct"]) - number(away["win_pct"]))
            return (100 if same_div else 0, combined + closeness)
        game = max(scheduled, key=matchup_score)
        home, away = by_team[game["home_team"]], by_team[game["away_team"]]
        same_div = home["division"] == away["division"]
        builder.add(
            hierarchy="game", signal="matchup_stakes", subject_type="matchup",
            subject_id=game["game_id"], subject_name=f"{away['team_name']} at {home['team_name']}",
            related=[str(away["team_name"]), str(home["team_name"])],
            headline=f"{away['team_name']} at {home['team_name']} puts {'division' if same_div else 'standings'} pressure on the field",
            stakes="The scheduled game offers direct evidence in a race with fewer games left to recover from each result.",
            summary=f"Scheduled {game['date']}; records {away['w']}-{away['l']} and {home['w']}-{home['l']}; same division={same_div}.",
            sources=[SOURCES["games"], SOURCES["teams"], SOURCES["divisions"]], score=76 if same_div else 70,
            confidence="high", evidence=[{"metric": "scheduled_matchup", "value": game["date"],
                                           "comparison": f"{away['w']}-{away['l']} vs {home['w']}-{home['l']}",
                                           "source_file": SOURCES["games"], "source_row_key": f"game_id={game['game_id']}",
                                           "notes": "Unplayed regular-season game within seven days after the cutoff."}],
        )

    current_by_abbr = by_abbr
    links = [row for row in champions if row["team_abbr"] in current_by_abbr]
    if links:
        past = max(links, key=lambda r: number(current_by_abbr[r["team_abbr"]]["win_pct"]))
        current = current_by_abbr[past["team_abbr"]]
        builder.add(
            hierarchy="tournament", signal="historical_champion_echo", subject_type="historical_link",
            subject_id=past["team_id"], subject_name=past["team_name"], related=[str(current["division"])],
            headline=f"1980 qualifier {past['team_name']} remains relevant in the 1981 race",
            stakes="Last season gives the current race historical weight, but it does not decide this season's finish.",
            summary=f"1980 champion-table record {past['wins']}-{past['losses']}; current record {current['w']}-{current['l']} through {builder.as_of}.",
            sources=[SOURCES["champions"], SOURCES["games"], SOURCES["teams"]], score=82, confidence="high",
            evidence=[
                {"metric": "prior_season_champion_record", "value": f"{past['wins']}-{past['losses']}",
                 "comparison": f"1980 {past['conference']} {past['slot']}", "source_file": SOURCES["champions"],
                 "source_row_key": f"team_id={past['team_id']}", "notes": "Prior-season context, not a forecast."},
                {"metric": "current_record", "value": f"{current['w']}-{current['l']}",
                 "comparison": str(current["division"]), "source_file": SOURCES["games"],
                 "source_row_key": f"team_id={current['team_id']};through={builder.as_of}", "notes": "Recomputed current record."},
            ],
        )

def write_menu(path: Path, candidates: list[dict[str, object]]) -> None:
    columns = ["menu_rank"] + CANDIDATE_COLUMNS
    rows = []
    ordered = sorted(candidates, key=lambda r: (HIERARCHY_ORDER[str(r["hierarchy_level"])], -number(r["signal_score"])))
    for rank, row in enumerate(ordered, start=1):
        rows.append({"menu_rank": rank, **row})
    write_csv(path, columns, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate evidence-backed weekly 1981 story signals.")
    parser.add_argument("--season", type=int, default=ACTIVE_SEASON)
    parser.add_argument("--week-label", default=ACTIVE_COVERAGE_LABEL)
    parser.add_argument("--as-of", default=ACTIVE_AS_OF_DATE)
    parser.add_argument("--expected-games-per-team", type=int, default=EXPECTED_GAMES_PER_TEAM)
    parser.add_argument("--run-id", default="sprint1_1981_week_15")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    date.fromisoformat(args.as_of)
    catalog_paths = load_catalog_paths()
    for source in SOURCES.values():
        if source not in catalog_paths:
            raise ValueError(f"Selected source is absent from authoritative catalog: {source}")
        if not (ROOT / source).exists():
            raise FileNotFoundError(ROOT / source)

    manifest = build_source_manifest(
        args.season, args.as_of, args.week_label, args.expected_games_per_team
    )
    manifest_paths = write_source_manifest(
        manifest, args.season, args.as_of, args.week_label, args.expected_games_per_team
    )
    builder = SignalBuilder(
        args.season, args.week_label, args.as_of, args.run_id,
        args.expected_games_per_team, catalog_paths,
    )
    build_latest_signals(builder)
    candidate_dir = ROOT / "csv" / "out" / "story" / "candidates"
    menu_dir = ROOT / "csv" / "out" / "story" / "menus"
    candidate_path = candidate_dir / f"story_candidates_{args.week_label}.csv"
    evidence_path = candidate_dir / f"story_evidence_{args.week_label}.csv"
    menu_path = menu_dir / f"story_menu_{args.week_label}.csv"
    write_csv(candidate_path, CANDIDATE_COLUMNS, builder.candidates)
    write_csv(evidence_path, EVIDENCE_COLUMNS, builder.evidence)
    write_menu(menu_path, builder.candidates)
    print(json.dumps({
        "candidate_path": str(candidate_path), "evidence_path": str(evidence_path),
        "menu_path": str(menu_path), "candidate_count": len(builder.candidates),
        "evidence_count": len(builder.evidence), "manifest_paths": manifest_paths,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
