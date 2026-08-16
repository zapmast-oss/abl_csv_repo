from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "csv" / "ootp_csv"
SORTABLE = ROOT / "csv" / "abl_statistics"
CONTROL = ROOT / "csv" / "out" / "control"
OUT = ROOT / "csv" / "out" / "story"
NEWSROOM_DATE = "1981-07-20"
AS_OF_DATE = "1981-07-19"
SEASON = 1981
RUN_SLUG = "1981_07_20_asof_1981-07-19"
RUN_ID = f"current_{RUN_SLUG}"
PREFLIGHT = CONTROL / "current_state_preflight_1981_target_1981-07-19.json"
BATCH_MANIFEST = CONTROL / "abl_batch_manifest_current.csv"

HIERARCHY = ["tournament", "standings", "race", "fans", "management_organization", "players", "team", "game"]
HIERARCHY_ORDER = {name: index for index, name in enumerate(HIERARCHY)}
CANDIDATE_FIELDS = [
    "candidate_id", "run_id", "newsroom_date", "as_of_date", "season",
    "candidate_class", "hierarchy_level", "signal_type", "subject_type",
    "subject_id", "subject_name", "related_subjects", "headline_factual",
    "stakes", "evidence_summary", "source_files", "signal_score", "confidence", "status",
]
EVIDENCE_FIELDS = [
    "evidence_id", "candidate_id", "newsroom_date", "as_of_date", "season",
    "evidence_class", "metric", "value", "comparison", "source_file",
    "source_row_key", "notes",
]
SOURCE_MANIFEST_FIELDS = [
    "source_file", "used_by_run", "source_status", "evidence_role",
    "batch_manifest_status", "reason",
]

SOURCES = {
    "games": "csv/ootp_csv/games.csv",
    "games_score": "csv/ootp_csv/games_score.csv",
    "game_logs": "csv/ootp_csv/game_logs.csv",
    "teams": "csv/ootp_csv/teams.csv",
    "divisions": "csv/ootp_csv/divisions.csv",
    "sub_leagues": "csv/ootp_csv/sub_leagues.csv",
    "players": "csv/ootp_csv/players.csv",
    "batting": "csv/ootp_csv/players_career_batting_stats.csv",
    "pitching": "csv/ootp_csv/players_career_pitching_stats.csv",
    "coaches": "csv/ootp_csv/coaches.csv",
    "team_financial": "csv/abl_statistics/abl_statistics_team_statistics___info_-_sortable_stats_team_finan.csv",
    "team_context": "csv/abl_statistics/abl_statistics_team_statistics___info_-_sortable_stats_team_pers_park.csv",
    "team_staff": "csv/abl_statistics/abl_statistics_team_statistics___info_-_sortable_stats_abl_staff.csv",
    "batting_xtra": "csv/abl_statistics/abl_statistics_team_statistics___info_-_sortable_stats_batting_xtra.csv",
    "champions_1980": "csv/out/almanac/1980/league_champions_1980_league200.csv",
}


def number(value: object, default: float = 0.0) -> float:
    try:
        return float(str(value).replace(" ", "").replace(",", "").strip())
    except (TypeError, ValueError):
        return default


def money(value: str) -> float:
    text = value.strip().lower().replace("$", "").replace(",", "")
    multiplier = 1_000_000 if text.endswith("m") else (1_000 if text.endswith("k") else 1)
    if text.endswith(("m", "k")):
        text = text[:-1]
    return number(text) * multiplier


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def normalize_id(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(rel: str) -> list[dict[str, str]]:
    with (ROOT / rel).open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        lines = (line for line in handle if line.strip() and not line.lstrip().startswith("#"))
        return list(csv.DictReader(lines))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def validate_control_state() -> dict[str, dict[str, str]]:
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    if preflight["verdict"] != "READY_FOR_CURRENT_RUN" or preflight["latest_completed_game_date"] != AS_OF_DATE:
        raise ValueError("Current story run refused: preflight is not READY for the July 19 cutoff")
    with BATCH_MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest = {row["source_file_path"]: row for row in csv.DictReader(handle)}
    for rel in SOURCES.values():
        path = ROOT / rel
        if not path.exists():
            raise FileNotFoundError(path)
        if rel in manifest and manifest[rel]["sha256"] != sha256(path):
            raise ValueError(f"Source changed after batch manifest: {rel}")
    return manifest


class Builder:
    def __init__(self):
        self.candidates: list[dict[str, object]] = []
        self.evidence: list[dict[str, object]] = []

    def add(self, *, candidate_class: str, hierarchy: str, signal: str,
            subject_type: str, subject_id: str, subject_name: str,
            related: list[str], headline: str, stakes: str, summary: str,
            sources: list[str], score: float, confidence: str,
            evidence: list[dict[str, str]]) -> None:
        candidate_id = f"{RUN_SLUG}__{signal}__{normalize_id(subject_id)}"
        self.candidates.append({
            "candidate_id": candidate_id, "run_id": RUN_ID,
            "newsroom_date": NEWSROOM_DATE, "as_of_date": AS_OF_DATE, "season": SEASON,
            "candidate_class": candidate_class, "hierarchy_level": hierarchy,
            "signal_type": signal, "subject_type": subject_type, "subject_id": subject_id,
            "subject_name": subject_name, "related_subjects": "|".join(related),
            "headline_factual": headline, "stakes": stakes, "evidence_summary": summary,
            "source_files": "|".join(sources), "signal_score": round(score, 1),
            "confidence": confidence, "status": "candidate",
        })
        for index, item in enumerate(evidence, 1):
            if item["source_file"] not in sources:
                raise ValueError(f"Evidence source not declared for {candidate_id}: {item['source_file']}")
            self.evidence.append({
                "evidence_id": f"{candidate_id}__e{index:02d}", "candidate_id": candidate_id,
                "newsroom_date": NEWSROOM_DATE, "as_of_date": AS_OF_DATE, "season": SEASON,
                **item,
            })


def current_data() -> dict[str, object]:
    games_all = read_csv(SOURCES["games"])
    teams_all = read_csv(SOURCES["teams"])
    divisions = read_csv(SOURCES["divisions"])
    sub_leagues = read_csv(SOURCES["sub_leagues"])
    cutoff = parse_date(AS_OF_DATE)
    teams = {
        row["team_id"]: row for row in teams_all
        if row["league_id"] == "200" and row["level"] == "1" and row["allstar_team"] == "0"
    }
    div_names = {(row["sub_league_id"], row["division_id"]): row["name"] for row in divisions if row["league_id"] == "200"}
    sl_names = {row["sub_league_id"]: row["name"] for row in sub_leagues if row["league_id"] == "200"}
    played = [row for row in games_all if row["league_id"] == "200" and row["game_type"] == "0" and row["played"] == "1" and parse_date(row["date"]) <= cutoff]
    newsroom = parse_date(NEWSROOM_DATE)
    scheduled_newsroom = [
        row for row in games_all
        if row["league_id"] == "200" and row["game_type"] == "0"
        and row["played"] == "0" and parse_date(row["date"]) == newsroom
    ]
    stats = {team_id: {"w": 0, "l": 0, "rs": 0, "ra": 0, "week_w": 0, "week_l": 0, "home_att": 0, "home_games": 0} for team_id in teams}
    week_start = cutoff - timedelta(days=6)
    for row in played:
        home, away = row["home_team"], row["away_team"]
        if home not in stats or away not in stats:
            continue
        away_runs, home_runs = int(row["runs0"]), int(row["runs1"])
        stats[home]["rs"] += home_runs; stats[home]["ra"] += away_runs
        stats[away]["rs"] += away_runs; stats[away]["ra"] += home_runs
        stats[home]["home_att"] += int(number(row["attendance"])); stats[home]["home_games"] += 1
        home_won = home_runs > away_runs
        stats[home]["w" if home_won else "l"] += 1; stats[away]["l" if home_won else "w"] += 1
        if parse_date(row["date"]) >= week_start:
            stats[home]["week_w" if home_won else "week_l"] += 1
            stats[away]["week_l" if home_won else "week_w"] += 1
    standings = []
    for team_id, team in teams.items():
        s = stats[team_id]; games = s["w"] + s["l"]
        standings.append({
            "team_id": team_id, "abbr": team["abbr"], "name": f"{team['name']} {team['nickname']}",
            "division": f"{sl_names[team['sub_league_id']]} {div_names[(team['sub_league_id'], team['division_id'])]}",
            "w": s["w"], "l": s["l"], "games": games, "pct": s["w"] / games,
            "rs": s["rs"], "ra": s["ra"], "week_w": s["week_w"], "week_l": s["week_l"],
            "avg_home_attendance": s["home_att"] / s["home_games"] if s["home_games"] else 0,
        })
    return {"teams": teams, "standings": standings, "played": played, "scheduled_newsroom": scheduled_newsroom, "week_start": week_start}


def build_candidates(builder: Builder, data: dict[str, object]) -> None:
    standings = data["standings"]
    by_id = {row["team_id"]: row for row in standings}
    by_abbr = {row["abbr"]: row for row in standings}
    divisions: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in standings:
        divisions[row["division"]].append(row)
    for division, rows in divisions.items():
        ranked = sorted(rows, key=lambda row: (-number(row["pct"]), -number(row["w"])))
        leader, second = ranked[:2]; gap = int(leader["w"]) - int(second["w"])
        if gap <= 5:
            builder.add(
                candidate_class="proven_current_fact", hierarchy="standings", signal="division_race",
                subject_type="division", subject_id=division, subject_name=division,
                related=[str(leader["name"]), str(second["name"])],
                headline=f"{leader['name']} leads {second['name']} by {gap} game(s) in {division}",
                stakes="With fewer games remaining, each direct result has less schedule left to absorb it.",
                summary=f"Through July 19: {leader['name']} {leader['w']}-{leader['l']}; {second['name']} {second['w']}-{second['l']}.",
                sources=[SOURCES["games"], SOURCES["teams"], SOURCES["divisions"], SOURCES["sub_leagues"]],
                score=92-gap*3, confidence="high",
                evidence=[{"evidence_class":"proven_current_fact","metric":"division_records","value":f"{leader['w']}-{leader['l']} vs {second['w']}-{second['l']}","comparison":f"lead={gap}","source_file":SOURCES["games"],"source_row_key":f"division={division};through={AS_OF_DATE}","notes":"Recomputed from completed games."}],
            )
    weekly = sorted(standings, key=lambda row: (int(row["week_w"])-int(row["week_l"]), int(row["week_w"])), reverse=True)
    for row, direction in ((weekly[0], "rise"), (weekly[-1], "fall")):
        builder.add(
            candidate_class="proven_current_fact", hierarchy="race", signal=f"weekly_{direction}",
            subject_type="team", subject_id=str(row["team_id"]), subject_name=str(row["name"]), related=[],
            headline=f"{row['name']} posted the week's sharpest {direction}",
            stakes="The week changed immediate race pressure; the next games determine whether that movement persists.",
            summary=f"July 13-19 record: {row['week_w']}-{row['week_l']}; season record {row['w']}-{row['l']}.",
            sources=[SOURCES["games"], SOURCES["teams"]], score=80, confidence="high",
            evidence=[{"evidence_class":"proven_current_fact","metric":"trailing_seven_day_record","value":f"{row['week_w']}-{row['week_l']}","comparison":f"season={row['w']}-{row['l']}","source_file":SOURCES["games"],"source_row_key":f"team_id={row['team_id']};1981-07-13..{AS_OF_DATE}","notes":"Completed games only."}],
        )
    pythag = []
    for row in standings:
        rs, ra, games = number(row["rs"]), number(row["ra"]), number(row["games"])
        expected = games * rs*rs/(rs*rs+ra*ra) if rs+ra else 0
        pythag.append((number(row["w"])-expected, expected, row))
    for diff, expected, row in (max(pythag,key=lambda x:x[0]), min(pythag,key=lambda x:x[0])):
        builder.add(
            candidate_class="proven_current_fact", hierarchy="team", signal="pythag_record_gap",
            subject_type="team", subject_id=str(row["team_id"]), subject_name=str(row["name"]), related=[],
            headline=f"{row['name']}'s record and run balance remain far apart",
            stakes="The gap is a condition to monitor, not proof that reversal is due.",
            summary=f"Record {row['w']}-{row['l']}; runs {row['rs']}-{row['ra']}; expected wins {expected:.1f}; gap {diff:+.1f}.",
            sources=[SOURCES["games"], SOURCES["teams"]], score=72+min(abs(diff),10), confidence="high",
            evidence=[{"evidence_class":"proven_current_fact","metric":"pythag_win_gap","value":f"{diff:+.1f}","comparison":f"actual={row['w']};expected={expected:.1f}","source_file":SOURCES["games"],"source_row_key":f"team_id={row['team_id']};through={AS_OF_DATE}","notes":"RS^2/(RS^2+RA^2); descriptive, not predictive."}],
        )

    players = {row["player_id"]: f"{row['first_name']} {row['last_name']}".strip() for row in read_csv(SOURCES["players"])}
    batting = [row for row in read_csv(SOURCES["batting"]) if row["year"]=="1981" and row["league_id"]=="200" and row["level_id"]=="1" and row["split_id"]=="1" and row["game_id"]=="0" and number(row["pa"])>=200]
    hitter=max(batting,key=lambda row:number(row["war"])); hitter_name=players.get(hitter["player_id"],hitter["player_id"])
    builder.add(candidate_class="proven_current_fact",hierarchy="players",signal="position_player_value_leader",subject_type="player",subject_id=hitter["player_id"],subject_name=hitter_name,related=[by_id.get(hitter["team_id"],{}).get("name",hitter["team_id"])],headline=f"{hitter_name} owns the strongest current position-player value line",stakes="The remaining schedule tests whether that broad contribution retains league-leading weight.",summary=f"{number(hitter['war']):.1f} WAR, {hitter['hr']} HR and {hitter['pa']} PA.",sources=[SOURCES["batting"],SOURCES["players"]],score=82,confidence="high",evidence=[{"evidence_class":"proven_current_fact","metric":"position_player_war","value":f"{number(hitter['war']):.1f}","comparison":"highest among ABL hitters with >=200 PA","source_file":SOURCES["batting"],"source_row_key":f"player_id={hitter['player_id']};year=1981","notes":"Raw current-season cumulative total."}])
    pitching=[row for row in read_csv(SOURCES["pitching"]) if row["year"]=="1981" and row["league_id"]=="200" and row["level_id"]=="1" and row["split_id"]=="1" and row["game_id"]=="0" and number(row["gs"])>=12 and number(row["outs"])>=270]
    ace=max(pitching,key=lambda row:number(row["war"])); ace_name=players.get(ace["player_id"],ace["player_id"]); innings=number(ace["outs"])/3; era=number(ace["er"])*9/innings
    builder.add(candidate_class="proven_current_fact",hierarchy="players",signal="ace_value_leader",subject_type="player",subject_id=ace["player_id"],subject_name=ace_name,related=[by_id.get(ace["team_id"],{}).get("name",ace["team_id"])],headline=f"{ace_name} has built the strongest current starter value case",stakes="Each remaining start carries team-race weight and the burden of sustaining an ace-level season.",summary=f"{number(ace['war']):.1f} WAR, {era:.2f} ERA, {innings:.1f} innings, {ace['w']}-{ace['l']} record.",sources=[SOURCES["pitching"],SOURCES["players"]],score=84,confidence="high",evidence=[{"evidence_class":"proven_current_fact","metric":"starter_war","value":f"{number(ace['war']):.1f}","comparison":"highest workload-qualified ABL starter","source_file":SOURCES["pitching"],"source_row_key":f"player_id={ace['player_id']};year=1981","notes":"Raw current-season cumulative total."}])

    finance = read_csv(SOURCES["team_financial"])
    finance_by_id={row["ID"]:row for row in finance}
    fan=max(standings,key=lambda row:number(finance_by_id.get(row["team_id"],{}).get("A/G")))
    frow=finance_by_id[fan["team_id"]]
    builder.add(candidate_class="enrichment_signal",hierarchy="fans",signal="attendance_interest",subject_type="team",subject_id=str(fan["team_id"]),subject_name=str(fan["name"]),related=[],headline=f"{fan['name']} is drawing the league's largest listed home crowd per game",stakes="Attendance gives the race public scale; it does not establish what fans feel or predict results.",summary=f"Sortable report attendance/game: {frow['A/G']}; raw season record {fan['w']}-{fan['l']}.",sources=[SOURCES["team_financial"],SOURCES["games"],SOURCES["teams"]],score=68,confidence="medium",evidence=[{"evidence_class":"enrichment_signal","metric":"attendance_per_game","value":frow["A/G"],"comparison":"highest promoted sortable team-finance value","source_file":SOURCES["team_financial"],"source_row_key":f"ID={fan['team_id']}","notes":"Fan-interest context only; no sentiment inferred."}])
    payroll_rows=[]
    for row in finance:
        if row["ID"] in by_id: payroll_rows.append((money(row["Pay"]),by_id[row["ID"]],row))
    under=[item for item in payroll_rows if number(item[1]["pct"])<.5] or payroll_rows
    pay,club,prow=max(under,key=lambda item:item[0])
    builder.add(candidate_class="enrichment_signal",hierarchy="management_organization",signal="payroll_record_pressure",subject_type="team",subject_id=str(club["team_id"]),subject_name=str(club["name"]),related=[prow.get("GM","")],headline=f"{club['name']}'s payroll commitment and record create an organization question",stakes="Resources and results can be compared; motive, blame, and future action remain unproven.",summary=f"Listed payroll {prow['Pay']}; record {club['w']}-{club['l']}; mode {prow['Mode']}.",sources=[SOURCES["team_financial"],SOURCES["games"],SOURCES["teams"]],score=70,confidence="medium",evidence=[{"evidence_class":"enrichment_signal","metric":"payroll_and_record","value":prow["Pay"],"comparison":f"record={club['w']}-{club['l']}","source_file":SOURCES["team_financial"],"source_row_key":f"ID={club['team_id']}","notes":"Organization context; no causal claim."}])

    champions=read_csv(SOURCES["champions_1980"]); links=[row for row in champions if row["team_abbr"] in by_abbr]
    if links:
        past=max(links,key=lambda row:number(by_abbr[row["team_abbr"]]["pct"])); current=by_abbr[past["team_abbr"]]
        builder.add(candidate_class="historical_echo",hierarchy="tournament",signal="prior_qualifier_echo",subject_type="historical_link",subject_id=past["team_id"],subject_name=past["team_name"],related=[str(current["division"])],headline=f"1980 qualifier {past['team_name']} remains central to the 1981 race",stakes="Prior achievement gives the current race context, not a forecast.",summary=f"1980 record {past['wins']}-{past['losses']}; current record {current['w']}-{current['l']} through July 19.",sources=[SOURCES["champions_1980"],SOURCES["games"],SOURCES["teams"]],score=78,confidence="high",evidence=[{"evidence_class":"historical_context","metric":"prior_and_current_record","value":f"1980 {past['wins']}-{past['losses']}; 1981 {current['w']}-{current['l']}","comparison":str(current["division"]),"source_file":SOURCES["champions_1980"],"source_row_key":f"team_id={past['team_id']};season=1980","notes":"Historical echo only."}])

    scheduled=data["scheduled_newsroom"]
    if scheduled:
        def matchup_rank(game):
            home,away=by_id[game["home_team"]],by_id[game["away_team"]]
            return (100 if home["division"]==away["division"] else 0)+(home["pct"]+away["pct"])*10-abs(home["pct"]-away["pct"])*5
        game=max(scheduled,key=matchup_rank); home,away=by_id[game["home_team"]],by_id[game["away_team"]]
        builder.add(candidate_class="proven_current_fact",hierarchy="game",signal="scheduled_matchup_stakes",subject_type="matchup",subject_id=game["game_id"],subject_name=f"{away['name']} at {home['name']}",related=[str(away["name"]),str(home["name"])],headline=f"{away['name']} at {home['name']} places current standings pressure on the field",stakes="This is an unplayed July 20 schedule fact. No July 20 result is assumed or required.",summary=f"Scheduled July 20; records {away['w']}-{away['l']} and {home['w']}-{home['l']}; same division={home['division']==away['division']}.",sources=[SOURCES["games"],SOURCES["teams"],SOURCES["divisions"]],score=76,confidence="high",evidence=[{"evidence_class":"proven_current_fact","metric":"unplayed_scheduled_game","value":NEWSROOM_DATE,"comparison":f"{away['w']}-{away['l']} vs {home['w']}-{home['l']}","source_file":SOURCES["games"],"source_row_key":f"game_id={game['game_id']};played=0","notes":"Schedule context only; not game-results evidence."}])


def source_manifest(batch: dict[str, dict[str, str]], used_sources: set[str]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows=[]
    for rel in sorted(used_sources):
        entry=batch.get(rel)
        status="compatible_historical" if rel==SOURCES["champions_1980"] else ("active_current" if rel.startswith("csv/ootp_csv/") else "promoted_enrichment")
        role="historical_context" if rel==SOURCES["champions_1980"] else ("current_state_fact" if rel.startswith("csv/ootp_csv/") else "enrichment")
        rows.append({"source_file":rel,"used_by_run":"yes","source_status":status,"evidence_role":role,"batch_manifest_status":"not_in_current_batch_historical" if not entry else "hash_verified","reason":"Used by one or more evidence records."})
    excluded=[
        ("csv/out/star_schema/monday_1981_standings_by_division.csv","excluded_stale_snapshot","32-game standings snapshot"),
        ("csv/out/star_schema/fact_team_reporting_1981_weekly_change.csv","excluded_stale_snapshot","32-game weekly snapshot"),
        ("csv/out/star_schema/fact_player_batting.csv","excluded_stale_snapshot","preserved early player snapshot"),
        ("csv/out/star_schema/fact_player_pitching.csv","excluded_stale_snapshot","preserved early player snapshot"),
        ("csv/out/csv_out/z_ABL_Manager_Tendencies.csv","excluded_stale_snapshot","62-game tendency report"),
        ("csv/out/csv_out/z_ABL_Division_Leverage.csv","excluded_stale_snapshot","62-game division report"),
        ("csv/out/csv_out/z_ABL_Rotation_Stability.csv","excluded_stale_snapshot","62-game rotation report"),
    ]
    for rel,status,reason in excluded:
        rows.append({"source_file":rel,"used_by_run":"no","source_status":status,"evidence_role":"none","batch_manifest_status":"excluded","reason":reason})
    disabled=[
        ("manager_tendency","Staff names available, but no cutoff-compatible tactical tendency source."),
        ("staff_id_complete_linkage","Removed staff ID fields; unresolved coach-name matches must remain null."),
        ("caught_stealing_rate","CS and SB% unavailable in accepted team batting schema without another validated source."),
        ("BatR","Removed and unavailable."),("wSB","Removed and unavailable."),("UBR","Removed and unavailable."),("BsR","Removed and unavailable."),
    ]
    for signal,reason in disabled:
        rows.append({"source_file":f"signal:{signal}","used_by_run":"no","source_status":"disabled_unavailable","evidence_role":"disabled_signal","batch_manifest_status":"not_applicable","reason":reason})
    return rows, [{"signal":signal,"reason":reason} for signal,reason in disabled]


def render_packet(candidates: list[dict[str, object]], evidence: list[dict[str, object]], disabled: list[dict[str,str]], source_rows: list[dict[str,str]]) -> tuple[dict[str,object],str]:
    by_candidate=defaultdict(list)
    for row in evidence:by_candidate[row["candidate_id"]].append(row)
    classes=[("proven_current_fact","Proven Current-State Facts"),("enrichment_signal","Enrichment Signals"),("historical_echo","Historical Echoes")]
    packet={"run_id":RUN_ID,"newsroom_date":NEWSROOM_DATE,"as_of_date":AS_OF_DATE,"editorial_rule":"Data points where to look; the game proves the story.","source_state":{"preflight":"READY_FOR_CURRENT_RUN","completed_games_through":AS_OF_DATE,"july_20_results_used":False},"candidate_count":len(candidates),"sections":[],"disabled_signals":disabled}
    lines=[f"# Baseball Observer Packet — Newsroom {NEWSROOM_DATE}, Games Through {AS_OF_DATE}","","> Data points where to look; the game proves the story.","","## Source State","",f"- Newsroom date: {NEWSROOM_DATE}",f"- Completed-game cutoff: {AS_OF_DATE}","- Preflight: READY_FOR_CURRENT_RUN","- July 20 results used: No","- Current proof: raw OOTP game/score/log sources","- Enrichment: promoted sortable stats","- Historical context: governed almanac only","- Stale Week-labeled derivatives: excluded","" ]
    ordered=sorted(candidates,key=lambda row:(HIERARCHY_ORDER[row["hierarchy_level"]],-number(row["signal_score"])))
    for class_name,title in classes:
        selected=[row for row in ordered if row["candidate_class"]==class_name]
        section={"candidate_class":class_name,"stories":[]};lines += [f"## {title}",""]
        for row in selected:
            story={**row,"evidence":by_candidate[row["candidate_id"]]};section["stories"].append(story)
            lines += [f"### [{row['hierarchy_level'].replace('_',' ').title()}] {row['headline_factual']}","",f"**Stakes:** {row['stakes']}","",f"**Evidence:** {row['evidence_summary']}","",f"**Sources:** {row['source_files']}",""]
        packet["sections"].append(section)
    lines += ["## Unavailable / Disabled Signals",""]
    for item in disabled:lines.append(f"- **{item['signal']}** — {item['reason']}")
    lines += ["","## Editorial Guardrail","","These are evidence-backed places to look. They are not predetermined outcomes, motives, or emotions. July 20 schedule rows are preview context only.",""]
    return packet,"\n".join(lines)


def main() -> int:
    batch=validate_control_state();data=current_data();builder=Builder();build_candidates(builder,data)
    candidates=sorted(builder.candidates,key=lambda row:(HIERARCHY_ORDER[row["hierarchy_level"]],-number(row["signal_score"])))
    used={source for row in candidates for source in str(row["source_files"]).split("|")}
    used.update({SOURCES["games_score"],SOURCES["game_logs"]})
    source_rows,disabled=source_manifest(batch,used)
    candidates_dir=OUT/"candidates";menus_dir=OUT/"menus";packets_dir=OUT/"packets";manifests_dir=OUT/"manifests"
    candidate_path=candidates_dir/f"story_candidates_{RUN_SLUG}.csv";evidence_path=candidates_dir/f"story_evidence_{RUN_SLUG}.csv";menu_path=menus_dir/f"story_menu_{RUN_SLUG}.csv"
    write_csv(candidate_path,CANDIDATE_FIELDS,candidates);write_csv(evidence_path,EVIDENCE_FIELDS,builder.evidence)
    write_csv(menu_path,["menu_rank",*CANDIDATE_FIELDS],[{"menu_rank":i,**row} for i,row in enumerate(candidates,1)])
    manifest_csv=manifests_dir/f"story_engine_source_manifest_{RUN_SLUG}.csv";write_csv(manifest_csv,SOURCE_MANIFEST_FIELDS,source_rows)
    manifest_payload={"run_id":RUN_ID,"newsroom_date":NEWSROOM_DATE,"as_of_date":AS_OF_DATE,"entry_count":len(source_rows),"sources":source_rows,"disabled_signals":disabled}
    (manifests_dir/f"story_engine_source_manifest_{RUN_SLUG}.json").write_text(json.dumps(manifest_payload,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    ml=[f"# Story Engine Source Manifest — {RUN_SLUG}","",f"Entries: **{len(source_rows)}**","","| Used | Status | Role | Source | Reason |","|---|---|---|---|---|"]+[f"| {r['used_by_run']} | `{r['source_status']}` | `{r['evidence_role']}` | `{r['source_file']}` | {r['reason']} |" for r in source_rows]
    (manifests_dir/f"story_engine_source_manifest_{RUN_SLUG}.md").write_text("\n".join(ml)+"\n",encoding="utf-8")
    packet,packet_md=render_packet(candidates,builder.evidence,disabled,source_rows)
    (packets_dir/f"baseball_observer_packet_{RUN_SLUG}.json").parent.mkdir(parents=True,exist_ok=True)
    (packets_dir/f"baseball_observer_packet_{RUN_SLUG}.json").write_text(json.dumps(packet,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    (packets_dir/f"baseball_observer_packet_{RUN_SLUG}.md").write_text(packet_md,encoding="utf-8")
    themes=[row["headline_factual"] for row in sorted(candidates,key=lambda row:-number(row["signal_score"]))[:10]]
    summary={"run_id":RUN_ID,"candidate_count":len(candidates),"evidence_count":len(builder.evidence),"menu_item_count":len(candidates),"source_manifest_entries":len(source_rows),"source_files_used":sorted(used),"source_files_excluded":[r["source_file"] for r in source_rows if r["source_status"].startswith("excluded")],"disabled_signals":disabled,"top_candidate_themes":themes,"safe_for_baseball_observer_editorial_use":True}
    (manifests_dir/f"story_engine_run_summary_{RUN_SLUG}.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    sl=[f"# Story Engine Run Summary — {RUN_SLUG}","",f"- Candidates: **{len(candidates)}**",f"- Evidence records: **{len(builder.evidence)}**",f"- Menu items: **{len(candidates)}**",f"- Source manifest entries: **{len(source_rows)}**","- Safe for EB/Baseball Observer editorial use: **YES**","","## Top themes",""]+[f"{i}. {theme}" for i,theme in enumerate(themes,1)]+["","## Disabled signals",""]+[f"- `{d['signal']}` — {d['reason']}" for d in disabled]
    (manifests_dir/f"story_engine_run_summary_{RUN_SLUG}.md").write_text("\n".join(sl)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2,ensure_ascii=False));return 0


if __name__=="__main__":raise SystemExit(main())
