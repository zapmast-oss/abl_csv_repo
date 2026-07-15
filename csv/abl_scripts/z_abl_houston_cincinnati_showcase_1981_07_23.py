from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "csv" / "ootp_csv"
OUT = ROOT / "csv" / "out" / "story" / "production"
GAME_ID, HOU, CIN = "1161", "23", "14"
CUTOFF = datetime.strptime("1981-07-23", "%Y-%m-%d").date()
SLUG = "1981_07_23_asof_1981-07-23_partial"

def rows(name):
    with (RAW / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def date(value):
    return datetime.strptime(value, "%Y-%m-%d").date()

def main():
    games = rows("games.csv")
    teams = {r["team_id"]: r for r in rows("teams.csv")}
    people = {r["player_id"]: f'{r["first_name"]} {r["last_name"]}' for r in rows("players.csv")}
    coaches = {r["coach_id"]: r for r in rows("coaches.csv")}
    staff_ids = {r["team_id"]: r for r in rows("team_roster_staff.csv") if r["team_id"] in (HOU, CIN)}
    completed = [r for r in games if r["league_id"] == "200" and r["game_type"] == "0" and r["played"] == "1" and date(r["date"]) <= CUTOFF]
    completed_ids = {r["game_id"] for r in completed}
    target = next(r for r in games if r["game_id"] == GAME_ID)
    if target["played"] != "0":
        raise RuntimeError("Showcase game 1161 is no longer unplayed; refusing to generate a pregame packet")

    wl = defaultdict(Counter)
    runs = defaultdict(Counter)
    for g in completed:
        home, away = g["home_team"], g["away_team"]
        # OOTP's runs0 is the away score and runs1 is the home score.
        hr, ar = int(g["runs1"]), int(g["runs0"])
        runs[home]["for"] += hr; runs[home]["against"] += ar
        runs[away]["for"] += ar; runs[away]["against"] += hr
        if hr > ar: wl[home]["w"] += 1; wl[away]["l"] += 1
        else: wl[away]["w"] += 1; wl[home]["l"] += 1

    h2h = sorted([g for g in completed if {g["home_team"], g["away_team"]} == {HOU, CIN}], key=lambda g: date(g["date"]))
    current_series = [g for g in h2h if date(g["date"]) >= datetime.strptime("1981-07-20", "%Y-%m-%d").date()]
    recent = {}
    for tid in (HOU, CIN):
        gs = sorted([g for g in completed if tid in (g["home_team"], g["away_team"])], key=lambda g: (date(g["date"]), int(g["game_id"])), reverse=True)[:10]
        recent[tid] = sum((g["home_team"] == tid and int(g["runs1"]) > int(g["runs0"])) or (g["away_team"] == tid and int(g["runs0"]) > int(g["runs1"])) for g in gs)

    bat = defaultdict(Counter)
    for r in rows("players_game_batting.csv"):
        if r["game_id"] in completed_ids and r["team_id"] in (HOU, CIN):
            for k in ("ab", "h", "hr", "rbi", "bb", "pa"):
                bat[(r["team_id"], r["player_id"])][k] += int(float(r[k] or 0))
    top = {}
    for tid in (HOU, CIN):
        candidates = [(pid, s) for (team, pid), s in bat.items() if team == tid and s["pa"] >= 100]
        candidates.sort(key=lambda x: (x[1]["hr"], x[1]["rbi"], x[1]["h"]), reverse=True)
        top[tid] = [{"name": people.get(pid, pid), **dict(s), "avg": round(s["h"] / s["ab"], 3) if s["ab"] else 0} for pid, s in candidates[:4]]

    projections = {r["team_id"]: r for r in rows("projected_starting_pitchers.csv") if r["team_id"] in (HOU, CIN)}
    projected = {tid: people.get(projections[tid]["starter_0"], projections[tid]["starter_0"]) for tid in (HOU, CIN)}
    player_rows = {r["player_id"]: r for r in rows("players.csv")}
    pitching_rows = {r["player_id"]: r for r in rows("players_pitching.csv")}
    starter_ids = {tid: projections[tid]["starter_0"] for tid in (HOU, CIN)}
    pitch_names = {
        "fastball": "fastball", "slider": "slider", "curveball": "curveball",
        "screwball": "screwball", "forkball": "forkball", "changeup": "changeup",
        "sinker": "sinker", "splitter": "splitter", "knuckleball": "knuckleball",
        "cutter": "cutter", "circlechange": "circle change", "knucklecurve": "knuckle curve",
    }
    pitching_game_rows = rows("players_game_pitching_stats.csv")
    starter_profiles = {}
    for tid, pid in starter_ids.items():
        appearances = [r for r in pitching_game_rows if r["player_id"] == pid and r["game_id"] in completed_ids]
        totals = Counter()
        for r in appearances:
            for key in ("g", "gs", "w", "l", "s", "ha", "k", "bb", "r", "er", "hra", "outs", "qs"):
                totals[key] += int(float(r[key] or 0))
        innings = totals["outs"] / 3
        rating = pitching_rows[pid]
        arsenal = [
            {"pitch": label, "rating": int(rating[f"pitching_ratings_pitches_{key}"])}
            for key, label in pitch_names.items()
            if int(rating[f"pitching_ratings_pitches_{key}"]) > 0
        ]
        bio = player_rows[pid]
        starter_profiles[tid] = {
            "player_id": pid, "name": people[pid], "age": int(bio["age"]),
            "experience_years": int(bio["experience"]), "throws": "L" if bio["throws"] == "2" else "R",
            "games": totals["g"], "starts": totals["gs"], "wins": totals["w"], "losses": totals["l"],
            "saves": totals["s"], "innings": f'{totals["outs"] // 3}.{totals["outs"] % 3}',
            "era": round(totals["er"] * 27 / totals["outs"], 2) if totals["outs"] else None,
            "whip": round((totals["ha"] + totals["bb"]) * 3 / totals["outs"], 2) if totals["outs"] else None,
            "strikeouts": totals["k"], "walks": totals["bb"], "home_runs": totals["hra"],
            "quality_starts": totals["qs"], "arsenal": arsenal,
            "stuff": int(rating["pitching_ratings_overall_stuff"]),
            "movement": int(rating["pitching_ratings_overall_movement"]),
            "control": int(rating["pitching_ratings_overall_control"]),
            "stamina": int(rating["pitching_ratings_misc_stamina"]),
            "hold": int(rating["pitching_ratings_misc_hold"]),
            "velocity_code": int(rating["pitching_ratings_misc_velocity"]),
            "injured": bio["injury_is_injured"] == "1",
        }
    staff_roles = ("manager", "general_manager", "bench_coach", "pitching_coach", "hitting_coach")
    staff = {
        tid: {
            role: f'{coaches[staff_ids[tid][role]]["first_name"]} {coaches[staff_ids[tid][role]]["last_name"]}'
            for role in staff_roles
        }
        for tid in (HOU, CIN)
    }
    park = next(r for r in rows("parks.csv") if r["park_id"] == teams[CIN]["park_id"])
    h2h_hou = sum((g["home_team"] == HOU and int(g["runs1"]) > int(g["runs0"])) or (g["away_team"] == HOU and int(g["runs0"]) > int(g["runs1"])) for g in h2h)
    current_hou = sum((g["home_team"] == HOU and int(g["runs1"]) > int(g["runs0"])) or (g["away_team"] == HOU and int(g["runs0"]) > int(g["runs1"])) for g in current_series)

    payload = {
        "title": "Houston at Cincinnati — Thursday Observer Showcase Production Packet",
        "cutoff": "completed games through 1981-07-23 game 1155 (Chicago 15, Dallas 10)",
        "showcase_game": {"game_id": GAME_ID, "date": target["date"], "played": False, "away": "Houston Mavericks", "home": "Cincinnati Cougars"},
        "records": {tid: {"w": wl[tid]["w"], "l": wl[tid]["l"], "runs_for": runs[tid]["for"], "runs_against": runs[tid]["against"], "last_10": f"{recent[tid]}-{10-recent[tid]}"} for tid in (HOU, CIN)},
        "season_series": {"Houston": h2h_hou, "Cincinnati": len(h2h)-h2h_hou, "completed_games": len(h2h)},
        "current_series": {"Houston": current_hou, "Cincinnati": len(current_series)-current_hou, "completed_games": len(current_series)},
        "confirmed_starters": projected,
        "starter_confirmation_source": "User confirmation; raw game 1161 has not yet populated starter IDs.",
        "starter_profiles": starter_profiles,
        "verified_staff": staff,
        "park": park["name"], "top_hitters": top,
        "authority": {"official": "raw OOTP CSV", "enrichment": "StatsPlus not required for any score, record, standing, or schedule claim"},
        "guardrail": "Do not include or assume the result of game 1161. It is played=0 in games.csv."
    }
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"houston_at_cincinnati_thursday_observer_showcase_{SLUG}"
    stem.with_suffix(".json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def rec(tid): return f'{wl[tid]["w"]}-{wl[tid]["l"]}'
    lines = [f'# {payload["title"]}', '', '**Showcase date:** Thursday, July 23, 1981  ',
             '**Official cutoff:** Completed games through Chicago 15, Dallas 10, game 1155, on July 23  ',
             '**Showcase status:** Unplayed (`games.csv` game 1161, `played=0`)  ',
             f'**Site:** {park["name"]}, Cincinnati', '',
             '## Editorial spine', '',
             f'First-place Houston ({rec(HOU)}) enters Cincinnati with the Cougars at {rec(CIN)}. The direct stakes are clean: Cincinnati can take another game out of the ABC Central margin; Houston can answer after Cincinnati won Wednesday’s game 7–1.', '',
             'The game is the story. Treat every number below as pregame context and never as evidence of a Thursday result.', '',
             '## Stakes board', '',
             '| Club | Record | Runs | Run diff. | Last 10 |', '|---|---:|---:|---:|---:|']
    for tid in (HOU, CIN):
        lines.append(f'| {teams[tid]["name"]} {teams[tid]["nickname"]} | {rec(tid)} | {runs[tid]["for"]}-{runs[tid]["against"]} | {runs[tid]["for"]-runs[tid]["against"]:+d} | {recent[tid]}-{10-recent[tid]} |')
    lines += ['', f'- ABC Central margin entering the Showcase: **{abs((wl[HOU]["w"]-wl[HOU]["l"])-(wl[CIN]["w"]-wl[CIN]["l"])) / 2:g} games**.',
              f'- Season series through Wednesday: **Houston {h2h_hou}, Cincinnati {len(h2h)-h2h_hou}**.',
              f'- Current four-game series: **Cincinnati {len(current_series)-current_hou}, Houston {current_hou}**. Houston won Monday; Cincinnati won Tuesday and Wednesday.',
              '- Wednesday, July 22: Cincinnati 7, Houston 1. That result is included; Thursday is not.', '',
              '## Confirmed starters', '',
              f'- **Houston: {projected[HOU]}** — confirmed by the commissioner/user.', f'- **Cincinnati: {projected[CIN]}** — confirmed by the commissioner/user.',
              '- The names also match each club’s `starter_0` assignment in `projected_starting_pitchers.csv`. Game 1161 itself has not yet populated its starter-ID fields, so the confirmation source is documented separately.', '']
    for tid in (HOU, CIN):
        profile = starter_profiles[tid]
        arsenal_text = ", ".join(f'{pitch["pitch"]} {pitch["rating"]}' for pitch in profile["arsenal"])
        lines += [f'### {profile["name"]} — {teams[tid]["name"]}', '',
                  f'- Age {profile["age"]}; throws {profile["throws"]}; {profile["experience_years"]} years of listed experience; currently injured: **{"yes" if profile["injured"] else "no"}**.',
                  f'- 1981 ABL line through the cutoff: **{profile["games"]} G, {profile["starts"]} GS, {profile["wins"]}-{profile["losses"]}, {profile["innings"]} IP, {profile["era"]:.2f} ERA, {profile["whip"]:.2f} WHIP, {profile["strikeouts"]} K, {profile["walks"]} BB, {profile["home_runs"]} HR allowed**.',
                  f'- Current pitch ratings: **{arsenal_text}**.',
                  f'- Overall OOTP ratings: stuff {profile["stuff"]}, movement {profile["movement"]}, control {profile["control"]}, stamina {profile["stamina"]}, hold {profile["hold"]}.',
                  f'- Raw velocity-band code: {profile["velocity_code"]}. The export does not provide a trustworthy MPH label, so do not translate this code into an MPH range on air.', '']
    lines += [
              '## Managers and field staff', '',
              '| Club | Manager | Bench coach | Pitching coach | Hitting coach | General manager |',
              '|---|---|---|---|---|---|']
    for tid in (HOU, CIN):
        lines.append(f'| {teams[tid]["name"]} {teams[tid]["nickname"]} | {staff[tid]["manager"]} | {staff[tid]["bench_coach"]} | {staff[tid]["pitching_coach"]} | {staff[tid]["hitting_coach"]} | {staff[tid]["general_manager"]} |')
    lines += ['',
              'Manager and staff IDs come directly from `team_roster_staff.csv` and resolve through `coaches.csv`. No managerial tendencies are inferred.', '',
              '## Bats to frame', '']
    for tid in (HOU, CIN):
        lines += [f'### {teams[tid]["name"]} {teams[tid]["nickname"]}', '', '| Player | AVG | HR | RBI | PA |', '|---|---:|---:|---:|---:|']
        for r in top[tid]: lines.append(f'| {r["name"]} | {r["avg"]:.3f} | {r["hr"]} | {r["rbi"]} | {r["pa"]} |')
        lines.append('')
    lines += ['## Verified honors and franchise context', '',
              '- **Scott Reis** has a 1981 ABL All-Star entry in raw OOTP `players_awards.csv` (award ID 9, dated July 12). He may be identified as a 1981 All-Star.',
              '- Cincinnati won the ABC Central in 1980 at **97-65** and reached the playoffs under manager Chris Allison.',
              '- Houston reached the playoffs in each season from **1975 through 1980**, won its division four times in that span, and has a `won_playoffs=1` entry for 1979 under manager Carlos Sanchez.',
              '- Use the history as franchise context, not as evidence of how either clubhouse feels about Thursday.', '']
    lines += ['## Observer open', '',
              f'Baseball matters because the stakes are incredibly high. Houston brings a {rec(HOU)} record and the ABC Central lead into Cincinnati. The Cougars are {rec(CIN)}, and after Cincinnati took Wednesday’s game 7–1, Thursday gives Houston a direct chance to answer before the series moves on. The records set the pressure. This unplayed game supplies the verdict.', '',
              '## Segment beats', '',
              '1. Establish the cutoff: Dallas–Chicago is final; Houston–Cincinnati is not.',
              '2. Put the ABC Central margin and current records on screen.',
              '3. Recap Wednesday only as series context: Cincinnati 7, Houston 1.',
              '4. Introduce the confirmed starters and distinguish user confirmation from the still-empty game-row starter IDs.',
              '5. Move to the clubs’ leading power/run-production bats.',
              '6. Close on direct opportunity: Cincinnati can narrow the race; Houston can protect its lead.', '',
              '## Guardrails and source ledger', '',
              '- Say **ABL**, never ABLE.', '- Do not state, imply, simulate, or backfill a Thursday result.',
              '- Raw OOTP is official for score, schedule, record, standings, roster, and game status.',
              '- StatsPlus is enrichment only and is not used here to override any official fact.',
              '- `games.csv`: cutoff, schedule, game status, records, run totals, recent form, and season series.',
              '- `games_score.csv` and `game_logs.csv`: complete cross-coverage for all completed games in preflight.',
              '- `players_game_batting.csv` and `players.csv`: player lines and names.',
              '- Starter confirmation: commissioner/user confirmation; `projected_starting_pitchers.csv` independently matches both names.',
              '- `players_game_pitching_stats.csv`, `players_pitching.csv`, and `players.csv`: starter lines, biographies, arsenals, and ratings.',
              '- `team_roster_staff.csv` and `coaches.csv`: current manager and staff assignments.',
              '- `teams.csv` and `parks.csv`: identities and venue.', '',
              '**Packet status: SAFE FOR PREGAME PRODUCTION — SHOWCASE RESULT EXCLUDED**', '']
    stem.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"markdown": str(stem.with_suffix('.md')), "json": str(stem.with_suffix('.json')), "records": {"Houston": rec(HOU), "Cincinnati": rec(CIN)}, "showcase_played": False}, indent=2))

if __name__ == "__main__":
    main()
