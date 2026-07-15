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
    park = next(r for r in rows("parks.csv") if r["park_id"] == teams[CIN]["park_id"])
    h2h_hou = sum((g["home_team"] == HOU and int(g["runs1"]) > int(g["runs0"])) or (g["away_team"] == HOU and int(g["runs0"]) > int(g["runs1"])) for g in h2h)

    payload = {
        "title": "Houston at Cincinnati — Thursday Observer Showcase Production Packet",
        "cutoff": "completed games through 1981-07-23 game 1155 (Chicago 15, Dallas 10)",
        "showcase_game": {"game_id": GAME_ID, "date": target["date"], "played": False, "away": "Houston Mavericks", "home": "Cincinnati Cougars"},
        "records": {tid: {"w": wl[tid]["w"], "l": wl[tid]["l"], "runs_for": runs[tid]["for"], "runs_against": runs[tid]["against"], "last_10": f"{recent[tid]}-{10-recent[tid]}"} for tid in (HOU, CIN)},
        "season_series": {"Houston": h2h_hou, "Cincinnati": len(h2h)-h2h_hou, "completed_games": len(h2h)},
        "projected_rotation_slot": projected,
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
             f'First-place Houston ({rec(HOU)}) enters Cincinnati with the Cougars at {rec(CIN)}. The direct stakes are clean: Cincinnati can take another game out of the ABC Central margin; Houston can answer after Cincinnati won Wednesday’s opener 7–1.', '',
             'The game is the story. Treat every number below as pregame context and never as evidence of a Thursday result.', '',
             '## Stakes board', '',
             '| Club | Record | Runs | Run diff. | Last 10 |', '|---|---:|---:|---:|---:|']
    for tid in (HOU, CIN):
        lines.append(f'| {teams[tid]["name"]} {teams[tid]["nickname"]} | {rec(tid)} | {runs[tid]["for"]}-{runs[tid]["against"]} | {runs[tid]["for"]-runs[tid]["against"]:+d} | {recent[tid]}-{10-recent[tid]} |')
    lines += ['', f'- ABC Central margin entering the Showcase: **{abs((wl[HOU]["w"]-wl[HOU]["l"])-(wl[CIN]["w"]-wl[CIN]["l"])) / 2:g} games**.',
              f'- Season series through Wednesday: **Houston {h2h_hou}, Cincinnati {len(h2h)-h2h_hou}**.',
              '- Wednesday, July 22: Cincinnati 7, Houston 1. That result is included; Thursday is not.', '',
              '## Probable-pitcher handling', '',
              f'- Houston projected rotation slot: **{projected[HOU]}**.', f'- Cincinnati projected rotation slot: **{projected[CIN]}**.',
              '- These names come from `projected_starting_pitchers.csv`; label them projected, not confirmed, because game 1161 has starter IDs set to 0.', '',
              '## Bats to frame', '']
    for tid in (HOU, CIN):
        lines += [f'### {teams[tid]["name"]} {teams[tid]["nickname"]}', '', '| Player | AVG | HR | RBI | PA |', '|---|---:|---:|---:|---:|']
        for r in top[tid]: lines.append(f'| {r["name"]} | {r["avg"]:.3f} | {r["hr"]} | {r["rbi"]} | {r["pa"]} |')
        lines.append('')
    lines += ['## Observer open', '',
              f'Baseball matters because the stakes are incredibly high. Houston brings a {rec(HOU)} record and the ABC Central lead into Cincinnati. The Cougars are {rec(CIN)}, and after Cincinnati took Wednesday’s game 7–1, Thursday gives Houston a direct chance to answer before the series moves on. The records set the pressure. This unplayed game supplies the verdict.', '',
              '## Segment beats', '',
              '1. Establish the cutoff: Dallas–Chicago is final; Houston–Cincinnati is not.',
              '2. Put the ABC Central margin and current records on screen.',
              '3. Recap Wednesday only as series context: Cincinnati 7, Houston 1.',
              '4. Introduce projected starters with the explicit “projected” label.',
              '5. Move to the clubs’ leading power/run-production bats.',
              '6. Close on direct opportunity: Cincinnati can narrow the race; Houston can protect its lead.', '',
              '## Guardrails and source ledger', '',
              '- Say **ABL**, never ABLE.', '- Do not state, imply, simulate, or backfill a Thursday result.',
              '- Raw OOTP is official for score, schedule, record, standings, roster, and game status.',
              '- StatsPlus is enrichment only and is not used here to override any official fact.',
              '- `games.csv`: cutoff, schedule, game status, records, run totals, recent form, and season series.',
              '- `games_score.csv` and `game_logs.csv`: complete cross-coverage for all completed games in preflight.',
              '- `players_game_batting.csv` and `players.csv`: player lines and names.',
              '- `projected_starting_pitchers.csv`: projected rotation slot only.',
              '- `teams.csv` and `parks.csv`: identities and venue.', '',
              '**Packet status: SAFE FOR PREGAME PRODUCTION — SHOWCASE RESULT EXCLUDED**', '']
    stem.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"markdown": str(stem.with_suffix('.md')), "json": str(stem.with_suffix('.json')), "records": {"Houston": rec(HOU), "Cincinnati": rec(CIN)}, "showcase_played": False}, indent=2))

if __name__ == "__main__":
    main()
