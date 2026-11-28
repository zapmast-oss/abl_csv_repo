"""The Baseball Observer desk. Compare races using the games in the admitted snapshot."""
from collections import defaultdict
from datetime import timedelta
from fractions import Fraction

from abl_newsroom_beats import day, game_records, record, rows, wl


def _gap(first, second):
    return ((first["wins"] - second["wins"]) + (second["losses"] - first["losses"])) / 2


def _margin(value):
    return f"{value:g} {'game' if value == 1 else 'games'}"


def _pretty(iso):
    date = day(iso)
    return f"{date:%B} {date.day}"


def _number(n):
    return ("zero","one","two","three","four","five","six","seven","eight","nine")[n] if 0<=n<10 else str(n)


def prepare_observer(raw, cutoff, beats, new, revised, candidates, east, slate):
    clubs = {int(t["team_id"]): t for t in rows(raw, "teams") if int(t["team_id"]) in beats}
    name = {t: row["name"] for t, row in clubs.items()}
    gamebook = game_records(raw, day(cutoff), clubs)
    games = sorted(gamebook.values(), key=lambda g: (g["date"], g["game_id"]))
    calendar = defaultdict(list)
    divisions = defaultdict(list)
    for t, b in beats.items():
        divisions[b["division"]].append(t)
    for g in games:
        for side in ("away_id", "home_id"):
            calendar[g[side]].append(g)
    lines = {t: record(calendar[t], t) for t in clubs}
    for t, line in lines.items():
        if (line["wins"], line["losses"]) != (beats[t]["wins"], beats[t]["losses"]):
            raise ValueError(f"Observer and beat records disagree for club {t}")
    def ranked(pair):
        return sorted(pair, key=lambda t: (-Fraction(lines[t]["wins"], max(1,lines[t]["wins"]+lines[t]["losses"])), t))
    future = sorted((day(g["date"]), int(g["time"]), int(g["game_id"]), int(g["away_team"]), int(g["home_team"]))
                    for g in rows(raw,"games") if g["league_id"] == "200" and g["game_type"] == "0"
                    and g["played"] == "0" and day(g["date"]).year == day(cutoff).year and day(g["date"]) >= day(cutoff))
    pairs = [ranked(members)[:2] for members in divisions.values()]
    first, second = min(pairs, key=lambda p: (_gap(lines[p[0]], lines[p[1]]), p))
    division = beats[first]["division"].replace("Western Division","West").replace("Eastern Division","East").replace("Central Division","Central")
    evidence = [f"games.csv and teams.csv — league 200, played regular-season games through {cutoff}. Standings, recent form and series records use the participating clubs and the final scores."]
    sections = []
    if new:
        meeting = next((g for g in future if {g[3],g[4]} == {first,second}), None)
        gap = _gap(lines[first],lines[second])
        equal = wl(lines[first]) == wl(lines[second])
        if equal:
            lead = f"{name[first]} and {name[second]} are tied at {wl(lines[first])} in the {division}."
        else:
            lead = (f"In the {division}, {name[first]} ({wl(lines[first])}) and {name[second]} ({wl(lines[second])}) "
                    f"are separated by {_margin(abs(gap))} in the standings.")
        head = [g for g in calendar[first] if second in (g["away_id"], g["home_id"])]
        week=(day(cutoff)-timedelta(days=7)).isoformat()
        past=[record([g for g in calendar[t] if g["date"]<=week],t) for t in (first,second)]
        then=_gap(*past)
        if equal and then:
            lead+=f" A week ago {name[first if then<0 else second]} trailed the other club by {_margin(abs(then))}."
        if head:
            matchup = record(head, first)
            lead += f" {name[first]} has gone {wl(matchup)} head-to-head."
        if meeting:
            lead += f" The schedule puts {name[meeting[3]]} at {name[meeting[4]]} on {_pretty(meeting[0])}."
            evidence.append(f"games.csv — game_id={meeting[2]}, date={meeting[0]}, played=0, visitor={meeting[3]}, home={meeting[4]}. Schedule only.")
        sections.append((f"Where the race is closest", [lead]))
        # A big final score can dominate a short-window run total. Demonstrate its actual weight.
        leaders = {t for division_members in divisions.values() for t in ranked(division_members)
                   if t in beats[t]["division_leaders"]}
        noteworthy = [g for g in games if g["game_id"] in new and abs(g["runs_home"]-g["runs_away"]) >= 8
                      and (g["away_id"] in leaders or g["home_id"] in leaders)]
        if noteworthy:
            g = max(noteworthy,key=lambda row: abs(row["runs_home"]-row["runs_away"]))
            winner, loser = (g["home_id"],g["away_id"]) if g["runs_home"] > g["runs_away"] else (g["away_id"],g["home_id"])
            recent = [row for row in calendar[winner] if row["date"] <= g["date"]][-7:]
            others = [row for row in recent if row["game_id"] != g["game_id"]]
            if len(others) and g in recent:
                line = record(others,winner)
                sections[0][1].append(
                    f"{name[winner]} beat {name[loser]} {max(g['runs_home'],g['runs_away'])}–{min(g['runs_home'],g['runs_away'])} on {_pretty(g['date'])}. "
                    f"Take that game out of {name[winner]}’s last {_number(len(recent))} through that date and the other {_number(len(others))} left the club "
                    f"{wl(line)}, with {line['runs']} runs scored and {line['allowed']} allowed. One game can change the look of a whole week’s run totals.")
                evidence.append(f"games.csv — game_id={g['game_id']}, team_id={winner}. Comparison game IDs "+", ".join(str(row["game_id"]) for row in others)+".")
        cross = [g for g in games if g["game_id"] in new and g["home_id"] in leaders and g["away_id"] in leaders
                 and beats[g["home_id"]]["division"] != beats[g["away_id"]]["division"]]
        if cross:
            g = min(cross,key=lambda g:(abs(g["runs_home"]-g["runs_away"]),-g["game_id"]))
            a,h = g["away_id"],g["home_id"]
            series = []
            for row in reversed([r for r in games if (a in (r["home_id"],r["away_id"]) or h in (r["home_id"],r["away_id"])) and r["date"] <= g["date"]]):
                if {row["home_id"],row["away_id"]} != {a,h}:
                    break
                series.append(row)
            comparison = []
            for t in (a,h):
                opponent = next(o for o in ranked(divisions[beats[t]["division"]]) if o != t)
                comparison.append(f"{name[t]} and {name[opponent]} are {_margin(abs(_gap(lines[t],lines[opponent])))} apart")
            points = f"The arithmetic matters differently in the two divisions. {comparison[0]}, while {comparison[1]}."
            win, loss = (h,a) if g["runs_home"] > g["runs_away"] else (a,h)
            series_note = (f" Across their {_number(len(series))} consecutive meetings, {name[win]} went {wl(record(series,win))}."
                           if g in series and len(series)>1 else "")
            sections.append(("Two division leaders, two different margins",[
                f"{name[win]} beat {name[loss]} {max(g['runs_home'],g['runs_away'])}–{min(g['runs_home'],g['runs_away'])} on {_pretty(g['date'])}."
                +series_note+" "+points]))
            evidence.append(f"games.csv — game_id={g['game_id']}; meeting IDs "+", ".join(str(x["game_id"]) for x in series)+".")
    if east and east.get("division") and new and "NBC Eastern Division" in divisions:
        top, challenger = ranked(divisions["NBC Eastern Division"])[:2]
        seasons = sorted((int(r["team_id"]),int(r["year"])) for r in rows(raw,"team_history")
                         if r["league_id"] == "200" and r["made_playoffs"] == "1" and int(r["year"]) < day(cutoff).year)
        visits = {t:[year for team,year in seasons if team==t] for t in (top,challenger)}
        lead = (f"In the NBC East, {name[top]} is {wl(lines[top])} and "
                f"{name[challenger]} is {wl(lines[challenger])}.")
        if len(visits[top])==1 and len(visits[challenger])>1:
            lead += (f" {name[top]} first reached the tournament in {visits[top][0]}. {name[challenger]} has already been there "
                     f"{_number(len(visits[challenger]))} times. A return this fall would give {name[top]} consecutive "
                     "appearances for the first time." if visits[top][0]==day(cutoff).year-1 else
                     f" Their histories include one tournament appearance for {name[top]} and {len(visits[challenger])} for {name[challenger]}.")
        lead += f" For {name[challenger]}, start with where the runs have gone. The club has scored {lines[challenger]['runs']} and allowed {lines[challenger]['allowed']}."
        sections.append(("The view from the NBC East", [lead]))
        evidence.append(f"team_history.csv — league_id=200, team_id={top} and {challenger}, made_playoffs=1 before {day(cutoff).year}; games.csv, final runs for both clubs through {cutoff}.")
    if revised:
        details = [c["evidence_summary"] for c in candidates if c["signal_type"] == "revised_game"]
        sections.append(("Corrections to the record",details))
        evidence.extend(f"games.csv, players_game_batting.csv, players_game_pitching_stats.csv and games_score.csv — compare game_id={gid} in the two verified exports." for gid in sorted(revised))
    if not new and not revised:
        personnel = [c for c in candidates if c["signal_type"] == "personnel_change"]
        sections.append(("At the desk", [c["evidence_summary"] for c in personnel] or
            ["The latest export adds no final scores. The previous briefing and its open assignments remain the reference for the next games."]))
    if new:
        pitches = [p for team in slate["teams"].values() for p in team["pitches"] if p["eligible"] and p["category"] in ("organization","individual")]
        if pitches:
            p = max(pitches,key=lambda p:p["score"])
            sections.append(("One player to follow",[p["observation"]+" "+p["question"]]))
            evidence.extend(p["evidence"])
    title = (f"{name[first]} visits {name[second]} with the {division} tied" if new and equal and meeting and meeting[3]==first and meeting[0]==day(cutoff)+timedelta(days=1)
             else f"{name[first]} and {name[second]} set the terms in the {division}" if new else "The Baseball Observer | Desk update")
    return dict(title=title, sections=sections, evidence=list(dict.fromkeys(evidence)),
                trigger=cutoff, key="observer", team_ids=[], kind="observer")
