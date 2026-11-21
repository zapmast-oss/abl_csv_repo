"""Build a franchise editorial board from one verified newsroom edition and its predecessor."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import timedelta
from fractions import Fraction
from pathlib import Path

from abl_newsroom_beats import day, game_records, innings, next_opponent, record, rows, wl


REPO = Path(__file__).resolve().parents[2]
BATTING = ("ab", "pa", "h", "hr", "rbi", "gs")
PITCHING = ("outs", "er", "k", "gs", "g", "s")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()[:24]


def gap(better, other):
    return ((better["wins"] - other["wins"]) + (other["losses"] - better["losses"])) / 2


def race_order(members, records):
    def pct(tid):
        line = records[tid]
        return Fraction(line["wins"], max(1, line["wins"] + line["losses"]))
    order = sorted(members, key=lambda tid: (-pct(tid), tid))
    return order, [tid for tid in order if pct(tid) == pct(order[0])]


def signed(n):
    return f"{n:+g}"


def count(n, singular, plural=None):
    return f"{n:g} {singular if n == 1 else plural or singular+'s'}"


def select_games(ids):
    return "games.csv game_id=" + ",".join(str(x) for x in ids) + " (ABL completed regular season)"


def names(tids, clubs):
    return " and ".join(clubs[t]["name"] for t in tids)


def archive_board(clubs, cutoff):
    """Look back at specific dated desk investigations. A planning row is not a publication receipt."""
    prior = defaultdict(list)
    for path in sorted((REPO / "csv/out/story/editorial").glob("observer_editorial_board_*_asof_*.csv")):
        match = re.search(r"asof_(\d{4}-\d{2}-\d{2})", path.stem)
        issue = re.search(r"board_(\d{4})_(\d{2})_(\d{2})", path.stem)
        if not match or not issue or day(match[1]) >= cutoff or day("-".join(issue.groups())) > cutoff:
            continue
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                for tid, t in clubs.items():
                    if t["name"] + " " + t["nickname"] not in row["candidate_title"]:
                        continue
                    prior[tid].append(dict(cutoff=match[1], kind=row["candidate_type"],
                                           title=row["candidate_title"], fact=row["evidence_summary"],
                                           source=f"{path.relative_to(REPO).as_posix()} candidate_id={row['candidate_id']}"))
    return prior


def game_lines(raw, games, year):
    """Keep dated appearances. A player can represent more than one club during a season."""
    by_player = {"bat": defaultdict(list), "pit": defaultdict(list)}
    by_club = {"bat": defaultdict(list), "pit": defaultdict(list)}
    for kind, file, columns in [("bat", "players_game_batting", BATTING),
                                ("pit", "players_game_pitching_stats", PITCHING)]:
        seen = set()
        for row in rows(raw, file):
            if row["league_id"] != "200" or row["level_id"] != "1" or int(row["year"]) != year:
                continue
            gid = int(row["game_id"])
            game = games.get(gid)
            if game is None:
                continue
            tid, pid = int(row["team_id"]), int(row["player_id"])
            key = (tid, pid, gid)
            if tid not in (game["home_id"], game["away_id"]) or key in seen:
                raise ValueError(f"Unresolved participant in {file}.csv at team, player, game {key}")
            seen.add(key)
            entry = (game["date"], gid, tid, pid, {c: int(row[c]) for c in columns})
            by_player[kind][pid].append(entry)
            by_club[kind][tid].append(entry)
    return by_player, by_club


def totals(entries):
    result = Counter()
    for row in entries:
        result.update(row[4])
    return result


def pitch(key, category, title, observation, significance, question, evidence, trigger, importance,
          stakes, history=0, *, related=(), qualified=True, direction=None, interpretation=None):
    return dict(key=key, category=category, title=title, observation=observation,
                significance=significance, question=question, evidence=evidence,
                trigger=trigger, importance=importance, stakes=stakes, historical_weight=history,
                related_team_ids=list(related), qualified=qualified, direction=direction,
                interpretation=interpretation or significance)


def division_pitch(tid, club, group, leagues, now, before, boundary, cutoff, views, archives, history):
    order, leaders = race_order(group, now)
    _, old_leaders = race_order(group, before)
    rivals = [t for t in order if t != tid]
    closest = rivals[0]
    lead = tid in leaders
    margin = gap(now[tid], now[closest]) if lead else -gap(now[order[0]], now[tid])
    old_diff = gap(before[tid], before[closest])
    compare = gap(now[tid], now[closest]) - old_diff
    record_text = wl(now[tid])
    if lead and len(leaders) > 1:
        title = f"{club} shares first place with {names([t for t in leaders if t != tid], leagues)}"
        position = f"shares the division lead with {names([t for t in leaders if t != tid], leagues)} by winning percentage"
    elif lead:
        title = f"{club} leads {names([closest], leagues)} by {count(margin, 'game')}"
        position = f"leads {names([closest], leagues)} by {margin:g} {'game' if margin == 1 else 'games'}"
    else:
        title = f"{club} trails {names(leaders, leagues)} by {count(abs(margin), 'game')}"
        position = f"trails {names(leaders, leagues)} by {abs(margin):g} {'game' if abs(margin) == 1 else 'games'}"
    statement = f"At {record_text}, {club} {position} in the {leagues[tid]['division']}."
    previous_play = all(before[t]["wins"] + before[t]["losses"] for t in group)
    changes = []
    if previous_play:
        then = f"{club} was {wl(before[tid])} and {leagues[closest]['name']} was {wl(before[closest])} through {boundary}."
        interval = (f"{club} gained {abs(compare):g} {'game' if abs(compare)==1 else 'games'} on {leagues[closest]['name']}."
                    if compare > 0 else (f"{club} lost {abs(compare):g} {'game' if abs(compare)==1 else 'games'} relative to {leagues[closest]['name']}."
                                         if compare < 0 else f"The distance between those two clubs held steady."))
        changes.append(then + " " + interval)
    if previous_play and set(leaders) != set(old_leaders):
        changes.append(f"{names(old_leaders, leagues)} occupied first place through {boundary}.")
        if tid not in old_leaders and tid in leaders:
            title = f"{club} {'joins '+names([t for t in leaders if t!=tid],leagues)+' atop' if len(leaders)>1 else 'takes the lead in'} the {leagues[tid]['division']}"
    reference = (f"games.csv league_id=200 game_type=0 played=1 division team IDs "
                 f"{','.join(map(str, sorted(group)))} through {boundary} and {cutoff}")
    evidence = [reference, f"teams.csv, divisions.csv, sub_leagues.csv league_id=200 team_id={tid}"]
    desk = next((r for r in reversed(archives) if r["kind"] == "division_race"), None)
    if desk:
        then = [(leagues[t]["name"],wl(record([g for g in views[t] if g["date"] <= desk["cutoff"]],t))) for t in group
                if leagues[t]["name"]+" "+leagues[t]["nickname"] in desk["title"]]
        if len(then)==2:
            changes.append(f"The two teams on the {day(desk['cutoff']):%B} {day(desk['cutoff']).day} race board were "
                           f"{then[0][0]} ({then[0][1]}) and {then[1][0]} ({then[1][1]}).")
            evidence.append(f"games.csv league_id=200 game_type=0 played=1 division team IDs {','.join(map(str,group))} through {desk['cutoff']}")
        evidence.append(desk["source"])
    context = history["context"]
    if history["berths"] is not None and (history["important"] or abs(margin) <= 3):
        changes.append(context)
        evidence.extend(history["evidence"])
    proximity = 16 if abs(margin) <= 3 else 13 if abs(margin) <= 6 else 7
    material = previous_play and (set(leaders) != set(old_leaders) and tid in {*old_leaders, *leaders} or abs(compare) >= 2)
    meaningful = (abs(margin) <= 3 or material and abs(margin) <= 7)
    band = 0 if margin == 0 else next((i for i in [1,3,5,8,12] if abs(margin) <= i), 15)
    facts = statement + " " + " ".join(changes)
    if previous_play and compare:
        impact = (f"The last seven days changed the gap by {count(abs(compare),'game')}. "
                  f"One head-to-head win for the trailing side would narrow it by a full game."
                  if margin else f"Those {count(abs(compare),'game')} of movement took away the earlier separation. The next decision by either club can break the tie.")
    else:
        impact = ("One head-to-head win separates two clubs tied at the top." if not margin else
                  f"A single head-to-head win for the trailing side would narrow the gap by a full game.")
    p = pitch(f"{tid}:{cutoff.year}:race", "competition", title, facts, impact,
                 f"Will {club} gain ground in its next division games?", evidence,
                 [sorted(leaders), closest, band, bool(lead)], 21 if material else 17, proximity,
                 7 if desk or history["important"] else 4, related=sorted(set([tid, closest, *leaders])),
                 qualified=meaningful, direction=("co-leader" if len(leaders)>1 else "leader") if lead else "chasing")
    # Exact mirror-image standings (e.g. both clubs in the same tie) give one article slot.
    p["event_key"] = "division:"+leagues[tid]["division"]+":"+str(sorted([tid,closest]))
    p["trigger"] = [sorted(leaders),sorted([tid,closest]),band]
    return p


def historical_context(tid, club, history_rows, results, year):
    seasons = sorted(history_rows)
    if not seasons:
        return dict(context=f"A prior tournament record for {club} is unavailable.", berths=None,
                    important=False, evidence=[])
    berths = [y for y in seasons if history_rows[y]["made_playoffs"] == "1"]
    titles = [y for y in seasons if history_rows[y]["won_playoffs"] == "1"]
    scope = f"{seasons[0]}–{seasons[-1]}"
    last = results.get(seasons[-1])
    if len(berths) == 1:
        context = f"{club} reached the tournament once in {scope}, in {berths[0]}."
    elif not berths:
        context = f"{club} went without a tournament berth in the {len(seasons)} recorded seasons from {scope}."
    else:
        context = f"{club} reached the tournament {len(berths)} times in {scope}, most recently in {berths[-1]}."
    if titles:
        context += f" Its most recent title came in {titles[-1]}."
    evidence = [f"team_history.csv league_id=200 team_id={tid} years={','.join(map(str,seasons))} made_playoffs,won_playoffs"]
    if last:
        context += f" Its {seasons[-1]} club finished {last['w']}-{last['l']}."
        evidence.append(f"team_history_record.csv league_id=200 year={seasons[-1]} team_id={tid}")
    return dict(context=context, berths=berths, titles=titles, important=len(berths)<=1 or year-1 in titles, evidence=evidence)


def run_pitch(tid, club, view, cutoff):
    line = record(view, tid)
    close = [g for g in view if abs(g["runs_home"]-g["runs_away"]) == 1]
    one = record(close, tid)
    rd = line["runs"] - line["allowed"]
    balance = line["wins"] - line["losses"]
    opposite = balance * rd < 0
    if not view:
        return None
    last = view[-7:]
    form = record(last, tid)
    statement = (f"{club} is {wl(line)} with {line['runs']} runs scored and {line['allowed']} allowed "
                 f"({signed(rd)}). Its one-run games account for {wl(one)} of that record. "
                 f"Over its last {len(last)} games it went {wl(form)} and was "
                 f"{signed(form['runs']-form['allowed'])} in runs.")
    sign = lambda n: (n > 0) - (n < 0)
    title = (f"{club} is winning with a negative run differential" if opposite and balance>0 else
             f"{club} has outscored opponents despite a losing record" if opposite else
             f"The one-run portion of {club}’s {wl(line)} season")
    reason = (f"The standings and scoring margin point in different directions. The close-game record shows how much of the difference came in narrow decisions."
              if opposite else f"The close-game record shows which part of the standings came down to a single run.")
    return pitch(f"{tid}:{cutoff.year}:runs", "performance", title, statement, reason,
                 f"Does {club} still have the same gap between its record and scoring margin after the next series?",
                 [f"games.csv league_id=200 game_type=0 played=1 team_id={tid} through {cutoff}; one-run subset abs(runs0-runs1)=1",
                  select_games(g['game_id'] for g in last)], [sign(rd), sign(balance), (abs(one['wins']-one['losses'])//5)],
                 21 if opposite else 11, 14 if opposite else 7, qualified=opposite and abs(balance)>=4 and abs(rd)>=8,
                 direction="opposing signs" if opposite else "aligned" if balance and rd else "even")


def fixture_pitch(tid, club, fixture, clubs, view, now, cutoff):
    if fixture is None:
        return None
    date_due, _, gid, other, home = fixture
    if clubs[tid]["division"] != clubs[other]["division"]:
        return None
    head = [g for g in view if other in (g["away_id"], g["home_id"])]
    if not head:
        return None
    line = record(head, tid)
    opponent = clubs[other]["name"]
    home_label = f"{opponent} at {club}" if home == tid else f"{club} at {opponent}"
    ahead = max(clubs[tid]["gb"], clubs[other]["gb"])
    close = min(clubs[tid]["gb"], clubs[other]["gb"]) <= 4
    statement = (f"{home_label} is on the schedule for {date_due}. Through {cutoff}, {club} is "
                 f"{wl(line)} against {opponent}, scoring {line['runs']} runs and allowing {line['allowed']}. "
                 f"Their season records are {wl(now[tid])} and {wl(now[other])}, respectively.")
    if close and ahead >= 7 and line["wins"] > line["losses"] and clubs[tid]["gb"] > clubs[other]["gb"]:
        meaning = f"{club} has won more often than it has lost against a club ahead of it. Another direct win counts against that opponent as well."
    else:
        meaning = (f"A decision moves the two clubs a full game relative to each other. Their "
                   f"head-to-head record supplies a better test of this matchup than their overall records alone.")
    spoiler = clubs[tid]["gb"] > clubs[other]["gb"] and line["wins"] >= line["losses"]+3
    closely_matched = abs(gap(now[tid],now[other]))<=3
    result = pitch(f"{tid}:{cutoff.year}:opponent:{other}", "division matchup",
                 f"{club} goes into the {opponent} game {wl(line)} in the season series", statement, meaning,
                 f"How does the next meeting change this series and the distance between the clubs?",
                 [f"games.csv game_id={gid} played=0 league_id=200 date={date_due}", select_games(g["game_id"] for g in head),
                  f"games.csv league_id=200 game_type=0 played=1 team IDs {tid},{other} through {cutoff}"],
                 [home, clubs[tid]["gb"] == 0, clubs[other]["gb"] == 0, (line["wins"]-line["losses"])//3],
                 20 if close else 12, 15 if close else 7, related=[tid,other], qualified=close and len(head)>=3 and (closely_matched or spoiler))
    result["event_key"] = f"matchup:{cutoff.year}:{min(tid,other)}:{max(tid,other)}"
    result["trigger"] = [home,sorted((clubs[tid]["gb"]==0,clubs[other]["gb"]==0)),abs(line["wins"]-line["losses"])//3]
    result["scheduled_game_id"] = gid
    return result


def trade_pitches(tid, clubs, trades, players, by_player, cutoff):
    club = clubs[tid]["name"]
    pitches = []
    for deal in trades:
        moved = day(deal["date"])
        if moved > cutoff or (cutoff - moved).days > 65:
            continue
        for pid in deal["incoming_ids"]:
            pid = int(pid)
            opportunities = []
            for kind, field in [("bat", "pa"), ("pit", "outs")]:
                entries = by_player[kind].get(pid, [])
                later = [e for e in entries if e[2] == tid and e[0] > deal["date"]]
                earlier = [e for e in entries if e[2] == deal["partner_id"] and e[0] < deal["date"]]
                before, after = totals(earlier), totals(later)
                opportunities.append((after[field], kind, earlier, later, before, after))
            work, kind, earlier, later, before, after = max(opportunities, key=lambda o: o[0])
            if not work:
                continue
            person = f"{players[pid]['first_name']} {players[pid]['last_name']}"
            surname = players[pid]["last_name"]
            partner = clubs.get(deal["partner_id"],{}).get("name",f"team #{deal['partner_id']}")
            if kind == "pit":
                file = "players_game_pitching_stats.csv"
                after_text = f"{innings(after['outs'])} {'inning' if after['outs']==3 else 'innings'}, {count(after['er'],'earned run')} and {count(after['k'],'strikeout')} across {count(len(later),'appearance')}"
                before_text = f"{innings(before['outs'])} {'inning' if before['outs']==3 else 'innings'} and {count(before['er'],'earned run')}"
                role = "pitching work"
                material = (15, 36, 63, 90)
                tally = count(after['gs'],'start')
            else:
                file = "players_game_batting.csv"
                after_text = f"{count(after['h'],'hit')} in {count(after['ab'],'at-bat')}, with {count(after['hr'],'home run')} and {after['rbi']} RBI"
                before_text = f"{count(before['h'],'hit')} in {count(before['ab'],'at-bat')}"
                role = "lineup work"
                material = (15, 40, 75, 110)
                tally = count(after['pa'],'plate appearance')
            statement = (f"{club} acquired {person} from {partner} on {moved:%B} {moved.day}. "
                         f"Since the next day, {surname} has recorded {after_text} for {club} ({tally}).")
            if earlier:
                statement += f" Before the trade date, his line for {partner} was {before_text} in {count(len(earlier),'game')}."
            else:
                statement += f" The {cutoff.year} game records show no earlier appearance in this role for {partner} before the trade date."
            threshold = sum(work>=mark for mark in material)
            if kind == "pit":
                rate = 27 * after["er"] / after["outs"]
                comparison = (f"That gives him a {rate:.2f} ERA with {club}."
                              if after["outs"] >= 15 else f"{club} has only a few innings to evaluate.")
            else:
                comparison = (f"He is batting {after['h']/after['ab']:.3f} for {club} over that stretch."
                              if after["ab"] else f"He has yet to record an at-bat for {club} after the trade.")
            statement += " " + comparison
            if kind == "pit" and earlier and before["outs"]>0 and after["outs"]>=15:
                meaning = (f"{partner} got a {27*before['er']/before['outs']:.2f} ERA from {surname} before the move. "
                           f"{club} has used him for {count(after['gs'],'start')} and {count(len(later)-after['gs'],'relief appearance')}. "
                           f"The two workloads give those rates different weight.")
            elif kind == "bat" and earlier and before["ab"] and after["ab"]:
                meaning = (f"The move has given {surname} {count(after['gs'],'start')} in the {club} lineup. "
                           f"He hit {before['h']/before['ab']:.3f} for {partner} before the trade. The unequal samples leave room for that comparison to change.")
            else:
                meaning = f"For this trade, the clearest measure so far is {role}. {club} has used {surname} in {count(len(later),'game')}."
            refs = [deal["source"], f"players.csv player_id={pid}",
                    f"{file} player_id={pid} team_id={tid} joined to played ABL regular-season games after {deal['date']} through {cutoff}, excluding the trade date",
                    f"{file} player_id={pid} team_id={deal['partner_id']} joined to played ABL regular-season games before {deal['date']} in {cutoff.year}"]
            if later:
                refs.append(select_games(e[1] for e in sorted(later)))
            item = pitch(f"{tid}:{cutoff.year}:{deal['id']}:{pid}", "organization", f"What {club} has received from {person}",
                         statement, meaning, f"Does the {role} for {person} extend beyond this first sample?", refs,
                         [kind, threshold, after["gs"]//3], 18+min(4,threshold), 11, related=[tid,deal["partner_id"]],
                         qualified=threshold>=2, direction=role)
            item["workload"] = work
            item["player_id"] = pid
            pitches.append(item)
    return sorted(pitches, key=lambda p: (-p["importance"],-p["workload"],p["key"]))


def individual_pitch(tid, club, view, by_club, people, cutoff):
    if not view:
        return None
    last_seven = {g["game_id"] for g in view[-7:]}
    qualified = []
    for kind in ("bat", "pit"):
        by_id = defaultdict(list)
        for row in by_club[kind][tid]:
            if row[1] in last_seven:
                by_id[row[3]].append(row)
        for pid, entries in by_id.items():
            stats = totals(entries)
            if kind == "bat" and stats["pa"] >= 12:
                grade = min(23, 2*stats["hr"] + stats["h"] + stats["rbi"]//2)
                exceptional = stats["hr"]>=3 or (stats["h"]>=12 and stats["ab"] and stats["h"]/stats["ab"]>=.375)
            elif kind == "pit" and stats["outs"] >= 15:
                grade = min(23, stats["outs"]//3 + stats["k"]//2 + max(0,5-stats["er"]))
                exceptional = (stats["k"]>=10 and 27*stats["er"]/stats["outs"] <= 2.5 or
                               stats["outs"]>=24 and stats["er"] == 0)
            else:
                continue
            qualified.append((exceptional, grade, -pid, kind, sorted(entries), stats))
    if not qualified:
        return None
    exceptional, _, negative_pid, kind, entries, recent = max(qualified, key=lambda x:x[:4])
    pid = -negative_pid
    player = people[pid]
    season_rows = [e for e in by_club[kind][tid] if e[3] == pid]
    earlier = totals(e for e in season_rows if e[1] not in last_seven)
    if kind == "bat":
        file = "players_game_batting.csv"
        fact = f"{count(recent['h'],'hit')} in {count(recent['ab'],'at-bat')}, including {count(recent['hr'],'home run')}, and drove in {count(recent['rbi'],'run')}"
        prior = f"{count(earlier['h'],'hit')} in {count(earlier['ab'],'at-bat')} and {count(earlier['hr'],'home run')}"
        focus = f"{player} supplied {recent['h']} hits as {club} went {wl(record(view[-7:],tid))}"
        question = f"Does {player} sustain this hitting over a longer stretch?"
        tag = recent["hr"] >= 3
    else:
        file = "players_game_pitching_stats.csv"
        fact = f"{innings(recent['outs'])} innings, allowed {count(recent['er'],'earned run')} and struck out {recent['k']}"
        prior = f"{innings(earlier['outs'])} {'inning' if earlier['outs']==3 else 'innings'} and {count(earlier['er'],'earned run')}"
        focus = f"{player} allowed {count(recent['er'],'earned run')} in {innings(recent['outs'])} innings for {club}"
        question = f"How much of {club}’s next pitching workload falls to {player}?"
        tag = recent["er"] == 0
    statement = (f"In {count(len(entries),'appearance')} during {club}’s last {count(len(last_seven),'game')}, "
                 f"{player} {'had' if kind == 'bat' else 'worked'} {fact}. Before those games, his season line for {club} included {prior}.")
    reason = f"The short stretch can be measured against the player's own earlier work. {club} went {wl(record(view[-7:],tid))} in those games."
    refs = [f"players.csv player_id={pid}",
            f"{file} team_id={tid} player_id={pid} joined to played ABL regular-season games through {cutoff}",
            select_games(sorted(last_seven))]
    item = pitch(f"{tid}:{cutoff.year}:player:{pid}:{kind}", "individual", focus, statement, reason, question,
                 refs, [kind, bool(exceptional), tag, recent["hr"]//3 if kind=="bat" else recent["k"]//8],
                 21 if exceptional else 13, 12 if exceptional else 7, qualified=exceptional, direction=kind)
    item["player_id"] = pid
    return item


def draft_pitch(tid, club, view, by_club, players, cutoff, old_watch):
    """Follow real opportunities for the club's own recent draftees; draft standing alone is a notebook fact."""
    recent = {g["game_id"] for g in view[-10:]}
    window = defaultdict(list)
    for kind in ("bat","pit"):
        for entry in by_club[kind][tid]:
            p = players[entry[3]]
            if p["draft_team_id"]==str(tid) and cutoff.year-2<=int(p["draft_year"])<=cutoff.year:
                window[entry[3],kind].append(entry)
    possibilities = []
    for (pid,kind), appearances in window.items():
        appearances.sort()
        if len(appearances)>16 or appearances[-1][1] not in recent:
            continue
        line = totals(appearances)
        work = line["pa"] if kind=="bat" else line["outs"]
        if work<3:
            continue
        watched = next((p for p in old_watch if int(p["player_id"])==pid),{})
        possibilities.append((bool(watched),work,-pid,kind,appearances,line,watched))
    if not possibilities:
        return None
    _,work,negative_pid,kind,entries,line,watched = max(possibilities,key=lambda o:o[:4])
    pid = -negative_pid
    person = players[pid]
    name = f"{person['first_name']} {person['last_name']}"
    file = "players_game_batting.csv" if kind=="bat" else "players_game_pitching_stats.csv"
    line_text = (f"{count(line['h'],'hit')} in {count(line['ab'],'at-bat')}, with {count(line['gs'],'start')}"
                 if kind=="bat" else f"{innings(line['outs'])} {'inning' if line['outs']==3 else 'innings'} and {count(line['er'],'earned run')}")
    draft = f"at overall pick {person['draft_overall_pick']}" if int(person['draft_overall_pick'])>0 else "in the amateur draft"
    statement = (f"{club} drafted {name} in {person['draft_year']} {draft}. His first {cutoff.year} game line "
                 f"in this role for the parent club was on {entries[0][0]}. Through {cutoff} he has {line_text} across "
                 f"{count(len(entries),'appearance')}.")
    if watched:
        statement += f" The previous franchise notebook followed him at {watched['affiliate']}."
    reason = (f"The organization can now assess its draft choice against ABL opponents. "
              f"The sample covers {count(len(entries),'appearance')} and leaves his longer-term role open.")
    evidence = [f"players.csv player_id={pid} draft_team_id={tid} draft_year={person['draft_year']}",
                f"{file} player_id={pid} team_id={tid} joined to played ABL regular-season games through {cutoff}",
                select_games(row[1] for row in entries)]
    if watched:
        evidence.append(f"Prior team notebook development_watch player_id={pid} affiliate team_id={watched['team_id']}")
    item = pitch(f"{tid}:{cutoff.year}:draft-usage:{pid}:{kind}","development",
                 f"A drafted player gets a chance in {club}: {name}", statement, reason,
                 f"Will {name} continue to receive {'plate appearances' if kind=='bat' else 'pitching work'} for {club}?",
                 evidence,[kind,sum(work>=n for n in (3,15,35,65)),line["gs"]//4],20 if watched else 16,
                 8,history=5, qualified=bool(watched) or work>=15)
    item["player_id"] = pid
    return item


def wildcard_pitch(tid, clubs, groups, records, cutoff, wild_cards):
    if wild_cards != 1:
        return None
    conference = clubs[tid]["conference"]
    conference_groups = [group for group in groups.values() if clubs[group[0]]["conference"] == conference]
    leaders = [race_order(g,records)[1] for g in conference_groups]
    # No provisional tie-breaking assumption for divisions sharing first place.
    if any(len(group) != 1 for group in leaders) or tid in {group[0] for group in leaders}:
        return None
    field = [t for g in conference_groups for t in g if all(t != lead[0] for lead in leaders)]
    order, front = race_order(field,records)
    other = next(t for t in order if t != tid)
    distance = abs(gap(records[tid],records[other]))
    if distance > 5:
        return None
    leader = tid in front
    if len(front)>1 and leader:
        phrase = f"shares first place in the provisional {conference} wild-card order with {names([x for x in front if x != tid],clubs)}"
    else:
        phrase = (f"leads {names([other],clubs)}" if leader else f"trails {names(front,clubs)}")
        phrase += f" by {distance:g} {'game' if distance==1 else 'games'} in the provisional {conference} wild-card order"
    title = f"{clubs[tid]['name']} {phrase}"
    observation = (f"{title}. {clubs[tid]['name']} is {wl(records[tid])} and {clubs[other]['name']} is {wl(records[other])}. "
                   f"Each division has a sole leader at this cutoff. The comparison ranks the remaining clubs by winning percentage.")
    return pitch(f"{tid}:{cutoff.year}:wildcard","wild card",title,observation,
                 f"The conference’s one wild card leaves a route into the tournament even if the division leader stays ahead.",
                 f"Can {clubs[tid]['name']} improve its position against the other non-division leaders?",
                 ["league_playoffs.csv league_id=200 num_wild_cards=1",f"games.csv league_id=200 game_type=0 played=1 through {cutoff}; "
                  f"teams.csv, sub_leagues.csv and divisions.csv conference={conference}, team IDs {','.join(str(t) for g in conference_groups for t in g)}"],
                 [sorted(front),other,0 if distance==0 else 1 if distance<=1 else 3 if distance<=3 else 5],
                 20,16,related=[tid,other],qualified=distance<=3,direction="leading" if leader else "chasing")


def engine_pitch(tid, club, candidate, view, cutoff):
    kind = candidate["signal_type"]
    if kind not in {"personnel_change","revised_game","completed_game"}:
        return None
    gid = candidate["subject_id"]
    if kind == "personnel_change":
        if str(tid) != str(gid):
            return None
        content = candidate["evidence_summary"]
        p = pitch(f"{tid}:{cutoff.year}:personnel", "personnel", f"A personnel change for {club}",
                  content, f"The updated list changes the personnel available to track for {club}.",
                  f"Which of the changed roster assignments is followed by a game appearance?",
                  [f"players_roster_status.csv, players.csv team_id={tid} from the two admitted exports",
                   f"trade_history.csv dated movement through {cutoff} involving team_id={tid}"],
                  content, 16, 8, qualified=True)
    else:
        game = next((g for g in view if g["game_id"]==int(gid)),None)
        if game is None:
            return None
        revised = kind == "revised_game"
        important = revised or abs(game["runs_away"]-game["runs_home"])>=8 or game["innings"]>=12
        if not important:
            return None
        files = [game["source"],f"games_score.csv game_id={gid}, away and home inning totals"]
        full = candidate["evidence_summary"]
        if revised:
            files.extend(f"{Path(f).name} game_id={gid} compare prior admitted snapshot" for f in candidate["source_files"].split("|"))
        # Key for a game article must be the same from either beat.
        key = f"game:{cutoff.year}:{gid}"
        title = (f"Revised scoring record for game {gid}" if revised else
                 f"{game['summary']} in {game['innings']} innings")
        statement = full if revised else f"On {game['date']}, the final was {game['summary']} in {game['innings']} innings."
        if not revised and abs(game["runs_away"]-game["runs_home"])>=8:
            seven = view[-7:]
            whole = record(seven,tid)
            others = [g for g in seven if g["game_id"] != int(gid)]
            if game in seven and others:
                rest = record(others,tid)
                statement += (f" {club} scored {whole['runs']} and allowed {whole['allowed']} in its last {len(seven)} games. "
                              f"In the other {len(others)}, the totals were {rest['runs']} and {rest['allowed']}.")
                files.append(select_games(g["game_id"] for g in seven))
        reason = ("The revised line supersedes the earlier scoring entry. Any account using the original attribution needs review."
                  if revised else f"The size of this result changes how {club}’s scoring over a short series should be read."
                  if abs(game["runs_away"]-game["runs_home"])>=8 else
                  "The extra innings put a longer game than usual on both pitching staffs' records.")
        p = pitch(key+f":{tid}","scoring revision" if revised else "game",title,statement,reason,
                  f"Which details of the earlier coverage require amendment?" if revised else
                  f"Is the same pattern present when {club} plays again?",
                  list(dict.fromkeys(files)), [gid,game["runs_away"],game["runs_home"],game["innings"],full if revised else ""],
                  24 if revised else 17, 16 if revised else 10,related=[game["away_id"],game["home_id"]],qualified=important)
        p["event_key"] = key
    p["story_engine_candidates"] = [candidate["candidate_id"]]
    return p


def assess(item, old, previous_cutoff, cutoff, edition, covered):
    """Keep raw changes observable, but require a material development for another article."""
    category = item["category"]
    item["trigger"] = fingerprint([item.get("event_key",item["key"]),item["trigger"]])
    observed = fingerprint(item["observation"])
    repeated = bool(old and item["trigger"] == old.get("trigger"))
    correction = bool(old and previous_cutoff == cutoff and observed != old.get("facts_signature"))
    revised = category == "scoring revision" or correction
    same_coverage = bool(covered and covered.get("trigger") == item["trigger"])
    earlier = old.get("direction") if old else None
    if not old:
        assessment = "Opened"
    elif revised:
        assessment = "Corrected evidence"
    elif repeated:
        assessment = "No material change"
    elif earlier and item.get("direction") != earlier:
        assessment = "Earlier assessment changed"
    else:
        assessment = "Further evidence"
    rubric = dict(relevance=13 if item["qualified"] else 8,
                  novelty=16 if revised else 12 if not old else 0 if repeated else 14,
                  significance=item["importance"], competitive_stakes=item["stakes"],
                  historical_context=item["historical_weight"],
                  evidence_strength=15 if len(item["evidence"])>=2 else 12,
                  coverage_freshness=0 if same_coverage and not revised else 8)
    score = min(100,sum(rubric.values()))
    eligible = bool(item["qualified"] and score>=70 and (not repeated or revised) and (not same_coverage or revised))
    item.update(score=score, score_components=rubric, assessment=assessment, eligible=eligible)
    if not covered:
        item["coverage_reference"] = None
    else:
        item["coverage_reference"] = f"{covered['edition']}/{covered['path']}"
    history = list(old.get("developments",[])) if old else []
    if not old or not repeated or revised:
        history.append(dict(cutoff=cutoff,edition=edition,assessment=assessment,trigger=item["trigger"],
                            observation=item["observation"],evidence=item["evidence"],direction=item.get("direction")))
    return dict(key=item["key"],category=category,first_seen=old["first_seen"] if old else cutoff,
                originating_event=old["originating_event"] if old else item["observation"],edition=edition,
                last_seen=cutoff, last_material_change=history[-1]["cutoff"], trigger=item["trigger"],
                facts_signature=observed, direction=item.get("direction"), developments=history,
                status="active",team_ids=item["related_team_ids"],why_it_matters=item["significance"],
                question=item["question"],evidence=item["evidence"],latest_assessment=assessment,
                coverage_reference=item["coverage_reference"], scheduled_game_id=item.get("scheduled_game_id"))


def build_slate(raw_dir, cutoff, beats, candidates, previous, edition):
    raw = Path(raw_dir)
    when = day(cutoff)
    cutoff = when.isoformat()
    beats = {int(k): b for k,b in beats.items()}
    clubs = {int(t["team_id"]):t for t in rows(raw,"teams") if int(t["team_id"]) in beats}
    if len(clubs) != 24 or len(beats) != 24:
        raise ValueError("The editorial slate requires 24 verified franchise notebooks.")
    players = {int(p["player_id"]):p for p in rows(raw,"players")}
    people = {pid:f"{p['first_name']} {p['last_name']}" for pid,p in players.items()}
    groups = defaultdict(list)
    for tid,b in beats.items():
        clubs[tid].update(division=b["division"],conference=b["conference"],gb=b["gb"])
        groups[b["division"]].append(tid)
    book = game_records(raw,when,clubs)
    views = defaultdict(list)
    for game in sorted(book.values(),key=lambda g:(g["date"],g["game_id"])):
        for side in ("home_id","away_id"):
            views[game[side]].append(game)
    boundary = when - timedelta(days=7)
    records = {t:record(views[t],t) for t in clubs}
    earlier = {t:record([g for g in views[t] if g["date"] <= boundary.isoformat()],t) for t in clubs}
    for t,beat in beats.items():
        if (beat["wins"],beat["losses"]) != (records[t]["wins"],records[t]["losses"]):
            raise ValueError(f"Slate and notebook records disagree for team {t}")
    upcoming = next_opponent(raw,when,clubs)
    by_player,by_club = game_lines(raw,book,when.year)
    old_history,old_results = defaultdict(dict),defaultdict(dict)
    for filename,index in [("team_history",old_history),("team_history_record",old_results)]:
        for row in rows(raw,filename):
            tid,year=int(row["team_id"]),int(row["year"])
            if row["league_id"]=="200" and year<when.year and tid in clubs:
                index[tid][year] = row
    archive = archive_board(clubs,when)
    wild_cards = next((int(r["num_wild_cards"]) for r in rows(raw,"league_playoffs") if r["league_id"]=="200"),0)
    old_threads = previous.get("threads",{})
    coverage = list(previous.get("coverage",[]))
    filed = {a["key"]:a for a in coverage}
    board,threads,queue = {},{},[]
    hierarchy = {"competition":"standings","wild card":"race","performance":"team","game":"game",
                 "scoring revision":"game","individual":"players","division matchup":"race",
                 "organization":"management_organization","personnel":"management_organization","development":"players"}
    for tid,b in sorted(beats.items()):
        club = clubs[tid]["name"]
        old_beat = previous.get("teams",{}).get(str(tid),{})
        historical = historical_context(tid,club,old_history[tid],old_results[tid],when.year)
        race = division_pitch(tid,club,groups[b["division"]],clubs,records,earlier,boundary,when,
                              views,archive[tid],historical)
        items = [race,run_pitch(tid,club,views[tid],when),
                 fixture_pitch(tid,club,upcoming.get(tid),clubs,views[tid],records,when),
                 individual_pitch(tid,club,views[tid],by_club,people,when),
                 wildcard_pitch(tid,clubs,groups,records,when,wild_cards),
                 draft_pitch(tid,club,views[tid],by_club,players,when,old_beat.get("development_watch",[]))]
        items += trade_pitches(tid,clubs,b["transactions"],players,by_player,when)
        items += [engine_pitch(tid,club,c,views[tid],when) for c in candidates]
        pitches = []
        for p in filter(None,items):
            if tid not in p["related_team_ids"]:
                p["related_team_ids"].append(tid)
            p["hierarchy"] = hierarchy[p["category"]]
            p.setdefault("story_engine_candidates",[])
            key = p["key"]
            if key in threads:
                raise ValueError("Duplicate editorial thread: "+key)
            threads[key] = assess(p,old_threads.get(key),previous.get("cutoff"),cutoff,edition,filed.get(key))
            if p["eligible"]:
                queue.append(dict(team_id=tid,key=key,score=p["score"],title=p["title"],trigger=p["trigger"]))
            pitches.append(p)
        pitches.sort(key=lambda p:(not p["eligible"],-p["score"],p["key"]))
        leading = pitches[0]
        anchor = (f"Follow {club} in the {b['division']}. {historical['context']}")
        assignment = f"{anchor} Current investigation: {leading['question']}"
        board[str(tid)] = dict(team=b["team"],abbr=b["abbr"],assignment=assignment,beat_identity=anchor,
            current_record=f"{b['wins']}-{b['losses']}", division=b["division"],
            previous_edition=previous.get("edition"), prior_assignment=old_beat.get("assignment"),
            notebook=f"teams/{tid:02}-{b['abbr']}.md", notebook_threads=b["stories"],archive_investigations=archive[tid],
            pitches=pitches, reporting_queue=[p["key"] for p in pitches if p["eligible"]],
            open_investigations=[p["key"] for p in pitches], prior_coverage=[a for a in coverage if tid in a["team_ids"]],
            development_watch=b["prospects"])
    for key,prior in old_threads.items():
        if key in threads:
            continue
        fixture = book.get(prior.get("scheduled_game_id"))
        if prior["category"] == "division matchup" and fixture is not None:
            status = "resolved"
            note = f"The scheduled meeting was completed on {fixture['date']}. {fixture['summary']}."
            evidence = [fixture["source"]]
        else:
            status = "archived"
            note = "The subject no longer meets this edition’s selection criteria. Prior evidence and coverage remain available."
            evidence = prior["evidence"]
        developments=list(prior["developments"])
        if prior.get("status") != status:
            developments.append(dict(cutoff=cutoff,edition=edition,assessment=status,observation=note,evidence=evidence))
        threads[key]=dict(prior,status=status,developments=developments)
    queue.sort(key=lambda r:(-r["score"],r["team_id"],r["key"]))
    return dict(edition=edition,cutoff=cutoff,previous_edition=previous.get("edition"),teams=board,threads=threads,
                coverage=coverage,reporting_queue=queue, internal=True)
