"""The internal daily reporting queue and division-by-division ABL 411."""
import csv
import json
from collections import defaultdict
from pathlib import Path

from abl_newsroom_beats import day, game_records, record, rows, wl
from abl_newsroom_print import write_report
from abl_newsroom_slate import build_slate, count, gap, race_order


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def store(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")


def when(value):
    date = day(value)
    return f"{date:%B} {date.day}"


def names(ids, clubs):
    return " and ".join(clubs[tid]["name"] for tid in ids)


def listing(words):
    if len(words)<3:
        return " and ".join(words)
    return ", ".join(words[:-1])+", and "+words[-1]


def result(game, clubs):
    away, home = clubs[game["away_id"]]["name"], clubs[game["home_id"]]["name"]
    a, h = game["runs_away"], game["runs_home"]
    if a == h:
        return f"{away} and {home} tied {a}-{h}"
    winner, loser = (away,home) if a > h else (home,away)
    return f"{winner} beat {loser} {max(a,h)}-{min(a,h)}"


def label(game, clubs):
    return f"{clubs[game['away_id']]['name']} at {clubs[game['home_id']]['name']}"


def game_ref(g):
    if "runs_home" in g:
        return g["source"]
    return f"games.csv game_id={g['game_id']} played=0 date={g['date']} visitor={g['away_id']} home={g['home_id']}"


def schedule(raw, cutoff, clubs):
    upcoming = []
    for r in rows(raw,"games"):
        if r["league_id"]!="200" or r["game_type"]!="0" or r["played"]!="0":
            continue
        d, a, h = day(r["date"]), int(r["away_team"]), int(r["home_team"])
        if d < cutoff or d.year != cutoff.year or a not in clubs or h not in clubs:
            continue
        upcoming.append(dict(game_id=int(r["game_id"]),date=d.isoformat(),time=int(r["time"]),
                             away_id=a,home_id=h,recorded_innings=int(r["innings"])))
    return sorted(upcoming,key=lambda g:(g["date"],g["time"],g["game_id"]))


def pair_next(front, challenger, next_games, now, clubs):
    """Describe the exact effect on a pair; accommodate off days and unfinished games."""
    left = [g for g in next_games if front in (g["home_id"],g["away_id"])]
    right = [g for g in next_games if challenger in (g["home_id"],g["away_id"])]
    leader, chaser = names([front],clubs), names([challenger],clubs)
    margin = gap(now[front],now[challenger])
    direct = next((g for g in left if g in right),None)
    if direct and len(left) == len(right) == 1:
        if margin == 0:
            return f"The winner of the {leader}–{chaser} game moves a full game ahead of the loser."
        if margin == 1:
            return f"A {chaser} win over {leader} would pull the two clubs level."
        return (f"A {chaser} win over {leader} would "
                + (f"cut the gap to {count(margin-1,'game')}." if margin>1 else f"put {chaser} ahead of {leader} in games back."))
    if len(left)==1 and len(right)==1 and margin>=1:
        return (f"A {chaser} win and a {leader} loss would bring them level." if margin==1 else
                f"A {chaser} win and a {leader} loss would cut the gap to {count(margin-1,'game')}.")
    if len(left)>1 or len(right)>1:
        narrowed=margin-(len(left)+len(right))/2
        if narrowed>0:
            return (f"If {chaser} wins all its listed games and {leader} loses all of its games that day, "
                    f"the margin falls to {count(narrowed,'game')}.")
        return f"Both the number of decisions and their outcomes will matter for {leader} and {chaser}."
    if left and not right:
        return f"With no {chaser} game listed that day, a {leader} win would gain half a game on {chaser}."
    if right and not left:
        return f"With no {leader} game listed that day, a {chaser} win would gain half a game on {leader}."
    return f"The next change in the margin depends on the next recorded decisions for {leader} and {chaser}."


def division_brief(division, members, now, before, finals, future, clubs, cutoff):
    order, leaders = race_order(members,now)
    _, old_leaders = race_order(members,before)
    front, second = order[:2]
    margin = gap(now[front],now[second])
    old_margin = gap(before[front],before[second])
    compared = any(before[t]["wins"]+before[t]["losses"] for t in members)
    if len(leaders)>1:
        mark = wl(now[front])
        same = all(wl(now[t])==mark for t in leaders)
        standing = f"{names(leaders,clubs)} {'remained tied' if set(leaders)==set(old_leaders) and compared else 'share first place'}"
        standing += f" at {mark}." if same else " by winning percentage."
    else:
        f,s = names([front],clubs), names([second],clubs)
        if not compared:
            standing=f"{f} ({wl(now[front])}) leads {s} ({wl(now[second])}) by {count(margin,'game')}."
        elif margin==old_margin:
            size="one" if margin==1 else f"{margin:g}"
            standing=f"{f} ({wl(now[front])}) kept its {size}-game lead over {s} ({wl(now[second])})."
        elif old_margin>=0:
            verb = "widened" if margin>old_margin else "shrank"
            possessive=f+"’" if f.endswith("s") else f+"’s"
            standing=f"{possessive} lead over {s} {verb} from {count(old_margin,'game')} to {count(margin,'game')}. {f} is {wl(now[front])} and {s} is {wl(now[second])}."
        else:
            standing=f"{f} passed {s}. It went from {count(abs(old_margin),'game')} behind to {count(margin,'game')} ahead."
    featured=set(order[:2])
    if compared and len(order)>2:
        third = next((t for t in order if t not in leaders and t!=second),None)
        if third is not None:
            at, formerly = gap(now[front],now[third]), gap(before[front],before[third])
            if at!=formerly and min(at,formerly)<=4 and formerly>=0:
                featured.add(third)
                standing += f" {names([third],clubs)} {'closed from' if at<formerly else 'slipped from'} {formerly:g} back to {at:g}."
    today = [g for g in finals if g["home_id"] in members or g["away_id"] in members]
    # Put the leaders' results first; each actual game appears once within a division.
    today.sort(key=lambda g:(min(order.index(t) for t in (g["away_id"],g["home_id"]) if t in members),g["game_id"]))
    decisive = [g for g in today if featured.intersection((g["home_id"],g["away_id"]))]
    scores = ". ".join(result(g,clubs) for g in decisive)+"." if decisive else "The clubs at the front had no completed game that day."
    games = [g for g in future if g["home_id"] in members or g["away_id"] in members]
    next_games = [g for g in games if g["date"]==games[0]["date"]] if games else []
    if next_games:
        listings=[]
        for g in next_games:
            line=label(g,clubs)
            if g["recorded_innings"]:
                line+=f" (unfinished, {count(g['recorded_innings'],'inning')} already recorded)"
            listings.append(line)
        preview=f"On {when(next_games[0]['date'])} the schedule lists "+listing(listings)+". "
        preview+=pair_next(front,second,next_games,now,clubs)
    else:
        preview="No further regular-season game for this division appears in the admitted schedule."
    restriction=f"games.csv league_id=200 game_type=0 played=1 through {cutoff}; compare rows with date before {cutoff}"
    refs=[restriction,f"teams.csv and divisions.csv team IDs {','.join(map(str,sorted(members)))}"]
    return dict(division=division,heading=division,movement=standing,results=scores,next=preview,
                team_ids=sorted(members),completed_game_ids=[g["game_id"] for g in today],
                scheduled_game_ids=[g["game_id"] for g in next_games],evidence=refs+[game_ref(g) for g in today+next_games])


def performances(raw, admitted, people):
    """Dated player evidence in the games this edition added or corrected."""
    by_player, actors = defaultdict(list), defaultdict(list)
    for file,kind in [("players_game_batting","batting"),("players_game_pitching_stats","pitching")]:
        for row in rows(raw,file):
            gid=int(row["game_id"])
            if gid not in admitted or row["league_id"]!="200" or row["level_id"]!="1":
                continue
            tid,pid=int(row["team_id"]),int(row["player_id"])
            g=admitted[gid]
            if tid not in (g["home_id"],g["away_id"]) or int(row["year"])!=day(g["date"]).year:
                continue
            source=f"{file}.csv game_id={gid} player_id={pid} team_id={tid}"
            if kind=="batting":
                if int(row["pa"])==0:
                    continue
                fact=f"{row['h']}-for-{row['ab']}, {count(int(row['hr']),'home run')}, {row['rbi']} RBI"
                grade=3*int(row["hr"])+2*int(row["rbi"])+int(row["h"])
            else:
                if int(row["g"])==0:
                    continue
                fact=f"{count(int(row['outs']),'out')}, {count(int(row['er']),'earned run')}, {count(int(row['k']),'strikeout')}"
                grade=int(row["outs"])//3+int(row["k"])-int(row["er"])
            line=dict(player_id=pid,name=people[pid],team_id=tid,game_id=gid,date=g["date"],role=kind,
                      fact=fact,evidence=[source,f"players.csv player_id={pid}"])
            by_player[pid].append(line)
            actors[gid,tid].append((grade,line))
    best={k:max(items,key=lambda row:(row[0],-row[1]["player_id"]))[1] for k,items in actors.items()}
    return by_player,best


def team_context(tid, views, schedule_rows, now, clubs):
    recent=views[tid][-7:]
    form=record(recent,tid)
    pending=[g for g in schedule_rows if tid in (g["home_id"],g["away_id"])]
    first=[g for g in pending if g["date"]==pending[0]["date"]] if pending else []
    return dict(team_id=tid,team=names([tid],clubs),record=wl(now[tid]),
                recent_form=dict(games=len(recent),record=wl(form),runs_scored=form["runs"],runs_allowed=form["allowed"],
                                 game_ids=[g["game_id"] for g in recent]),
                next_games=first)


def next_test(item,tid,upcoming,clubs,people):
    if not upcoming:
        return "The admitted schedule has no next regular-season game for this club. Await its next verified event."
    game=upcoming[0]
    fixture=f"{label(game,clubs)} on {when(game['date'])}"
    category=item["category"]
    player=people.get(item.get("player_id"))
    if player:
        return f"Check the participation and stat line for {player} in {fixture}. An appearance extends the sample for this investigation."
    if category=="performance":
        return f"Check the margin in {fixture}. A one-run decision adds to the close-game record, and the final score changes the club’s run balance."
    if category=="division matchup" and item.get("scheduled_game_id"):
        direct=next((g for g in upcoming if g["game_id"]==item["scheduled_game_id"]),None)
        if direct:
            return f"The winner of {label(direct,clubs)} on {when(direct['date'])} gains a full game relative to the losing opponent."
    return f"Check the final of {fixture} and the same-day decisions of the clubs ahead or alongside {names([tid],clubs)}. Recalculate the margin once those scores are in."


def queue_item(p,tid,context,book,by_player,best,clubs,people,previous,slate,native,changes,cutoff):
    base=context[tid]
    related=p["related_team_ids"]
    key=p["key"]
    native_ids=p.get("story_engine_candidates",[])
    native_rows=[native[c] for c in native_ids if c in native]
    revised=[g for g in changes["revised"] if any(t in (book[g]["away_id"],book[g]["home_id"]) for t in related)]
    added=sorted((book[gid] for gid in changes["new"] if tid in (book[gid]["away_id"],book[gid]["home_id"])),
                 key=lambda g:(g["date"],g["game_id"]))
    refs=list(p["evidence"])
    pid=p.get("player_id")
    actor_lines = sorted((x for x in by_player.get(pid,[]) if x["team_id"]==tid),key=lambda x:(x["date"],x["game_id"],x["role"]))
    for row in actor_lines:
        refs.extend(row["evidence"])
    event=""
    if native_rows:
        event=" ".join(c["evidence_summary"] for c in native_rows)
    elif actor_lines:
        last=actor_lines[-1]
        event=f"{last['name']} in {when(last['date'])} game {last['game_id']}: {last['fact']}."
    elif pid and added:
        event=f"No batting plate appearance or pitching appearance for {people[pid]} in {names([tid],clubs)}’s newly admitted games."
        refs.append(f"players_game_batting.csv and players_game_pitching_stats.csv team_id={tid} player_id={pid} game IDs "+",".join(str(g['game_id']) for g in added))
    elif added:
        event=" ".join(f"{when(g['date'])}: {result(g,clubs)}." for g in added)
    elif revised:
        event="Review the revised record for "+", ".join(f"game {gid}" for gid in sorted(revised))+"."
    else:
        event="No new completed game or relevant scoring correction for this club in this edition."
    actor_records=actor_lines
    if not pid and added:
        latest=best.get((added[-1]["game_id"],tid))
        actor_records=[latest] if latest else []
    refs.extend(game_ref(g) for g in added)
    for line in actor_records:
        refs.extend(line["evidence"])
    upcoming=base["next_games"]
    refs.extend(game_ref(g) for g in upcoming)
    for t in related:
        refs.append(f"games.csv league_id=200 game_type=0 played=1 team_id={t} through {cutoff}")
        sample=context[t]["recent_form"]["game_ids"]
        if sample:
            refs.append("games.csv game IDs "+",".join(map(str,sample))+f" team_id={t} recent form")
        refs.extend(game_ref(g) for g in context[t]["next_games"])
    old=previous.get("threads",{}).get(key)
    facts_before=(old["developments"][-1]["observation"] if old and old.get("developments") else None)
    return dict(thread_id=key,team_ids=related,desk_team_id=tid,category=p["category"],title=p["title"],priority=p["score"],
                assessment=p["assessment"],consider_for_review=p["eligible"],event=event,
                change_since_previous=dict(previous_cutoff=previous.get("cutoff"),previous_observation=facts_before,
                    assessment=p["assessment"],first_seen=slate["threads"].get(key,{}).get("first_seen",cutoff)),
                baseball_question=p["question"],why_now=p["significance"],known=p["observation"],uncertain=p["question"],
                next_condition=next_test(p,tid,upcoming,clubs,people),teams=[context[t] for t in related],
                player_ids=[pid] if pid else sorted({r["player_id"] for r in actor_records}),
                people=actor_records or ([dict(player_id=pid,name=people[pid])] if pid else []),
                new_game_ids=[g["game_id"] for g in added],related_revised_game_ids=sorted(revised),
                scheduled_game_ids=[g["game_id"] for g in upcoming],source_candidates=native_ids,
                earlier_coverage=[dict(edition=a["edition"],title=a["title"],path=a["path"]) for a in previous.get("coverage",[])
                                  if key==a.get("key") or key in a.get("story_ids",[])],
                evidence=list(dict.fromkeys(refs)), disposition="reporting desk")


def daily_state(raw,cutoff,beats):
    clubs={int(r["team_id"]):r for r in rows(raw,"teams") if int(r["team_id"]) in beats}
    book=game_records(raw,day(cutoff),clubs)
    views=defaultdict(list)
    for g in sorted(book.values(),key=lambda g:(g["date"],g["game_id"])):
        for tid in (g["away_id"],g["home_id"]):
            views[tid].append(g)
    future=schedule(raw,day(cutoff),clubs)
    now={t:record(views[t],t) for t in clubs}
    before={t:record([g for g in views[t] if g["date"]<cutoff],t) for t in clubs}
    groups=defaultdict(list)
    for tid,b in beats.items():
        if (now[tid]["wins"],now[tid]["losses"]) != (b["wins"],b["losses"]):
            raise ValueError(f"Daily desk disagrees with the verified record for {b['team']}")
        groups[b["division"]].append(tid)
    if len(groups)!=6 or sum(map(len,groups.values()))!=24:
        raise ValueError("The daily desk requires the six ABL divisions and all 24 clubs.")
    finals=[g for g in book.values() if g["date"]==cutoff]
    sections=[division_brief(d,groups[d],now,before,finals,future,clubs,cutoff) for d in
              sorted(groups,key=lambda d:(not d.startswith("NBC"),0 if "Eastern" in d else 1 if "Central" in d else 2,d))]
    return clubs,book,sections,{t:team_context(t,views,future,now,clubs) for t in clubs}


def table(path):
    with path.open(encoding="utf-8-sig",newline="") as handle:
        return list(csv.DictReader(handle))


def build_desk(stage: Path, intel: Path, previous: dict, cycle: str):
    """Write into the caller's staging directory. All products here are internal desk work."""
    from abl_newsroom_editorial import record_pitches
    stage=Path(stage)
    stage.mkdir(parents=True,exist_ok=True)
    v,ledger=read(intel/"verification.json"),read(intel/"coverage.json")
    raw,cutoff=Path(v["raw_dir"]),v["cutoff"]
    beats={b["team_id"]:b for b in (read(p) for p in (intel/"teams").glob("*.json"))}
    sources=table(intel/"story_candidates.csv")
    native={r["candidate_id"]:r for r in sources}
    native_evidence=defaultdict(list)
    for row in table(intel/"story_evidence.csv"):
        native_evidence[row["candidate_id"]].append(row)
    slate=build_slate(raw,cutoff,beats,sources,previous,cycle)
    record_pitches(stage,slate,v,cycle,raw)
    clubs,book,divisions,context=daily_state(raw,cutoff,beats)
    changes=dict(new=set(ledger["completed_game_ids"]),revised=set(ledger["corrected_game_ids"]))
    people={int(r["player_id"]):f"{r['first_name']} {r['last_name']}" for r in rows(raw,"players")}
    by_player,best=performances(raw,{g:book[g] for g in changes["new"]|changes["revised"]},people)
    items=[]
    for team,b in slate["teams"].items():
        for p in b["pitches"]:
            items.append(queue_item(p,int(team),context,book,by_player,best,clubs,people,previous,slate,native,changes,cutoff))
    linked={cid for item in items for cid in item["source_candidates"]}
    for c in sources:
        cid=c["candidate_id"]
        if cid in linked:
            continue
        if c["subject_type"]=="matchup":
            gid=int(c["subject_id"])
            g=book[gid]
            members=[g["home_id"],g["away_id"]]
            key=f"game:{v['season']}:{gid}"
        else:
            members=[int(c["subject_id"])]
            key=f"{members[0]}:{v['season']}:{c['signal_type']}"
        selectors=[f"{Path(r['source_file']).name} {r['source_row_key']}" for r in native_evidence[cid]]
        p=dict(key=key,category=c["signal_type"],related_team_ids=members,story_engine_candidates=[cid],evidence=selectors,
               title=c["headline_factual"],score=int(c["signal_score"]),assessment="Newly admitted" if c["signal_type"]!="revised_game" else "Correction",
               eligible=False,question=f"How did this result affect {names(members,clubs)}?",significance=c["stakes"],observation=c["evidence_summary"])
        items.append(queue_item(p,members[0],context,book,by_player,best,clubs,people,previous,slate,native,changes,cutoff))
    items.sort(key=lambda i:(i["category"] not in ("scoring revision","revised_game"),not i["consider_for_review"],-i["priority"],i["thread_id"]))
    corrections=[]
    for g in sorted(changes["revised"]):
        c=next((r for r in sources if r["signal_type"]=="revised_game" and int(r["subject_id"])==g),None)
        if c is None:
            raise ValueError(f"Missing Story Engine correction for game {g}")
        evidence=native_evidence[c["candidate_id"]]
        notes=[r["value"] for r in evidence if r["metric"]=="record revision"]
        if not notes:
            raise ValueError(f"Missing before-and-after evidence for correction to game {g}")
        corrections.append(dict(game_id=g,date=book[g]["date"],game=result(book[g],clubs),changes=notes,
                                evidence=[f"{Path(r['source_file']).name} {r['source_row_key']}" for r in evidence]))
    subtitle=f"ABL Newsroom | {when(v['league_date'])}, {day(v['league_date']).year} | Results through {when(cutoff)}"
    update=[]
    if corrections:
        update.append(("Scoring corrections",[f"Game {r['game_id']} on {when(r['date'])}. {r['game']}. "+" ".join(r["changes"]) for r in corrections]))
    update.extend((d["heading"],[d["movement"]+" "+d["results"],d["next"]]) for d in divisions)
    write_report(stage/"abl-411","ABL 411 | Around the six divisions",update,subtitle)
    team_slates=[]
    for tid,b in sorted(beats.items(),key=lambda x:(x[1]["division"],x[1]["team"])):
        picks=[i for i in items if i["desk_team_id"]==tid]
        lead=picks[0]
        team_slates.append((b["team"],[lead["title"]+". "+lead["event"],lead["next_condition"]]))
    write_report(stage/"story-slate","Reporting desk | 24 franchise assignments",team_slates,subtitle)
    store(stage/"queue.json",dict(issue_date=v["league_date"],cutoff=cutoff,source_id=v["source_id"],edition=cycle,
                                  prior_edition=ledger["previous_edition"],source_directory=str(raw),divisions=divisions,
                                  corrections=corrections,candidates=items))
    store(stage/"slate.json",slate)
