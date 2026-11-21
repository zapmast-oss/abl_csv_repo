"""Run one verified, internal ABL edition. Production Control invokes this entry point."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import hashlib
from html import escape
import json
import os
from pathlib import Path
import shutil
import sys
from uuid import uuid4
from datetime import datetime, timezone

from abl_newsroom_beats import build_beats
from abl_newsroom_corrections import changed_lines
from abl_newsroom_editorial import prepare_editorial, audit_editorial
from abl_newsroom_print import write_report
from abl_newsroom_verify import verify_export
from z_abl_story_engine_current_1981 import Builder, CANDIDATE_FIELDS, EVIDENCE_FIELDS, HIERARCHY_ORDER, write_csv


REPO = Path(__file__).resolve().parents[2]
ARCHIVE = REPO / "csv/out/story/newsroom"
PUBLICATION = REPO / "csv/out/story/manifests/july20_publication_checkpoint_1981_07_20_asof_1981-07-19.md"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def journal(archive, status, **details):
    with (archive / "operations.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(), status=status, **details)) + "\n")


@contextmanager
def exclusive(archive):
    """An OS lock releases automatically on a crash."""
    lock = (archive / "newsroom.lock").open("a+b")
    try:
        lock.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        lock.close()


def game_index(verified):
    return {g["game_id"]: {k: g[k] for k in ("date", "home_team", "away_team", "runs0", "runs1", "innings",
            "hits0", "hits1", "errors0", "errors1", "starter0", "starter1", "winning_pitcher", "losing_pitcher", "save_pitcher")}
            for g in verified["games"]}


def game_delta(v, previous):
    games = game_index(v)
    if previous is None or previous["season"] != v["season"]:
        if previous and v["season"] < previous["season"]:
            raise ValueError("Source would reverse the season")
        return games, {int(i) for i,g in games.items() if g["date"] == v["cutoff"]}, set()
    old = previous["games"]
    if v["cutoff"] < previous["cutoff"] or v["league_date"] < previous["league_date"] or old.keys()-games.keys():
        raise ValueError("Source withdraws completed decisions or reverses the verified calendar. Prior edition remains current.")
    return games, {int(i) for i in games.keys()-old.keys()}, {int(i) for i in games.keys() & old.keys() if games[i] != old[i]}


def assignments(beats):
    remit = ["franchise record", "current roster and farm system", "division race", "game and player performance",
             "transactions", "open questions", "coverage history"]
    return {str(t): {"assignment": f"{b['team']} beat", "franchise_id": t, "remit": remit} for t,b in beats.items()}


def continuity(beat, prior, cycle, cutoff):
    """Carry observations forward. Each archive stores its predecessor and the first/last evidence for each question."""
    old = prior or {}
    threads = dict(old.get("threads", {}))
    current_keys = set()
    for s in beat["stories"]:
        key = s["key"]
        current_keys.add(key)
        entry = dict(threads.get(key, {"opened": cycle, "first_cutoff": cutoff, "observations": 0}))
        if entry.get("evidence") != s["evidence"]:
            entry.update(last_update=cycle, observations=entry["observations"] + 1)
        entry.update(question=s["question"], evidence=s["evidence"], status="watching")
        threads[key] = entry
    for key in threads.keys()-current_keys:
        threads[key] = dict(threads[key], status="carried from earlier coverage")

    ids = set(beat["roster_ids"])
    old_ids = set(old.get("roster_ids", ids))
    names = {str(p["player_id"]):p["name"] for p in beat["roster"]}
    flags = {str(p["player_id"]):[p["injury_flag"],p["day_to_day_flag"]] for p in beat["roster"]}
    roster_changes = []
    for change, diff in [("entered the active list", ids-old_ids), ("left the active list", old_ids-ids)]:
        for pid in sorted(diff):
            name = names.get(str(pid)) or old.get("names",{}).get(str(pid)) or f"Player #{pid}"
            roster_changes.append(f"{name} {change} between exports. [players_roster_status.csv, team_id={beat['team_id']}, player_id={pid}]")
    for pid in flags.keys() & old.get("flags",{}).keys():
        if flags[pid]!=old["flags"][pid]:
            roster_changes.append(f"{names[pid]} has updated injury/day-to-day flags {flags[pid]}, previously {old['flags'][pid]}. [players.csv player_id={pid}, two export snapshots]")
    deals = {str(x["id"]): x["summary"] for x in beat["transactions"]}
    old_deals = old.get("deals", deals)
    transfers = [f"{x['date']} — {x['summary']}. [{x['source']}]" for x in beat["transactions"]
                 if old_deals.get(str(x["id"])) != x["summary"]]
    for key in old_deals.keys()-deals.keys():
        transfers.append(f"The current trade ledger omits earlier entry {key}. Compare the prior notebook before citing it.")
    return {"threads": threads, "roster_ids": sorted(ids), "names":names, "flags":flags, "deals": deals}, roster_changes+transfers


def game_label(g, beats):
    visitor, host = beats[int(g["away_team"])], beats[int(g["home_team"])]
    return f"{visitor['team']} {g['runs0']}, {host['team']} {g['runs1']}"


def build_menu(v, cycle, beats, new, corrected, changes, revisions):
    """One story per game, including both beat perspectives and exact keys into the archived export."""
    builder = Builder(slug=cycle, run_id=cycle, newsroom_date=v["league_date"], as_of=v["cutoff"], season=v["season"])
    raw = Path(v["raw_dir"])
    base = raw.relative_to(REPO).as_posix()
    for g in v["games"]:
        gid = int(g["game_id"])
        if gid not in new | corrected:
            continue
        a, h = beats[int(g["away_team"])], beats[int(g["home_team"])]
        margin = abs(int(g["runs0"])-int(g["runs1"]))
        weight = 45 + (8 if margin <= 1 else 0) + (20 if margin >= 10 else 0) + (8 if int(g["innings"]) > 9 else 0)
        race_facts = [b["stories"][0]["evidence"] for b in (a,h)]
        latest = g["date"] == v["cutoff"]
        movement = latest and any(b["division_leaders_change"] for b in (a,h)) and all(len(b["game_ids"])==1 for b in (a,h))
        label = game_label(g, beats)
        sources = [f"{base}/{name}.csv" for name in ("games","games_score","teams","divisions","sub_leagues")]
        detail = f"{g['date']}, game #{gid}, {g['innings']} innings. {label}."
        correction_evidence = []
        if gid in corrected:
            for note in revisions[gid]:
                filename = next((n for n in ("players_game_batting.csv","players_game_pitching_stats.csv","games_score.csv") if n in note), "games.csv")
                source = f"{base}/{filename}"
                if source not in sources:
                    sources.append(source)
                correction_evidence.append(dict(evidence_class="official",metric="record revision",value=note,
                    comparison="Prior verified export",source_file=source,source_row_key=f"game_id={gid}",notes=note))
            detail += " " + " ".join(revisions[gid])
        # Name the game before comparing its result with standings at the cycle cutoff.
        builder.add(candidate_class="proven_current_fact", hierarchy="standings" if movement else "game",
            signal="revised_game" if gid in corrected else "completed_game", subject_type="matchup", subject_id=str(gid),
            subject_name=label, related=[a["team"],h["team"]], headline=("Revised game record — " if gid in corrected else "")+label,
            stakes=" ".join(race_facts), summary=detail, sources=sources, score=min(99,weight+30*movement), confidence="high",
            evidence=[dict(evidence_class="official",metric="away-home runs",value=f"{g['runs0']}-{g['runs1']}",
                           comparison=f"{g['innings']} innings",source_file=sources[0],source_row_key=f"league_id=200 game_id={gid} date={g['date']}",notes="Checked against inning scoring."),*correction_evidence])
    for tid, facts in changes.items():
        if not facts:
            continue
        b = beats[tid]
        sources = [f"{base}/players_roster_status.csv", f"{base}/trade_history.csv",f"{base}/players.csv"]
        builder.add(candidate_class="proven_current_fact", hierarchy="management_organization", signal="personnel_change",
            subject_type="team",subject_id=str(tid),subject_name=b["team"],related=[],headline=f"{b['team']} personnel update",
            stakes=b["stories"][0]["evidence"],summary=" ".join(facts),sources=sources,score=55,confidence="high",
            evidence=[dict(evidence_class="official",metric="observed change",value=f,comparison="Previous admitted export",
                source_file=sources[0 if "active list" in f else 2 if "injury/day-to-day" in f else 1],source_row_key=f"team_id={tid}; {f}",notes="See archived beat ledger.") for f in facts])
    ranked = sorted(builder.candidates,key=lambda r:(HIERARCHY_ORDER[r["hierarchy_level"]],-r["signal_score"],r["subject_id"]))
    return ranked, builder.evidence


def write_edition(stage, v, previous, cycle, previous_cycle, previous_raw=None):
    games, new, corrected = game_delta(v, previous)
    baseline = previous is None or previous["season"] != v["season"]
    revisions = defaultdict(list)
    if not baseline:
        revisions.update(changed_lines(previous_raw,v["raw_dir"],games.keys() & previous["games"].keys(),v["season"]))
        for gid in corrected:
            old = previous["games"][str(gid)]
            latest = games[str(gid)]
            fields = ", ".join(f"{k} {old[k]} to {val}" for k,val in latest.items() if old[k]!=val)
            revisions[gid].insert(0,f"Revised game record ({fields}). [games.csv game_id={gid}]")
        corrected.update(revisions)
    beats = build_beats(Path(v["raw_dir"]), v["cutoff"], new, v["league_date"])
    if set(beats) != {int(t) for t in v["teams"]} or len(beats)!=24:
        raise ValueError("The franchise assignment register does not cover the 24 verified teams")
    save(stage/"assignments.json", assignments(beats))
    edition = f"League date {v['league_date']} | Completed regular-season games through {v['cutoff']} | Internal edition"
    opening = ("Opening edition. Coverage begins with the latest completed game date. Earlier decisions establish the season records."
               if baseline else f"Previous edition {previous_cycle}.")
    team_states, changes = {}, {}
    for tid, beat in beats.items():
        key = str(tid)
        prior = previous["beats"].get(key) if previous and not baseline else None
        state, changed = continuity(beat, prior, cycle, v["cutoff"])
        changes[tid] = changed
        team_states[key] = state
        path = stage/"teams"/f"{tid:02}-{beat['abbr']}"
        path.parent.mkdir(parents=True,exist_ok=True)
        save(path.with_suffix(".json"),dict(beat,continuity=state,previous_edition=previous_cycle))
        sections = [(s["title"],s["paragraphs"]) for s in beat["sections"]]
        corrections = [note for gid in sorted(corrected) if key in (games[str(gid)]["away_team"],games[str(gid)]["home_team"]) for note in revisions[gid]]
        sections.insert(1,("Changes to earlier results and personnel", corrections+changed))
        carried = [s["question"]+" Last recorded evidence: "+s["evidence"] for s in state["threads"].values() if s["status"].startswith("carried")]
        sections.append(("Earlier open questions",carried))
        sections.append(("Edition record",[opening, f"Verified source archive {v['source_id']}. Source filenames in this notebook refer to that export. Club #{tid}."]))
        write_report(path, beat["team"]+" | Franchise beat", sections,edition)

    candidates,evidence = build_menu(v,cycle,beats,new,corrected,changes,revisions)
    write_csv(stage/"story_candidates.csv",CANDIDATE_FIELDS,candidates)
    write_csv(stage/"story_evidence.csv",EVIDENCE_FIELDS,evidence)
    write_csv(stage/"story_menu.csv",["menu_rank",*CANDIDATE_FIELDS],[dict(menu_rank=i,**row) for i,row in enumerate(candidates,1)])
    leads = candidates[:4]
    lead_notes = []
    for c in leads:
        lead_notes.append(c["evidence_summary"]+" "+c["stakes"])
        if c["signal_type"]=="completed_game":
            gid = int(c["subject_id"])
            for tid in [int(games[str(gid)]["away_team"]),int(games[str(gid)]["home_team"])]:
                sections = beats[tid]["sections"]
                focus = next(s for s in sections if s["title"]=="Reporting focus")
                notes = [s for s in focus["paragraphs"] if f"game_id={gid} " in s]
                lead_notes.extend(notes)
    divisions = defaultdict(list)
    for b in beats.values():
        divisions[b["division"]].append(b)
    raceboard = []
    for division, group in sorted(divisions.items()):
        group.sort(key=lambda b:(-b["wins"]/max(1,b["wins"]+b["losses"]), b["team"]))
        raceboard.append(division+" — "+", ".join(f"{b['team']} {b['wins']}-{b['losses']}" for b in group)+
                         f". [teams.csv and completed games.csv through {v['cutoff']}]")
    slate = [f"{games[str(i)]['date']} | {game_label(games[str(i)],beats)}. [games.csv #{i}]" for i in sorted(new|corrected,key=lambda i:(games[str(i)]["date"],i))]
    activity = f"{len(games)} completed games. {v['day']['completed']} of {v['day']['scheduled']} on the cutoff date. {len(new)} games in this edition’s reporting window. {len(corrected)} revised game records."
    exceptions = ", ".join(str(x["game_id"]) for x in v["log_exceptions"])
    source_note = f"Source archive {v['source_id']}. Preflight {v['preflight']['verdict']}. Full raw hashes, exact paths and log exceptions are in verification.json."
    write_report(stage/"ernie_briefing","Ernie Bewell | League briefing",
        [("At the desk",[activity,opening]),("Lead stories",lead_notes or ["No fresh game or personnel story met the reporting criteria."]),
         ("Division races",raceboard),("Results on the assignment board",slate),
         ("Follow next",[b["stories"][1]["question"]+" "+b["stories"][1]["evidence"] for b in sorted(beats.values(),key=lambda b:-b["priority"])[:3]] if leads else []),
         ("Verification",[source_note]+([f"Game-log exceptions omitted from coverage: {exceptions}."] if exceptions else []))],edition)
    significant = [c for c in candidates if c["signal_type"]=="revised_game" or c["signal_score"]>=65][:3]
    if significant:
        write_report(stage/"abl_411","ABL 411 | Internal update",[("Around the league",[c["evidence_summary"]+" "+c["stakes"] for c in significant])],edition)
    state = dict(cutoff=v["cutoff"],league_date=v["league_date"],season=v["season"],source_id=v["source_id"],games=games,beats=team_states)
    save(stage/"state.json",state)
    save(stage/"verification.json",{k:value for k,value in v.items() if k not in ("teams","games")})
    save(stage/"coverage.json",dict(previous_edition=previous_cycle,baseline=baseline,completed_game_ids=sorted(new),corrected_game_ids=sorted(corrected),
        source_archive=v["raw_dir"],internal=True,externally_published=False,publication_receipts=[str(PUBLICATION.relative_to(REPO))] if PUBLICATION.exists() else [],
        story_count=len(candidates),abl_411=bool(significant),assignments=len(beats),reports={str(t):f"teams/{t:02}-{b['abbr']}.md" for t,b in beats.items()}))


def audit(path):
    receipt=load(path/"files.json")
    for name,expected in receipt.items():
        if digest(path/name)!=expected:
            raise ValueError(f"Archive integrity failed: {path/name}")
    recorded=load(path/"coverage.json")
    if recorded["assignments"]!=24 or len(list((path/"teams").glob("*.json")))!=24:
        raise ValueError("Missing beat notebooks")
    for p in path.rglob("*.md"):
        if not p.with_suffix(".pdf").is_file():
            raise ValueError(f"Missing PDF for {p}")


def summary_for(path):
    v=load(path/"verification.json")
    return dict(edition=path.name,cutoff=v["cutoff"],league_date=v["league_date"],season=v["season"],
                source_id=v["source_id"],reports=str(path/"teams"),ernie=str(path/"ernie_briefing.pdf"),
                story_menu=str(path/"story_menu.csv"),raw_dir=v["raw_dir"])


def last_edition(archive):
    pointer=archive/"current.json"
    if pointer.is_file():
        return load(pointer)
    editions=archive/"editions"
    if not editions.exists():
        return {}
    named={p.name:p for p in editions.iterdir() if p.is_dir()}
    if not named:
        return {}
    parents={name:load(p/"coverage.json")["previous_edition"] for name,p in named.items()}
    tips=set(named)-set(parents.values())
    if len(tips)!=1:
        raise ValueError("Edition history has multiple tips and requires review")
    tip=tips.pop()
    visited=set()
    name=tip
    while name and name in named and name not in visited:
        visited.add(name)
        name=parents[name]
    if name or visited!=set(named):
        raise ValueError("Incomplete edition history. Restore the missing archive before continuing.")
    audit(named[tip])
    return summary_for(named[tip])


def reading_index(archive, scratch, summary):
    edition=summary["edition"]
    path=archive/"editions"/edition
    coverage=load(path/"coverage.json")
    registered=load(path/"assignments.json")
    prefix=f"editions/{edition}/"
    def link(filename,title):
        return f'<a href="{escape(prefix+filename,quote=True)}">{escape(title)}</a>'
    teams=[]
    for tid,team in sorted(registered.items(),key=lambda item:item[1]["assignment"]):
        md=coverage["reports"][tid]
        teams.append(f'<li>{link(md[:-3]+".pdf",team["assignment"])} <small>{link(md,"Markdown")}</small></li>')
    index=load(Path(summary["editorial"])/"articles.json")
    slate=load(Path(summary["editorial"])/"slate.json")
    def article_link(item):
        stem=f"journalism/{item['edition']}/{item['path'][:-3]}"
        return f'<li><a href="{escape(stem)}.pdf">{escape(item["title"])}</a> <small><a href="{escape(stem)}.md">Markdown</a></small></li>'
    def section(kind, heading):
        items=[item for item in index["articles"] if item["kind"]==kind]
        if not items:
            past=next((item for item in reversed(slate["coverage"]) if item["kind"]==kind),None)
            items=[past] if past else []
            if past:
                heading+=f" · most recent, {past['cutoff']}"
        return f"<h2>{heading}</h2><ul>{''.join(article_link(item) for item in items)}</ul>" if items else ""
    top=f'journalism/{edition}/'
    masthead=[f'<a href="{top}story-slate.pdf">24-club story slate</a>',f'<a href="{top}story-slate.md">Slate in Markdown</a>',
              f'<a href="{top}slate.json">Investigations and coverage history</a>',f'<a href="{top}articles.json">Article evidence</a>',
              f'<a href="{top}story_candidates.csv">Editorial Story Engine candidates</a>',f'<a href="{top}story_evidence.csv">Editorial Story Engine evidence</a>',
              link("story_menu.csv","Game and personnel signals"),link("verification.json","Source verification")]
    update=(f'<h2>ABL 411</h2><p>{link("abl_411.pdf","League update")} · {link("abl_411.md","Markdown")}</p>'
            if coverage["abl_411"] else '<h2>ABL 411</h2><p>No new bulletin in this edition.</p>')
    corrections=[]
    for item in reversed(slate.get("correction_register",[])):
        source=f"journalism/{item['edition']}/{item['desk'][:-3]}.pdf"
        corrections.append(f'<li><a href="{escape(source)}">Game {item["game_id"]} · amended reporting, {item["cutoff"]}</a></li>')
    correction_box="<h2>Scoring corrections</h2><ul>"+"".join(corrections)+"</ul>" if corrections else ""
    page=f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ABL Newsroom | Current edition</title><style>
body{{font:1.05rem/1.65 system-ui,sans-serif;color:#142e3b;max-width:72rem;margin:2rem auto;padding:0 1.5rem}}h1,h2{{font-family:Georgia,serif}}h1{{margin-bottom:0}}a{{color:#075e63}}ul{{columns:3;column-width:18rem;padding-left:1.25rem}}li{{break-inside:avoid;padding:.25rem 0}}small{{margin-left:.5rem}}nav a{{display:inline-block;margin:0 1.5rem .5rem 0}}
</style><main><h1>ABL Newsroom</h1><p>League date {summary['league_date']} · Games through {summary['cutoff']}</p>
{section('observer','The Baseball Observer')}{section('beat','Beat reporting')}{section('league','League reporting')}{update}{correction_box}
<h2>Franchise intelligence</h2><ul>{''.join(teams)}</ul><h2>Reporting desk</h2><nav>{' '.join(masthead)}</nav>
<p><a href="journalism/">Earlier journalism</a> · <a href="editions/">Intelligence archive</a> · <a href="current.json">Current edition record</a> · <a href="operations.jsonl">Operations log</a></p></main></html>"""
    target=archive/"index.html"
    if not target.is_file() or target.read_text(encoding="utf-8")!=page:
        tmp=scratch/"index.html"
        tmp.write_text(page,encoding="utf-8")
        os.replace(tmp,target)


@contextmanager
def workspace(archive):
    path = archive / (".building-" + uuid4().hex)
    path.mkdir()
    try:
        yield path
    finally:
        if path.resolve().parent == archive.resolve():
            shutil.rmtree(path)


def run(archive, source):
    archive.mkdir(parents=True,exist_ok=True)
    with exclusive(archive), workspace(archive) as scratch:
        stage = Path(scratch)/"edition"
        stage.mkdir()
        pointer = archive/"current.json"
        pointer_data = last_edition(archive)
        previous_cycle = pointer_data.get("edition")
        prior_path = archive/"editions"/previous_cycle if previous_cycle else None
        if prior_path:
            audit(prior_path)
            if not pointer.is_file():
                recovered=scratch/"recovered.json"
                save(recovered,pointer_data)
                os.replace(recovered,pointer)
                journal(archive,"recovered_pointer",edition=previous_cycle)
        v = verify_export(REPO,stage/"preflight",source,snapshot_root=None if archive==ARCHIVE else archive/"snapshots")
        same = v["source_id"] == pointer_data.get("source_id")
        cycle = v["cutoff"]+"_"+v["source_id"][:16]
        dest = archive/"editions"/cycle
        dest.parent.mkdir(exist_ok=True)
        if dest.exists():
            audit(dest)
            if not same and load(dest/"coverage.json")["previous_edition"] != previous_cycle:
                raise ValueError("Previously archived export replayed out of sequence")
        else:
            previous = load(prior_path/"state.json") if prior_path else None
            write_edition(stage,v,previous,cycle,previous_cycle,pointer_data.get("raw_dir"))
            save(stage/"files.json",{p.relative_to(stage).as_posix():digest(p) for p in sorted(stage.rglob("*")) if p.is_file()})
            audit(stage)
        intel = dest if dest.exists() else stage
        supplements=archive/"journalism"
        supplements.mkdir(exist_ok=True)
        editorial=supplements/cycle
        added=not editorial.exists()
        if added:
            parent=load(intel/"coverage.json")["previous_edition"]
            prior_slate=supplements/parent/"slate.json" if parent else None
            past=load(prior_slate) if prior_slate and prior_slate.is_file() else {}
            pending=scratch/"journalism"
            pending.mkdir()
            prepare_editorial(pending,intel,past,cycle)
            pending.rename(editorial)
        else:
            audit_editorial(editorial,v["source_id"])
        if intel==stage:
            stage.rename(dest)
        summary=summary_for(dest)
        summary.update(editorial=str(editorial),observer=str(editorial/"observer/ernie-briefing.pdf"),slate=str(editorial/"story-slate.pdf"))
        reading_index(archive,scratch,summary)
        if pointer_data!=summary or not pointer.is_file():
            staged_pointer = Path(scratch)/"current.json"
            save(staged_pointer,summary)
            os.replace(staged_pointer,pointer)
        status = "editorial_activated" if same and added else "unchanged" if same else "completed"
        journal(archive,status,**summary)
        return dict(status=status,**summary)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,help="OOTP export folder (defaults to the live ABL export)")
    parser.add_argument("--archive",type=Path,default=ARCHIVE,help="Alternate newsroom archive for isolated replay")
    parser.add_argument("--status",action="store_true")
    args=parser.parse_args()
    archive=args.archive.resolve()
    if archive!=ARCHIVE and not archive.is_relative_to(REPO/"tmp"):
        parser.error("Isolated replays must use a directory beneath repo/tmp")
    try:
        if args.status:
            current=load(archive/"current.json")
            audit(archive/"editions"/current["edition"])
            if current.get("editorial"):
                audit_editorial(Path(current["editorial"]),current["source_id"])
            print(json.dumps(current,indent=2))
        else:
            print(json.dumps(run(archive,args.source),indent=2))
    except Exception as exc:
        archive.mkdir(parents=True,exist_ok=True)
        journal(archive,"failed",error=str(exc),source=str(args.source or "live"))
        print(f"Newsroom failed: {exc}",file=sys.stderr)
        return 1
    return 0


if __name__=="__main__":
    raise SystemExit(main())
