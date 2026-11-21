"""Select, fact-check and file internal reporting alongside an immutable intelligence edition."""
import csv
import hashlib
import json
import re
from pathlib import Path

from abl_newsroom_beats import day, rows
from abl_newsroom_observer import prepare_observer
from abl_newsroom_factcheck import ClaimCheck
from abl_newsroom_print import write_report
from z_abl_story_engine_current_1981 import Builder, CANDIDATE_FIELDS, EVIDENCE_FIELDS, write_csv


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def store(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def audit_editorial(path, source):
    receipt = read(path/"files.json")
    contents = {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()} - {"files.json"}
    if set(receipt) != contents or any(sha(path/k)!=v for k,v in receipt.items()):
        raise ValueError(f"Editorial archive integrity failure: {path}")
    manifest = read(path/"articles.json")
    if manifest["source_id"]!=source or len(read(path/"slate.json")["teams"])!=24:
        raise ValueError("Editorial source or beat register mismatch")
    for article in manifest["articles"]:
        if not {article["path"],article["path"][:-3]+".pdf"} <= contents:
            raise ValueError(f"Incomplete reading copies: {article['path']}")
    if not {"observer/ernie-briefing.pdf","story-slate.pdf","story_candidates.csv","story_evidence.csv"} <= contents:
        raise ValueError("Editorial desk records are incomplete")


def body_digest(piece):
    return hashlib.sha256(json.dumps([piece["title"],piece["sections"]],ensure_ascii=False).encode()).hexdigest()


def release_problem(piece, v):
    if not (piece.get("title") and piece.get("sections") and piece.get("evidence")):
        return "Missing copy or supporting evidence"
    names = set(re.findall(r"[\w]+\.csv", " ".join(piece["evidence"])))
    if not names.intersection(v["hashes"]):
        return "Source references do not identify a verified export table"
    text = " ".join([piece["title"], *(p for _, ps in piece["sections"] for p in ps)])
    if re.search(r"\b(?:told reporters|according to a clubhouse source|fans were furious|PLACEHOLDER|TBD)\b",text,re.I):
        return "Unsupported attribution or unfinished copy"
    if re.search(r"\b(?:1 earned runs|1 innings|1 games|1 home runs)\b",text):
        return "Singular/plural mismatch"
    return None


def file_article(stage, piece, stem, edition, v):
    # Full row selectors travel with the archive. Readers also get the source tables in their copy.
    tables = sorted(set(re.findall(r"[\w]+\.csv"," ".join(piece["evidence"]))).intersection(v["hashes"]))
    copy = list(piece["sections"]) + [("Reporting record", [
        "Sources in the verified ABL export: "+", ".join(tables)+". Full row references accompany this edition in articles.json."])]
    credit="The Baseball Observer  |  Ernie Bewell" if piece["kind"]=="observer" else piece.get("byline","League desk")
    write_report(stage/stem,piece["title"],copy,credit+"  |  "+edition,article=True)
    return dict(path=stem+".md",title=piece["title"],team_ids=piece["team_ids"],kind=piece["kind"],
                key=piece["key"],trigger=piece["trigger"],evidence=piece["evidence"],body_sha256=body_digest(piece))


def record_pitches(stage, slate, v, cycle, raw):
    builder = Builder(slug=cycle,run_id=cycle,newsroom_date=v["league_date"],as_of=v["cutoff"],season=v["season"])
    for tid, team in slate["teams"].items():
        for p in team["pitches"]:
            records = []
            for selector in p["evidence"]:
                for name in dict.fromkeys(re.findall(r"[\w]+\.csv",selector)):
                    if name in v["hashes"]:
                        records.append(dict(evidence_class="official",metric=p.get("category","beat reporting"),value=p["observation"],
                            comparison=p.get("assessment",""),source_file=str(raw/name),source_row_key=selector,notes=p["significance"]))
            if not records:
                raise ValueError(f"Pitch without a verified source table: {p['key']}")
            files = list(dict.fromkeys(r["source_file"] for r in records))
            builder.add(candidate_class="editorial_investigation",hierarchy=p.get("hierarchy","team_identity"),
                signal="beat_pitch",subject_type="franchise_investigation",subject_id=p["key"],
                subject_name=team["assignment"],related=[str(tid)],headline=p["title"],stakes=p["significance"],
                summary=p["observation"],sources=files,score=p["score"],confidence="documented",evidence=records)
    write_csv(stage/"story_candidates.csv",CANDIDATE_FIELDS,builder.candidates)
    write_csv(stage/"story_evidence.csv",EVIDENCE_FIELDS,builder.evidence)


def reported_brief(p, tid, checker, beats, raw):
    evidence=list(p["evidence"])
    title,lead,reading=p["title"],p["observation"],p["significance"]
    question=p["question"]
    if p["category"]=="game":
        gid=int(p["key"].split(":")[2])
        g=checker.games[gid]
        a,h=int(g["runs0"]),int(g["runs1"])
        if a!=h:
            side=int(h>a)
            winner,loser=(int(g["home_team"]),int(g["away_team"])) if side else (int(g["away_team"]),int(g["home_team"]))
            city,other=checker.clubs[winner]["name"],checker.clubs[loser]["name"]
            line=[r for r in rows(raw,"games_score") if int(r["game_id"])==gid and int(r["team"])==side]
            frame=max(line,key=lambda r:(int(r["score"]),-int(r["inning"])))
            tally,inning=int(frame["score"]),int(frame["inning"])
            ordinal={1:"first",2:"second",3:"third",4:"fourth",5:"fifth",6:"sixth",7:"seventh",8:"eighth",9:"ninth"}.get(inning,f"inning {inning}")
            title=f"{city} scores {tally} in the {ordinal} to beat {other}" if inning<=9 else f"{city} beats {other} in {g['innings']} innings"
            when=day(g["date"])
            lead=f"{city} beat {other} {max(a,h)}–{min(a,h)} on {when:%B} {when.day}. The biggest inning for {city} was the {ordinal}, when it scored {tally}."
            hits=[r for r in rows(raw,"players_game_batting") if r["league_id"]=="200" and r["level_id"]=="1" and int(r["year"])==when.year and int(r["game_id"])==gid and int(r["team_id"])==winner]
            if hits:
                hit=max(hits,key=lambda r:(int(r["rbi"]),int(r["hr"]),int(r["h"])))
                person=next(r for r in rows(raw,"players") if r["player_id"]==hit["player_id"])
                if int(hit["rbi"]):
                    lead+=f" {person['first_name']} {person['last_name']} went {hit['h']}-for-{hit['ab']} and drove in {hit['rbi']}."
                    evidence.append(f"players.csv and players_game_batting.csv, player_id={hit['player_id']}, team_id={winner}, game_id={gid}.")
            evidence.append(f"games_score.csv, game_id={gid}, team={side}, inning={inning}.")
            reading=re.sub(r"^On .*? innings\.\s*","",p["observation"])
            if reading==p["observation"]:
                reading=p["significance"]
    return dict(title=title,sections=[("",[lead,reading]),("What to watch",[question])],
                evidence=evidence,team_ids=[tid],kind="beat",key=p["key"],byline=beats[tid]["team"]+" beat")


def prepare_editorial(stage, intel, previous, cycle):
    from abl_newsroom_east import build_east
    from abl_newsroom_slate import build_slate

    v, ledger = read(intel/"verification.json"), read(intel/"coverage.json")
    raw, cutoff = Path(v["raw_dir"]), v["cutoff"]
    beats = {b["team_id"]:b for b in (read(path) for path in (intel/"teams").glob("*.json"))}
    with (intel/"story_candidates.csv").open(encoding="utf-8-sig",newline="") as stream:
        candidates = list(csv.DictReader(stream))
    new, revised = set(ledger["completed_game_ids"]),set(ledger["corrected_game_ids"])
    slate = build_slate(raw,cutoff,beats,candidates,previous,cycle)
    checker=ClaimCheck(raw,cutoff,v["league_date"],beats)
    if set(slate["teams"]) != {str(t) for t in beats} or any(not b["pitches"] for b in slate["teams"].values()):
        raise ValueError("Each franchise needs a monitored story slate")
    record_pitches(stage,slate,v,cycle,raw)
    history = list(previous.get("coverage",[]))
    latest = {r["key"]:r for r in history}
    edition = f"ABL Newsroom  |  League date {v['league_date']}  |  Games through {cutoff}"
    articles, held = [], []
    previous_reads=previous.get("assessment_digests",{})
    reads=dict(previous_reads)
    def release(piece, stem, force=False):
        old = latest.get(piece["key"])
        signature=body_digest(piece)
        reads[piece["key"]]=signature
        if old and not force:
            if old["body_sha256"] == signature:
                return False
            observed_change=signature!=previous_reads.get(piece["key"])
            references=" ".join(piece["evidence"])
            linked_game=any(re.search(rf"\b{gid}\b",references) for gid in revised)
            correction=observed_change and (linked_game or (cutoff==previous.get("cutoff") and not new and bool(revised)))
            if piece["trigger"]==old["trigger"] and not correction:
                return False
        problem = release_problem(piece,v)
        checked={}
        if not problem:
            try:
                checked=checker.evaluate(piece)
            except ValueError as error:
                problem=str(error)
        if problem:
            held.append(dict(key=piece["key"],title=piece["title"],reason=problem))
            return False
        entry = file_article(stage,piece,stem,edition,v)
        entry.update(edition=cycle,cutoff=cutoff,prior_edition=old["edition"] if old else None,checked=checked)
        history.append(entry)
        articles.append(entry)
        latest[entry["key"]]=entry
        return True

    east = build_east(raw,cutoff,new,beats,snapshot_date=v["league_date"])
    for tid, piece in east.get("teams",{}).items():
        tid = int(tid)
        piece.update(team_ids=[tid],kind="beat",key=f"east-{tid}",byline=beats[tid]["team"]+" beat")
        release(piece,f"beat/{tid:02}-{beats[tid]['abbr']}")
    if east.get("division"):
        piece = east["division"]
        piece.update(team_ids=sorted(int(t) for t in east["teams"]),kind="league",key="nbc-east")
        release(piece,"league/nbc-east")
    filed = {t for a in articles if a["kind"]=="beat" for t in a["team_ids"]}
    ranking = sorted(((p["score"],int(t),p) for t,b in slate["teams"].items() for p in b["pitches"] if p["eligible"]),
                     key=lambda x:(-x[0],x[1],x[2]["key"]))
    covered_events = set()
    # Give different investigations a hearing before commissioning a second version of the same race.
    subjects=[]
    for _,_,p in ranking:
        if p["category"] not in subjects:
            subjects.append(p["category"])
    varied=[next(row for row in ranking if row[2]["category"]==kind and row[1] not in filed)
            for kind in subjects if any(row[2]["category"]==kind and row[1] not in filed for row in ranking)]
    for _, tid, p in varied+ranking:
        if tid in filed or len(covered_events)>=4:
            continue
        event = p.get("trigger",p["key"])
        if not isinstance(event,str):
            event=json.dumps(event,sort_keys=True)
        if event in covered_events:
            continue
        piece=reported_brief(p,tid,checker,beats,raw)
        piece["trigger"]=event
        if release(piece,f"beat/{tid:02}-{beats[tid]['abbr']}"):
            filed.add(tid)
            covered_events.add(event)
    observer = prepare_observer(raw,cutoff,beats,new,revised,candidates,east,slate)
    if not release(observer,"observer/ernie-briefing",force=True):
        raise ValueError(f"Observer briefing held: {held[-1]['reason']}")
    corrections=list(previous.get("correction_register",[]))
    for gid in sorted(revised):
        affected=[dict(edition=a["edition"],path=a["path"],title=a["title"]) for a in previous.get("coverage",[])
                  if gid in a.get("checked",{}).get("games",[])]
        corrections.append(dict(game_id=gid,edition=cycle,cutoff=cutoff,desk="observer/ernie-briefing.md",
                                earlier_articles_citing_game=affected))
    slate["correction_register"]=corrections
    slate["coverage"]=history
    slate["assessment_digests"]=reads
    slate["cutoff"]=cutoff
    investigations=dict(previous.get("commissioned_investigations",{}))
    monitors={**{f"east-{t}":piece for t,piece in east["teams"].items()},**({"nbc-east":east["division"]} if east.get("division") else {})}
    for key,piece in monitors.items():
        prior=investigations.get(key,{})
        reports=[a for a in history if a["key"]==key]
        updates=list(prior.get("developments",[]))
        updated=any(a["key"]==key for a in articles)
        if updated:
            updates.append(dict(edition=cycle,cutoff=cutoff,title=piece["title"],evidence=piece["evidence"]))
        investigations[key]=dict(first_edition=prior.get("first_edition",cycle),originating_event=prior.get("originating_event",piece["title"]),
            team_ids=piece["team_ids"],status="active",last_examined=cycle,trigger=piece["trigger"],developments=updates,
            assessment="Updated coverage" if updated else "Examined and carried forward",
            latest_coverage=reports[-1]["edition"]+"/"+reports[-1]["path"] if reports else None)
    for key in investigations.keys()-monitors.keys():
        investigations[key]=dict(investigations[key],status="archived",last_examined=cycle,assessment="The current record no longer supports this angle")
    slate["commissioned_investigations"]=investigations
    # Full reporting history, including carried articles, lives at the desk for each club.
    slate_sections = []
    for tid, t in sorted(slate["teams"].items(),key=lambda item:beats[int(item[0])]["team"]):
        own = [a for a in history if int(tid) in a["team_ids"]]
        t["coverage_history"]=[{"edition":a["edition"],"key":a["key"],"path":a["path"],"title":a["title"]} for a in own]
        t["ready_to_read"]=[link for link in t["coverage_history"] if link["edition"]==cycle]
        for pitch in t["pitches"]:
            pitch["coverage_reference"] = next((a["edition"]+"/"+a["path"] for a in reversed(own) if a["key"]==pitch["key"]),None)
        slate_sections.append((beats[int(tid)]["team"],[t["assignment"]]+[
            f"{i}. {p['title']} — {p['observation']} {p['question']}" for i,p in enumerate(t["pitches"][:2],1)]))
    write_report(stage/"story-slate","Newsroom story slate | 24 franchise assignments",slate_sections,edition)
    store(stage/"slate.json",slate)
    store(stage/"articles.json",dict(edition=cycle,source_id=v["source_id"],cutoff=cutoff,league_date=v["league_date"],
        previous_edition=ledger["previous_edition"],articles=articles,held=held,external_publication=False,
        source_directory=str(raw),evidence_policy="Scores, appearances and standings use completed regular-season games through cutoff. Unplayed rows support schedules only. Personnel attributes use the identified export date."))
    store(stage/"files.json",{p.relative_to(stage).as_posix():sha(p) for p in stage.rglob("*") if p.is_file()})
    audit_editorial(stage,v["source_id"])
