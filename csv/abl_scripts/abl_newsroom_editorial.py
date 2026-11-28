"""Select, fact-check and file internal reporting alongside an immutable intelligence edition."""
import hashlib
import json
import re
from datetime import datetime, timezone
from html import escape
import os
from pathlib import Path

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


def seal(folder):
    store(folder/"files.json",{p.relative_to(folder).as_posix():sha(p) for p in sorted(folder.rglob("*")) if p.is_file()})


def audit_files(folder):
    receipt=read(folder/"files.json")
    present={p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}-{"files.json"}
    if set(receipt)!=present or any(sha(folder/k)!=v for k,v in receipt.items()):
        raise ValueError(f"Archive hash mismatch: {folder}")
    return present


def audit_desk(folder, source):
    contents=audit_files(folder)
    needed={"desk.json","slate.json","queue.json","story-slate.md","story-slate.pdf","abl-411.md","abl-411.pdf",
            "story_candidates.csv","story_evidence.csv"}
    if not needed<=contents or read(folder/"desk.json")["source_id"]!=source or len(read(folder/"slate.json")["teams"])!=24:
        raise ValueError(f"Incomplete reporting desk: {folder}")


def audit_issue(folder, source):
    contents=audit_files(folder)
    manifest=read(folder/"articles.json")
    if manifest["source_id"]!=source or not manifest["articles"] or "index.html" not in contents:
        raise ValueError(f"Incomplete approved issue: {folder}")
    for article in manifest["articles"]:
        stem=article["path"][:-3]
        if not {stem+suffix for suffix in (".html",".md",".pdf")}<=contents:
            raise ValueError(f"Approved reading copy missing: {stem}")
    return manifest


def copy_hash(value):
    source={k:v for k,v in value.items() if k!="review"}
    return hashlib.sha256(json.dumps(source,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()


def checked_review(doc, v, checker, threads):
    if any(doc.get(k)!=v[k] for k in ("source_id","cutoff","league_date")):
        raise ValueError("Review belongs to a different verified export or issue date")
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*",str(doc.get("slug",""))) or doc.get("kind") not in ("observer","beat","league"):
        raise ValueError("Review needs a valid article slug and desk")
    review=doc.get("review",{})
    if (review.get("decision")!="approve" or review.get("copy_sha256")!=copy_hash(doc)
            or not all(isinstance(review.get(k),str) and review[k].strip() for k in ("reviewer","question","promise","sequence","stakes","ending"))):
        raise ValueError("Editorial approval of this exact copy is missing or incomplete")
    if not doc.get("byline") or not doc.get("story_ids") or not set(doc["story_ids"])<=set(threads):
        raise ValueError("Approved copy must name its desk and link to current investigations")
    if not doc.get("team_ids") or not set(doc["team_ids"])<=checker.clubs.keys():
        raise ValueError("Approved copy must identify its ABL franchises")
    problem=release_problem(doc,v)
    if problem:
        raise ValueError(problem)
    checked=checker.evaluate(doc)
    named=set(doc.get("claim_games",[]))
    if not named or not named<=set(checked["games"]):
        raise ValueError("Article game claims need matching, verified row selectors")
    for item in doc.get("claim_map",[]):
        if not item.get("claim") or not item.get("games") or not set(item["games"])<=named:
            raise ValueError("A correction reference does not identify a verified article claim")
    excluded={int(row["game_id"]) for row in v.get("log_exceptions",[])}
    for ref in doc["evidence"]:
        if "game_logs.csv" in ref:
            matches=re.findall(r"game[ _]ids?\s*[=:]?\s*(\d+(?:\s*,\s*\d+)*)",ref,re.I)
            cited={int(gid) for ids in matches for gid in ids.split(",")}
            if not cited or cited & excluded:
                raise ValueError("Game-log citation is ambiguous or refers to an excluded block")
    return checked


STYLE="""*{box-sizing:border-box}body{margin:0;background:#f9f7f0;color:#172e31;font:17px/1.55 Georgia,serif}
main{max-width:960px;margin:auto;padding:26px clamp(18px,4vw,48px) 60px}header{border-top:5px solid #214a46;border-bottom:1px solid #73847c;margin-bottom:24px}
h1,h2,h3,p{margin:0 0 .72em}h1{font-size:clamp(2.1rem,5vw,3.25rem);line-height:1.1;letter-spacing:-.035em}
h2{font-size:1.55rem;line-height:1.25;margin-top:1.3em}h3{font-size:1.15rem;margin-top:1em}.date,nav,summary,footer,.meta{font: .91rem/1.55 system-ui,sans-serif}
.date{margin:.65em 0;color:#435d56}a{color:#12594d;text-underline-offset:.15em}article{max-width:680px}article p{font-size:1.12rem}
.lead{font-size:1.8rem;line-height:1.17}ul{padding-left:1.3rem}li{margin:.3rem 0}details{margin:1rem 0}summary{cursor:pointer;font-weight:650}
.divisions{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,320px),1fr));gap:0 1.6rem}
.divisions section{border-top:1px solid #bbc5bc;padding:.5rem 0}.divisions p{font-size:.96rem}.clubs{columns:3 15rem}.clubs li{break-inside:avoid}footer{border-top:1px solid #a5b3a8;margin-top:2rem;padding-top:1rem}
@media print{body{background:white}main{padding:0}.clubs{columns:2}}"""


def page(title, body):
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{escape(title)} | ABL Newsroom</title><style>{STYLE}</style></head><body><main>{body}</main></body></html>')


def link(target,label):
    return f'<a href="{escape(target,quote=True)}">{escape(label)}</a>'


def render_article(stage, doc, checked, v):
    stem=doc["kind"]+"/"+doc["slug"]
    target=stage/stem
    credit=doc["byline"]+"  |  "+v["league_date"]+"  |  Games through "+v["cutoff"]
    files=", ".join(sorted(checked["source_tables"]))
    brief=f"Verified against {files}. Game and player references are in the companion HTML reporting record."
    write_report(target,doc["title"],list(doc["sections"])+[("Reporting record",[brief])],credit,article=True)
    text="".join((f"<h2>{escape(h)}</h2>" if h else "")+"".join(f"<p>{escape(p)}</p>" for p in ps) for h,ps in doc["sections"])
    evidence="".join("<li>"+escape(ref)+"</li>" for ref in doc["evidence"])
    name=doc["slug"]
    body=(f'<header><p class="date">{escape(credit)}</p><h1>{escape(doc["title"])}</h1></header><article>{text}</article>'
          f'<details><summary>Reporting record</summary><ul>{evidence}</ul></details><footer>'
          +link(name+".pdf","PDF")+ " · "+link(name+".md","Markdown")+" · "+link("../index.html","This issue")+"</footer>")
    target.with_suffix(".html").write_text(page(doc["title"],body),encoding="utf-8")
    return dict(key=name,slug=name,title=doc["title"],kind=doc["kind"],team_ids=doc["team_ids"],
                path=stem+".md",lead=doc.get("lead",False),story_ids=doc["story_ids"],review=doc["review"],
                evidence=doc["evidence"],checked=checked,claim_games=doc["claim_games"],claim_map=doc.get("claim_map",[]))


def active_issue(archive, cycle):
    folder=archive/"issues"/cycle
    active=folder/"active.json"
    if active.exists():
        revision=read(active)["revision"]
        if not re.fullmatch(r"[0-9a-f]{20}",revision):
            raise ValueError("Invalid issue revision")
        return folder/revision
    # Recover from a crash between admitting a complete directory and writing its pointer.
    if folder.exists():
        revisions={p.name:read(p/"articles.json")["previous_revision"] for p in folder.iterdir() if p.is_dir()}
        tips=set(revisions)-set(revisions.values())
        if len(tips)==1:
            return folder/tips.pop()
        if revisions:
            raise ValueError("Ambiguous issue history. Review the archived revisions.")
    return None


def publish(archive, scratch, intel, desk, cycle, paths):
    v=read(intel/"verification.json")
    prior=active_issue(archive,cycle)
    if prior:
        audit_issue(prior,v["source_id"])
        if not (prior.parent/"active.json").exists():
            recovered=scratch/"recovered-issue.json"
            store(recovered,dict(revision=prior.name,source_id=v["source_id"]))
            os.replace(recovered,prior.parent/"active.json")
    if not paths:
        return prior
    docs=read(prior/"reviews.json") if prior else {}
    incoming=[read(path) for path in paths]
    if len({doc.get("slug") for doc in incoming})!=len(incoming):
        raise ValueError("Two reviews use the same slug in this submission")
    if all(docs.get(doc.get("slug"))==doc for doc in incoming):
        return prior
    beats={b["team_id"]:b for b in (read(p) for p in (intel/"teams").glob("*.json"))}
    checker=ClaimCheck(Path(v["raw_dir"]),v["cutoff"],v["league_date"],beats)
    threads=read(desk/"slate.json")["threads"]
    for doc in incoming:
        checked_review(doc,v,checker,threads)
        docs[doc["slug"]]=doc
    if sum(bool(d.get("lead")) for d in docs.values())>1:
        raise ValueError("More than one lead was approved. Submit the updated lead decisions together.")
    predecessor=prior.name if prior else None
    revision=hashlib.sha256(json.dumps([predecessor,docs],sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:20]
    work=scratch/"reviewed-issue"
    work.mkdir()
    entries=[render_article(work,d,checked_review(d,v,checker,threads),v) for d in sorted(docs.values(),key=lambda d:(not d.get("lead",False),d["slug"]))]
    store(work/"reviews.json",docs)
    store(work/"articles.json",dict(edition=cycle,revision=revision,previous_revision=predecessor,source_id=v["source_id"],
        cutoff=v["cutoff"],league_date=v["league_date"],approved_at_utc=datetime.now(timezone.utc).isoformat(),
        articles=entries,externally_published=False))
    nav="".join("<li>"+link(a["path"][:-3]+".html",a["title"])+"</li>" for a in entries)
    body=(f'<header><p class="date">Games through {v["cutoff"]}</p><h1>ABL Newsroom · {v["league_date"]}</h1></header><ul>{nav}</ul>'
          +link(f"../../../desks/{cycle}/abl-411.pdf","ABL 411 · all six divisions")+" · "+link("../../../index.html","Current newsroom"))
    (work/"index.html").write_text(page("Reviewed issue",body),encoding="utf-8")
    seal(work)
    dest=archive/"issues"/cycle/revision
    dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():
        work.rename(dest)
    audit_issue(dest,v["source_id"])
    marker=scratch/"active.json"
    store(marker,dict(revision=revision,source_id=v["source_id"]))
    os.replace(marker,dest.parent/"active.json")
    return dest
