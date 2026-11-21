"""Reading page, previous bylines and notices for scoring changes."""
import csv
from html import escape
import os
from pathlib import Path
import re

from abl_newsroom_beats import rows
from abl_newsroom_editorial import active_issue, link, page, read


def previous_coverage(archive, cycle, slate):
    prior=list(slate.get("coverage",[]))
    issue=active_issue(archive,cycle)
    if issue:
        manifest=read(issue/"articles.json")
        for a in manifest["articles"]:
            for key in a["story_ids"]:
                prior.append(dict(key=key,slug=a["slug"],title=a["title"],team_ids=a["team_ids"],edition=cycle,
                    path=(issue/a["path"]).relative_to(archive).as_posix(),cutoff=manifest["cutoff"],
                    trigger=slate.get("threads",{}).get(key,{}).get("trigger"),story_ids=a["story_ids"],
                    claim_games=a["claim_games"],claim_map=a.get("claim_map",[])))
    return prior


def correction_register(archive, intel, prior_desk, previous):
    notices=read(prior_desk/"desk.json")["correction_register"] if prior_desk and prior_desk.is_dir() else []
    v=read(intel/"verification.json")
    with (intel/"story_candidates.csv").open(encoding="utf-8-sig",newline="") as f:
        updates=[r for r in csv.DictReader(f) if r["signal_type"]=="revised_game"]
    if not updates:
        return notices
    players={r["player_id"]:r for r in rows(Path(v["raw_dir"]),"players")}
    game_index=read(intel/"state.json")["games"]
    parent=read(intel/"coverage.json")["previous_edition"]
    earlier_games=read(archive/"editions"/parent/"state.json")["games"] if parent else {}
    words={"rbi":("RBI","drove in|runs? batted in|RBI"),"h":("hits","hits?|for[- ]|batting"),
           "hr":("home runs","homers?|homered|home runs?"),"k":("strikeouts","strikeouts?|struck out"),
           "bb":("walks","walks?|walked"),"outs":("outs","innings?|outs?"),"er":("earned runs","earned runs?|ERA")}
    for row in updates:
        gid=int(row["subject_id"])
        details=row["evidence_summary"]
        changes=[]
        for pid,values in re.findall(r"player #(\d+), (?:batting|pitching) line revised \(([^)]+)\)",details):
            player=players.get(pid)
            if player:
                for field,before,after in re.findall(r"(\w+) (\S+) → ([^, ]+)",values):
                    if field in words:
                        stat,pattern=words[field]
                        changes.append((player["last_name"],pattern,f'{player["first_name"]} {player["last_name"]}, {stat}: {before} to {after}.'))
        g=game_index[str(gid)]
        old=earlier_games.get(str(gid),g)
        def winner(line):
            margin=int(line["runs0"])-int(line["runs1"])
            return int(line["away_team"]) if margin>0 else int(line["home_team"]) if margin<0 else None
        decision_changed=winner(old)!=winner(g) or set((g["home_team"],g["away_team"]))!=set((old["home_team"],old["away_team"]))
        teams={int(line[k]) for line in (g,old) for k in ("home_team","away_team")}
        affected={}
        for story in previous.get("coverage",[]):
            cited=gid in story.get("claim_games",[])
            race=decision_changed and min(g["date"],old["date"])<=story["cutoff"] and teams.intersection(story["team_ids"])
            if cited or race:
                sentences=[s for m in story.get("claim_map",[]) if gid in m["games"] for s in re.split(r"(?<=\.)\s+",m["claim"])]
                precise=[s for s in sentences if any(surname in s and re.search(pattern,s,re.I) for surname,pattern,_ in changes)]
                context=[s for s in sentences if s not in precise]
                if race:
                    context.append("Recheck the standings at the earlier article’s cutoff against the changed decision.")
                affected[story["path"]]=dict(title=story["title"],path=story["path"][:-3]+".html",claims=precise,context_to_review=context)
        notices.append(dict(game_id=gid,cutoff=v["cutoff"],source_id=v["source_id"],details=details,
                            changed_player_lines=[c[2] for c in changes],
                            affected_articles=list(affected.values())))
    return notices


def reading_index(archive,scratch,summary):
    cycle=summary["edition"]
    desk=Path(summary["editorial"])
    intel=archive/"editions"/cycle
    issue=Path(summary["publication"]) if summary["publication"] else None
    entries=read(issue/"articles.json")["articles"] if issue else []
    queue=read(desk/"queue.json")
    coverage=read(intel/"coverage.json")
    def local(path,title):
        return link(Path(path).relative_to(archive).as_posix(),title)
    def pair(base):
        return local(base.with_suffix(".pdf"),"PDF")+" · "+local(base.with_suffix(".md"),"Markdown")
    body=[f'<header><p class="date">{escape(summary["league_date"])} · Completed games through {escape(summary["cutoff"])}</p><h1>ABL Newsroom</h1></header>']
    if not entries:
        body.append('<p class="meta">The daily desk is current. This edition has no approved articles.</p>')
    for number,a in enumerate(entries):
        doc=read(issue/"reviews.json")[a["slug"]]
        section={"observer":"The Baseball Observer","beat":"Beat reporting","league":"League reporting"}[a["kind"]]
        stem=issue/a["path"]
        opening=next(p for _,pars in doc["sections"] for p in pars)
        if number==0:
            body.append(f'<section><p class="date">{section} · {escape(doc["byline"])}</p><h2 class="lead">'
                        +local(stem.with_suffix(".html"),a["title"])+"</h2><p>"+escape(opening)+"</p><p class=\"meta\">"
                        +local(stem.with_suffix(".html"),"Read the article")+" · "+pair(stem)+"</p></section>")
        else:
            body.append(f'<p><strong>{section}</strong> · '+local(stem.with_suffix(".html"),a["title"])
                        +' <span class="meta">('+pair(stem)+')</span></p>')
    notices=read(desk/"desk.json")["correction_register"]
    if notices:
        body.append("<h2>Scoring corrections</h2>")
        for n in reversed(notices):
            plain=" ".join(n.get("changed_player_lines",[])) or n["details"]
            body.append(f'<p><strong>Game {n["game_id"]} · {escape(n["cutoff"])}</strong> {escape(plain)}</p>')
            for a in n["affected_articles"]:
                body.append("<p>Earlier coverage: "+link(a["path"],a["title"])+". "+escape(" ".join(a["claims"]))+"</p>")
                if a.get("context_to_review"):
                    body.append("<details><summary>Related context to recheck</summary><p>"+escape(" ".join(a["context_to_review"]))+"</p></details>")
    body.append('<h2>ABL 411</h2><p class="meta">Six division races · '+pair(desk/"abl-411")+'</p><div class="divisions">')
    for div in queue["divisions"]:
        body.append(f'<section><h3>{escape(div["division"])}</h3><p>{escape(div["movement"])}</p><p>{escape(div["next"])}</p></section>')
    body.append("</div><h2>Reporting desk</h2><p>"+local(desk/"story-slate.pdf","24-team story slate")+" · "
                +local(desk/"story-slate.md","Markdown")+" · "+local(desk/"queue.json","Editorial queue and follow-ups")+"</p>")
    registered=read(intel/"assignments.json")
    body.append('<details><summary>24 franchise notebooks</summary><ul class="clubs">')
    for tid,team in sorted(registered.items(),key=lambda item:item[1]["assignment"]):
        md=intel/coverage["reports"][tid]
        body.append("<li>"+local(md.with_suffix(".pdf"),team["assignment"])+" · "+local(md,"MD")+"</li>")
    body.append("</ul></details><details><summary>Evidence and verification</summary><p>"+
                " · ".join([local(desk/"slate.json","Story histories"),local(desk/"story_candidates.csv","Ranked candidates"),
                            local(desk/"story_evidence.csv","Candidate evidence"),local(intel/"story_menu.csv","Game and transaction signals"),
                            local(intel/"verification.json","Verified export"),local(desk/"desk.json","Correction register")]+
                           ([local(issue/"articles.json","Publication decisions and sources")] if issue else []))+"</p></details>")
    history=[]
    for manifest in sorted((archive/"issues").glob("*/*/articles.json"),key=lambda p:p.as_posix(),reverse=True):
        m=read(manifest)
        current=active_issue(archive,m["edition"])
        label="current revision" if current==manifest.parent else "earlier revision"
        history.append(local(manifest.parent/"index.html",f'{m["league_date"]} · {label} · {m["revision"][:8]}'))
    for record in sorted((archive/"desks").glob("*/desk.json"),reverse=True):
        meta=read(record)
        if record.parent!=desk:
            history.append(local(record.parent/"abl-411.pdf",meta["league_date"]+" · archived ABL 411")+" · "+local(record.parent/"queue.json","Reporting queue"))
    for old in sorted((archive/"journalism").glob("*/articles.json"),reverse=True):
        m=read(old)
        if m["articles"]:
            history.append("<details><summary>"+escape(m["league_date"])+" · superseded automated drafts</summary><ul>"+
                           "".join("<li>"+local(old.parent/a["path"],a["title"])+"</li>" for a in m["articles"])+"</ul></details>")
    if history:
        body.append("<details><summary>Issue archive and prior versions</summary><ul>"+"".join("<li>"+s+"</li>" for s in history)+"</ul></details>")
    body.append('<footer>'+local(archive/"current.json","Current edition record")+" · "+local(archive/"operations.jsonl","Operations log")+"</footer>")
    html=page("Daily issue","".join(body))
    target=archive/"index.html"
    if not target.is_file() or target.read_text(encoding="utf-8")!=html:
        draft=scratch/"index.html"
        draft.write_text(html,encoding="utf-8")
        os.replace(draft,target)
