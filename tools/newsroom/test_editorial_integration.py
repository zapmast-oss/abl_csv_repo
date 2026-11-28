"""Replay two real, pinned exports and one synthetic correction below repo/tmp.

Run directly with the PDF-capable production Python. The default unittest discovery
stays fast. This acceptance takes several minutes and requires the local LFS history.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import date, datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.parse import unquote, urlsplit
from uuid import uuid4

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[2]
SCRATCH = ROOT / "tmp"
NEWSROOM = ROOT / "csv/abl_scripts/abl_newsroom.py"
OLD = "4ef0889cf75d5dd089507678fa34b6ae719710d2"
NEW = "c844e18a9216c78e55cfbe4c0fda6f9c17794790"
EAST = {1: "MIA", 3: "ATL", 6: "CHA", 8: "TB"}
PRODUCTION = ROOT / "csv/out/story/newsroom"
LEGACY_CYCLE = "1981-08-19_2acae29146076f62"
LEGACY_DIGESTS = {
    "editions": "642ac5386544828572516291eca3e247432d8488d780827d9d76af0dd988e94f",
    "journalism": "b6e458ea30228d23fa8fded05d501b07e4f37b7db4402d1ffe58773473442e9a",
}


def require(condition, reason):
    if not condition:
        raise AssertionError(reason)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def hash_tree(folder):
    return {p.relative_to(folder).as_posix(): digest(p) for p in folder.rglob("*") if p.is_file()}


def table(folder, name):
    with (folder / f"{name}.csv").open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        yield from csv.DictReader(handle)


def game_date(value):
    return date(*(int(n) for n in value.split("-")))


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def prepare_export(commit, folder):
    folder.mkdir()
    csv_tree = f"{commit}:csv/ootp_csv"
    names = [n for n in git("ls-tree", "--name-only", csv_tree).decode().splitlines() if n.endswith(".csv")]
    require(len(names) == 72, f"Expected 72 committed export files in {csv_tree}")
    git_dir = (ROOT / git("rev-parse", "--git-common-dir").decode().strip()).resolve()
    for name in names:
        content = git("show", f"{csv_tree}/{name}")
        if content.startswith(b"version https://git-lfs.github.com/spec/v1"):
            oid = re.search(rb"^oid sha256:([0-9a-f]{64})$", content, re.M).group(1).decode()
            local = git_dir / "lfs/objects" / oid[:2] / oid[2:4] / oid
            if not local.is_file():
                local = ROOT / "csv/ootp_csv" / name
            require(digest(local) == oid, f"Verified local LFS payload unavailable: {name}, {oid}")
            shutil.copyfile(local, folder / name)
        else:
            (folder / name).write_bytes(content)


def newsroom(archive, source, expect_success=True, publish=()):
    command = [sys.executable, "-B", str(NEWSROOM), "--archive", str(archive), "--source", str(source)]
    if publish:
        command.extend(["--publish", *map(str, publish)])
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    require((result.returncode == 0) == expect_success, f"Newsroom exit {result.returncode}: {result.stderr}")
    if expect_success:
        return read_json(archive / "current.json")
    return result.stderr.strip()


class LocalLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []
        self.words = []

    def handle_starttag(self, tag, attrs):
        if tag == "a" and "href" in dict(attrs):
            self.hrefs.append(dict(attrs)["href"])

    def handle_data(self, text):
        self.words.append(text)


def inspect_page(page):
    html = LocalLinks()
    html.feed(page.read_text(encoding="utf-8"))
    targets = set()
    for href in html.hrefs:
        url = urlsplit(href)
        if url.scheme or url.netloc or not url.path:
            continue
        target = (page.parent / unquote(url.path)).resolve()
        require(target.exists(), f"Broken link in {page}: {href}")
        targets.add(target)
    return targets, " ".join(" ".join(html.words).split())


def legacy_archives():
    result = {}
    for component, expected in LEGACY_DIGESTS.items():
        files = hash_tree(PRODUCTION / component / LEGACY_CYCLE)
        canonical = json.dumps(files, sort_keys=True, separators=(",",":")).encode()
        actual = hashlib.sha256(canonical).hexdigest()
        require(actual == expected, f"Changed immutable {component} archive, previously {expected}, now {actual}")
        result[component] = {"file_count":len(files),"tree_sha256":actual}
    return result


def reference_facts(source):
    finals = [g for g in table(source, "games") if g["league_id"] == "200" and g["date"].startswith("1981-")
              and g["game_type"] == "0" and g["played"] == "1"]
    history = list(table(source, "team_history"))
    team_stats = {}
    for tid in EAST:
        results = []
        for g in sorted(finals, key=lambda g: (game_date(g["date"]), int(g["game_id"]))):
            if str(tid) not in (g["home_team"], g["away_team"]):
                continue
            away = str(tid) == g["away_team"]
            gf, ga = int(g["runs0" if away else "runs1"]), int(g["runs1" if away else "runs0"])
            opponent = int(g["home_team" if away else "away_team"])
            results.append((game_date(g["date"]), opponent, gf - ga))
        def record(selected):
            return (sum(r > 0 for _, _, r in selected), sum(r < 0 for _, _, r in selected))
        team_stats[tid] = dict(record=record(results), differential=sum(r for _, _, r in results),
                               division=record([g for g in results if g[1] in EAST]),
                               outside=record([g for g in results if g[1] not in EAST]),
                               june=record([g for g in results if g[0].month <= 6]),
                               since_july=record([g for g in results if g[0].month >= 7]),
                               last_seven=record(results[-7:]),
                               one_run=record([g for g in results if abs(g[2]) == 1]),
                               five_plus=record([g for g in results if abs(g[2]) >= 5]),
                               playoffs=sorted(int(h["year"]) for h in history if h["team_id"] == str(tid)
                                               and h["league_id"] == "200" and int(h["year"]) < 1981
                                               and h["made_playoffs"] == "1"))
    trades = [r["message_id"] for r in table(source, "trade_history") if
              "1" in (r["team_id_0"], r["team_id_1"]) and date(1981, 7, 21) <= game_date(r["date"]) <= date(1981, 7, 31)]
    next_east = [(g["game_id"], g["away_team"], g["home_team"]) for g in table(source, "games")
                 if game_date(g["date"]) == date(1981, 8, 20) and g["played"] == "0" and g["game_type"] == "0"
                 and g["league_id"] == "200" and int(g["away_team"]) in EAST]
    return team_stats, trades, next_east


def team_story(editorial, tid):
    return (editorial / "beat" / f"{tid:02}-{EAST[tid]}.md").read_text(encoding="utf-8")


def challenged_claims(confirmed, source):
    sys.path.insert(0, str(NEWSROOM.parent))
    from abl_newsroom_factcheck import ClaimCheck
    clubs = [read_json(p) for p in Path(confirmed["reports"]).glob("*.json")]
    checker = ClaimCheck(source, confirmed["cutoff"], confirmed["league_date"], {c["team_id"]:c for c in clubs})
    def check(text, ref):
        return checker.evaluate(dict(title="Desk review", sections=[("",[text])], evidence=[ref]))
    source_ref = "games.csv game_id=1446 date=1981-08-19 visitor=8 home=10"
    require(check("San Francisco beat Tampa Bay 18-2 on August 19.",source_ref)["score_claims"] == 1,
            "The correct score must clear the release gate")
    require(check("Charlotte is 69-52.","games.csv team_id=6")["record_claims"] == 1,
            "The correct standing must clear the release gate")
    require(check("Miami visits Atlanta on August 20.","games.csv scheduled game_id=1453 date=1981-08-20 visitor=1 home=3")["games"] == [1453],
            "A future fixture must be usable as a schedule reference")
    cases = {
        "wrong_score":("San Francisco beat Tampa Bay 19-2 on August 19.",source_ref),
        "wrong_record":("Charlotte is 68-53.","games.csv team_id=6"),
        "score_attached_to_wrong_game":("San Francisco beat Tampa Bay 18-2 on August 19.","games.csv game_id=1445"),
        "wrong_game_date":("Game record.",source_ref.replace("08-19","08-18")),
        "reversed_clubs":("Game record.",source_ref.replace("visitor=8 home=10","visitor=10 home=8")),
        "future_game_as_result":("Game record.","games.csv game_id=1453"),
        "nonexistent_game":("Game record.","games.csv game_id=999999"),
        "nonexistent_player":("Player line.","players.csv player_id=999999"),
        "wrong_trade_date":("Trade record.","trade_history.csv message_id=19852 date=1981-07-28"),
        "nonexistent_table":("Player line.","invented_history.csv"),
    }
    rejected = {}
    for key,(text, ref) in cases.items():
        try:
            check(text,ref)
        except ValueError as exc:
            rejected[key] = str(exc)
        else:
            raise AssertionError(f"Release gate accepted a false {key} fixture")
    return rejected


def editorial_quality(editorial, source):
    stats, trades, next_games = reference_facts(source)
    manifest = read_json(editorial / "articles.json")
    articles = manifest["articles"]
    numbers = dict(zip(("zero","one","two","three","four","five","six","seven","eight","nine"),map(str,range(10))))
    def normalized(text):
        lowered = re.sub(r"[–—−]", "-", text).lower()
        return re.sub(r"\b(?:" + "|".join(numbers) + r")\b", lambda m:numbers[m[0]], lowered)
    def contains_record(text, wl):
        win, loss = wl
        return re.search(rf"(?<!\d){win}\s*(?:-|to|and|games and lost)\s*{loss}(?!\d)", text)
    findings = {}
    for tid, abbr in EAST.items():
        path = editorial / "beat" / f"{tid:02}-{abbr}.md"
        require(path.is_file(), f"Missing {path.name}; held submissions: {manifest['held']}")
        text = normalized(team_story(editorial, tid))
        selected = "last_seven" if tid == 8 else "record"
        require(contains_record(text, stats[tid][selected]), f"{abbr}: reported {selected} differs from official games")
        require(any(a["path"] == path.relative_to(editorial).as_posix() and a["evidence"] for a in articles),
                f"{abbr}: missing article or traceable sources in article index")
        pdf_text = " ".join(page.extract_text() for page in PdfReader(path.with_suffix(".pdf")).pages)
        require(len(pdf_text.split()) >= 90, f"{abbr}: article PDF lacks readable text")
        findings[abbr] = stats[tid]
    charlotte, miami, atlanta, tampa = (normalized(team_story(editorial, t)) for t in (6, 1, 3, 8))
    require(stats[6]["playoffs"] == [1980] and ("1980" in charlotte or "1979" in charlotte and "a year later" in charlotte),
            "Charlotte debut must agree with history")
    require(contains_record(charlotte, stats[6]["division"]) and contains_record(charlotte, stats[6]["outside"]),
            "Charlotte article must support the division-versus-elsewhere comparison")
    require(stats[1]["differential"] < 0 and contains_record(miami, stats[1]["one_run"])
            and contains_record(miami, stats[1]["five_plus"]), "Miami sustainability margins disagree with games.csv")
    miami_evidence = " ".join(next(a["evidence"] for a in articles if a["key"] == "east-1"))
    require(len(trades) == 3 and all(i in miami_evidence for i in trades), "Miami trades must trace to each native transaction ID")
    require(contains_record(atlanta, stats[3]["june"]) and contains_record(atlanta, stats[3]["since_july"]),
            "Atlanta split at July 1 disagrees with finals")
    tampa_evidence = " ".join(next(a["evidence"] for a in articles if a["key"] == "east-8"))
    require("smeltz" in tampa and "1446" in tampa_evidence, "Tampa's individual reporting or game citation absent")
    # Verify the organizational and individual lines in their real context using
    # actual game IDs and clubs at the time, independently of the article builder.
    games = {r["game_id"]:r for r in table(source,"games") if r["league_id"] == "200" and r["game_type"] == "0"
             and r["played"] == "1" and r["date"].startswith("1981-")}
    def totals(filename, pid, tid, cols, *, from_date=None, to_date=None, opponent=None, only=None):
        acc = Counter()
        for row in table(source,filename):
            g = games.get(row["game_id"])
            if not g or (int(row["player_id"]),int(row["team_id"]),row["year"],row["league_id"],row["level_id"]) != (pid,tid,"1981","200","1"):
                continue
            d = game_date(g["date"])
            if (from_date and d < from_date or to_date and d > to_date or
                    opponent and str(opponent) not in (g["away_team"],g["home_team"]) or only and int(g["game_id"]) != only):
                continue
            acc.update({k:int(row[k]) for k in cols})
        return acc
    bat, pit = "players_game_batting", "players_game_pitching_stats"
    checks = []
    for pid,former,trade_day in [(8655,21,28),(1849,2,29)]:
        trade_date = date(1981,7,trade_day)
        for tid, cut in [(former, {"to_date":trade_date}),(1, {"from_date":trade_date})]:
            values = totals(bat,pid,tid,["h","ab"],**cut)
            fact = f"{values['h']}-for-{values['ab']}"
            require(fact in miami, f"Miami: unsupported player #{pid} split {fact}")
            checks.append(dict(player_id=pid,team_id=tid,line=fact))
    arm = totals(pit,6333,3,["outs","er","k"],opponent=1)
    require(f"{arm['outs']//3} 2/3 innings" in atlanta and f"{arm['er']} earned runs" in atlanta and
            f"striking out {arm['k']}" in atlanta, "Atlanta: Ybanez vs. Miami differs from appearances")
    hitter = totals(bat,1005,8,["h","ab","hr","rbi"])
    require(f"{hitter['h']/hitter['ab']:.3f}"[1:] in tampa and f"{hitter['hr']} home runs" in tampa and
            f"{hitter['rbi']} rbi" in tampa, "Tampa: Smeltz season line differs from game records")
    funches = totals(pit,6782,8,["outs","bb","er"],only=1446)
    require(f"walked {funches['bb']} hitters" in tampa and f"{funches['er']} earned runs" in tampa,
            "Tampa: Funches outing differs from #1446")
    require({(a,h) for _,a,h in next_games} == {("1","3"),("6","8")}, "Upcoming East schedule changed")
    division = (editorial / "league/nbc-east.md").read_text(encoding="utf-8").lower()
    require(all(city in division for city in ("charlotte","miami","atlanta","tampa bay")),
            "Division article misses a club")
    require("4.5" in division or "4½" in division or "four and a half" in division, "Division lead unsupported")
    observer = (editorial / "observer/ernie-briefing.md").read_text(encoding="utf-8")
    require(len(observer.split()) >= 180, "The Observer briefing lacks substantive reporting")
    require(all(p.with_suffix(".pdf").is_file() for p in editorial.rglob("*.md")), "Missing print counterpart")
    return {"checked_team_facts": findings, "miami_transaction_ids": trades, "miami_player_splits":checks,
            "atlanta_pitcher_6333_vs_miami":arm, "tampa_hitter_1005":hitter, "tampa_pitcher_6782_game_1446":funches,
            "next_games_away_home": next_games}


def synthetic_correct(source, test_source):
    test_source.mkdir()
    for file in source.glob("*.csv"):
        if file.name != "players_game_batting.csv":
            shutil.copyfile(file, test_source / file.name)
    name = "players_game_batting.csv"
    transfers = {
        ("10","2404"): ("rbi",5,4), ("10","5582"): ("rbi",0,1),
        ("8","1005"): ("h",1,0), ("8","481"): ("h",0,1),
    }
    changed = set()
    with (source / name).open(encoding="utf-8-sig", newline="") as old, (test_source / name).open("w", encoding="utf-8", newline="") as new:
        reader = csv.DictReader(old)
        writer = csv.DictWriter(new, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            key = row["team_id"], row["player_id"]
            if row["game_id"] == "1446" and key in transfers:
                column, before, after = transfers[key]
                require(int(row[column]) == before and key not in changed, f"Unexpected fixture for {key}")
                row[column] = str(after)
                changed.add(key)
            writer.writerow(row)
    require(changed == set(transfers), "Incomplete synthetic scorer fixture")


def reviewed_issue(archive, current, reviews):
    publication = Path(current["publication"])
    articles = read_json(publication / "articles.json")["articles"]
    approved = {r["slug"]:r for r in (read_json(path) for path in reviews)}
    require(len(articles) == len(reviews), "Reviewed list and published article count disagree")
    top_links, visible = inspect_page(archive / "index.html")
    issue_links, _ = inspect_page(publication / "index.html")
    for article in articles:
        md = publication / article["path"]
        pdf, html = md.with_suffix(".pdf"), md.with_suffix(".html")
        require(md.is_file() and pdf.is_file() and html.is_file(), f"Missing reading copy for {md}")
        require(html in top_links or (publication / "index.html" in top_links and html in issue_links),
                f"Approved article is unreachable from the reading page: {html}")
        text = md.read_text(encoding="utf-8")
        _, browser = inspect_page(html)
        printable = " ".join(p.extract_text() for p in PdfReader(pdf).pages)
        title = article["title"]
        pdf_body = " ".join(printable.split())
        require(title in text and title in browser, f"Incomplete edition of {title}")
        for _, paragraphs in approved[article["slug"]]["sections"]:
            for paragraph in paragraphs:
                require(paragraph in text and paragraph in browser and paragraph in pdf_body,
                        f"Approved paragraph missing from a reading copy of {title}: {paragraph[:45]}")
        require(article["evidence"] and article["review"], f"Missing evidence/review for {title}")
    require("411" in visible and "desk" in visible.lower(), "Reading page lacks 411 or reporting desk")
    return articles


def division_update(desk, source):
    sections = read_json(desk / "queue.json")["divisions"]
    require(len(sections) == 6, "ABL 411 needs all six divisions")
    rows = [g for g in table(source,"games") if g["league_id"] == "200" and g["game_type"] == "0"]
    expected_today = {int(g["game_id"]) for g in rows if game_date(g["date"]) == date(1981,8,19) and g["played"] == "1"}
    expected_next = {int(g["game_id"]) for g in rows if game_date(g["date"]) == date(1981,8,20) and g["played"] == "0"}
    actual_today, actual_next = set(), set()
    for s in sections:
        require(s["movement"] and s["results"] and s["next"] and s["evidence"], f"Incomplete 411 division {s['division']}")
        actual_today.update(map(int,s["completed_game_ids"]))
        actual_next.update(map(int,s["scheduled_game_ids"]))
    require(expected_today == actual_today and expected_next == actual_next, "411 game/window mismatch")
    require(1113 in actual_next and 1113 not in actual_today, "Open game #1113 presented as a result")
    return {"divisions":len(sections),"scheduled_games":sorted(actual_next),"played_games":sorted(actual_today)}


def run(work):
    old, new, synthetic, archive = [work / s for s in ("aug18","aug19","synthetic","archive")]
    initial_archives = legacy_archives()
    for sha, source in [(OLD, old), (NEW, new)]:
        prepare_export(sha, source)
    # These four old articles are evidence for regression only. They are never
    # approved or published by a normal news cycle.
    archival_facts = editorial_quality(PRODUCTION / "journalism" / LEGACY_CYCLE, new)
    current_reviews = sorted((ROOT / "tools/newsroom/reviews/1981-08-20").glob("*.json"))
    require(current_reviews, "No actual reviewed August 20 issue submissions found")
    prior = newsroom(archive, old)
    now = newsroom(archive, new)
    require((prior["cutoff"],now["cutoff"],now["league_date"]) == ("1981-08-18","1981-08-19","1981-08-20"), "Wrong cutoff")
    for cycle in [prior, now]:
        require(not cycle.get("publication") and not cycle.get("observer"),
                "Ordinary cycle published an unreviewed story or forced an Observer article")
    inspect_page(archive / "index.html")
    edition = archive / "editions" / now["edition"]
    desk = Path(now["editorial"])
    require(desk.parent == archive / "desks", "Queue entered the publication archive")
    coverage = read_json(edition / "coverage.json")
    require(coverage["completed_game_ids"] == list(range(1441,1453)) and not coverage["corrected_game_ids"], "Wrong baseball delta")
    require(coverage["assignments"] == 24 and len(coverage["reports"]) == 24, "Incomplete notebooks")
    slate = read_json(desk / "slate.json")
    previous = read_json(Path(prior["editorial"]) / "slate.json")
    require(len(slate["teams"]) == len(previous["teams"]) == 24, "Incomplete beat slate")
    require(set(previous["threads"]) <= set(slate["threads"]), "Prior investigations disappeared")
    require(all((desk / f).is_file() for f in ("queue.json","abl-411.md","abl-411.pdf","story-slate.md","story-slate.pdf")), "Reporting desk incomplete")
    dispatch = division_update(desk,new)
    sys.path.insert(0,str(ROOT / "scripts"))
    from abl_newsroom_control import current_issue_matches
    require(current_issue_matches(now["source_id"],archive), "Controller rejected complete desk-only cycle")
    require(not (desk / "observer/ernie-briefing.pdf").exists(), "Templated copy presented as a finished Observer article")
    rejected = challenged_claims(now, new)
    initial_pointer = (archive / "current.json").read_bytes()
    bad = work / "unreviewed.json"
    draft = read_json(current_reviews[0])
    draft.pop("review", None)
    bad.write_text(json.dumps(draft), encoding="utf-8")
    unreviewed = newsroom(archive, new, False, [bad])
    draft = read_json(current_reviews[0])
    draft["title"] += " with an unreviewed change"
    bad.write_text(json.dumps(draft), encoding="utf-8")
    changed_copy = newsroom(archive, new, False, [bad])
    draft = read_json(current_reviews[0])
    draft["slug"] += "-synthetic-false"
    draft["sections"][0][1].append("Charlotte is 68-53.")
    signable = {k:v for k,v in draft.items() if k != "review"}
    draft["review"]["copy_sha256"] = hashlib.sha256(json.dumps(signable, ensure_ascii=False, sort_keys=True,
                                                              separators=(",",":")).encode()).hexdigest()
    bad.write_text(json.dumps(draft), encoding="utf-8")
    false_story = newsroom(archive, new, False, [current_reviews[0],bad])
    require("record" in false_story.lower() or "standing" in false_story.lower(), "Bad score/record cleared factual gate")
    require((archive / "current.json").read_bytes() == initial_pointer, "A rejected review moved the pointer")
    first = newsroom(archive, new, publish=current_reviews[:1])
    first_issue = Path(first["publication"])
    first_hashes = hash_tree(first_issue)
    accepted = newsroom(archive, new, publish=current_reviews)
    published = Path(accepted["publication"])
    articles = reviewed_issue(archive, accepted, current_reviews)
    if len(current_reviews) > 1:
        require(read_json(published / "articles.json")["previous_revision"] == first_issue.name and
                first_issue != published and hash_tree(first_issue) == first_hashes,
                "Publishing a companion did not preserve and identify its predecessor")
    require(current_issue_matches(now["source_id"],archive), "Controller rejected approved issue")
    before, legacy_before, desk_before = hash_tree(published), hash_tree(edition), hash_tree(desk)
    pointer = (archive / "current.json").read_bytes()
    pointer_mtime = (archive / "current.json").stat().st_mtime_ns
    for requested in ((), current_reviews):
        newsroom(archive, new, publish=requested)
        require((archive / "current.json").read_bytes() == pointer and (archive / "current.json").stat().st_mtime_ns == pointer_mtime
                and hash_tree(published) == before,
                "Rerun moved current pointer or rewrote approved copy")
    quiet = work / "quiet"
    quiet.mkdir()
    for item in new.glob("*.csv"):
        if item.name != "games.csv":
            shutil.copyfile(item, quiet / item.name)
    missing = newsroom(archive, quiet, False)
    require("games.csv" in missing and (archive / "current.json").read_bytes() == pointer,
            "Incomplete source disturbed the admitted reporting desk")
    shutil.copyfile(new / "games.csv", quiet / "games.csv")
    with (quiet / "languages.csv").open("ab") as sink:
        sink.write(b"\r\n")
    last = newsroom(archive, quiet)
    require(not last.get("publication"), "A harmless byte change created a newly published issue")
    synthetic_correct(new, synthetic)
    amended = newsroom(archive, synthetic)
    corrected_dir = archive / "editions" / amended["edition"]
    recorded = read_json(corrected_dir / "coverage.json")
    require(recorded["completed_game_ids"] == [] and recorded["corrected_game_ids"] == [1446],
            "Scorer fixture must affect precisely game #1446")
    require(not amended.get("publication"), "New correction silently published or overwrote article copy")
    notices = read_json(Path(amended["editorial"]) / "desk.json")["correction_register"]
    require(notices and "1446" in json.dumps(notices), "Game #1446 missing from published-claim correction register")
    changed_game = next(n for n in notices if n["game_id"] == 1446)
    affected = {a["title"]:a for a in changed_game["affected_articles"]}
    expected = {a["title"] for a in articles if 1446 in a["claim_games"]}
    require(affected.keys() == expected and expected, "Scorer notice omitted the West lead or flagged unrelated reviewed copy")
    require(any("Margalit" in claim for a in affected.values() for claim in a["claims"]),
            "Correction failed to identify the previously reported RBI claim")
    require(not any("64–57" in line for a in affected.values() for line in a["claims"]) and
            any("64–57" in line for a in affected.values() for line in a["context_to_review"]),
            "A batting-only correction treated an unchanged standing as a corrected result")
    require(all((archive / a["path"]).is_file() for a in affected.values()), "Unreachable affected earlier coverage")
    require("2404" in changed_game["details"] and "5 → 4" in changed_game["details"],
            "Correction does not expose the old and revised player line")
    _, visible = inspect_page(archive / "index.html")
    require("correction" in visible.lower() and "1446" in visible, "Correction not visible to readers")
    newsroom(archive, synthetic)
    require(hash_tree(published) == before and hash_tree(desk) == desk_before and hash_tree(edition) == legacy_before,
            "Correction overwrote approved copy or a previously admitted desk")
    require(legacy_archives() == initial_archives, "Original August 19 archives changed")
    return dict(test_scope="Two pinned real OOTP exports and one invented hit/RBI scorer fixture in a disposable archive.",
                tested_at=datetime.now(timezone.utc).isoformat(), commits=[OLD,NEW], historical_archive_integrity=initial_archives,
                real_source_digests=[prior["source_id"],now["source_id"]], active_teams=len(slate["teams"]),
                cutoffs=[prior["cutoff"],now["cutoff"]], real_delta_games=coverage["completed_game_ids"],
                prior_investigations=len(previous["threads"]), approved_titles=[a["title"] for a in articles], division_update=dispatch,
                protected_revision_files=len(before), checks=["regular cycle generates no articles","24 continuing beat assignments",
                    "six-division 411","reject missing editorial review","reject altered reviewed copy","explicit reviewed issue",
                    "HTML links and matching article formats","immutable same-snapshot editorial revision","duplicate publish and ordinary rerun preserve issue",
                    "incomplete export rejected","quiet export generates no articles","scorer correction carried through past quiet edition",
                    "original archive hashes unchanged"], archived_beat_source_checks=archival_facts,
                incomplete_source_response=missing, unreviewed_response=unreviewed, altered_review_response=changed_copy,
                signed_false_story_response=false_story,
                synthetic_revised_game=1446, synthetic_correction_register=notices, false_claims_rejected=rejected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, help="Optional JSON acceptance result")
    options = parser.parse_args()
    work = SCRATCH / f"journalism-integration-{uuid4().hex}"
    work.mkdir(parents=True)
    try:
        receipt = run(work)
        content = json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
        if options.receipt:
            options.receipt.write_text(content, encoding="utf-8")
        print(content)
    finally:
        if work.resolve().parent == SCRATCH.resolve() and work.name.startswith("journalism-integration-"):
            shutil.rmtree(work)
