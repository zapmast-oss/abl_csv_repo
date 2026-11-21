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
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from uuid import uuid4

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[2]
SCRATCH = ROOT / "tmp"
NEWSROOM = ROOT / "csv/abl_scripts/abl_newsroom.py"
OLD = "4ef0889cf75d5dd089507678fa34b6ae719710d2"
NEW = "c844e18a9216c78e55cfbe4c0fda6f9c17794790"
EAST = {1: "MIA", 3: "ATL", 6: "CHA", 8: "TB"}


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


def newsroom(archive, source, expect_success=True):
    result = subprocess.run([sys.executable, "-B", str(NEWSROOM), "--archive", str(archive),
                             "--source", str(source)], cwd=ROOT, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    require((result.returncode == 0) == expect_success, f"Newsroom exit {result.returncode}: {result.stderr}")
    if expect_success:
        return read_json(archive / "current.json")
    return result.stderr.strip()


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


def run(work):
    old, new, synthetic, archive = [work / s for s in ("aug18","aug19","synthetic","archive")]
    for sha, source in [(OLD, old), (NEW, new)]:
        prepare_export(sha, source)
    # A clean current baseline is the actual activation path. In the chronological
    # replay, a previously investigated East angle can legitimately be held over.
    clean = newsroom(work / "baseline-archive", new)
    facts = editorial_quality(Path(clean["editorial"]), new)
    rejected = challenged_claims(clean, new)
    prior = newsroom(archive, old)
    now = newsroom(archive, new)
    require((prior["cutoff"],now["cutoff"],now["league_date"]) == ("1981-08-18","1981-08-19","1981-08-20"), "Wrong cutoff")
    edition = archive / "editions" / now["edition"]
    editorial = Path(now["editorial"])
    coverage = read_json(edition / "coverage.json")
    require(coverage["completed_game_ids"] == list(range(1441,1453)) and not coverage["corrected_game_ids"], "Wrong baseball delta")
    require(coverage["assignments"] == 24 and len(coverage["reports"]) == 24, "Incomplete legacy reports")
    slate = read_json(editorial / "slate.json")
    previous_slate = read_json(Path(prior["editorial"]) / "slate.json")
    require(len(slate["teams"]) == len(previous_slate["teams"]) == 24, "Incomplete beat slates")
    require(set(previous_slate["threads"]) <= set(slate["threads"]), "Prior editorial investigations disappeared")
    before, legacy_before = hash_tree(editorial), hash_tree(edition)
    pointer = (archive / "current.json").read_bytes()
    newsroom(archive, new)
    require((archive / "current.json").read_bytes() == pointer and hash_tree(editorial) == before,
            "Repeat produced duplicate or revised journalism")
    quiet = work / "quiet"
    quiet.mkdir()
    for item in new.glob("*.csv"):
        if item.name != "games.csv":
            shutil.copyfile(item, quiet / item.name)
    missing = newsroom(archive, quiet, expect_success=False)
    require("games.csv" in missing and (archive / "current.json").read_bytes() == pointer and hash_tree(edition) == legacy_before,
            "Export missing games.csv disturbed published work or omitted the failure reason")
    shutil.copyfile(new / "games.csv", quiet / "games.csv")
    synthetic_correct(new, synthetic)
    # A new source identity without a new baseball event.
    with (quiet / "languages.csv").open("ab") as sink:
        sink.write(b"\r\n")
    last = newsroom(archive, quiet)
    quiet_articles = read_json(Path(last["editorial"]) / "articles.json")["articles"]
    require(not any(a["kind"] in ("beat","league") for a in quiet_articles),
            "Reformatted source bytes generated a new feature with no baseball change")
    amended = newsroom(archive, synthetic)
    corrected_dir = archive / "editions" / amended["edition"]
    recorded = read_json(corrected_dir / "coverage.json")
    require(recorded["completed_game_ids"] == [] and recorded["corrected_game_ids"] == [1446],
            "Scorer fixture must affect exactly game #1446")
    corrected_editorial = Path(amended["editorial"])
    metadata = read_json(corrected_editorial / "articles.json")
    current_copy = "\n".join(p.read_text(encoding="utf-8") for p in corrected_editorial.rglob("*.md"))
    require("1446" in current_copy and "2404" in current_copy and "5582" in current_copy and "rbi" in current_copy.lower(),
            "Correction must identify the game and both revised player lines")
    require("Nate Smeltz went 1-for-3" in team_story(editorial,8) and
            "Nate Smeltz went 0-for-3" in team_story(corrected_editorial,8),
            "A previously published description of Smeltz's performance was not corrected")
    require(not any(a["key"] in ("east-1","east-3","east-6") for a in metadata["articles"]),
            "One San Francisco/Tampa Bay scoring change refiled unrelated East beat articles")
    require(hash_tree(editorial) == before and hash_tree(edition) == legacy_before,
            "Correction altered the previous archived beat, Observer or edition")
    newsroom(archive, synthetic)
    require(len(list((archive / "journalism").iterdir())) == 4, "Replay produced duplicate editorial editions")
    return dict(test_scope="Two real local OOTP exports; invented RBI/hit transfers for the same game only inside a disposable archive.",
                tested_at=datetime.now(timezone.utc).isoformat(), commits=[OLD, NEW], real_source_digests=[prior["source_id"],now["source_id"]],
                cutoffs=[prior["cutoff"],now["cutoff"]], real_delta_games=coverage["completed_game_ids"],
                prior_investigations=len(previous_slate["threads"]), active_teams=len(slate["teams"]),
                immutable_prior_files={"legacy":len(legacy_before),"editorial":len(before)},
                checks=["source-derived four-team fact checks","division synthesis","PDF text","24 monitored beats","real historical replay",
                        "prior threads retained","unchanged replay","incomplete source rejected","quiet export held","scorer correction propagated","earlier journalism intact"],
                source_checks=facts,incomplete_source_response=missing, synthetic_revised_game=1446,
                synthetic_corrected_articles=len(metadata["articles"]), false_claims_rejected=rejected)


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
