"""Replay EAB opening day using two immutable OOTP exports and disposable approvals."""

import argparse
import json
import shutil
import sys
from uuid import uuid4
from zipfile import ZipFile

from abl_eab_data import ROOT, races
from abl_eab_release import PublicPage, Publisher, catalog_for, file_hash, route_ok
from abl_newsroom_verify import rows

sys.path.insert(0, str(ROOT / "tools/newsroom"))
from test_editorial_integration import LEGACY_CYCLE, NEW, OLD, prepare_export, require
from pypdf import PdfReader


def refuse(action, message):
    try:
        action()
    except ValueError as exc:
        require(message in str(exc), str(exc))
    else:
        raise AssertionError(f"Expected rejection mentioning {message}")


def test_postseason():
    teams, divisions = {}, {}
    for offset, conference in ((0, "ABC"), (12, "NBC")):
        wins = [80, 74, 60, 58, 73, 57, 56, 55, 70, 54, 53, 52]
        for i, count in enumerate(wins):
            group = ("East", "Central", "West")[i // 4]
            team = offset + i + 1
            teams[team] = {"w": count, "l": 120 - count, "division": group, "code": str(team)}
            divisions.setdefault((conference, group), []).append(team)
    _, classic = races(teams, divisions, 1980)
    _, protected = races(teams, divisions, 1981)
    require(classic["ABC"]["dcs"] == [(1, 2), (5, 9)], "Pre-1981 DCS was changed")
    require(protected["ABC"]["dcs"] == [(1, 9), (5, 2)], "Missing 1981 same-division protection")
    teams[2].update(w=80, l=40)
    _, tied = races(teams, divisions, 1981)
    require(len(tied["ABC"]["options"]) == 2 and not tied["ABC"]["dcs"], "Tied race treated as a fixed bracket")


def acceptance(root):
    snapshots = root / "snapshots"
    old_export, new_export = root / "august-18", root / "august-19"
    prepare_export(OLD, old_export)
    prepare_export(NEW, new_export)
    ledger = root / "ledger"
    newsroom = root / "newsroom"
    current = json.loads((ROOT / "csv/out/story/newsroom/current.json").read_text(encoding="utf-8"))
    # Use a pinned issue even when the real newsroom advances past this test window.
    sealed = ROOT / f"csv/out/story/newsroom/issues/{LEGACY_CYCLE}/d486b29935a56f9145e3"
    issue = newsroom / "issues" / sealed.parent.name / sealed.name
    shutil.copytree(sealed, issue)
    manifest = json.loads((issue / "articles.json").read_text(encoding="utf-8"))
    require(current["league_date"] >= manifest["league_date"], "Verified newsroom is behind the acceptance fixture")
    approvals = []
    receipt_path = ledger / "fixture" / "session.json"
    receipt_path.parent.mkdir(parents=True)

    def save_approvals():
        receipt_path.write_text(json.dumps({"releases": approvals}), encoding="utf-8")

    def approve(item, cutoff):
        path = issue / item["path"]
        digest = file_hash(path)
        return {"lane": "editorial", "channel": "EAB", "artifact": str(path), "sha256": digest,
                "source_cutoff": cutoff, "approval": {"reviewer": "Disposable test approval",
                                                     "at": "2026-11-23T00:00:00Z", "sha256": digest}}

    publisher = Publisher(output=root / "site", newsroom=newsroom, ledger=ledger)
    publish = publisher.publish
    index = publisher.dist / "index.html"
    catalog = lambda: catalog_for(publisher.dist)
    first, outcome = publish(old_export, snapshots=snapshots)
    require(outcome == "published" and first["cutoff"] == "1981-08-18", "Replay must start at August 18 finals")
    old_home = publisher.dist / "releases" / first["id"] / "index.html"
    old_digest, index_digest = file_hash(old_home), file_hash(index)
    repeated, outcome = publish(old_export, snapshots=snapshots)
    require(outcome == "unchanged" and repeated == first and file_hash(index) == index_digest, "Repeat altered the edition")
    print("A passed: unchanged validated export preserved the edition and pointer", flush=True)

    approvals.append(approve(manifest["articles"][0], "1981-08-20"))
    save_approvals()
    second, outcome = publish(new_export, expected=manifest["source_id"], snapshots=snapshots)
    require(outcome == "published" and second["cutoff"] == "1981-08-19" and second["league_date"] == "1981-08-20"
            and second["finals"] == 1450 and second["finals"] > first["finals"], "New batch did not advance the record")
    require(not catalog()["issues"] and "editorial held" in (publisher.logs / "publications.jsonl").read_text(encoding="utf-8"),
            "An invalid editorial receipt reached the new public edition")
    base = publisher.dist / "releases" / second["id"]
    main = index.read_text(encoding="utf-8")
    require(first["id"] in main and file_hash(old_home) == old_digest, "Current pointer erased the previous edition")
    print("B and I passed: 1,450 verified finals went live in the fixture with an ineligible issue and no new media", flush=True)

    index_digest = file_hash(index)
    incomplete = root / "incomplete"
    incomplete.mkdir()
    shutil.copyfile(new_export / "games.csv", incomplete / "games.csv")
    refuse(lambda: publish(incomplete, snapshots=snapshots), "expected 72 immediate CSVs")
    require(file_hash(index) == index_digest, "Rejected input changed the pointer")
    refuse(lambda: publish(old_export, snapshots=snapshots), "earlier or incomplete baseball capture")
    require(file_hash(index) == index_digest, "Earlier capture changed the pointer")
    print("D passed: partial and regressing captures left the verified current pointer intact", flush=True)

    approvals[0] = approve(manifest["articles"][0], manifest["cutoff"])
    save_approvals()

    def interrupt():
        raise ValueError("simulated failure immediately before switching the public pointer")

    refuse(lambda: publish(new_export, snapshots=snapshots, before_switch=interrupt), "simulated failure")
    require(file_hash(index) == index_digest and catalog()["current"] == second["id"], "Interrupted run changed the pointer")
    pending = set((publisher.dist / "releases").iterdir())
    issued, outcome = publish(new_export, snapshots=snapshots)
    require(outcome == "published" and set((publisher.dist / "releases").iterdir()) == pending,
            "Retry duplicated or lost the sealed release")
    archive_issue = catalog()["issues"][0]
    previous_issue = publisher.dist / archive_issue["path"]
    previous_issue_hash = file_hash(previous_issue)
    article = archive_issue["articles"][0]
    require(article["cutoff"] == manifest["cutoff"] and article["byline"] == "Ernie Bewell", "Article date/byline drifted")
    reading = (publisher.dist / article["path"]).read_text(encoding="utf-8")
    require("August 19, 1981" in reading and "Games through August 19, 1981" in reading
            and "Reporting record" not in reading and "See who leaves" in reading, "Article lost cutoff or public copy")
    reader = PdfReader(publisher.dist / article["pdf"])
    pdf = " ".join(p.extract_text() for p in reader.pages)
    require("Games through 1981-08-19" in pdf and "Reporting record" not in pdf and "Ernie Bewell" in pdf,
            "PDF diverged from the public manuscript")

    approvals.append(approve(manifest["articles"][1], manifest["cutoff"]))
    save_approvals()
    latest, _ = publish(new_export, snapshots=snapshots)
    require(catalog()["current"] == latest["id"] and len(catalog()["issues"]) == 2
            and file_hash(previous_issue) == previous_issue_hash and file_hash(old_home) == old_digest,
            "New issue failed to preserve the earlier issue and baseball edition")
    require(catalog()["issues"][-1]["articles"][0]["path"] == article["path"], "Existing approved article was duplicated")
    publisher.rollback(first["id"])
    require(catalog()["current"] == first["id"] and file_hash(previous_issue) == previous_issue_hash, "Rollback lost history")
    restored, _ = publish(new_export, snapshots=snapshots)
    require(restored == latest and len(catalog()["editions"]) == 4, "Restoration duplicated an edition")
    require(issued["cutoff"] == restored["cutoff"], "Editorial-only release advanced the baseball cutoff")
    print("C and E passed: earlier editions and issues survived an interruption, retry and verified rollback", flush=True)

    archive = publisher.package()
    included = {"index.html"}
    for edition in catalog()["editions"]:
        route = f"releases/{edition['id']}/manifest.json"
        included.add(route)
        included.update(json.loads((publisher.dist / route).read_text(encoding="utf-8"))["files"])
    with ZipFile(archive) as bundle:
        require(bundle.namelist() == sorted(included) and all(route_ok(p) for p in included) and bundle.testzip() is None,
                "Deployment archive contains invalid, unapproved or corrupt entries")
        require(article["pdf"] in included and not any(p.endswith((".csv", ".ps1", ".md")) for p in included),
                "Sensitive sources or missing reading PDF in deployment archive")
    refuse(lambda: PublicPage('<script src="https://www.youtube.com/"></script>', "index.html"), "Unexpected script")
    refuse(lambda: PublicPage('<p onclick="open()">click</p>', "index.html"), "Unsafe HTML")
    print("F passed: hashed public allowlist, HTML/link checks and packaging exclude production files", flush=True)

    verify_published_facts(base, publisher.dist, new_export)
    print(publisher.dist, flush=True)


def verify_published_facts(base, dist, new_export):
    teams = {row["abbr"].lower(): f"{row['name']} {row['nickname']}" for row in rows(new_export / "teams.csv")
             if (row["league_id"], row["level"], row["allstar_team"]) == ("200", "1", "0")}
    team_pages = set((base / "teams").glob("*/index.html"))
    require(len(teams) == len(team_pages) == 24, "24 independent team URLs required")
    for path in team_pages:
        name = teams[path.parent.name]
        body = path.read_text(encoding="utf-8")
        require(f"<h1>{name}</h1>" in body and f"{name} beat" in body and (path.parent / "stats.html").is_file(),
                f"Wrong identity or stats route for {name}")
    print("G passed: 24 unique franchise routes, beat identities and team statistics", flush=True)

    unfinished = (base / "games/1981-08-20.html").read_text(encoding="utf-8").split('id="g-1113"')[1].split('</section>', 1)[0]
    scheduled = (base / "games/1981-08-20.html").read_text(encoding="utf-8").split('id="g-1458"')[1].split('</section>', 1)[0]
    completed = (base / "games/1981-08-19.html").read_text(encoding="utf-8")
    require("Unfinished game" in unfinished and "Final" not in unfinished and "Box score" not in unfinished,
            "Unfinished game entered the official record")
    require("Scheduled" in scheduled and "Final" not in scheduled and "Final · 2–18" in completed and "Box score" in completed,
            "Final and scheduled rows crossed the verified cutoff")
    for title in ("1980 player records", "1980 season almanac", "1972 season almanac"):
        require(any(f"<h1>{title}</h1>" in p.read_text(encoding="utf-8")
                    for p in (dist / "almanac").glob("*/index.html")), f"Missing {title}")
    test_postseason()
    print("H passed: newspaper/PDF cutoff, scheduled/final separation, stable historical records and DCS rules", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true", help="Retain the disposable publication for browser QA")
    args = parser.parse_args()
    scratch = (ROOT / "tmp").resolve()
    scratch.mkdir(exist_ok=True)
    root = scratch / ("eab-acceptance-" + uuid4().hex)
    root.mkdir()
    try:
        acceptance(root)
    finally:
        if not args.keep and root.resolve().parent == scratch and root.name.startswith("eab-acceptance-"):
            shutil.rmtree(root)


if __name__ == "__main__":
    main()
