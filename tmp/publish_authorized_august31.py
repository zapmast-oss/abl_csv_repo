"""File the commissioner's 26 approved manuscripts and build the August 31 edition."""

from collections import Counter
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/eab"))

from abl_eab_editorial import eligible_issues
from abl_eab_release import Publisher, audit_manifest, catalog_for, file_hash, read_json
from abl_newsroom import ARCHIVE, run
from abl_newsroom_editorial import audit_issue, copy_hash, store
from abl_newsroom_verify import inventory


REVIEW = ROOT / "tools/newsroom/reviews/1981-08-31"
PREVIEW = ROOT / "output/eab/august-31-complete-edition-review"
WORK = ROOT / "tmp/august31-complete-edition-review"
STAGE = WORK / "commissioner-approved"
LEAGUE_DATE = "1981-08-31"
CUTOFF = "1981-08-30"
SOURCE = "461bc955c6db88b42b31da7c12c95fc8e1d15796ddd1d26de6f51af30296118d"
BASE = "1981-08-31_1981-08-30_0843e0ccb03d0525"

preview = read_json(PREVIEW / "coverage.json")
assert preview["preview_only"] and preview["base_public_release"] == BASE
assert (preview["league_date"], preview["cutoff"], preview["finals"]) == (LEAGUE_DATE, CUTOFF, 1571)
checked = {item["slug"]: item for item in preview["articles"]}
assert len(checked) == 26 and Counter(a["kind"] for a in checked.values()) == {"beat": 24, "observer": 1, "league": 1}
club_assignments = {item["slug"]: item for item in preview["clubs"]}
assert len(club_assignments) == 24 and len({item["team_id"] for item in club_assignments.values()}) == 24
assert set(club_assignments) == {slug for slug, item in checked.items() if item["kind"] == "beat"}
current = read_json(ARCHIVE / "current.json")
assert (current["source_id"], current["league_date"], current["cutoff"]) == (SOURCE, LEAGUE_DATE, CUTOFF)
raw = Path(current["raw_dir"])
verify = read_json(ARCHIVE / "editions" / current["edition"] / "verification.json")
expected = verify["hashes"]
assert len(expected) == 72 and inventory(raw, set(expected)) == expected
promoted = ROOT / "csv/ootp_csv"
promoted_before = inventory(promoted, set(expected))

p = Publisher()
prior = catalog_for(p.dist)
assert prior["current"] == BASE
audit_manifest(p.dist, BASE)
preserved = {f"releases/{e['id']}/manifest.json" for e in prior["editions"]}
for issue in prior["issues"]:
    for a in issue["articles"]:
        preserved.add(a["path"])
        if a.get("pdf"):
            preserved.add(a["pdf"])
preserved = {path: file_hash(p.dist / path) for path in preserved}

STAGE.mkdir(exist_ok=True)
originals = {}
submissions = []
for slug, check in checked.items():
    origin = REVIEW / f"{slug}.json"
    doc = read_json(origin)
    originals[origin] = file_hash(origin)
    assert (doc["slug"], doc["title"], doc["kind"], doc["source_id"], doc["league_date"], doc["cutoff"]) == (
        slug, check["title"], check["kind"], SOURCE, LEAGUE_DATE, CUTOFF)
    fingerprint = copy_hash(doc)
    assert fingerprint == check["source_draft_sha256"]
    if slug in club_assignments:
        assert doc["team_ids"] == [club_assignments[slug]["team_id"]]
    doc["review"].update(decision="approve", reviewer="Commissioner", copy_sha256=fingerprint)
    target = STAGE / origin.name
    store(target, doc)
    submissions.append(target)

filed = run(ARCHIVE, raw, submissions)
assert (filed["source_id"], filed["edition"], filed["league_date"], filed["cutoff"]) == (
    SOURCE, current["edition"], LEAGUE_DATE, CUTOFF)
issue_dir = Path(filed["publication"])
issue = audit_issue(issue_dir, SOURCE)
assert set(checked) == {item["slug"] for item in issue["articles"]}
for item in issue["articles"]:
    assert file_hash(issue_dir / item["path"]) == checked[item["slug"]]["reading_copy_sha256"]
print("Filed all 26 exact commissioner-approved newsroom manuscripts.", flush=True)

for item in issue["articles"]:
    approved = p.approve_article(issue_dir / item["path"], checked[item["slug"]]["reading_copy_sha256"], "Commissioner")
    assert approved["title"] == item["title"] and approved["source_id"] == SOURCE

world = SimpleNamespace(source_id=SOURCE, league_date=LEAGUE_DATE, cutoff=CUTOFF,
                        teams={row["team_id"]: row for row in preview["clubs"]})
warnings = []
eligible = eligible_issues(ROOT, world, prior["editions"], warnings, standalone=p.approvals)
assert not warnings, warnings
known = {item["key"] for item in prior["issues"]}
new_issues = [i for i in eligible if i["key"] not in known]
assert len(new_issues) == 1 and {a.slug for a in new_issues[0]["articles"]} == set(checked)
print("Recorded 26 exact EAB approvals. Building the existing baseball edition with the reporting issue.", flush=True)

entry, status = p.publish(raw, expected=SOURCE)
assert status in {"published", "unchanged"}
assert (entry["league_date"], entry["cutoff"], entry["finals"], entry["source_id"]) == (
    LEAGUE_DATE, CUTOFF, 1571, SOURCE)
catalog = catalog_for(p.dist)
release = entry["id"]
assert catalog["current"] == release and release != BASE
sealed = audit_manifest(p.dist, release)
published, = sealed["issues"]
assert set(checked) == {item["slug"] for item in published["articles"]} and published.get("category") is None
assert all(file_hash(p.dist / path) == digest for path, digest in preserved.items())
assert all(file_hash(path) == digest for path, digest in originals.items())
assert inventory(raw, set(expected)) == expected
assert inventory(promoted, set(expected)) == promoted_before
assert {a["title"] for i in catalog["issues"] if i.get("category") == "The Observer Conversation"
        for a in i["articles"]} == {"The Pennant Races Come Home", "The Forecast Meets the Facts"}
groups = {a["slug"]: a for a in published["articles"]}
prefix = f"releases/{release}/"
clubs = [{**row, "club_path": prefix + row["club_path"].split("/", 2)[2],
          "article_path": groups[row["slug"]]["path"]} for row in preview["clubs"]]
proof = {
    "release": release, "previous": BASE, "finals": 1571,
    "newsroom": prefix + "newsroom/index.html", "current_issue": prefix + "newsroom/current.html",
    "issue": published["path"], "archive": prefix + "newsroom/observer-conversation/index.html",
    "clubs": clubs, "articles": [dict(article, route=article["path"]) for article in published["articles"]],
    "preserved": preserved,
}
store(WORK / "authorized-release.json", proof)
print("Verified local release:", release, "with 24 club reports, the Observer lead, and The Desk.", flush=True)
