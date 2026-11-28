"""Rehearse the August 31 editorial release with approvals confined to a local preview."""

from collections import Counter
from html import escape
import os
from pathlib import Path
from shutil import copytree
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/eab"))

from abl_eab_editorial import approved_copy, eligible_issues, release_approval
from abl_eab_release import Publisher, audit_manifest, catalog_for, file_hash, now, read_json
from abl_eab_site import href
from abl_newsroom import audit
from abl_newsroom_editorial import audit_desk, audit_issue, copy_hash, link, page, publish, store
from abl_newsroom_verify import inventory


WORK = ROOT / "tmp/august31-complete-edition-review"
PREVIEW = ROOT / "output/eab/august-31-complete-edition-review"
REVIEWS = ROOT / "tools/newsroom/reviews/1981-08-31"
NEWSROOM = ROOT / "csv/out/story/newsroom"
REVIEWER = "Preview fixture only (publication approval pending)"
EXPECTED = "1981-08-31_1981-08-30_0843e0ccb03d0525"


def public_path(preview_root, route):
    return os.path.relpath(preview_root / route, PREVIEW).replace(os.sep, "/")


current = read_json(NEWSROOM / "current.json")
source = current["source_id"]
date, cutoff = current["league_date"], current["cutoff"]
cycle = current["edition"]
assert (date, cutoff) == ("1981-08-31", "1981-08-30")
intel = NEWSROOM / "editions" / cycle
desk = Path(current["editorial"])
assert desk.parent == NEWSROOM / "desks"
audit(intel)
audit_desk(desk, source)
verified = read_json(intel / "verification.json")
raw = Path(verified["raw_dir"])
assert inventory(raw, set(verified["hashes"])) == verified["hashes"]
assert len(read_json(intel / "state.json")["games"]) == 1571
teams = {data["team_id"]: data for path in (intel / "teams").glob("*.json") if (data := read_json(path))}
assignments = read_json(REVIEWS / "challenge-evidence.json")
docs = {path: document for path in sorted(REVIEWS.glob("*.json"))
        if "slug" in (document := read_json(path))}
assert Counter(doc["kind"] for doc in docs.values()) == {"beat": 24, "league": 1, "observer": 1}
assert all(doc["source_id"] == source and doc["cutoff"] == cutoff and doc["league_date"] == date
           and doc["review"]["decision"] == "pending"
           and copy_hash(doc) == assignments["article_checks"][doc["slug"]]["copy_sha256"]
           for doc in docs.values())
beats = {doc["team_ids"][0]: doc for doc in docs.values() if doc["kind"] == "beat"}
assert all(len(doc["team_ids"]) == 1 for doc in beats.values()) and set(beats) == set(teams)

production = Publisher()
catalog = catalog_for(production.dist)
assert catalog["current"] == EXPECTED
audit_manifest(production.dist, EXPECTED)
official_world = SimpleNamespace(source_id=source, league_date=date, cutoff=cutoff, teams=teams)
warnings = []
official = eligible_issues(ROOT, official_world, catalog["editions"], warnings, standalone=production.approvals)
assert [a.title for issue in official if issue["league_date"] == date for a in issue["articles"]] == [
    "The Forecast Meets the Facts"]
assert not warnings, warnings

protected = [*REVIEWS.glob("*"), NEWSROOM / "current.json", production.dist / "index.html",
             *production.approvals.glob("*.json")]
protected = {path: file_hash(path) for path in protected if path.is_file()}
approvals_before = set(production.approvals.glob("*.json"))
archived = {f"releases/{item['id']}/manifest.json" for item in catalog["editions"]}
archived.update(a[key] for issue in catalog["issues"] for a in issue["articles"]
                for key in ("path", "pdf") if a.get(key))

WORK.mkdir(parents=True, exist_ok=True)
staged_docs = WORK / "simulated-reviews"
staged_docs.mkdir(exist_ok=True)
staged = []
for path, doc in docs.items():
    digest = copy_hash(doc)
    # Use exact manuscripts with a test-only review flag. Production never reads this directory.
    doc["review"].update(decision="approve", reviewer=REVIEWER, copy_sha256=digest)
    draft = staged_docs / path.name
    store(draft, doc)
    staged.append(draft)
isolated_newsroom = WORK / "newsroom"
sealed = publish(isolated_newsroom, WORK, intel, desk, cycle, staged)
manifest = audit_issue(sealed, source)
assert len(manifest["articles"]) == 26
print("Newsroom verified and sealed 26 unchanged manuscripts for preview only.", flush=True)

approval_dir = WORK / "simulated-eab-approvals"
approval_dir.mkdir(exist_ok=True)
isolated_ledger = WORK / "empty-production-ledger"
publisher = Publisher(output=PREVIEW, ledger=isolated_ledger, newsroom=isolated_newsroom, approval_dir=approval_dir)
entries = []
for item in manifest["articles"]:
    manuscript = sealed / item["path"]
    digest = file_hash(manuscript)
    article = approved_copy(manuscript, item, manifest, digest)
    original = next(doc for doc in docs.values() if doc["slug"] == article.slug)
    assert tuple((h, tuple(" ".join(line.splitlines()) for line in lines)) for h, lines in original["sections"]) == article.sections
    entry = release_approval(manuscript, (isolated_newsroom / "issues").resolve(), catalog["editions"], digest, REVIEWER, now())
    assert entry["artifact"] == str(manuscript)
    entries.append(entry)
store(approval_dir / "simulation.json", {"releases": entries})
if not publisher.dist.exists():
    copytree(production.dist, publisher.dist)
print("Approved-article gate passed inside preview. Building the existing EAB layout.", flush=True)
entry, status = publisher.publish(raw, expected=source)
assert status in {"published", "unchanged"}
assert (entry["league_date"], entry["cutoff"], entry["finals"]) == (date, cutoff, 1571)
release = entry["id"]
test_catalog = catalog_for(publisher.dist)
assert test_catalog["current"] == release and release != EXPECTED
preview_issue, = [issue for issue in test_catalog["issues"]
                  if issue["league_date"] == date and issue.get("category") is None]
displayed = {a["slug"]: a for a in preview_issue["articles"]}
assert set(displayed) == {doc["slug"] for doc in docs.values()}
assert sum(a["kind"] == "beat" for a in displayed.values()) == 24
assert {a["title"] for issue in test_catalog["issues"] if issue.get("category") == "The Observer Conversation"
        for a in issue["articles"]} == {"The Pennant Races Come Home", "The Forecast Meets the Facts"}
assert all(file_hash(production.dist / route) == file_hash(publisher.dist / route) for route in archived)
assert all(file_hash(path) == digest for path, digest in protected.items())
assert set(production.approvals.glob("*.json")) == approvals_before
assert inventory(raw, set(verified["hashes"])) == verified["hashes"]
print("All 72 sealed source CSVs, existing public articles, production receipts, and original drafts are unchanged.", flush=True)

route_prefix = f"releases/{release}/"
rows = []
for team_id, doc in beats.items():
    data = teams[team_id]
    publication = displayed[doc["slug"]]
    club_path = route_prefix + f"teams/{data['abbr'].lower()}/index.html"
    club_markup = (publisher.dist / club_path).read_text(encoding="utf-8")
    assert f'href="{href(club_path, publication["path"])}"' in club_markup
    rows.append(dict(club=data["team"], division=data["division"], title=doc["title"], team_id=team_id,
                     club_path=club_path, article_path=publication["path"], slug=doc["slug"]))
assert len({row["club_path"] for row in rows}) == len({row["article_path"] for row in rows}) == 24
proof = {"preview_only": True, "preview_release": release, "base_public_release": EXPECTED,
         "league_date": date, "cutoff": cutoff, "finals": 1571, "issue": preview_issue["path"],
         "newsroom": route_prefix + "newsroom/index.html",
         "current_issue": route_prefix + "newsroom/current.html", "clubs": rows,
         "articles": [{"title": article["title"], "slug": article["slug"], "kind": article["kind"],
                       "route": article["path"], "pdf": article.get("pdf"),
                       "source_draft_sha256": copy_hash(next(doc for doc in docs.values() if doc["slug"] == article["slug"])),
                       "reading_copy_sha256": article["digest"]} for article in displayed.values()]}
store(PREVIEW / "coverage.json", proof)

def review_link(route, label):
    return link(public_path(publisher.dist, route), label)


def coverage_table(items):
    return '<ul>' + "".join(
        f'<li><b>{escape(row["club"])}</b> · {review_link(row["article_path"], row["title"])} '
        f'· {review_link(row["club_path"] + "#beat", "Franchise beat")}</li>'
        for row in sorted(items, key=lambda item: item["club"])
    ) + '</ul>'


body = ('<header><p class="date">Action Baseball League · August 31, 1981 · Games through August 30</p>'
        '<h1>Complete EAB edition for review</h1></header>'
        '<p><strong>Local review only.</strong> The original manuscripts remain pending. This preview exercises '
        'the review and publication gates with test decisions kept outside the production archive and approval ledger.</p>')
body += ('<p>' + review_link(preview_issue["path"], "Open the complete 26-article issue") + " · "
         + review_link(proof["newsroom"], "Open the EAB Newsroom") + " · "
         + review_link("index.html", "Open the preview homepage") + " · "
         + link("coverage.json", "Article fingerprints and coverage register") + "</p>")
body += '<h2>The Baseball Observer and The Desk</h2><ul>'
for item in displayed.values():
    if item["kind"] != "beat":
        body += f'<li>{review_link(item["path"], item["title"])}</li>'
body += '</ul><h2>All 24 franchise beats</h2>'
for division in ("NBC Eastern Division", "NBC Central Division", "NBC Western Division",
                 "ABC Eastern Division", "ABC Central Division", "ABC Western Division"):
    body += f'<h3>{division}</h3>' + coverage_table(row for row in rows if row["division"] == division)
body += ('<h2>Observer Conversation</h2><p>Both published conversations remain in '
         + review_link(route_prefix + "newsroom/observer-conversation/index.html", "the series archive") + ".</p>")
(PREVIEW / "index.html").write_text(page("Complete August 31 EAB edition for review", body), encoding="utf-8")
print("Preview release:", release, flush=True)
print("Review index:", PREVIEW / "index.html", flush=True)
