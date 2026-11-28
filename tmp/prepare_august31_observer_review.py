"""Stage the supplied August 31 excerpts using EAB's sealed issue format."""
from pathlib import Path
import re
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools/eab"))

from abl_eab_editorial import Article, OBSERVER_CONVERSATION, approved_copy, article_pdf, identity
from abl_eab_release import Publisher, read_json
from abl_eab_site import Site
from abl_newsroom_editorial import audit_issue, page, seal, store

INPUT = Path("C:/Users/earld/.codex/attachments/eab244ff-934d-44d3-826a-4cf739bfa2a5/Pasted text.txt")
PREVIEW = REPO / "output/eab/august-31-observer-review"
SCRIPT_STAGING = REPO / "tmp/august-31-observer-staging"
PROD = Publisher()
TITLE = "The Forecast Meets the Facts"
SUBTITLE = "Ernie Bewell and Big Earl Look Back at the August Division Challenges"
SLUG = "the-forecast-meets-the-facts"
RELATED = "this-is-the-test"
BYLINE = "The Baseball Observer"
TEAMS = (1, 2, 4, 5, 6, 9, 10, 12, 13, 17, 19, 20, 22, 23, 24)
GAMES = (1113, 1494, 1525, 1537, 1549, 1561, 1563, 1566, 1568, 1572)
SECTION_TITLES = (
    "Miami: A Believer Is Born",
    "Chicago: Is This Finally the Year?",
    "San Francisco: Big Earl Isn't Budging",
    "St. Louis: The Pitching Was Always There",
    "Las Vegas: Different Ways to Win",
    "Boston and Philadelphia: The Whisper and the Answer",
    "Houston: The Luxury of a Lead",
    "The Final Test",
)

supplied = INPUT.read_text(encoding="utf-8-sig")
content = supplied.split("# THE OBSERVER CONVERSATION", 1)[1].split("\n---\n\n## 5.", 1)[0].strip()
supplied_body = content.split("*The Baseball Observer | August 31, 1981*", 1)[1].strip()
copy = supplied_body
copy = re.sub(r"(?m)^---\n\n", "", copy)
copy = re.sub(r"(?m)^### ", "## ", copy)
assert tuple(re.findall(r"(?m)^## (.+)$", copy)) == SECTION_TITLES
paragraphs = [("", [])]
for block in re.split(r"\n\s*\n", copy):
    if block.startswith("## "):
        paragraphs.append((block[3:], []))
    else:
        paragraphs[-1][1].append(block)
sections = tuple((heading, tuple(blocks)) for heading, blocks in paragraphs)

current = read_json(REPO / "csv/out/story/newsroom/current.json")
assert (current["league_date"], current["cutoff"]) == ("1981-08-31", "1981-08-30")
manifest = {k: current[k] for k in ("edition", "league_date", "cutoff", "source_id")}
warnings = []
pdf = article_pdf(Article(TITLE, SLUG, BYLINE, "observer", TEAMS, "", False, sections, GAMES,
                          True, related_slug=RELATED, subtitle=SUBTITLE), manifest, warnings)
assert pdf and not warnings
credit = f"{BYLINE} | {manifest['league_date']} | Games through {manifest['cutoff']}"
text = (f"# {TITLE}\n\n{credit}\n\n{copy}\n\n## Reporting record\n\n"
        f"Public PDF SHA-256: {identity(pdf)}\n"
        f"Related article: {RELATED}\n"
        f"Subtitle: {SUBTITLE}\n")
digest = identity(text.encode("utf-8"))
manifest.update(revision=digest[:20], previous_revision=None, externally_published=False, articles=[])
item = dict(title=TITLE, slug=SLUG, kind="observer", path=f"observer/{SLUG}.md", category=OBSERVER_CONVERSATION,
            related_slug=RELATED, subtitle=SUBTITLE, team_ids=TEAMS, lead=False, claim_games=GAMES,
            review={"decision": "approve", "reviewer": "Commissioner",
                    "scope": "Exact conversation copy supplied for release, with mechanical formatting only."})
manifest["articles"].append(item)
archive = SCRIPT_STAGING / "newsroom"
issue = archive / "issues" / f"{current['edition']}-observer-conversation" / manifest["revision"]
path = issue / item["path"]
if not issue.exists():
    path.parent.mkdir(parents=True)
    path.write_bytes(text.encode("utf-8"))
    path.with_suffix(".pdf").write_bytes(pdf)
    path.with_suffix(".html").write_text(page(TITLE, f"<h1>{TITLE}</h1><p>{SUBTITLE}</p>" + Site.coverage_body(sections)),
                                        encoding="utf-8")
    (issue / "index.html").write_text(page(OBSERVER_CONVERSATION, f'<h1>{OBSERVER_CONVERSATION}</h1>'
                                         f'<p><a href="{item["path"][:-3]}.html">{TITLE}</a></p>'), encoding="utf-8")
    store(issue / "articles.json", manifest)
    seal(issue)
audit_issue(issue, current["source_id"])
article = approved_copy(path, item, manifest, digest)
assert article.sections == sections
assert article.subtitle == SUBTITLE
assert [line for line in supplied_body.splitlines() if line and line != "---" and not line.startswith("### ")] == [
    line for line in copy.splitlines() if line and not line.startswith("## ")]
print("Sealed preview manuscript and reading PDF:", path, flush=True)
if "--seal-only" in sys.argv:
    sys.exit(0)

publisher = Publisher(output=PREVIEW, newsroom=archive, ledger=SCRIPT_STAGING / "ledger",
                      approval_dir=SCRIPT_STAGING / "approvals" / digest)
if not publisher.dist.exists():
    shutil.copytree(PROD.dist, publisher.dist)
# The isolated preview receipt exercises the same approval data contract. Production approval
# can use the normal approve-article workflow against this verified local preview.
publisher.approvals.mkdir(parents=True, exist_ok=True)
store(publisher.approvals / f"{manifest['revision']}-{digest}.json", {
    "releases": [dict(artifact=str(path), title=TITLE, channel="EAB", lane="editorial",
                      source_id=current["source_id"], source_cutoff=current["cutoff"], sha256=digest,
                      revision=manifest["revision"], approval=dict(reviewer="Commissioner",
                      at="2026-11-28", sha256=digest))],
})
result, action = publisher.publish(Path(current["raw_dir"]), expected=current["source_id"])
print(action, result["id"], "preview", publisher.dist / "index.html", flush=True)
