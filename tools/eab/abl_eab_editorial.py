"""Admit only sealed newsroom copy explicitly approved for EAB."""

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
from xml.sax.saxutils import escape

from abl_eab_coverage import coverage_block, feature_inline
from abl_eab_data import ROOT
from abl_newsroom_editorial import audit_issue


SLUG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
OBSERVER_CONVERSATION = "The Observer Conversation"
PRIVATE = re.compile(r"\b[a-z]:[\\/]|file://|\\\\|(?:^|[ /])(?:csv/out/|output/newsroom/|logs/|\.env\b)",
                     re.I | re.M)
PROHIBITED = re.compile(r"\.ps1\b|\b(?:password\s*=|bearer\s+[a-z0-9_-]{16,})", re.I)


@dataclass(frozen=True)
class Article:
    title: str
    slug: str
    byline: str
    kind: str
    teams: tuple[int, ...]
    digest: str
    lead: bool
    sections: tuple[tuple[str, tuple[str, ...]], ...]
    game_ids: tuple[int, ...]
    pdf: bool
    feature_through: str | None = None
    sealed_pdf: bytes | None = None
    related_slug: str | None = None
    subtitle: str | None = None


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def identity(data):
    return hashlib.sha256(data).hexdigest()


def approved_paths(ledger, archive, warnings, standalone):
    receipts = sorted(ledger.glob("*/session.json"))
    if standalone is not None:
        receipts.extend(sorted(standalone.glob("*.json")))
    for receipt in receipts:
        try:
            for entry in read_json(receipt).get("releases", []):
                approval = entry.get("approval") or {}
                if (entry.get("lane") != "editorial" or entry.get("channel", "").casefold() != "eab"
                        or not approval.get("reviewer") or not approval.get("at")
                        or approval.get("sha256") != entry.get("sha256")):
                    continue
                path = Path(entry["artifact"])
                if path.suffix.lower() == ".md" and path.resolve().is_relative_to(archive):
                    yield path.resolve(), entry["sha256"], entry.get("source_cutoff")
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            warnings.append(f"Unable to read an editorial approval receipt: {exc}")


def approved_copy(path, item, manifest, digest):
    content = path.read_bytes()
    if len(content) > 180_000 or identity(content) != digest:
        raise ValueError("The exact EAB-approved manuscript is unavailable.")
    heading, credit, *blocks = re.split(r"\n\s*\n", content.decode("utf-8-sig").replace("\r\n", "\n").strip())
    if heading != "# " + item["title"] or not item["title"].strip():
        raise ValueError("Manuscript and sealed issue title do not match.")
    byline, when, cutoff = (part.strip() for part in credit.split("|"))
    if when != manifest["league_date"] or cutoff != "Games through " + manifest["cutoff"]:
        raise ValueError("Manuscript and sealed issue cutoff do not match.")
    sections = [("", [])]
    special = item.get("category") == "Special Coverage"
    conversation = item.get("category") == OBSERVER_CONVERSATION
    feature_through = None
    sealed_pdf = None
    related_slug = None
    subtitle = None
    if special or conversation:
        sealed_pdf = path.with_suffix(".pdf").read_bytes()
        metadata = "\n".join(blocks)
        record = re.search(r"(?m)^Public PDF SHA-256: ([a-f0-9]{64})$", metadata)
        if not record or record[1] != identity(sealed_pdf) or not sealed_pdf.startswith(b"%PDF-"):
            raise ValueError("The reading PDF differs from the approved manuscript.")
    if special:
        feature_through = date.fromisoformat(item["feature_through"]).isoformat()
        window = re.search(r"(?m)^Homepage feature through: (\d{4}-\d\d-\d\d)$", metadata)
        if feature_through < when or not window or window[1] != feature_through:
            raise ValueError("Special Coverage must match the approved PDF and feature dates.")
    if conversation:
        related = re.search(r"(?m)^Related (?:briefing|article): ([a-z0-9-]+)$", metadata)
        related_slug = item["related_slug"]
        if item["kind"] != "observer" or not related or related[1] != related_slug or not SLUG.fullmatch(related_slug):
            raise ValueError("An Observer Conversation must match its approved companion reference.")
        subtitle = item.get("subtitle")
        if subtitle:
            signed = re.search(r"(?m)^Subtitle: (.+)$", metadata)
            if not signed or signed[1] != subtitle:
                raise ValueError("The Observer Conversation subtitle differs from the approved manuscript.")
    for block in blocks:
        if block.startswith("## "):
            label = block[3:].strip()
            if label.casefold() == "reporting record":
                break
            sections.append((label, []))
        else:
            if special or conversation:
                coverage_block(block)
                sections[-1][1].append(block)
            else:
                sections[-1][1].append(" ".join(block.splitlines()))
    exposed = "\n".join([item["title"], subtitle or "", byline, *(value for h, ps in sections for value in [h, *ps])])
    if PRIVATE.search(exposed) or PROHIBITED.search(exposed) or not any(ps for _, ps in sections):
        raise ValueError("Manuscript includes private material or has no eligible copy.")
    return Article(item["title"], item["slug"], byline, item["kind"], tuple(item["team_ids"]), digest,
                   item["lead"], tuple((h, tuple(ps)) for h, ps in sections),
                   tuple(int(value) for value in item["claim_games"]), path.with_suffix(".pdf").is_file(),
                   feature_through, sealed_pdf, related_slug, subtitle)


def release_approval(manuscript, archive, editions, digest, reviewer, at):
    """Record a human decision for already-verified copy outside a playing session."""
    path = manuscript.resolve()
    if (not path.is_relative_to(archive) or len(path.relative_to(archive).parts) != 4
            or path.suffix != ".md"):
        raise ValueError("Select a sealed Newsroom manuscript.")
    if not re.fullmatch(r"[0-9a-f]{64}", digest) or not reviewer.strip():
        raise ValueError("Record the exact manuscript hash and the approving reviewer.")
    folder = path.parent.parent
    manifest = read_json(folder / "articles.json")
    if manifest["revision"] != folder.name or not re.fullmatch(r"[0-9a-f]{20}", folder.name):
        raise ValueError("The manuscript is outside a sealed issue revision.")
    if not any((item["source_id"], item["league_date"], item["cutoff"]) ==
               (manifest["source_id"], manifest["league_date"], manifest["cutoff"]) for item in editions):
        raise ValueError("Publish verified baseball for this source before approving a standalone EAB article.")
    audit_issue(folder, manifest["source_id"])
    article = next((item for item in manifest["articles"] if item["path"] == path.relative_to(folder).as_posix()), None)
    if (article is None or article["review"]["decision"] != "approve"
            or article["kind"] not in {"observer", "beat", "league"} or not SLUG.fullmatch(article["slug"])
            or not article["team_ids"]):
        raise ValueError("The manuscript needs Newsroom review before EAB release approval.")
    approved_copy(path, article, manifest, digest)
    return {"lane": "editorial", "channel": "EAB", "title": article["title"], "artifact": str(path),
            "source_id": manifest["source_id"], "source_cutoff": manifest["cutoff"], "revision": folder.name,
            "sha256": digest, "approval": {"reviewer": reviewer.strip(), "at": at, "sha256": digest}}


def eligible_issues(repo, world, prior_editions, warnings, *, ledger=None, newsroom=None, standalone=None):
    archive = ((newsroom or repo / "csv/out/story/newsroom") / "issues").resolve()
    ledger = ledger or repo / "logs/production_control/sessions"
    known = {m["source_id"]: (m["league_date"], m["cutoff"]) for m in prior_editions}
    known[world.source_id] = (world.league_date, world.cutoff)
    requested = defaultdict(dict)
    for path, digest, cutoff in approved_paths(ledger, archive, warnings, standalone):
        if len(path.relative_to(archive).parts) == 4:
            requested[path.parent.parent][path] = (digest, cutoff)
    result = []
    for folder, entries in sorted(requested.items()):
        try:
            manifest = read_json(folder / "articles.json")
            source_id = manifest["source_id"]
            if known.get(source_id) != (manifest["league_date"], manifest["cutoff"]):
                raise ValueError("Issue differs from a verified EAB date and cutoff.")
            if (str(date.fromisoformat(manifest["cutoff"])) > world.cutoff
                    or str(date.fromisoformat(manifest["league_date"])) > world.league_date):
                raise ValueError("Issue cutoff is later than the current baseball release.")
            if manifest["externally_published"] not in (True, False):
                raise ValueError("Issue publication metadata is invalid.")
            revision = manifest["revision"]
            if not re.fullmatch(r"[0-9a-f]{20}", revision) or revision != folder.name:
                raise ValueError("Invalid sealed issue revision.")
            audit_issue(folder, source_id)
            safe = []
            for item in manifest["articles"]:
                if (not SLUG.fullmatch(item["slug"]) or item["kind"] not in ("observer", "beat", "league")
                        or not item["team_ids"] or not set(item["team_ids"]) <= world.teams.keys()
                        or item["review"]["decision"] != "approve"):
                    continue
                candidate = folder / f"{item['kind']}/{item['slug']}.md"
                approved = entries.get(candidate)
                if item["path"] != candidate.relative_to(folder).as_posix() or not approved:
                    continue
                digest, cutoff = approved
                try:
                    if cutoff != manifest["cutoff"]:
                        raise ValueError("The EAB receipt cutoff differs from the manuscript cutoff.")
                    safe.append(approved_copy(candidate, item, manifest, digest))
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    warnings.append(f"Manuscript held: {exc}")
            if safe:
                key = identity(("|".join([source_id, revision, *(a.digest for a in safe)])).encode())
                result.append({
                    "key": key, "source_id": source_id, "edition": manifest["edition"], "revision": revision,
                    "league_date": manifest["league_date"], "cutoff": manifest["cutoff"], "articles": safe,
                })
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            warnings.append(f"Editorial held: {exc}")
    return result


def article_pdf(article, issue, warnings):
    """Render the approved public text. Internal newsroom PDFs carry different metadata."""
    if article.sealed_pdf is not None:
        return article.sealed_pdf
    if not article.pdf:
        return None
    try:
        from reportlab.lib import colors
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.pdfgen.canvas import Canvas
        from reportlab.platypus import Paragraph, SimpleDocTemplate
        from reportlab.platypus.doctemplate import LayoutError
    except ImportError:
        return None
    output = BytesIO()
    ink = colors.HexColor("#152419")
    title = ParagraphStyle("title", fontName="Times-Bold", fontSize=26, leading=30, textColor=ink, spaceAfter=13)
    heading = ParagraphStyle("heading", fontName="Helvetica-Bold", fontSize=13, leading=17,
                             textColor=ink, spaceBefore=10, spaceAfter=6,
                             keepWithNext=bool(article.related_slug))
    body = ParagraphStyle("body", fontName="Times-Roman", fontSize=12, leading=17, spaceAfter=10)
    story = [
        Paragraph(escape(article.title), title),
        Paragraph(escape(f"{article.byline} | {issue['league_date']} | Games through {issue['cutoff']}"), heading),
    ]
    if article.subtitle:
        story.append(Paragraph(escape(article.subtitle), heading))
    for label, paragraphs in article.sections:
        if label:
            story.append(Paragraph(escape(label), heading))
        story.extend(Paragraph(feature_inline(text, escape) if article.related_slug else escape(text), body)
                     for text in paragraphs)
    try:
        SimpleDocTemplate(output, title=article.title, author=article.byline).build(
            story, canvasmaker=lambda *a, **kw: Canvas(*a, **{**kw, "invariant": True}))
    except (LayoutError, OSError, ValueError) as exc:
        warnings.append(f"Optional reading PDF unavailable: {exc}")
        return None
    return output.getvalue()
