"""Validate, archive and activate an atomic static EAB publication."""

import copy
from datetime import datetime, timezone
from hashlib import sha256
from html import escape, unescape
from html.parser import HTMLParser
import json
from pathlib import Path
from posixpath import dirname, join, normpath
import re
import shutil
from urllib.parse import unquote, urlsplit, urlunsplit
from uuid import uuid4
import zipfile

from abl_eab_data import ROOT, load_world
from abl_eab_editorial import PRIVATE, PROHIBITED, eligible_issues, identity, read_json, release_approval
from abl_eab_site import HERE, Site, pretty
from abl_newsroom_control import claim_operation
from abl_newsroom_verify import default_live_dir, verify_export


ENTRY = re.compile(r"\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}_[0-9a-f]{16}\Z")
ASSET = re.compile(r"assets/[0-9a-f]{24}\.(png|css|js)\Z")
REGISTER = re.compile(r'<script type="application/json" id="eab-publications">(.*?)</script>', re.S)
HOME_URL = re.compile(r'(?<=\s)(href|src)="([^"]*)"')


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def file_hash(path):
    with path.open("rb") as content:
        from hashlib import file_digest
        return file_digest(content, "sha256").hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def atomic_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = path.with_name("." + path.name + "." + uuid4().hex + ".tmp")
    try:
        import os
        with candidate.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        candidate.replace(path)
    finally:
        candidate.unlink(missing_ok=True)


def catalog_for(dist):
    path = dist / "index.html"
    if not path.is_file():
        return {"current": None, "editions": [], "issues": []}
    doc = REGISTER.search(path.read_text(encoding="utf-8"))
    if not doc:
        raise ValueError("The current EAB publication catalog is missing.")
    data = json.loads(doc[1])
    keys = [entry["id"] for entry in data["editions"]]
    if len(keys) != len(set(keys)) or any(not ENTRY.fullmatch(key) for key in keys) or data["current"] not in keys:
        raise ValueError("The current EAB publication catalog is inconsistent.")
    return data


def root_page(data, dist):
    """Serve the sealed baseball homepage at the root and keep the timeline anchor."""
    current = data["current"]
    target = f"releases/{current}/index.html"

    def rebase(match):
        attr, value = match.groups()
        parts = urlsplit(unescape(value))
        if parts.scheme or parts.netloc or not parts.path:
            return match[0]
        path = normpath(join(dirname(target), parts.path))
        if path == target:
            path = "index.html"
        if not route_ok(path):
            raise ValueError(f"Invalid link on the current EAB homepage: {value}")
        if path == "index.html" and parts.fragment and not parts.query:
            path = ""
        return f'{attr}="{escape(urlunsplit(("", "", path, parts.query, parts.fragment)), quote=True)}"'

    entries = []
    for entry in reversed(data["editions"]):
        title = f"{pretty(entry['league_date'])} · Finals through {pretty(entry['cutoff'], year=False)}"
        active = entry["id"] == current
        selected = " · Current edition" if active else ""
        anchor = ' id="current-edition"' if active else ""
        entries.append(f'<li data-item><a{anchor} href="releases/{entry["id"]}/index.html">{escape(title)}</a>'
                       f'<span>{escape(entry["published_at"].replace("T", " ").replace("Z", " UTC") + selected)}</span></li>')
    embedded = canonical(data).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    register = f'<script type="application/json" id="eab-publications">{embedded}</script>'
    timeline = f"""<section id="editions" aria-labelledby="timeline-title">
<div class="section-head"><h2 id="timeline-title">Publication timeline</h2></div>
<label class="search">Find a league date
<input type="search" data-filter="edition-list" placeholder="Try August 20"></label>
<p class="sr-only" role="status" data-count="edition-list"></p><ul id="edition-list" class="date-grid">{''.join(entries)}</ul></section>"""
    home = HOME_URL.sub(rebase, (dist / target).read_text(encoding="utf-8"))
    if "</head>" not in home or "</main>" not in home:
        raise ValueError("The sealed EAB homepage is missing its document landmarks.")
    return home.replace("</head>", register + "</head>", 1).replace("</main>", timeline + "</main>", 1)


class PublicPage(HTMLParser):
    def __init__(self, text, route):
        super().__init__(convert_charrefs=True)
        if PRIVATE.search(text) or PROHIBITED.search(text):
            raise ValueError(f"Private material in public page {route}")
        self.route, self.links, self.ids = route, set(), set()
        self.feed(text)

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if (tag in {"base", "iframe", "object", "embed", "form"} or
                any(key.startswith("on") for key in attrs)):
            raise ValueError(f"Unsafe HTML in {self.route}")
        if tag == "script":
            src = attrs.get("src", "")
            local_script = re.fullmatch(r"assets/[0-9a-f]{24}\.js", normpath(join(dirname(self.route), src)))
            catalog_script = (not src and self.route == "index.html" and attrs.get("id") == "eab-publications"
                              and attrs.get("type") == "application/json")
            if not (local_script or catalog_script):
                raise ValueError(f"Unexpected script in {self.route}")
        for key in ("href", "src"):
            if attrs.get(key):
                self.links.add(attrs[key])
        if value := attrs.get("id"):
            if value in self.ids:
                raise ValueError(f"Duplicate public anchor {self.route}#{value}")
            self.ids.add(value)


def route_ok(path):
    if path == "index.html" or ASSET.fullmatch(path):
        return True
    if re.fullmatch(r"almanac/\d{4}-[0-9a-f]{16}/index\.html", path):
        return True
    parts = path.split("/")
    return (len(parts) >= 3 and parts[0] == "releases" and ENTRY.fullmatch(parts[1])
            and (parts[-1] == "manifest.json" or re.fullmatch(r"[a-z0-9][a-z0-9._-]*\.(html|pdf)", parts[-1]))
            and all(re.fullmatch(r"[a-z0-9][a-z0-9_-]*", p) for p in parts[2:-1]))


def validate_pages(stage, dist, routes, external):
    parsed = {}
    for route in routes:
        if not route_ok(route):
            raise ValueError(f"Unapproved public file {route}")

    def located(route):
        for tree in (stage, dist):
            target = tree / route
            if target.is_file() and not target.is_symlink() and target.resolve().is_relative_to(tree.resolve()):
                return target
        raise ValueError(f"Missing public page or asset {route}")

    def parsed_page(route):
        if route not in parsed:
            parsed[route] = PublicPage(located(route).read_text(encoding="utf-8"), route)
        return parsed[route]

    for route in sorted(routes):
        if not route.endswith(".html"):
            continue
        page = parsed_page(route)
        for link in page.links:
            target = urlsplit(link)
            if target.scheme:
                if link not in external:
                    raise ValueError(f"Unapproved external link {link} in {route}")
                continue
            if target.netloc or target.path.startswith("/") or "\\" in link:
                raise ValueError(f"Invalid local link {link} in {route}")
            other = normpath(join(dirname(route), unquote(target.path))) if target.path else route
            if not route_ok(other):
                raise ValueError(f"Link leaves the EAB public surface in {route}: {link}")
            located(other)
            if target.fragment and (not other.endswith(".html") or unquote(target.fragment) not in parsed_page(other).ids):
                raise ValueError(f"Broken anchor {route}: {link}")


def external_urls(urls):
    links = set(urls)
    for url in links:
        target = urlsplit(url)
        if target.scheme != "https" or target.netloc not in ("www.youtube.com", "forums.ootpdevelopments.com",
                                                              "statsplus.net", "www.statsplus.net"):
            raise ValueError("An EAB media link has an unapproved public origin.")
    return links


def media_links():
    media = read_json(HERE / "media.json")
    external_urls([entry["url"] for entry in media])
    return media


def signature(world, issue_keys, media):
    digest = sha256()
    digest.update(canonical([world.source_id, sorted(issue_keys), world.history, media]).encode())
    for path in sorted([*HERE.glob("abl_eab_*.py"), HERE / "site.css", HERE / "site.js"]):
        digest.update(path.read_bytes())
    for assets in (world.logos, {tid: blob for tid, (_, blob) in world.photos.items()}):
        for key in sorted(assets):
            digest.update(str(key).encode() + bytes.fromhex(identity(assets[key])))
    return digest.hexdigest()


def audit_manifest(dist, release_id):
    if not ENTRY.fullmatch(release_id):
        raise ValueError("Select a valid EAB release ID.")
    prefix = f"releases/{release_id}/"
    manifest_path = dist / prefix / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest["entry"]["id"] != release_id or not manifest["files"]:
        raise ValueError("Invalid EAB release manifest.")
    for path, digest in manifest["files"].items():
        file = dist / path
        if (not route_ok(path) or not (path.startswith(prefix) or path.startswith(("assets/", "almanac/")))
                or file.is_symlink() or not file.resolve().is_relative_to(dist.resolve())
                or not file.is_file() or file_hash(file) != digest):
            raise ValueError(f"EAB release file failed integrity verification: {path}")
    if prefix + "index.html" not in manifest["files"]:
        raise ValueError("EAB release homepage is missing from the manifest.")
    return manifest


class Publisher:
    def __init__(self, repo=ROOT, *, output=None, ledger=None, newsroom=None, assets=None, approval_dir=None):
        self.repo = repo
        self.output = output or repo / "output/eab"
        self.dist = self.output / "dist"
        self.work = self.output / "work"
        self.logs = self.output / "receipts" if output else repo / "logs/eab"
        self.approvals = approval_dir or self.logs / "article-approvals"
        self.assets = assets or default_live_dir().parents[1] / "news/html/images"
        self.ledger = ledger
        self.newsroom = newsroom

    def log(self, status, **values):
        self.logs.mkdir(parents=True, exist_ok=True)
        with (self.logs / "publications.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(canonical({"at": now(), "status": status, **values}) + "\n")

    def approve_article(self, manuscript, digest, reviewer):
        self.work.mkdir(parents=True, exist_ok=True)
        with claim_operation(self.work) as acquired:
            if not acquired:
                raise ValueError("An EAB operation is already running.")
            archive = ((self.newsroom or self.repo / "csv/out/story/newsroom") / "issues").resolve()
            editions = catalog_for(self.dist)["editions"]
            entry = release_approval(manuscript, archive, editions, digest, reviewer, now())
            edition = next(item for item in editions if item["source_id"] == entry["source_id"])
            audit_manifest(self.dist, edition["id"])
            receipt = self.approvals / f"{entry['revision']}-{digest}.json"
            if receipt.is_file():
                saved = read_json(receipt)["releases"][0]
                if saved["artifact"] != entry["artifact"] or saved["sha256"] != digest:
                    raise ValueError("The existing EAB approval refers to a different manuscript.")
                return saved
            atomic_write(receipt, (canonical({"releases": [entry]}) + "\n").encode())
            return entry

    def publish(self, source=None, *, expected=None, snapshots=None, before_switch=None):
        """Validate the source using the existing newsroom verifier, then publish locally."""
        self.work.mkdir(parents=True, exist_ok=True)
        with claim_operation(self.work) as acquired:
            if not acquired:
                raise ValueError("An EAB update is already running.")
            stage = self.work / ("build-" + uuid4().hex)
            stage.mkdir()
            try:
                verified = verify_export(self.repo, stage / "preflight", source, snapshot_root=snapshots)
                if expected and verified["source_id"] != expected:
                    raise ValueError("The validated EAB source differs from the production checkpoint.")
                world = load_world(self.repo, verified, self.assets)
                receipt = self.publish_world(world, stage / "site", before_switch)
            except Exception as exc:
                self.log("failed", error=str(exc))
                raise
            finally:
                if stage.resolve().parent == self.work.resolve() and stage.name.startswith("build-"):
                    shutil.rmtree(stage)
            return receipt

    def publish_world(self, world, stage, before_switch):
        catalog = catalog_for(self.dist)
        editions = catalog["editions"]
        newest = max(((e["league_date"], e["cutoff"], e["finals"]) for e in editions), default=None)
        if newest and (world.league_date < newest[0] or world.cutoff < newest[1]
                       or (world.finals < newest[2] and world.season == int(newest[0][:4]))):
            raise ValueError("EAB rejected an earlier or incomplete baseball capture. Use explicit rollback for past editions.")
        warnings = []
        eligible = eligible_issues(self.repo, world, editions, warnings, ledger=self.ledger, newsroom=self.newsroom,
                                   standalone=self.approvals)
        for detail in warnings:
            self.log("editorial held", detail=detail)
        warnings.clear()
        media = media_links()
        build = signature(world, {i["key"] for i in [*catalog["issues"], *eligible]}, media)
        release_id = f"{world.league_date}_{world.cutoff}_{build[:16]}"
        release = self.dist / "releases" / release_id
        if release.is_dir():
            sealed = audit_manifest(self.dist, release_id)
            if catalog["current"] == release_id:
                self.log("unchanged", release=release_id, cutoff=world.cutoff)
                return sealed["entry"], "unchanged"
            entry, added = sealed["entry"], sealed["issues"]
        else:
            entry = dict(id=release_id, signature=build, source_id=world.source_id, season=world.season,
                         league_date=world.league_date, cutoff=world.cutoff, finals=world.finals, published_at=now())
            temporary = copy.deepcopy(catalog)
            site = Site(stage, world, entry, temporary)
            entry.update(style=site.styles, script=site.script, mark=site.logos.get(0))
            site.build(eligible, media, warnings)
            added = temporary["issues"][len(catalog["issues"]):]
            outputs = {path.relative_to(stage).as_posix(): file_hash(path) for path in sorted(stage.rglob("*"))
                       if path.is_file()}
            self.attach(catalog, entry, added)
            site.put("index.html", root_page(catalog, stage))
            links = sorted(m["url"] for m in media)
            validate_pages(stage, self.dist, {*outputs, "index.html"}, set(links))
            manifest = {"entry": entry, "issues": added, "files": outputs, "external_links": links}
            site.put(site.route("manifest.json"), canonical(manifest) + "\n")
            self.install(stage, release_id)
        for detail in warnings:
            self.log("editorial warning", detail=detail)
        self.attach(catalog, entry, added)
        audit_manifest(self.dist, release_id)
        if before_switch:
            before_switch()
        atomic_write(self.dist / "index.html", root_page(catalog, self.dist).encode())
        self.log("published", release=release_id, source=world.source_id, cutoff=world.cutoff)
        return entry, "published"

    @staticmethod
    def attach(catalog, entry, issues):
        if entry["id"] not in {e["id"] for e in catalog["editions"]}:
            catalog["editions"].append(entry)
        for issue in issues:
            if issue["key"] not in {i["key"] for i in catalog["issues"]}:
                catalog["issues"].append(issue)
        catalog["current"] = entry["id"]

    def install(self, staging, release_id):
        for folder in ("assets", "almanac"):
            for child in (staging / folder).glob("*"):
                target = self.dist / folder / child.name
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    item = child / "index.html" if child.is_dir() else child
                    existing = target / "index.html" if target.is_dir() else target
                    if not existing.is_file() or file_hash(item) != file_hash(existing):
                        raise ValueError(f"An archived public asset changed: {target.name}")
                else:
                    child.rename(target)
        destination = self.dist / "releases" / release_id
        destination.parent.mkdir(parents=True, exist_ok=True)
        (staging / "releases" / release_id).rename(destination)

    def rollback(self, release_id):
        self.work.mkdir(parents=True, exist_ok=True)
        with claim_operation(self.work) as acquired:
            if not acquired:
                raise ValueError("An EAB update is already running.")
            catalog = catalog_for(self.dist)
            if release_id not in {entry["id"] for entry in catalog["editions"]}:
                raise ValueError("The selected release was never published.")
            manifest = audit_manifest(self.dist, release_id)
            catalog["current"] = release_id
            atomic_write(self.dist / "index.html", root_page(catalog, self.dist).encode())
            self.log("rollback", release=release_id, cutoff=manifest["entry"]["cutoff"])
            return manifest["entry"]

    def paths_for_package(self):
        catalog = catalog_for(self.dist)
        if not catalog["current"]:
            raise ValueError("Publish a validated local edition before packaging.")
        included = {"index.html"}
        external = {m["url"] for m in media_links()}
        for entry in catalog["editions"]:
            manifest = audit_manifest(self.dist, entry["id"])
            included.update(manifest["files"])
            included.add(f"releases/{entry['id']}/manifest.json")
            external.update(external_urls(manifest.get("external_links", [])))
        validate_pages(self.dist, self.dist, included, external)
        return catalog["current"], sorted(included)

    def package(self):
        self.work.mkdir(parents=True, exist_ok=True)
        with claim_operation(self.work) as acquired:
            if not acquired:
                raise ValueError("An EAB update is already running.")
            current, paths = self.paths_for_package()
            deliverable = self.output / f"eab-{current}.zip"
            scratch = self.work / (uuid4().hex + ".zip")
            try:
                with zipfile.ZipFile(scratch, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
                    for route in paths:
                        archive.write(self.dist / route, route)
                scratch.replace(deliverable)
            finally:
                scratch.unlink(missing_ok=True)
            self.log("packaged", release=current, count=len(paths))
        return deliverable
