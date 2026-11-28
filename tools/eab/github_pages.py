"""Publish the existing EAB file allowlist to its dedicated GitHub Pages repository."""

from hashlib import sha1, sha256
import json
import os
import subprocess
from time import monotonic, sleep, time_ns
from urllib.request import Request, urlopen
from uuid import uuid4

from abl_eab_release import ENTRY, REGISTER, canonical, catalog_for, now
from abl_newsroom_control import claim_operation


REPOSITORY = "https://github.com/zapmast-oss/experiencing-action-baseball.git"
SITE = "https://zapmast-oss.github.io/experiencing-action-baseball/"
ACTIONS_API = "https://api.github.com/repos/zapmast-oss/experiencing-action-baseball/actions/runs"
BRANCH = "refs/heads/main"
PAGES_LIMIT = 1_000_000_000
PAGES_WARNING = 800_000_000


def manifest_path(release):
    if not ENTRY.fullmatch(release):
        raise ValueError("Invalid EAB edition ID.")
    return f"releases/{release}/manifest.json"


def public_catalog(pointer):
    register = REGISTER.search(pointer.decode("utf-8"))
    if not register:
        raise ValueError("Expected an EAB edition timeline.")
    return json.loads(register[1])


class GitHubPages:
    def __init__(self, publisher, *, remote=REPOSITORY):
        self.publisher = publisher
        self.remote = remote
        self.work = publisher.output / "github-pages"
        self.repository = self.work / "git"
        self.log = publisher.logs / "github-pages.jsonl"

    def record(self, status, **details):
        result = {"at": now(), "status": status, "url": SITE, **details}
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with self.log.open("a", encoding="utf-8") as stream:
            stream.write(canonical(result) + "\n")
        return result

    def git(self, *args, data=None, index=None, timeout=40):
        env = os.environ.copy()
        for key in ("GIT_INDEX_FILE", "GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR"):
            env.pop(key, None)
        env.update(GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never")
        if index:
            env["GIT_INDEX_FILE"] = str(index)
        options = [f"--git-dir={self.repository}", f"--work-tree={self.publisher.dist}"]
        if args[0] == "init":
            options = []
        command = ["git", *options, "-c", "core.autocrlf=false", "-c", "credential.username=zapmast-oss",
                   "-c", "credential.interactive=never", "-c", "user.name=Experiencing Action Baseball",
                   "-c", "user.email=227528030+zapmast-oss@users.noreply.github.com", "-c", "commit.gpgsign=false", *args]
        try:
            run = subprocess.run(command, cwd=self.publisher.dist, env=env, input=data, capture_output=True,
                                 check=False, timeout=timeout)
        except subprocess.TimeoutExpired as error:
            raise ValueError(f"Git {args[0]} timed out. Check GitHub Pages status before retrying.") from error
        if run.returncode:
            raise ValueError(f"Git {args[0]} failed (exit {run.returncode}). Check connectivity, the saved GitHub "
                             "credential and the public branch before retrying.")
        return run.stdout

    def tree(self, revision):
        entries = {}
        for row in self.git("ls-tree", "-rz", revision).split(b"\0"):
            if row:
                blob, path = row.decode().split("\t", 1)
                entries[path] = blob
        return entries

    def download(self, path):
        request = Request(f"{SITE}{path}?eab-check={time_ns()}", headers={"Cache-Control": "no-cache"})
        with urlopen(request, timeout=6) as response:
            if not response.geturl().startswith(SITE):
                raise ValueError("GitHub Pages redirected outside the EAB website.")
            return response.read(16 * 1024 * 1024)

    def live(self):
        """Read the actual public pointer and its manifest without GitHub credentials."""
        try:
            pointer = self.download("")
            public = public_catalog(pointer)
            release = public["current"]
            entry = next(item for item in public["editions"] if item["id"] == release)
            manifest = self.download(manifest_path(release))
            if json.loads(manifest)["entry"] != entry:
                raise ValueError("The public timeline and edition manifest disagree.")
        except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
            raise ValueError("Could not verify the public EAB timeline and manifest over HTTPS.") from error
        return entry, sha256(pointer).digest(), sha256(manifest).digest()

    def confirmed(self, expected):
        try:
            return self.live() == expected
        except ValueError:
            return False

    def failed_run(self, commit):
        request = Request(f"{ACTIONS_API}?head_sha={commit}&per_page=10",
                          headers={"Accept": "application/vnd.github+json"})
        try:
            with urlopen(request, timeout=6) as response:
                runs = json.load(response)["workflow_runs"]
            run = next(item for item in runs if item["path"] == "dynamic/pages/pages-build-deployment"
                       and item["head_sha"] == commit)
            if run["status"] == "completed" and run["conclusion"] != "success":
                url = f"{REPOSITORY.removesuffix('.git')}/actions/runs/{int(run['id'])}"
                return f"GitHub reported {run['conclusion']} for the Pages build. {url}"
        except (OSError, ValueError, KeyError, TypeError, StopIteration):
            # A missing or rate-limited job record leaves the HTTP verification pending.
            pass
        return None

    def stage(self, parent, paths):
        original = self.tree(parent)
        allowed = set(paths) | {".nojekyll"}
        empty_blob = "100644 blob " + sha1(b"blob 0\0").hexdigest()
        if original.get(".nojekyll") != empty_blob or not original.keys() <= allowed:
            raise ValueError("The GitHub branch has files outside this EAB package. Preserve and reconcile its archive before publishing.")
        index = self.work / f"{uuid4().hex}.index"
        try:
            self.git("read-tree", parent, index=index)
            self.git("add", "-f", "--pathspec-from-file=-", "--pathspec-file-nul",
                     data=b"\0".join(path.encode() for path in paths) + b"\0", index=index)
            tree = self.git("write-tree", index=index).decode().strip()
        finally:
            index.unlink(missing_ok=True)
        staged = self.tree(tree)
        if staged.keys() != allowed or any(staged.get(path) != blob for path, blob in original.items()
                                           if path != "index.html"):
            raise ValueError("Git staging would change an archived EAB file or contain an unexpected path.")
        for path in paths:
            content = (self.publisher.dist / path).read_bytes()
            digest = sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
            if staged[path] != f"100644 blob {digest}":
                raise ValueError(f"Git staging altered the verified bytes of {path}.")
        return tree

    def push(self, parent, tree, release, retry):
        action = "Retry GitHub Pages for" if retry else "Publish"
        commit = self.git("commit-tree", tree, "-p", parent,
                          "-m", f"{action} EAB {release}").decode().strip()
        self.git("push", self.remote, f"{commit}:{BRANCH}")
        return commit

    def prepare(self, expected_source, rollback, retry):
        publisher = self.publisher
        catalog = catalog_for(publisher.dist)
        release = catalog["current"]
        if not release:
            raise ValueError("Publish a verified local edition before deploying to GitHub Pages.")
        entry = next(item for item in catalog["editions"] if item["id"] == release)
        if expected_source and expected_source != entry["source_id"]:
            raise ValueError("The local EAB edition differs from the verified production capture.")
        pointer = (publisher.dist / "index.html").read_bytes()
        route = manifest_path(release)
        manifest = (publisher.dist / route).read_bytes()
        expected = entry, sha256(pointer).digest(), sha256(manifest).digest()
        if not self.repository.is_dir():
            self.git("init", "--bare", "--object-format=sha1", str(self.repository))
        if self.git("ls-remote", "--get-url", self.remote).decode().strip() != self.remote:
            raise ValueError("Git rewrites the authorized EAB repository address.")
        self.git("fetch", "--no-tags", "--no-recurse-submodules", self.remote, BRANCH)
        parent = self.git("rev-parse", "FETCH_HEAD").decode().strip()
        branch_pointer = self.git("show", f"{parent}:index.html")
        previous = public_catalog(branch_pointer)["current"]
        editions = [item["id"] for item in catalog["editions"]]
        if previous not in editions:
            raise ValueError("The GitHub edition is missing from the local timeline. Preserve its archive before publishing.")
        if editions.index(previous) > editions.index(release) and not rollback:
            raise ValueError("GitHub already has a later edition. Use the explicit restore operation to publish an earlier one.")
        same_pointer = branch_pointer == pointer
        info = {"release": release, "cutoff": entry["cutoff"], "commit": parent}
        if same_pointer:
            if self.git("show", f"{parent}:{route}") != manifest:
                raise ValueError("GitHub has different bytes for the active edition's manifest.")
            if self.confirmed(expected):
                return expected, info, "unchanged"
        if not same_pointer or retry or rollback:
            _, paths = publisher.paths_for_package()
            size = sum((publisher.dist / path).stat().st_size for path in paths)
            if size >= PAGES_LIMIT:
                raise ValueError("The verified package has reached the 1 GB GitHub Pages site limit. The public edition is unchanged.")
            info.update(bytes=size, capacity_warning=size >= PAGES_WARNING)
            tree = self.stage(parent, paths)
            old_tree = self.git("rev-parse", f"{parent}^{{tree}}").decode().strip()
            if retry or tree != old_tree:
                info["commit"] = self.push(parent, tree, release, retry and tree == old_tree)
                self.record("submitted", **info)
                return expected, info, "published"
        return expected, info, "verified"

    def deploy(self, *, expected_source=None, rollback=False, retry=False, wait=45):
        self.work.mkdir(parents=True, exist_ok=True)
        self.publisher.work.mkdir(parents=True, exist_ok=True)
        info = {}
        try:
            with claim_operation(self.work) as acquired:
                if not acquired:
                    raise ValueError("Another GitHub Pages publication is running.")
                with claim_operation(self.publisher.work) as local:
                    if not local:
                        raise ValueError("EAB is preparing a local edition. Retry when it is ready.")
                    expected, info, status = self.prepare(expected_source, rollback, retry)
                if status == "unchanged":
                    return self.record(status, **info)
                deadline = monotonic() + wait
                while True:
                    if self.confirmed(expected):
                        return self.record(status, **info)
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        if failure := self.failed_run(info["commit"]):
                            return self.record("failed", **info, error=failure)
                        return self.record("pending", **info)
                    sleep(min(4, remaining))
        except (OSError, ValueError, KeyError) as error:
            self.record("failed", **info, error=str(error))
            raise
