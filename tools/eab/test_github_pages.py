"""Exercise the GitHub Pages publication boundary with a disposable Git repository."""

import os
import shutil
import stat
import subprocess
import unittest
from uuid import uuid4

from abl_eab_data import ROOT
from abl_eab_release import Publisher, canonical, file_hash, root_page
from github_pages import GitHubPages


def git(directory, *args):
    result = subprocess.run(["git", "-C", str(directory), *args], capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace"))
    return result.stdout


def remove_readonly(operation, path, error):
    if not isinstance(error, PermissionError):
        raise error
    os.chmod(path, stat.S_IWRITE)
    operation(path)


class FixturePages(GitHubPages):
    served = "main"
    failure = None

    def download(self, path):
        return git(self.remote, "show", f"{self.served}:{path or 'index.html'}")

    def failed_run(self, commit):
        return self.failure


class PagesTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / "tmp"
        scratch.mkdir(exist_ok=True)
        self.folder = scratch / ("eab-pages-test-" + uuid4().hex)
        self.folder.mkdir()
        self.addCleanup(self.clean_fixture, scratch.resolve())
        self.publisher = Publisher(output=self.folder / "local")
        self.catalog = {"current": None, "editions": [], "issues": []}
        self.put("assets/" + "1" * 24 + ".css", "body{}")
        self.put("assets/" + "2" * 24 + ".js", '"use strict";')
        self.first = self.edition("1981-08-19", "1981-08-18", 1438)
        seed = self.folder / "seed"
        shutil.copytree(self.publisher.dist, seed)
        (seed / ".nojekyll").touch()
        git(seed, "init", "-b", "main")
        git(seed, "-c", "core.autocrlf=false", "add", "--all")
        git(seed, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "-c", "commit.gpgsign=false",
            "commit", "-m", "Test opening edition")
        self.remote = self.folder / "remote"
        git(self.folder, "init", "--bare", "-b", "main", str(self.remote))
        git(seed, "push", self.remote.as_posix(), "main")
        self.pages = FixturePages(self.publisher, remote=self.remote.as_posix())

    def clean_fixture(self, scratch):
        if self.folder.resolve().parent == scratch and self.folder.name.startswith("eab-pages-test-"):
            shutil.rmtree(self.folder, onexc=remove_readonly)

    def put(self, route, content):
        target = self.publisher.dist / route
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="\n")

    def edition(self, day, cutoff, finals, body="<h1>Game Center</h1>"):
        key = f"{day}_{cutoff}_{finals:016x}"
        entry = dict(id=key, source_id=f"{finals:064x}", league_date=day, cutoff=cutoff, season=1981, finals=finals,
                     published_at="2026-11-23T23:00:00Z", style="assets/" + "1" * 24 + ".css",
                     script="assets/" + "2" * 24 + ".js", mark=None)
        home = f"releases/{key}/index.html"
        self.put(home, '<!doctype html><html lang="en"><head><title>EAB</title></head><body>'
                 '<main id="content">' + body + '</main></body></html>')
        files = {path: file_hash(self.publisher.dist / path) for path in (home, entry["style"], entry["script"])}
        self.put(f"releases/{key}/manifest.json", canonical({"entry": entry, "files": files, "issues": []}))
        self.catalog["editions"].append(entry)
        self.catalog["current"] = key
        self.put("index.html", root_page(self.catalog, self.publisher.dist))
        return entry

    def head(self):
        return git(self.remote, "rev-parse", "main").decode().strip()

    def test_deploy_verify_retry_and_restore_preserve_editions(self):
        opening = self.head()
        self.assertEqual(self.pages.deploy(wait=0)["status"], "unchanged")
        self.assertEqual(self.head(), opening)
        landing = self.pages.download("")
        self.assertLess(landing.index(b'<h1>Game Center</h1>'), landing.index(b'id="editions"'))
        self.assertNotIn(b'data-open', landing)

        later = self.edition("1981-08-20", "1981-08-19", 1450)
        self.put("private.csv", "local output outside the public allowlist")
        self.pages.served = opening
        pending = self.pages.deploy(expected_source=later["source_id"], wait=0)
        self.assertEqual(pending["status"], "pending")
        self.assertEqual(self.pages.live()[0], self.first)
        self.assertNotEqual(pending["commit"], opening)
        old_home = f"releases/{self.first['id']}/index.html"
        self.assertEqual(git(self.remote, "show", f"main:{old_home}"), (self.publisher.dist / old_home).read_bytes())
        self.assertNotIn(b"private.csv", git(self.remote, "ls-tree", "-r", "--name-only", "main"))
        self.assertEqual(self.pages.deploy(wait=0)["status"], "pending")
        self.assertEqual(self.head(), pending["commit"])

        self.pages.failure = "Disposable Pages build failure"
        self.assertEqual(self.pages.deploy(wait=0)["status"], "failed")
        self.assertEqual(self.pages.live()[0], self.first)
        self.assertEqual(self.head(), pending["commit"])
        self.pages.failure = None
        retry = self.pages.deploy(retry=True, wait=0)
        self.assertNotEqual(retry["commit"], pending["commit"])
        self.assertEqual(self.pages.live()[0], self.first)
        self.pages.served = "main"
        self.assertEqual(self.pages.deploy(wait=0)["status"], "unchanged")
        self.assertEqual(self.pages.live()[0], later)

        self.publisher.rollback(self.first["id"])
        with self.assertRaisesRegex(ValueError, "explicit restore"):
            self.pages.deploy(wait=0)
        self.assertEqual(self.pages.deploy(rollback=True, wait=0)["status"], "published")
        self.assertEqual(self.pages.live()[0], self.first)
        self.assertTrue(git(self.remote, "show", f"main:releases/{later['id']}/index.html"))

    def test_invalid_local_release_cannot_advance_the_remote(self):
        opening = self.head()
        self.edition("1981-08-20", "1981-08-19", 1450, '<p onclick="open()">Unsafe content</p>')
        with self.assertRaises(ValueError):
            self.pages.deploy(wait=0)
        self.assertEqual(self.head(), opening)
        self.assertEqual(self.pages.live()[0], self.first)


if __name__ == "__main__":
    unittest.main()
