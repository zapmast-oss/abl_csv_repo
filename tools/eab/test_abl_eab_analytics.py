"""Check that an optional public beacon preserves existing publications."""

from pathlib import Path
from posixpath import dirname, normpath, join
import re
import shutil
from types import SimpleNamespace
import unittest
from uuid import uuid4

from abl_eab_analytics import BEACON_URL, TOKEN_FILE, beacon, load_token, normalize_token
from abl_eab_data import ROOT
from abl_eab_release import PublicPage, signature, validate_pages
from abl_eab_site import Site


TEST_TOKEN = "1" * 32


class AnalyticsTests(unittest.TestCase):
    def setUp(self):
        self.temp = ROOT / "tmp" / f"eab-analytics-test-{uuid4().hex}"
        self.temp.mkdir(parents=True)
        self.addCleanup(self.remove_fixture)

    def remove_fixture(self):
        if self.temp.resolve().parent == (ROOT / "tmp").resolve():
            shutil.rmtree(self.temp)

    def test_token_is_optional_validated_and_part_of_revision_signature(self):
        self.assertIsNone(load_token(self.temp))
        self.assertEqual(beacon(None), "")
        with self.assertRaises(ValueError):
            normalize_token('<script src="unexpected"></script>')
        (self.temp / TOKEN_FILE).write_text(TEST_TOKEN + "\n", encoding="ascii")
        self.assertEqual(load_token(self.temp), TEST_TOKEN)
        world = SimpleNamespace(source_id="fixture", history={}, logos={}, photos={})
        self.assertNotEqual(signature(world, [], []), signature(world, [], [], TEST_TOKEN))
        self.assertEqual(signature(world, [], [], load_token(self.temp)), signature(world, [], [], TEST_TOKEN))

    def test_validator_accepts_only_the_expected_external_script(self):
        tag = beacon(TEST_TOKEN)
        self.assertTrue(PublicPage(tag, "index.html").has_beacon)
        for html in (tag + tag, tag.replace(BEACON_URL, "https://example.com/beacon.js"),
                     tag.replace(TEST_TOKEN, "wrong"), tag.replace('type="module"', 'type="module" async')):
            with self.assertRaises(ValueError):
                PublicPage(html, "index.html")

    def test_today_from_an_edition_page_targets_the_public_homepage(self):
        world = SimpleNamespace(logos={}, photos={}, season=1981, league_date="1981-08-20", cutoff="1981-08-19")
        site = Site(self.temp, world, {"id": "1981-08-20_1981-08-19_" + "a" * 16}, {})
        for path in ("index.html", "teams/index.html"):
            route = site.route(path)
            html = site.frame(route, "", "", "Today" if path == "index.html" else "Teams")
            today = re.search(r'<a(?: aria-current="page")? href="([^"]+)">Today</a>', html)[1]
            self.assertEqual(normpath(join(dirname(route), today)), "index.html")

    def test_copies_track_legacy_newsroom_without_changing_the_original(self):
        old = "1981-08-20_1981-08-19_" + "a" * 16
        new = "1981-08-20_1981-08-19_" + "b" * 16
        dist, stage = self.temp / "published", self.temp / "staged"
        issue = f"releases/{old}/newsroom/issues/story/index.html"
        article = str(Path(issue).with_name("article.html")).replace("\\", "/")
        (dist / issue).parent.mkdir(parents=True)
        (dist / "index.html").write_text("<p>Current edition</p>", encoding="utf-8")
        old_today = '<nav><a href="../../../index.html">Today</a></nav>'
        issue_html = f'<body>{old_today}<a href="article.html#copy">Read the story</a></body>'
        article_html = f'<body>{old_today}<p id="copy">Story text</p>{beacon(TEST_TOKEN)}</body>'
        (dist / issue).write_text(issue_html, encoding="utf-8")
        (dist / article).write_text(article_html, encoding="utf-8")
        catalog = {"issues": [{"path": issue, "articles": [{"path": article}]}]}
        world = SimpleNamespace(logos={}, photos={})
        site = Site(stage, world, {"id": new}, catalog, dist=dist, analytics_token=TEST_TOKEN)
        site.newsroom_reading_copies()
        self.assertEqual(set(site.reading_routes), {issue, article})
        validate_pages(stage, dist, set(site.reading_routes.values()), set())
        for destination in site.reading_routes.values():
            html = (stage / destination).read_text(encoding="utf-8")
            self.assertTrue(PublicPage(html, destination).has_beacon)
            today = re.search(r'<a href="([^"]+)">Today</a>', html)[1]
            self.assertEqual(normpath(join(dirname(destination), today)), "index.html")
        self.assertEqual((dist / issue).read_text(encoding="utf-8"), issue_html)
        self.assertEqual((dist / article).read_text(encoding="utf-8"), article_html)
        self.assertEqual(catalog["issues"][0]["articles"][0]["path"], article)
