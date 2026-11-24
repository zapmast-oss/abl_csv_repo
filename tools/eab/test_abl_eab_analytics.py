"""Check that an optional public beacon preserves existing publications."""

from pathlib import Path
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

    def test_copies_track_legacy_newsroom_without_changing_the_original(self):
        old = "1981-08-20_1981-08-19_" + "a" * 16
        new = "1981-08-20_1981-08-19_" + "b" * 16
        dist, stage = self.temp / "published", self.temp / "staged"
        issue = f"releases/{old}/newsroom/issues/story/index.html"
        article = str(Path(issue).with_name("article.html")).replace("\\", "/")
        (dist / issue).parent.mkdir(parents=True)
        issue_html = '<body><a href="article.html#copy">Read the story</a></body>'
        (dist / issue).write_text(issue_html, encoding="utf-8")
        (dist / article).write_text('<body><p id="copy">Story text</p></body>', encoding="utf-8")
        catalog = {"issues": [{"path": issue, "articles": [{"path": article}]}]}
        world = SimpleNamespace(logos={}, photos={})
        site = Site(stage, world, {"id": new}, catalog, dist=dist, analytics_token=TEST_TOKEN)
        site.tracked_newsroom()
        validate_pages(stage, dist, set(site.reading_routes.values()), set())
        for destination in site.reading_routes.values():
            self.assertTrue(PublicPage((stage / destination).read_text(encoding="utf-8"), destination).has_beacon)
        self.assertEqual((dist / issue).read_text(encoding="utf-8"), issue_html)
        self.assertEqual(catalog["issues"][0]["articles"][0]["path"], article)
