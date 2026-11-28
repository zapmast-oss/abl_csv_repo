"""Check the recurring conversation archive and both sides of a companion link."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from abl_eab_editorial import Article
from abl_eab_site import Site


class ObserverConversationTests(unittest.TestCase):
    def test_later_episode_reuses_archive_and_cross_links(self):
        with TemporaryDirectory() as folder:
            world = SimpleNamespace(source_id="new", league_date="1981-09-04", cutoff="1981-09-03",
                                    teams={1: {"name": "Miami Hurricanes"}}, logos={}, photos={})
            entry = {"id": "verified-release"}
            earlier = dict(slug="the-pennant-races-come-home", title="The Pennant Races Come Home",
                           path="releases/earlier/newsroom/issues/first/the-pennant-races-come-home.html",
                           league_date="1981-08-20", cutoff="1981-08-19", related_slug="august-division-challenges-1981",
                           kind="observer", teams=[1], digest="earlier", dek="Preview discussion.",
                           byline="The Baseball Observer")
            catalog = {"editions": [entry | {"source_id": world.source_id}], "issues": [
                {"path": "releases/earlier/newsroom/issues/first/index.html",
                 "source_id": "old", "key": "earlier", "league_date": "1981-08-20",
                 "cutoff": "1981-08-19", "articles": [earlier]},
            ]}
            site = Site(Path(folder), world, entry, catalog)
            html = {}
            site.page = lambda path, title, body, *args, **kwargs: html.update({path: body})
            site.team_tile = lambda *_, **__: ""
            site.team = lambda *_: ""
            report = Article("Independent reporting", "independent-reporting", "Ernie Bewell",
                             "observer", (1,), "report", True, (("", ("Preserved reporting paragraph.",)),), (), False)
            episode = Article("A later episode", "a-later-episode", "The Baseball Observer", "observer", (1,),
                              "episode", False, (("The discussion", ('**ERNIE:** "What changed?"',)),), (), True,
                              sealed_pdf=b"%PDF-test", related_slug=report.slug, subtitle="The September discussion")
            base = dict(source_id="new", edition="desk", revision="edition", league_date=world.league_date, cutoff=world.cutoff)
            site.editorial([dict(base, key="discussion", articles=[episode]),
                            dict(base, key="report", articles=[report])], [])
            site.newsroom()
            report_page = html[f"{site.issue_route('report')}/{report.slug}.html"]
            new_page = html[f"{site.issue_route('discussion')}/{episode.slug}.html"]
            archive = html[site.route("newsroom/observer-conversation/index.html")]
            self.assertIn("Preserved reporting paragraph.", report_page)
            self.assertIn("a-later-episode.html", report_page)
            self.assertIn("independent-reporting.html", new_page)
            self.assertIn("the-pennant-races-come-home.html", new_page)
            self.assertIn("observer-conversation/index.html", new_page)
            self.assertIn("<b>ERNIE:</b>", new_page)
            self.assertIn("The September discussion", new_page)
            self.assertLess(archive.index("A later episode"), archive.index("The Pennant Races Come Home"))
            self.assertIn("September 4, 1981", archive)
            self.assertIn("August 20, 1981", archive)

    def test_companion_can_be_published_after_the_episode(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            dist = root / "public"
            world = SimpleNamespace(source_id="new", league_date="1981-08-31", cutoff="1981-08-30",
                                    teams={1: {"name": "Miami Hurricanes"}}, logos={}, photos={})
            entry = dict(id="second-release", source_id=world.source_id)
            original = "releases/first/newsroom/issues/first/the-forecast-meets-the-facts.html"
            issue = "releases/first/newsroom/issues/first/index.html"
            episode = dict(path=original, title="The Forecast Meets the Facts", slug="the-forecast-meets-the-facts",
                           league_date=world.league_date, cutoff=world.cutoff, kind="observer",
                           related_slug="this-is-the-test", teams=[1], digest="episode")
            source = '<html><body><article class="reading coverage"><p>Approved dialogue.</p></article></body></html>'
            (dist/original).parent.mkdir(parents=True)
            (dist/original).write_text(source, encoding="utf-8")
            (dist/issue).write_text("<html><body>Issue</body></html>", encoding="utf-8")
            catalog = {"editions": [entry], "issues": [{"path": issue, "source_id": "new", "key": "episode",
                                                         "articles": [episode], "category": "The Observer Conversation"}]}
            site = Site(root/"first", world, entry, catalog, dist=dist)
            site.newsroom_reading_copies()
            before = (site.stage/site.reading_routes[original]).read_text(encoding="utf-8")
            self.assertIn("The Observer Conversation archive", before)
            self.assertNotIn("this-is-the-test.html", before)

            report = Article("This Is the Test", "this-is-the-test", "Ernie Bewell", "observer", (1,),
                             "report", True, (("", ("Independent reporting.",)),), (), False)
            eligible = [dict(key="report", source_id=world.source_id, edition="desk", revision="report",
                             league_date=world.league_date, cutoff=world.cutoff, articles=[report])]
            later = Site(root/"second", world, entry, catalog, dist=dist)
            later.newsroom_reading_copies(eligible)
            after = (later.stage/later.reading_routes[original]).read_text(encoding="utf-8")
            self.assertIn("this-is-the-test.html", after)
            self.assertIn("Approved dialogue.", after)
            html = {}
            later.page = lambda path, title, body, *args, **kwargs: html.update({path: body})
            later.team = lambda *_: ""
            later.editorial(eligible, [])
            self.assertIn("the-forecast-meets-the-facts.html", html[f"{later.issue_route('report')}/{report.slug}.html"])
            self.assertEqual((dist/original).read_text(encoding="utf-8"), source)


if __name__ == "__main__":
    unittest.main()
