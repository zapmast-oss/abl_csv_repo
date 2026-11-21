"""Replay the first editorial assignment from its immutable source capture."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "csv" / "abl_scripts"))
from abl_newsroom_beats import build_beats
from abl_newsroom_east import CHARLOTTE, MIAMI, ATLANTA, TAMPA, EastDesk, build_east
from abl_newsroom_factcheck import ClaimCheck


CAPTURE = ROOT / "output/newsroom/snapshots/2acae29146076f6298fcafecf8afc73fb47bfdc8e8362ea73a5c38454767d02d"


@unittest.skipUnless(CAPTURE.is_dir(), "August 19 source capture is unavailable")
class EastReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.beats = build_beats(CAPTURE, "1981-08-19", set())
        cls.edition = build_east(CAPTURE, "1981-08-19", {1446}, cls.beats)

    def test_edition_claims(self):
        self.assertEqual(set(self.edition["teams"]), {CHARLOTTE, MIAMI, ATLANTA, TAMPA})
        for club, expected in {
            CHARLOTTE: ("28-15", "216-149", "Jose Hernandez", "2.58 ERA"),
            MIAMI: ("20-16", "10-16", "17-for-49", "14-for-56", "12-8"),
            ATLANTA: ("57-46", "11 innings", "30 2/3 innings", "31-48", "24-18"),
            TAMPA: ("18-2", ".328", "31 2/3 innings", "2 1/3 innings"),
        }.items():
            p = self.edition["teams"][club]
            story = " ".join(par for _, paragraphs in p["sections"] for par in paragraphs)
            for fact in expected:
                self.assertIn(fact, story)
            self.assertTrue(p["evidence"])
            self.assertTrue(p["trigger"])
        league = self.edition["division"]
        self.assertIn("1850", league["evidence"][-1])
        self.assertIn("eight meetings", str(league["sections"]))

    def test_future_performances_and_trade_investigations_excluded(self):
        before_deals = build_beats(CAPTURE, "1981-07-22", set(), "1981-07-22")
        edition = build_east(CAPTURE, "1981-07-22", set(), before_deals, "1981-07-22")
        self.assertNotIn(MIAMI, edition["teams"])
        self.assertNotIn(TAMPA, edition["teams"])
        self.assertIsNone(edition["division"])
        self.assertNotIn("1446", str(edition))

    def test_release_checker_distinguishes_completed_games_and_scheduled_fixtures(self):
        claims = ClaimCheck(CAPTURE, "1981-08-19", "1981-08-20", self.beats)
        for p in (*self.edition["teams"].values(), self.edition["division"]):
            claims.evaluate(p)
        checked = claims.evaluate(self.edition["teams"][TAMPA])
        self.assertTrue({1446, 1454, 1466, 1478, 1490} <= set(checked["games"]))

    def test_altered_notebook_and_unadmitted_game_block(self):
        altered = {**self.beats, MIAMI: {**self.beats[MIAMI], "wins": 65}}
        with self.assertRaisesRegex(ValueError, "disagree"):
            build_east(CAPTURE, "1981-08-19", set(), altered)
        with self.assertRaisesRegex(ValueError, "unverified"):
            build_east(CAPTURE, "1981-08-19", {1453}, self.beats)

    def test_acquisition_anchor_survives_routine_game(self):
        old = build_beats(CAPTURE, "1981-08-18", set(), "1981-08-18")
        earlier = EastDesk(CAPTURE, "1981-08-18", old)
        self.assertEqual(earlier.miami()["trigger"], self.edition["teams"][MIAMI]["trigger"])
        self.assertIsNone(earlier.tampa())


if __name__ == "__main__":
    unittest.main()
