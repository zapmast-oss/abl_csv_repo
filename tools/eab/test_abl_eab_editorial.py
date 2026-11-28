"""Check that the public copy keeps the newsroom's full credit line."""

from pathlib import Path
import tempfile
import unittest

from abl_eab_editorial import approved_copy, identity


class EditorialCopyTests(unittest.TestCase):
    def test_pipe_in_byline_keeps_full_credit(self):
        manifest = {"league_date": "1981-08-31", "cutoff": "1981-08-30"}
        item = {"title": "Test report", "slug": "test-report", "kind": "observer",
                "team_ids": [13], "lead": True, "claim_games": [1572]}
        for byline in ("Ernie Bewell | The Baseball Observer", "The Desk | Action Baseball League"):
            with self.subTest(byline=byline), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "test-report.md"
                content = (f"# Test report\n\n{byline} | 1981-08-31 | Games through 1981-08-30\n\n"
                           "A preserved paragraph.\n").encode()
                path.write_bytes(content)
                article = approved_copy(path, item, manifest, identity(content))
                self.assertEqual(article.byline, byline)
                self.assertEqual(article.sections, (("", ("A preserved paragraph.",)),))
                with self.assertRaisesRegex(ValueError, "cutoff"):
                    approved_copy(path, item, {**manifest, "cutoff": "1981-08-29"}, identity(content))
