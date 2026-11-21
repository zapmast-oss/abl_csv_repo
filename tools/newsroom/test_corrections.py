"""Exercise scoring revisions without touching league files, published editions or pointers."""
import csv
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "csv/abl_scripts"))
from abl_newsroom_corrections import changed_lines


def write_table(folder, name, header, records):
    with (folder / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header.split())
        writer.writerows(records)


class TestScoringCorrections(unittest.TestCase):
    def test_revised_rbi_inning_and_reordered_lines(self):
        work = Path(tempfile.gettempdir()) / f"abl-correction-test-{uuid4().hex}"
        old, new = work / "old", work / "new"
        old.mkdir(parents=True)
        new.mkdir()
        self.addCleanup(shutil.rmtree, work)
        header = "league_id year level_id team_id player_id game_id ab h rbi wpa"
        batting = [(200, 1981, 1, 17, 923, 1450, 4, 3, 1, "0.200"),
                   (200, 1981, 1, 17, 9402, 1450, 3, 2, 0, "0.10")]
        scores = [(1450, 0, 1, 1), (1450, 0, 2, 0), (1450, 1, 1, 2)]
        for folder in (old, new):
            write_table(folder, "players_game_batting.csv", header, batting if folder == old else reversed(batting))
            write_table(folder, "players_game_pitching_stats.csv", "league_id year level_id team_id player_id game_id er", [(200,1981,1,17,8003,1450,2)])
            write_table(folder, "games_score.csv", "game_id team inning score", scores if folder == old else reversed(scores))
        self.assertEqual(changed_lines(old, new, [1450], 1981), {})
        # Reassign an RBI, retain another numeric value, and move a run between innings.
        write_table(new, "players_game_batting.csv", header, [
            (200,1981,1,17,9402,1450,3,2,0,".100"), (200,1981,1,17,923,1450,4,3,2,".2"),
            (200,1981,1,17,9402,1451,3,2,3,"0.4")])
        write_table(new, "games_score.csv", "game_id team inning score", [(1450,0,1,0),(1450,0,2,1),(1450,1,1,2)])
        diff = changed_lines(old, new, ["1450"], 1981)
        self.assertEqual(set(diff), {1450})
        self.assertEqual(len(diff[1450]), 3)
        self.assertIn("rbi 1 → 2", diff[1450][0])
        self.assertTrue(any("team=0 inning=2" in note for note in diff[1450]))
        self.assertNotIn("wpa", " ".join(diff[1450]))


if __name__ == "__main__":
    unittest.main()
