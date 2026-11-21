"""A day with a shared lead and two unfinished fixtures for the same pair of clubs."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"csv/abl_scripts"))
from abl_newsroom_dailydesk import division_brief, pair_next


def rec(w,l):
    return dict(wins=w,losses=l,ties=0)


def due(gid,a,h,innings=0):
    return dict(game_id=gid,date="1981-08-20",away_id=a,home_id=h,recorded_innings=innings)


class DailyDeskTests(unittest.TestCase):
    def test_shared_lead_does_not_get_reported_as_a_new_lead_change(self):
        clubs={1:{"name":"Phoenix"},2:{"name":"San Francisco"},3:{"name":"Los Angeles"},4:{"name":"San Diego"}}
        prior={1:rec(63,57),2:rec(63,57),3:rec(62,58),4:rec(44,76)}
        current={1:rec(64,57),2:rec(64,57),3:rec(62,59),4:rec(45,76)}
        division=division_brief("NBC Western Division",list(clubs),current,prior,[],
            [due(1457,4,3),due(1458,1,2)],clubs,"1981-08-19")
        self.assertIn("remained tied at 64-57",division["movement"])
        self.assertIn("Los Angeles slipped",division["movement"])
        self.assertIn("a full game ahead of the loser",division["next"])
        self.assertEqual(division["scheduled_game_ids"],[1457,1458])
        self.assertEqual(division["completed_game_ids"],[])

    def test_double_listing_and_unfinished_game_change_possible_margin(self):
        clubs={1:{"name":"Houston"},2:{"name":"Cincinnati"},3:{"name":"St. Louis"},4:{"name":"Nashville"}}
        prior={1:rec(72,47),2:rec(62,58),3:rec(57,62),4:rec(58,62)}
        current={1:rec(72,48),2:rec(63,58),3:rec(58,62),4:rec(59,62)}
        listing=[due(1113,3,1,5),due(1461,4,2),due(1462,1,3)]
        division=division_brief("ABC Central Division",list(clubs),current,prior,[],listing,clubs,"1981-08-19")
        self.assertIn("10.5 games to 9.5 games",division["movement"])
        self.assertIn("unfinished, 5 innings already recorded",division["next"])
        self.assertIn("8 games",division["next"])
        self.assertEqual(set(division["scheduled_game_ids"]),{1113,1461,1462})
        self.assertEqual(division["completed_game_ids"],[])
        self.assertTrue(any("game_id=1113 played=0" in ref for ref in division["evidence"]))

    def test_offday_allows_only_half_a_game_of_movement(self):
        clubs={1:{"name":"Dallas"},2:{"name":"Detroit"},3:{"name":"Chicago"}}
        note=pair_next(1,2,[due(1,2,3)],{1:rec(66,55),2:rec(65,56)},clubs)
        self.assertIn("no Dallas game listed",note)
        self.assertIn("Detroit win would gain half a game",note)


if __name__ == "__main__":
    unittest.main()
