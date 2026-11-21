"""Editorial regressions that require neither OOTP nor changes to an edition."""
import copy
import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"csv/abl_scripts"))
from abl_newsroom_slate import assess, pitch, trade_pitches, wildcard_pitch


class SlateTests(unittest.TestCase):
    def test_material_followup_quiet_replay_and_same_cutoff_correction(self):
        original = pitch("2:1981:race","competition","A shared lead","Both clubs are 64-57.",
                         "The next decision could separate the clubs.","Who leads after the series?",
                         ["games.csv completed regular-season games","teams.csv division membership"],
                         [2,10],21,16,related=[2,10],direction="co-leader")
        opening = copy.deepcopy(original)
        first = assess(opening,None,None,"1981-08-19","A",None)
        self.assertTrue(opening["eligible"])
        publication = dict(edition="A",path="beat/02-PHO.md",trigger=opening["trigger"])

        quiet = copy.deepcopy(original)
        second = assess(quiet,first,"1981-08-19","1981-08-20","B",publication)
        self.assertFalse(quiet["eligible"])
        self.assertEqual(first["developments"],second["developments"])
        self.assertEqual(quiet["coverage_reference"],"A/beat/02-PHO.md")

        amended = copy.deepcopy(original)
        amended["observation"] = "Both clubs are 63-57."
        third = assess(amended,first,"1981-08-19","1981-08-19","C",publication)
        self.assertTrue(amended["eligible"])
        self.assertEqual(amended["assessment"],"Corrected evidence")
        self.assertEqual(third["first_seen"],"1981-08-19")
        self.assertEqual(third["developments"][0]["edition"],"A")

        changed = copy.deepcopy(original)
        changed.update(direction="leader",trigger=[2],observation="Phoenix now leads outright.")
        fourth = assess(changed,second,"1981-08-20","1981-08-21","D",publication)
        self.assertTrue(changed["eligible"])
        self.assertEqual(fourth["latest_assessment"],"Earlier assessment changed")

    def test_trade_comparison_excludes_trade_day_other_teams_and_wrong_role(self):
        def line(when,gid,tid,outs,er):
            return (when,gid,tid,77,dict(outs=outs,er=er,k=3,g=1,gs=1,s=0))
        appearances = [
            line("1981-07-24",1,11,36,8),line("1981-07-28",2,8,27,10),
            line("1981-07-31",3,8,27,12),line("1981-07-31",4,24,27,13),
            line("1981-08-01",5,24,18,1),line("1981-08-02",6,24,18,0),
        ]
        deal = dict(id="trade:44",date="1981-07-31",partner_id=8,incoming_ids=[77],
                    source="trade_history.csv message_id=44")
        result, = trade_pitches(24,{24:{"name":"Seattle"},8:{"name":"Tampa Bay"}},[deal],{77:dict(first_name="Example",last_name="Pitcher")},
            {"bat":{},"pit":{77:appearances}},date(1981,8,2))
        self.assertIn("12 innings, 1 earned run",result["observation"])
        self.assertIn("9 innings and 10 earned runs in 1 game",result["observation"])
        self.assertIn("from Tampa Bay",result["observation"])
        self.assertIn("ERA from Pitcher",result["significance"])
        self.assertIn("0.75 ERA",result["observation"])
        self.assertEqual(result["workload"],36)
        self.assertEqual(result["evidence"][-1],"games.csv game_id=5,6 (ABL completed regular season)")

    def test_wild_card_holds_if_division_assignment_is_ambiguous(self):
        clubs = {i:{"name":f"Club {i}","conference":"ABC"} for i in range(1,5)}
        groups = {"east":[1,2],"west":[3,4]}
        records = {i:dict(wins=w,losses=l,ties=0) for i,(w,l) in enumerate([(60,40),(60,40),(61,39),(59,41)],1)}
        self.assertIsNone(wildcard_pitch(4,clubs,groups,records,date(1981,8,19),1))
        records[2]["wins"]=59
        result = wildcard_pitch(4,clubs,groups,records,date(1981,8,19),1)
        self.assertIn("provisional ABC wild-card order",result["observation"])
        self.assertIn("league_playoffs.csv",result["evidence"][0])


if __name__ == "__main__":
    unittest.main()
