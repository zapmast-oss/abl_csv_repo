# StatsPlus Story-Enrichment Plan

No source in this plan is enabled before promotion.

| File | Family | Candidate signal | State | Validation required |
|---|---|---|---|---|
| 01_abl_transactions_personnel_-_owner.csv | owner info | organization context | candidate_after_promotion | cutoff, schema, identity, duplicate, and field-semantic validation |
| 03_statsplus_Finantials.csv | financials | resource/results pressure | candidate_after_promotion | cutoff, schema, identity, duplicate, and field-semantic validation |
| 04_statsplus_Historical_Fan_Interest.csv | historical fan interest | historical fan-pressure context | candidate_after_promotion | cutoff, schema, identity, duplicate, and field-semantic validation |
| 05_statsplus_Fan_Data.csv | fan data | fan-pressure context | candidate_after_promotion | cutoff, schema, identity, duplicate, and field-semantic validation |
| 07_statsplus_Playoff_Odds_Div.csv | playoff odds by division | division-race probability and pressure | candidate_after_promotion | cutoff, schema, identity, duplicate, and field-semantic validation |
| 08_statsplus_Playoff_Odds_League.csv | playoff odds by league | league qualification probability | candidate_after_promotion | cutoff, schema, identity, duplicate, and field-semantic validation |
| 09_statsplus_Base_Runs.csv | BaseRuns | expected-performance gap | candidate_after_promotion | cutoff, schema, identity, duplicate, and field-semantic validation |
| 10_statsplus_Elo.csv | ELO ratings | team strength and momentum | candidate_after_promotion | cutoff, schema, identity, duplicate, and field-semantic validation |
| 11_statsplus_Team_WAR.csv | team WAR | team strength | candidate_after_promotion | cutoff, schema, identity, duplicate, and field-semantic validation |
| 12_statsplus_injury_sumary.csv | injury summary | injury burden | candidate_after_promotion | cutoff, schema, identity, duplicate, and field-semantic validation |
| 19_statsplus_Team_Baserunning.csv | team baserunning | team baserunning | candidate_after_promotion | cutoff, schema, identity, duplicate, and field-semantic validation |
| 22_statsplus_Player_Baserunning.csv | player baserunning | player baserunning | candidate_after_promotion | cutoff, schema, identity, duplicate, and field-semantic validation |
| 24_statsplus_Team_Age.csv | team age | organization age profile | candidate_after_promotion | cutoff, schema, identity, duplicate, and field-semantic validation |
| 25_statsplus_Best_Batting_Game.csv | best batting game | game-performance discovery | candidate_after_promotion | cutoff, schema, identity, duplicate, and field-semantic validation |
| 26_statsplus_Best_Pitching_Game.csv | best pitching game | game-performance discovery | candidate_after_promotion | cutoff, schema, identity, duplicate, and field-semantic validation |

StatsPlus supplies model/context enrichment. Raw OOTP remains the proof layer. Manager tendencies remain disabled without validated same-cutoff staff identity and fields.
