# StatsPlus Legacy-vs-Fresh Reconciliation

| Legacy table | Family | Fresh table | Status | Confidence | Drift | Recommendation |
|---|---|---|---|---|---|---|
| 01_ABL_Owner_Info.csv | owner info | 01_abl_transactions_personnel_-_owner.csv | LIKELY_MATCH_REVIEW | 0.6317 | yes | REVIEW_SCHEMA_DRIFT |
| 02_ABL_Front_Office_and_Coaches.csv | front office/coaches | 02_abl_transactions_personnel_-_all_coaches.csv | RENAMED_STRUCTURAL_MATCH | 1.0 | no | MATCHED_PENDING_CUTOFF_VALIDATION |
| 03_statsplus_financials_ABL.csv | financials | 03_statsplus_Finantials.csv | RENAMED_STRUCTURAL_MATCH | 1.0 | no | MATCHED_PENDING_CUTOFF_VALIDATION |
| 04_statsplus_Historical_Fan_Interenst_ABL.csv | historical fan interest | 04_statsplus_Historical_Fan_Interest.csv | RENAMED_STRUCTURAL_MATCH | 1.0 | yes | REVIEW_SCHEMA_DRIFT |
| 05_statsplus_fan_data_ABL.csv | fan data | 05_statsplus_Fan_Data.csv | RENAMED_STRUCTURAL_MATCH | 1.0 | no | MATCHED_PENDING_CUTOFF_VALIDATION |
| 06_League Standings.csv | league standings |  | INTENTIONALLY_EXCLUDED | 0.2724 | not_applicable | INTENTIONAL_EXCLUSION_ACCEPTED |
| 07_Playoff_Odds_Div.csv | playoff odds by division | 07_statsplus_Playoff_Odds_Div.csv | RENAMED_STRUCTURAL_MATCH | 1.0 | no | MATCHED_PENDING_CUTOFF_VALIDATION |
| 08_Playoff_Odds_League.csv | playoff odds by league | 08_statsplus_Playoff_Odds_League.csv | RENAMED_STRUCTURAL_MATCH | 1.0 | no | MATCHED_PENDING_CUTOFF_VALIDATION |
| 09_Base Runs.csv | BaseRuns | 09_statsplus_Base_Runs.csv | RENAMED_STRUCTURAL_MATCH | 1.0 | no | MATCHED_PENDING_CUTOFF_VALIDATION |
| 10_statsplus_Elo_ABL.csv | ELO ratings | 10_statsplus_Elo.csv | RENAMED_STRUCTURAL_MATCH | 1.0 | no | MATCHED_PENDING_CUTOFF_VALIDATION |
| 11_Team WAR.csv | team WAR | 11_statsplus_Team_WAR.csv | RENAMED_STRUCTURAL_MATCH | 1.0 | no | MATCHED_PENDING_CUTOFF_VALIDATION |
| 12_Injury Summary.csv | injury summary | 12_statsplus_injury_sumary.csv | RENAMED_STRUCTURAL_MATCH | 1.0 | no | MATCHED_PENDING_CUTOFF_VALIDATION |
| 13_Team_Batting_Div.csv | team batting by division | 13_statsplus_Team_Batting_Div.csv | RENAMED_STRUCTURAL_MATCH | 1.0 | no | MATCHED_PENDING_CUTOFF_VALIDATION |
| 14_Team_Batting_League.csv | team batting by league | 14_statsplus_Team_Batting_League.csv | RENAMED_STRUCTURAL_MATCH | 1.0 | no | MATCHED_PENDING_CUTOFF_VALIDATION |
| 15_Team_Pitching_Div.csv | team pitching by division | 15_statsplus_Team_Pitching_Div.csv | RENAMED_STRUCTURAL_MATCH | 1.0 | yes | REVIEW_SCHEMA_DRIFT |
| 16_Team_Pitching_League.csv | team pitching by league | 16_statsplus_Team_Pitching_league.csv | RENAMED_STRUCTURAL_MATCH | 1.0 | yes | REVIEW_SCHEMA_DRIFT |
| 17_Team_Fielding_Div.csv | team fielding by division | 17_statsplus_Team_Fielding_Div.csv | RENAMED_STRUCTURAL_MATCH | 1.0 | no | MATCHED_PENDING_CUTOFF_VALIDATION |
| 18_Team_Fielding_League.csv | team fielding by league | 18_statsplus_Team_Fielding_League.csv | RENAMED_STRUCTURAL_MATCH | 1.0 | no | MATCHED_PENDING_CUTOFF_VALIDATION |
| 19_Team_Baserunning.csv | team baserunning | 19_statsplus_Team_Baserunning.csv | RENAMED_STRUCTURAL_MATCH | 1.0 | yes | REVIEW_SCHEMA_DRIFT |
| 20_Player_Batting.csv | player batting | 20_statsplus_Player_Batting.csv | RENAMED_STRUCTURAL_MATCH | 1.0 | no | MATCHED_PENDING_CUTOFF_VALIDATION |
| 21_Player_PItching.csv | player pitching | 21_statsplus_Player_Pitching.csv | RENAMED_STRUCTURAL_MATCH | 1.0 | yes | REVIEW_SCHEMA_DRIFT |
| 22_Player_BaseRunning.csv | player baserunning | 22_statsplus_Player_Baserunning.csv | RENAMED_STRUCTURAL_MATCH | 1.0 | no | MATCHED_PENDING_CUTOFF_VALIDATION |
| 23_Player_Fielding.csv | player fielding | 23_statsplus_Player_Fielding.csv | RENAMED_STRUCTURAL_MATCH | 1.0 | no | MATCHED_PENDING_CUTOFF_VALIDATION |
| 24_ABL_Team_Age_Data.csv | team age | 24_statsplus_Team_Age.csv | LIKELY_MATCH_REVIEW | 1.0 | yes | REVIEW_SCHEMA_DRIFT |
| 25_Best_Batting_Game.csv | best batting game | 25_statsplus_Best_Batting_Game.csv | RENAMED_STRUCTURAL_MATCH | 1.0 | no | MATCHED_PENDING_CUTOFF_VALIDATION |
| 26_Best_Pitching_Game.csv | best pitching game | 26_statsplus_Best_Pitching_Game.csv | RENAMED_STRUCTURAL_MATCH | 1.0 | no | MATCHED_PENDING_CUTOFF_VALIDATION |
| 27_gtoc_table_combined.csv | Grand Tournament of Champions |  | INTENTIONALLY_EXCLUDED | 0.15 | not_applicable | INTENTIONAL_EXCLUSION_ACCEPTED |

Missing legacy tables, new tables, renamed tables, duplicate views, and schema drift require explicit review. A structural match does not establish current-date compatibility.
