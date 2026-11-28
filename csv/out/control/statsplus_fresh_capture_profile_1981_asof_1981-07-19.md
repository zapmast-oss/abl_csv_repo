# StatsPlus Fresh-Capture Profile

**Target:** 1981 as of 1981-07-19  
**Cutoff caution:** Structural matching does not prove that table values reach the target date.

| # | File | Source type | Family | Grain | Legacy match | Confidence | Drift | Authority | Recommendation |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 01_abl_transactions_personnel_-_owner.csv | OOTP-derived table | owner info | team | LIKELY_LEGACY_MATCH | 0.6317 | yes | STATSPLUS_SPECIFIC_AFTER_PROMOTION | PROFILE_FOR_POSSIBLE_PROMOTION |
| 2 | 02_abl_transactions_personnel_-_all_coaches.csv | OOTP-derived table | front office/coaches | staff/front office | RENAMED_STRUCTURAL_MATCH | 1.0 | no | CROSSCHECK_ONLY | CROSSCHECK_ONLY |
| 3 | 03_statsplus_Finantials.csv | StatsPlus export | financials | team | RENAMED_STRUCTURAL_MATCH | 1.0 | no | STATSPLUS_SPECIFIC_AFTER_PROMOTION | PROFILE_FOR_POSSIBLE_PROMOTION |
| 4 | 04_statsplus_Historical_Fan_Interest.csv | StatsPlus export | historical fan interest | historical/franchise | RENAMED_STRUCTURAL_MATCH | 1.0 | yes | STATSPLUS_SPECIFIC_AFTER_PROMOTION | PROFILE_FOR_POSSIBLE_PROMOTION |
| 5 | 05_statsplus_Fan_Data.csv | StatsPlus export | fan data | team | RENAMED_STRUCTURAL_MATCH | 1.0 | no | STATSPLUS_SPECIFIC_AFTER_PROMOTION | PROFILE_FOR_POSSIBLE_PROMOTION |
| 7 | 07_statsplus_Playoff_Odds_Div.csv | StatsPlus export | playoff odds by division | model/projection | RENAMED_STRUCTURAL_MATCH | 1.0 | no | STATSPLUS_SPECIFIC_AFTER_PROMOTION | PROFILE_FOR_POSSIBLE_PROMOTION |
| 8 | 08_statsplus_Playoff_Odds_League.csv | StatsPlus export | playoff odds by league | model/projection | RENAMED_STRUCTURAL_MATCH | 1.0 | no | STATSPLUS_SPECIFIC_AFTER_PROMOTION | PROFILE_FOR_POSSIBLE_PROMOTION |
| 9 | 09_statsplus_Base_Runs.csv | StatsPlus export | BaseRuns | team | RENAMED_STRUCTURAL_MATCH | 1.0 | no | STATSPLUS_SPECIFIC_AFTER_PROMOTION | PROFILE_FOR_POSSIBLE_PROMOTION |
| 10 | 10_statsplus_Elo.csv | StatsPlus export | ELO ratings | team | RENAMED_STRUCTURAL_MATCH | 1.0 | no | STATSPLUS_SPECIFIC_AFTER_PROMOTION | PROFILE_FOR_POSSIBLE_PROMOTION |
| 11 | 11_statsplus_Team_WAR.csv | StatsPlus export | team WAR | team | RENAMED_STRUCTURAL_MATCH | 1.0 | no | STATSPLUS_SPECIFIC_AFTER_PROMOTION | PROFILE_FOR_POSSIBLE_PROMOTION |
| 12 | 12_statsplus_injury_sumary.csv | StatsPlus export | injury summary | team | RENAMED_STRUCTURAL_MATCH | 1.0 | no | STATSPLUS_SPECIFIC_AFTER_PROMOTION | PROFILE_FOR_POSSIBLE_PROMOTION |
| 13 | 13_statsplus_Team_Batting_Div.csv | StatsPlus export | team batting by division | team | RENAMED_STRUCTURAL_MATCH | 1.0 | no | CROSSCHECK_ONLY | CROSSCHECK_ONLY |
| 14 | 14_statsplus_Team_Batting_League.csv | StatsPlus export | team batting by league | team | RENAMED_STRUCTURAL_MATCH | 1.0 | no | CROSSCHECK_ONLY | CROSSCHECK_ONLY |
| 15 | 15_statsplus_Team_Pitching_Div.csv | StatsPlus export | team pitching by division | team | RENAMED_STRUCTURAL_MATCH | 1.0 | yes | CROSSCHECK_ONLY | CROSSCHECK_ONLY |
| 16 | 16_statsplus_Team_Pitching_league.csv | StatsPlus export | team pitching by league | team | RENAMED_STRUCTURAL_MATCH | 1.0 | yes | CROSSCHECK_ONLY | CROSSCHECK_ONLY |
| 17 | 17_statsplus_Team_Fielding_Div.csv | StatsPlus export | team fielding by division | team | RENAMED_STRUCTURAL_MATCH | 1.0 | no | CROSSCHECK_ONLY | CROSSCHECK_ONLY |
| 18 | 18_statsplus_Team_Fielding_League.csv | StatsPlus export | team fielding by league | team | RENAMED_STRUCTURAL_MATCH | 1.0 | no | CROSSCHECK_ONLY | CROSSCHECK_ONLY |
| 19 | 19_statsplus_Team_Baserunning.csv | StatsPlus export | team baserunning | team | RENAMED_STRUCTURAL_MATCH | 1.0 | yes | STATSPLUS_SPECIFIC_AFTER_PROMOTION | PROFILE_FOR_POSSIBLE_PROMOTION |
| 20 | 20_statsplus_Player_Batting.csv | StatsPlus export | player batting | player | RENAMED_STRUCTURAL_MATCH | 1.0 | no | CROSSCHECK_ONLY | CROSSCHECK_ONLY |
| 21 | 21_statsplus_Player_Pitching.csv | StatsPlus export | player pitching | player | RENAMED_STRUCTURAL_MATCH | 1.0 | yes | CROSSCHECK_ONLY | CROSSCHECK_ONLY |
| 22 | 22_statsplus_Player_Baserunning.csv | StatsPlus export | player baserunning | player | RENAMED_STRUCTURAL_MATCH | 1.0 | no | STATSPLUS_SPECIFIC_AFTER_PROMOTION | PROFILE_FOR_POSSIBLE_PROMOTION |
| 23 | 23_statsplus_Player_Fielding.csv | StatsPlus export | player fielding | player | RENAMED_STRUCTURAL_MATCH | 1.0 | no | CROSSCHECK_ONLY | CROSSCHECK_ONLY |
| 24 | 24_statsplus_Team_Age.csv | StatsPlus export | team age | team | LIKELY_LEGACY_MATCH | 1.0 | yes | STATSPLUS_SPECIFIC_AFTER_PROMOTION | PROFILE_FOR_POSSIBLE_PROMOTION |
| 25 | 25_statsplus_Best_Batting_Game.csv | StatsPlus export | best batting game | game | RENAMED_STRUCTURAL_MATCH | 1.0 | no | STATSPLUS_SPECIFIC_AFTER_PROMOTION | PROFILE_FOR_POSSIBLE_PROMOTION |
| 26 | 26_statsplus_Best_Pitching_Game.csv | StatsPlus export | best pitching game | game | RENAMED_STRUCTURAL_MATCH | 1.0 | no | STATSPLUS_SPECIFIC_AFTER_PROMOTION | PROFILE_FOR_POSSIBLE_PROMOTION |

## Assessment

- Fresh files inspected: 25.
- Readable tables/files: 25; unreadable: 0.
- Numbered tables found: 25.
- Intentionally excluded: tables 6 and 27 (2 total).
- Unexpected missing legacy tables: 0.
- Same/likely legacy tables: 25; renamed structural matches: 23.
- New fresh tables: 0.
- Exact duplicate files: 0; reordered overlaps: 1.
- Schema drift tables: 7.
- High-value story-engine sources: 15.
- Crosscheck-only sources: 10; hold/review sources: 0.
- Capture coherent: **Yes**.
- Ready for a separate dry-run promotion task: **Yes**.

Promotion remains unauthorized. A dry run must retain table-level authority and schema-drift decisions.
