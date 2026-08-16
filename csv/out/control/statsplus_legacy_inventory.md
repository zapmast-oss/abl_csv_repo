# StatsPlus Legacy Inventory

**Mode:** Read-only; nothing copied or promoted.

| Measure | Count |
|---|---|
| Total Files | 36 |
| Csv Files | 34 |
| Txt Files | 2 |
| Folders | 1 |
| Likely Statsplus Tables | 27 |
| Coach Staff Division Files | 6 |
| Deep Dive 25 Txt Files | 2 |

## Files

| Relative path | Rows | Cols | Family | Grain | Role | Value |
|---|---|---|---|---|---|---|
| 01_ABL_Owner_Info.csv | 24 | 14 | owner info | team | StatsPlus legacy source | high |
| 02_ABL_Front_Office_and_Coaches.csv | 240 | 17 | front office/coaches | staff/front office | StatsPlus legacy source | medium |
| 03_statsplus_financials_ABL.csv | 25 | 11 | financials | team | StatsPlus legacy source | medium |
| 04_statsplus_Historical_Fan_Interenst_ABL.csv | 24 | 12 | historical fan interest | historical/franchise | StatsPlus legacy source | high |
| 05_statsplus_fan_data_ABL.csv | 25 | 11 | fan data | team | StatsPlus legacy source | high |
| 06_League Standings.csv | 24 | 17 | league standings | team | StatsPlus legacy source | medium |
| 07_Playoff_Odds_Div.csv | 24 | 13 | playoff odds by division | model/projection | StatsPlus legacy source | high |
| 08_Playoff_Odds_League.csv | 24 | 13 | playoff odds by league | model/projection | StatsPlus legacy source | high |
| 09_Base Runs.csv | 24 | 22 | BaseRuns | team | StatsPlus legacy source | high |
| 10_statsplus_Elo_ABL.csv | 24 | 11 | ELO ratings | team | StatsPlus legacy source | high |
| 11_Team WAR.csv | 24 | 8 | team WAR | team | StatsPlus legacy source | high |
| 12_Injury Summary.csv | 24 | 5 | injury summary | team | StatsPlus legacy source | high |
| 13_Team_Batting_Div.csv | 25 | 22 | team batting by division | team | StatsPlus legacy source | medium |
| 14_Team_Batting_League.csv | 25 | 22 | team batting by league | team | StatsPlus legacy source | medium |
| 15_Team_Pitching_Div.csv | 25 | 19 | team pitching by division | team | StatsPlus legacy source | medium |
| 16_Team_Pitching_League.csv | 25 | 19 | team pitching by league | team | StatsPlus legacy source | medium |
| 17_Team_Fielding_Div.csv | 25 | 20 | team fielding by division | team | StatsPlus legacy source | medium |
| 18_Team_Fielding_League.csv | 25 | 20 | team fielding by league | team | StatsPlus legacy source | medium |
| 19_Team_Baserunning.csv | 24 | 9 | team baserunning | team | StatsPlus legacy source | high |
| 20_Player_Batting.csv | 100 | 20 | player batting | player | StatsPlus legacy source | medium |
| 21_Player_PItching.csv | 96 | 22 | player pitching | player | StatsPlus legacy source | medium |
| 22_Player_BaseRunning.csv | 100 | 9 | player baserunning | player | StatsPlus legacy source | medium |
| 23_Player_Fielding.csv | 100 | 15 | player fielding | player | StatsPlus legacy source | medium |
| 24_ABL_Team_Age_Data.csv | 24 | 14 | team age | team | StatsPlus legacy source | high |
| 25_Best_Batting_Game.csv | 100 | 13 | best batting game | game | StatsPlus legacy source | high |
| 26_Best_Pitching_Game.csv | 100 | 13 | best pitching game | game | StatsPlus legacy source | high |
| 27_gtoc_table_combined.csv | 9 | 11 | Grand Tournament of Champions | historical/franchise | StatsPlus legacy source | high |
| Front_Office_and_Coaches_by_Div\ABC_Central.csv | 40 | 17 | coach/staff division | staff/front office | coach/staff division source | medium |
| Front_Office_and_Coaches_by_Div\ABC_East.csv | 40 | 17 | coach/staff division | staff/front office | coach/staff division source | medium |
| Front_Office_and_Coaches_by_Div\ABC_West.csv | 40 | 17 | coach/staff division | staff/front office | coach/staff division source | medium |
| Front_Office_and_Coaches_by_Div\NBC_Central.csv | 40 | 17 | coach/staff division | staff/front office | coach/staff division source | medium |
| Front_Office_and_Coaches_by_Div\NBC_East.csv | 40 | 17 | coach/staff division | staff/front office | coach/staff division source | medium |
| Front_Office_and_Coaches_by_Div\NBC_West.csv | 40 | 17 | coach/staff division | staff/front office | coach/staff division source | medium |
| Phoenix_Firebirds_Season_History.csv | 10 | 17 | franchise season history | historical/franchise | derived output | medium |
| Points not being used but valuable.txt |  |  | Deep Dive 25 notes | text/deep-dive notes | Deep Dive 25 source | medium |
| Team Profile Prompts and Tables.txt |  |  | Deep Dive 25 notes | text/deep-dive notes | Deep Dive 25 source | medium |

## Warnings

- Early-1981 baseline, not July 19 current proof.
- Team batting 13/14 and pitching 15/16 are byte-identical. Fielding 17/18 contain the same row set in different sort order.
- The six staff division files exactly reconstruct the 240-row master but lack IDs. TXT files are reference-only design material.
