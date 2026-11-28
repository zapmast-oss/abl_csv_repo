# StatsPlus Authority Crosscheck

**Target:** 1981 as of 1981-07-19  
**Promotion safe now:** No—profiling is not promotion approval.  
**Ready for separate dry-run promotion task:** Yes.

| Authority class | Tables |
|---|---|
| CROSSCHECK_ONLY | 10 |
| STATSPLUS_SPECIFIC_AFTER_PROMOTION | 15 |

## Raw OOTP current-record crosscheck

| Outcome | Rows |
|---|---|
| CROSSCHECK_AGREES | 120 |

Conflicts or identity holds: 0.

## Crosscheck-only/overlapping tables

| File | Family | Outcome | Closest raw schema | Closest sortable schema |
|---|---|---|---|---|
| 02_abl_transactions_personnel_-_all_coaches.csv | front office/coaches | CROSSCHECK_ONLY | csv\ootp_csv\league_events.csv | csv\abl_statistics\abl_statistics_player_statistics_-_sortable_stats_player_indicative_1.csv |
| 13_statsplus_Team_Batting_Div.csv | team batting by division | CROSSCHECK_ONLY | csv\ootp_csv\team_batting_stats.csv | csv\abl_statistics\abl_statistics_team_statistics___info_-_sortable_stats_batting_stats.csv |
| 14_statsplus_Team_Batting_League.csv | team batting by league | CROSSCHECK_ONLY | csv\ootp_csv\team_batting_stats.csv | csv\abl_statistics\abl_statistics_team_statistics___info_-_sortable_stats_batting_stats.csv |
| 15_statsplus_Team_Pitching_Div.csv | team pitching by division | CROSSCHECK_ONLY | csv\ootp_csv\team_starting_pitching_stats.csv | csv\abl_statistics\abl_statistics_player_statistics_-_sortable_stats_player_pitch_stats_1.csv |
| 16_statsplus_Team_Pitching_league.csv | team pitching by league | CROSSCHECK_ONLY | csv\ootp_csv\team_starting_pitching_stats.csv | csv\abl_statistics\abl_statistics_player_statistics_-_sortable_stats_player_pitch_stats_1.csv |
| 17_statsplus_Team_Fielding_Div.csv | team fielding by division | CROSSCHECK_ONLY | csv\ootp_csv\team_fielding_stats_stats.csv | csv\abl_statistics\abl_statistics_team_statistics___info_-_sortable_stats_c_fielding_1.csv |
| 18_statsplus_Team_Fielding_League.csv | team fielding by league | CROSSCHECK_ONLY | csv\ootp_csv\team_fielding_stats_stats.csv | csv\abl_statistics\abl_statistics_team_statistics___info_-_sortable_stats_c_fielding_1.csv |
| 20_statsplus_Player_Batting.csv | player batting | CROSSCHECK_ONLY | csv\ootp_csv\team_batting_stats.csv | csv\abl_statistics\abl_statistics_team_statistics___info_-_sortable_stats_batting_stats.csv |
| 21_statsplus_Player_Pitching.csv | player pitching | CROSSCHECK_ONLY | csv\ootp_csv\team_record.csv | csv\abl_statistics\abl_statistics_player_statistics_-_sortable_stats_player_pitch_stats_1.csv |
| 23_statsplus_Player_Fielding.csv | player fielding | CROSSCHECK_ONLY | csv\ootp_csv\team_fielding_stats_stats.csv | csv\abl_statistics\abl_statistics_team_statistics___info_-_sortable_stats_c_fielding_1.csv |

Raw OOTP governs current game/score/record proof. Sortable stats govern overlapping current enrichment. StatsPlus governs only validated StatsPlus-specific fields after explicit promotion. Conflicts must be reported, not silently resolved.
