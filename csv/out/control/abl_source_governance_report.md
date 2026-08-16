# ABL Source Governance Report

Registered sources: **509**  
Can drive current state: **3**  
Can support current state: **93**

Current-state authority is intentionally narrow: only raw OOTP game/result sources can prove the active date and results. Supplemental reports can enrich a validated capture but cannot advance its cutoff.

## Sources by family

| Family | Count |
|---|---:|
| `documentation` | 26 |
| `editorial_config` | 5 |
| `generated_output` | 103 |
| `historical_almanac` | 282 |
| `ootp_csv` | 73 |
| `sortable_stats` | 20 |

## Sources by authority

| Authority | Count |
|---|---:|
| `derived_output` | 103 |
| `documentation` | 26 |
| `editorial_config` | 5 |
| `historical_context` | 282 |
| `supplemental_report_extract` | 20 |
| `system_of_record_extract` | 73 |

## Files that can drive current state

- `csv/ootp_csv/game_logs.csv` — Filter league/game type and completed rows; detect maximum played game date and completed games per team.
- `csv/ootp_csv/games.csv` — Filter league/game type and completed rows; detect maximum played game date and completed games per team.
- `csv/ootp_csv/games_score.csv` — Filter league/game type and completed rows; detect maximum played game date and completed games per team.

## Sortable stats roles

| Path | Subject | Volatility | Curated target |
|---|---|---|---|
| `csv/abl_statistics/abl_statistics_player_statistics_-_sortable_stats_player_bat_ratings.csv` | batting | slow-changing | curated_current_player_batting or curated_current_team_stats |
| `csv/abl_statistics/abl_statistics_player_statistics_-_sortable_stats_player_bat_stats.csv` | batting | changes every export | curated_current_player_batting or curated_current_team_stats |
| `csv/abl_statistics/abl_statistics_player_statistics_-_sortable_stats_player_field_ratings.csv` | fielding_and_defense | slow-changing | curated_current_team_stats |
| `csv/abl_statistics/abl_statistics_player_statistics_-_sortable_stats_player_field_stats.csv` | fielding_and_defense | changes every export | curated_current_team_stats |
| `csv/abl_statistics/abl_statistics_player_statistics_-_sortable_stats_player_indicative_1.csv` | player_indicative_and_misc | changes every export | curated_current_player_indicative |
| `csv/abl_statistics/abl_statistics_player_statistics_-_sortable_stats_player_indicative_2.csv` | player_indicative_and_misc | changes every export | curated_current_player_indicative |
| `csv/abl_statistics/abl_statistics_player_statistics_-_sortable_stats_player_misc_info.csv` | player_indicative_and_misc | changes every export | curated_current_player_indicative |
| `csv/abl_statistics/abl_statistics_player_statistics_-_sortable_stats_player_pitch_ratings.csv` | pitching | slow-changing | curated_current_player_pitching or curated_current_team_stats |
| `csv/abl_statistics/abl_statistics_player_statistics_-_sortable_stats_player_pitch_stats_1.csv` | pitching | changes every export | curated_current_player_pitching or curated_current_team_stats |
| `csv/abl_statistics/abl_statistics_player_statistics_-_sortable_stats_player_pitch_stats_2.csv` | pitching | changes every export | curated_current_player_pitching or curated_current_team_stats |
| `csv/abl_statistics/abl_statistics_team_statistics___info_-_sortable_stats_abl_staff.csv` | staff_and_management | changes every export | curated_current_staff |
| `csv/abl_statistics/abl_statistics_team_statistics___info_-_sortable_stats_batting_stats.csv` | batting | changes every export | curated_current_player_batting or curated_current_team_stats |
| `csv/abl_statistics/abl_statistics_team_statistics___info_-_sortable_stats_batting_xtra.csv` | batting | changes every export | curated_current_player_batting or curated_current_team_stats |
| `csv/abl_statistics/abl_statistics_team_statistics___info_-_sortable_stats_c_fielding_1.csv` | fielding_and_defense | changes every export | curated_current_team_stats |
| `csv/abl_statistics/abl_statistics_team_statistics___info_-_sortable_stats_c_fielding_2.csv` | fielding_and_defense | changes every export | curated_current_team_stats |
| `csv/abl_statistics/abl_statistics_team_statistics___info_-_sortable_stats_pitching_1.csv` | pitching | changes every export | curated_current_player_pitching or curated_current_team_stats |
| `csv/abl_statistics/abl_statistics_team_statistics___info_-_sortable_stats_pitching_2.csv` | pitching | changes every export | curated_current_player_pitching or curated_current_team_stats |
| `csv/abl_statistics/abl_statistics_team_statistics___info_-_sortable_stats_team_cur_rec_hist.csv` | teams | changes every export | curated_current_team_context |
| `csv/abl_statistics/abl_statistics_team_statistics___info_-_sortable_stats_team_finan.csv` | financial_and_team_context | changes every export | curated_current_team_context |
| `csv/abl_statistics/abl_statistics_team_statistics___info_-_sortable_stats_team_pers_park.csv` | financial_and_team_context | changes every export | curated_current_team_context |

## Generated outputs: never source of truth

103 generated files are registered as `derived_output`. They may be rebuilt, audited, or rendered, but they cannot prove standings, results, or the latest game date.

## Stale or risky sources

123 sources require caution because they are derived, historical, unknown, or lack independent current-date authority. Consult the registry row before use.

## Recommended curated current tables

- `curated_current_games or curated_current_schedule`
- `curated_current_player_batting or curated_current_team_stats`
- `curated_current_player_batting; curated_current_player_pitching; curated_current_staff`
- `curated_current_player_indicative`
- `curated_current_player_pitching or curated_current_team_stats`
- `curated_current_player_ratings`
- `curated_current_staff`
- `curated_current_standings`
- `curated_current_team_context`
- `curated_current_team_stats`
- `curated_historical_context`
- `curated_reference`
