# Story Engine Source Manifest — 1981_07_20_asof_1981-07-19

Entries: **25**

| Used | Status | Role | Source | Reason |
|---|---|---|---|---|
| yes | `promoted_enrichment` | `enrichment` | `csv/abl_statistics/abl_statistics_team_statistics___info_-_sortable_stats_team_finan.csv` | Used by one or more evidence records. |
| yes | `active_current` | `current_state_fact` | `csv/ootp_csv/divisions.csv` | Used by one or more evidence records. |
| yes | `active_current` | `current_state_fact` | `csv/ootp_csv/game_logs.csv` | Used by one or more evidence records. |
| yes | `active_current` | `current_state_fact` | `csv/ootp_csv/games.csv` | Used by one or more evidence records. |
| yes | `active_current` | `current_state_fact` | `csv/ootp_csv/games_score.csv` | Used by one or more evidence records. |
| yes | `active_current` | `current_state_fact` | `csv/ootp_csv/players.csv` | Used by one or more evidence records. |
| yes | `active_current` | `current_state_fact` | `csv/ootp_csv/players_career_batting_stats.csv` | Used by one or more evidence records. |
| yes | `active_current` | `current_state_fact` | `csv/ootp_csv/players_career_pitching_stats.csv` | Used by one or more evidence records. |
| yes | `active_current` | `current_state_fact` | `csv/ootp_csv/sub_leagues.csv` | Used by one or more evidence records. |
| yes | `active_current` | `current_state_fact` | `csv/ootp_csv/teams.csv` | Used by one or more evidence records. |
| yes | `compatible_historical` | `historical_context` | `csv/out/almanac/1980/league_champions_1980_league200.csv` | Used by one or more evidence records. |
| no | `excluded_stale_snapshot` | `none` | `csv/out/star_schema/monday_1981_standings_by_division.csv` | 32-game standings snapshot |
| no | `excluded_stale_snapshot` | `none` | `csv/out/star_schema/fact_team_reporting_1981_weekly_change.csv` | 32-game weekly snapshot |
| no | `excluded_stale_snapshot` | `none` | `csv/out/star_schema/fact_player_batting.csv` | preserved early player snapshot |
| no | `excluded_stale_snapshot` | `none` | `csv/out/star_schema/fact_player_pitching.csv` | preserved early player snapshot |
| no | `excluded_stale_snapshot` | `none` | `csv/out/csv_out/z_ABL_Manager_Tendencies.csv` | 62-game tendency report |
| no | `excluded_stale_snapshot` | `none` | `csv/out/csv_out/z_ABL_Division_Leverage.csv` | 62-game division report |
| no | `excluded_stale_snapshot` | `none` | `csv/out/csv_out/z_ABL_Rotation_Stability.csv` | 62-game rotation report |
| no | `disabled_unavailable` | `disabled_signal` | `signal:manager_tendency` | Staff names available, but no cutoff-compatible tactical tendency source. |
| no | `disabled_unavailable` | `disabled_signal` | `signal:staff_id_complete_linkage` | Removed staff ID fields; unresolved coach-name matches must remain null. |
| no | `disabled_unavailable` | `disabled_signal` | `signal:caught_stealing_rate` | CS and SB% unavailable in accepted team batting schema without another validated source. |
| no | `disabled_unavailable` | `disabled_signal` | `signal:BatR` | Removed and unavailable. |
| no | `disabled_unavailable` | `disabled_signal` | `signal:wSB` | Removed and unavailable. |
| no | `disabled_unavailable` | `disabled_signal` | `signal:UBR` | Removed and unavailable. |
| no | `disabled_unavailable` | `disabled_signal` | `signal:BsR` | Removed and unavailable. |
