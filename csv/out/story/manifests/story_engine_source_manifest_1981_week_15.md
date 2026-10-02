# Story Engine Source Manifest — 1981_week_15

- Season: 1981
- As of: 1981-07-12
- Expected coverage: 89 games per team

| Used | Status | Source | Rows | Date range | Games/team | Reason |
|---|---|---|---:|---|---:|---|
| yes | `active_current` | `csv/ootp_csv/games.csv` | 9127 | 1981-04-06 to 1981-07-12 | 89 | Authoritative date-filterable game source; completed ABL games are filtered through the active cutoff. |
| yes | `compatible_static` | `csv/ootp_csv/teams.csv` | 158 | n/a | n/a | Team identity and league/division keys are dimensions, not performance snapshots. |
| yes | `compatible_static` | `csv/ootp_csv/divisions.csv` | 36 | n/a | n/a | Division names and keys are static structural dimensions. |
| yes | `compatible_static` | `csv/ootp_csv/players.csv` | 12549 | n/a | n/a | Player names and team identifiers support current raw statistical rows. |
| yes | `active_current` | `csv/ootp_csv/players_career_batting_stats.csv` | 89192 | n/a | n/a | Raw 1981 cumulative batting totals belong to the coordinated current OOTP export. |
| yes | `active_current` | `csv/ootp_csv/players_career_pitching_stats.csv` | 56555 | n/a | n/a | Raw 1981 cumulative pitching totals belong to the coordinated current OOTP export. |
| yes | `compatible_historical` | `csv/out/almanac/1980/league_champions_1980_league200.csv` | 8 | n/a | n/a | 1980 context is intentionally historical and cannot replace current-season evidence. |
| no | `excluded_wrong_games_count` | `csv/out/star_schema/monday_1981_standings_by_division.csv` | 24 | n/a | 32 | Preserved standings snapshot contains 32 games per team, not the active 89-game state. |
| no | `excluded_wrong_games_count` | `csv/out/star_schema/fact_team_reporting_1981_weekly_change.csv` | 24 | n/a | 32 | Preserved weekly-change rows end at 32 games, not the active 89-game state. |
| no | `excluded_stale_snapshot` | `csv/out/star_schema/fact_player_batting.csv` | 470 | n/a | 32 | Preserved early-season player fact is not used for active Week 15 leaders. |
| no | `excluded_stale_snapshot` | `csv/out/star_schema/fact_player_pitching.csv` | 470 | n/a | n/a | Preserved early-season player fact is not used for active Week 15 leaders. |
| no | `excluded_stale_snapshot` | `csv/story_candidates_1981_week_05.csv` | 24 | n/a | n/a | Preserved Week 5 story candidates are historical artifacts, not active inputs. |
| no | `excluded_stale_snapshot` | `csv/story_menu_1981_week_05.csv` | 24 | n/a | n/a | Preserved Week 5 story menu is not an active input. |
| no | `excluded_stale_snapshot` | `csv/story_menu_1981_week_07.csv` | 24 | n/a | n/a | Preserved early-season story menu is not an active input. |
| no | `disabled_incompatible` | `csv/out/star_schema/fact_manager_scorecard_1981_current.csv` | 24 | n/a | 32 | Manager scorecard is tied to a 32-game standings snapshot; management signals are disabled. |
| no | `disabled_incompatible` | `csv/out/csv_out/z_ABL_Manager_Tendencies.csv` | 24 | n/a | 62 | Manager tendencies stop at 62 games; management signals require the 89-game cutoff. |
| no | `disabled_incompatible` | `csv/out/csv_out/z_ABL_Division_Leverage.csv` | 24 | n/a | 62 | Division leverage stops at 62 games and cannot support active race signals. |
| no | `disabled_incompatible` | `csv/out/csv_out/z_ABL_Rotation_Stability.csv` | 24 | n/a | 62 | Rotation stability stops at 62 games and cannot support active pitcher/team signals. |
