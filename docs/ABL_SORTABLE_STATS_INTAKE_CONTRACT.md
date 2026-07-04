# ABL Sortable-Stats Intake Contract

## Family contract

All files under `csv/abl_statistics/` are `sortable_stats` / `supplemental_report_extract`. They may support current state only after capture-batch validation and linkage to a raw OOTP batch. None can independently drive current state or prove the latest game date.

Common validation: file/schema/row-count checks; reject mixed capture dates; join names/IDs to raw players/teams; verify the expected 24-team or active-player universe; test numeric parsing and duplicate grain; retain checksum and batch ID.

## File classification

| File suffix | Subject / class | Likely grain | Volatility | Allowed role | Recommended validation additions | Curated target |
|---|---|---|---|---|---|---|
| `player_bat_ratings.csv` | batting ratings; player, ratings-level | player | slow-changing | support/enrichment | unique player/name-team; rating ranges | `curated_current_player_ratings` |
| `player_bat_stats.csv` | batting statistics; player-level | player-season | changes every export | support current | recover/verify header; PA/AB/H reconciliation | `curated_current_player_batting` |
| `player_field_ratings.csv` | fielding ratings; player, ratings-level | player-position | slow-changing | support/enrichment | position uniqueness; rating ranges | `curated_current_player_ratings` |
| `player_field_stats.csv` | fielding statistics; player-level | player-position-season | changes every export | support current | innings/chances/errors reconciliation | `curated_current_team_stats` |
| `player_indicative_1.csv` | indicative attributes; player-level | player | slow-changing | context/enrichment | join coverage; categorical domain checks | `curated_current_player_indicative` |
| `player_indicative_2.csv` | indicative attributes; player-level | player | slow-changing | context/enrichment | join coverage; categorical domain checks | `curated_current_player_indicative` |
| `player_misc_info.csv` | player misc/identity; player-level | player | slow-changing | context/enrichment | identity collision and null checks | `curated_current_player_indicative` |
| `player_pitch_ratings.csv` | pitching ratings; player, ratings-level | pitcher | slow-changing | support/enrichment | role and arsenal/rating ranges | `curated_current_player_ratings` |
| `player_pitch_stats_1.csv` | pitching statistics; player-level | pitcher-season | changes every export | support current | IP/outs, BF, ERA-component checks | `curated_current_player_pitching` |
| `player_pitch_stats_2.csv` | advanced pitching statistics; player-level | pitcher-season | changes every export | support current | FIP/rate denominators and workload checks | `curated_current_player_pitching` |
| `abl_staff.csv` | staff/manager; staff-level | team-staff snapshot | slow-changing | context; conditional support | exactly 24 teams; manager IDs/names; batch coverage | `curated_current_staff` |
| `batting_stats.csv` | team batting; team-level | team-season snapshot | changes every export | support current | exactly 24 teams; G matches raw batch; totals reconcile | `curated_current_team_stats` |
| `batting_xtra.csv` | team advanced batting; team-level | team-season snapshot | changes every export | support current | exactly 24 teams; denominator/rate checks | `curated_current_team_stats` |
| `c_fielding_1.csv` | catcher fielding; team-level | team-season catcher summary | changes every export | support current | attempts, CS/SB/PB reconciliation | `curated_current_team_stats` |
| `c_fielding_2.csv` | catcher advanced fielding; team-level | team-season catcher summary | changes every export | support current | rate denominators and 24-team coverage | `curated_current_team_stats` |
| `pitching_1.csv` | team pitching; team-level | team-season snapshot | changes every export | support current | G/IP/outs and run-component reconciliation | `curated_current_team_stats` |
| `pitching_2.csv` | team advanced pitching; team-level | team-season snapshot | changes every export | support current | rate denominators; 24-team coverage | `curated_current_team_stats` |
| `team_cur_rec_hist.csv` | current record/history; team-level | team-season snapshot | changes every export | enrichment only for current | compare G/W/L to raw standings; never establish cutoff | `curated_current_team_context` |
| `team_finan.csv` | finances; team, financial/context-level | team-season snapshot | changes every export | context/enrichment | currency parsing; 24-team coverage; batch timestamp | `curated_current_team_context` |
| `team_pers_park.csv` | personality/park; team, context-level | team | slow-changing | context/enrichment | park/team joins; categorical and park-factor ranges | `curated_current_team_context` |

## Intake disposition

- **Drive current state:** no sortable-stat file.
- **Support current state:** statistical, ratings, indicative, staff, finance, and context files after batch linkage and validation.
- **Context only:** personality, park, financial, indicative, and misc fields when they do not represent game results.
- **Quarantine:** files with missing/shifted headers, unexpected universe counts, mixed exports, or game counts that disagree with the raw batch.

