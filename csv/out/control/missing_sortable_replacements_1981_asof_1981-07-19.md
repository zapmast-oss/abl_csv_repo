# Missing Sortable Replacements — 1981 as of 1981-07-19

Missing governed sources: **10**

These files remain in the official source set but have no counterpart in the player-statistics staging capture.

| Priority | OOTP family | Existing file | Grain | Required for current enrichment | Why it matters |
|---|---|---|---|---|---|
| P0 - critical | team stats | `abl_statistics_team_statistics___info_-_sortable_stats_batting_stats.csv` | one row per team-season/snapshot | yes | Provides the 24-team offensive totals needed to reconcile and enrich team identity. |
| P0 - critical | team stats | `abl_statistics_team_statistics___info_-_sortable_stats_batting_xtra.csv` | one row per team-season/snapshot | yes | Adds advanced team batting/baserunning denominators not guaranteed by the player-only capture. |
| P0 - critical | team stats | `abl_statistics_team_statistics___info_-_sortable_stats_c_fielding_1.csv` | table-specific; inspect keys | yes | Provides team catcher-defense and running-control evidence for defensive context. |
| P0 - critical | team stats | `abl_statistics_team_statistics___info_-_sortable_stats_c_fielding_2.csv` | table-specific; inspect keys | yes | Provides team catcher-defense and running-control evidence for defensive context. |
| P0 - critical | team stats | `abl_statistics_team_statistics___info_-_sortable_stats_pitching_1.csv` | one row per team-season/snapshot | yes | Provides core 24-team pitching totals and workload needed for team run-prevention enrichment. |
| P0 - critical | team stats | `abl_statistics_team_statistics___info_-_sortable_stats_pitching_2.csv` | one row per team-season/snapshot | yes | Provides advanced 24-team pitching rates and components for validated run-prevention profiles. |
| P1 - high | staff | `abl_statistics_team_statistics___info_-_sortable_stats_abl_staff.csv` | table-specific; inspect keys | yes | Supplies the team-to-manager/staff layer required for management context and later manager signals. |
| P1 - high | team context | `abl_statistics_team_statistics___info_-_sortable_stats_team_cur_rec_hist.csv` | table-specific; inspect keys | no - validation/context only | Provides a supplemental OOTP team-record/history report for reconciliation against raw-game standings; it cannot establish the cutoff. |
| P2 - context | financials | `abl_statistics_team_statistics___info_-_sortable_stats_team_finan.csv` | table-specific; inspect keys | no - context only | Provides payroll, budget, attendance, and financial context for organization and fan stories. |
| P2 - context | park/personality | `abl_statistics_team_statistics___info_-_sortable_stats_team_pers_park.csv` | table-specific; inspect keys | no - context only | Provides park and team personality context unavailable in a player-statistics-only capture. |

A separate team/staff/context staging capture is required. Absence here does not authorize retirement or deletion.
