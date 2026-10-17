# Team Sortable-Stats Reconciliation — 1981 as of 1981-07-19

- Existing sortable files inspected: **20**
- Team-staging files inspected: **17**
- Exact filename/schema matches: **8**
- Exact filename/schema changes: **2**
- Likely renamed matches: **0**
- Previously missing sources now safely replaced: **8**
- Existing sources still lacking safe replacements: **2**
- New-source candidates: **0**
- Duplicates/overlaps: **7**
- Unknown/manual-review staging files: **2**
- Staff/manager reports still unresolved: **YES**
- Promotion safe: **NO**

## Previously missing sources now safely replaced

- `abl_statistics_team_statistics___info_-_sortable_stats_batting_xtra.csv`
- `abl_statistics_team_statistics___info_-_sortable_stats_c_fielding_1.csv`
- `abl_statistics_team_statistics___info_-_sortable_stats_c_fielding_2.csv`
- `abl_statistics_team_statistics___info_-_sortable_stats_pitching_1.csv`
- `abl_statistics_team_statistics___info_-_sortable_stats_pitching_2.csv`
- `abl_statistics_team_statistics___info_-_sortable_stats_team_cur_rec_hist.csv`
- `abl_statistics_team_statistics___info_-_sortable_stats_team_finan.csv`
- `abl_statistics_team_statistics___info_-_sortable_stats_team_pers_park.csv`

## Existing sources still lacking safe replacements

- `abl_statistics_team_statistics___info_-_sortable_stats_abl_staff.csv`
- `abl_statistics_team_statistics___info_-_sortable_stats_batting_stats.csv`

## Staging files with no exact existing match

| Staging | Existing match | Type | Confidence | Recommendation | Warning |
|---|---|---|---:|---|---|
| `abl_statistics_team_statistics___info_-_sortable_stats_default.csv` | `abl_statistics_team_statistics___info_-_sortable_stats_team_cur_rec_hist.csv` | `OVERLAPPING_VIEW` | 57.2 | `HOLD_DUPLICATE_OR_OVERLAP` | do not register a subset view without a distinct curated consumer |
| `abl_statistics_team_statistics___info_-_sortable_stats_fielding_stats.csv` | `abl_statistics_team_statistics___info_-_sortable_stats_c_fielding_1.csv` | `OVERLAPPING_VIEW` | 61.3 | `HOLD_DUPLICATE_OR_OVERLAP` | do not register a subset view without a distinct curated consumer |
| `abl_statistics_team_statistics___info_-_sortable_stats_financials.csv` | `abl_statistics_team_statistics___info_-_sortable_stats_team_finan.csv` | `OVERLAPPING_VIEW` | 70.6 | `HOLD_DUPLICATE_OR_OVERLAP` | do not register a subset view without a distinct curated consumer |
| `abl_statistics_team_statistics___info_-_sortable_stats_history.csv` | `abl_statistics_team_statistics___info_-_sortable_stats_team_cur_rec_hist.csv` | `OVERLAPPING_VIEW` | 58.6 | `HOLD_DUPLICATE_OR_OVERLAP` | do not register a subset view without a distinct curated consumer |
| `abl_statistics_team_statistics___info_-_sortable_stats_park_info.csv` | `abl_statistics_team_statistics___info_-_sortable_stats_team_pers_park.csv` | `OVERLAPPING_VIEW` | 57.8 | `HOLD_DUPLICATE_OR_OVERLAP` | do not register a subset view without a distinct curated consumer |
| `abl_statistics_team_statistics___info_-_sortable_stats_pitching_stats.csv` | `abl_statistics_team_statistics___info_-_sortable_stats_pitching_1.csv` | `OVERLAPPING_VIEW` | 55.9 | `HOLD_DUPLICATE_OR_OVERLAP` | do not register a subset view without a distinct curated consumer |
| `abl_statistics_team_statistics___info_-_sortable_stats_staff.csv` | `abl_statistics_team_statistics___info_-_sortable_stats_abl_staff.csv` | `OVERLAPPING_VIEW` | 72.8 | `HOLD_DUPLICATE_OR_OVERLAP` | do not register a subset view without a distinct curated consumer |

## Likely new source candidates

None.

## Duplicates or overlaps

| Staging | Existing match | Type | Confidence | Recommendation | Warning |
|---|---|---|---:|---|---|
| `abl_statistics_team_statistics___info_-_sortable_stats_default.csv` | `abl_statistics_team_statistics___info_-_sortable_stats_team_cur_rec_hist.csv` | `OVERLAPPING_VIEW` | 57.2 | `HOLD_DUPLICATE_OR_OVERLAP` | do not register a subset view without a distinct curated consumer |
| `abl_statistics_team_statistics___info_-_sortable_stats_fielding_stats.csv` | `abl_statistics_team_statistics___info_-_sortable_stats_c_fielding_1.csv` | `OVERLAPPING_VIEW` | 61.3 | `HOLD_DUPLICATE_OR_OVERLAP` | do not register a subset view without a distinct curated consumer |
| `abl_statistics_team_statistics___info_-_sortable_stats_financials.csv` | `abl_statistics_team_statistics___info_-_sortable_stats_team_finan.csv` | `OVERLAPPING_VIEW` | 70.6 | `HOLD_DUPLICATE_OR_OVERLAP` | do not register a subset view without a distinct curated consumer |
| `abl_statistics_team_statistics___info_-_sortable_stats_history.csv` | `abl_statistics_team_statistics___info_-_sortable_stats_team_cur_rec_hist.csv` | `OVERLAPPING_VIEW` | 58.6 | `HOLD_DUPLICATE_OR_OVERLAP` | do not register a subset view without a distinct curated consumer |
| `abl_statistics_team_statistics___info_-_sortable_stats_park_info.csv` | `abl_statistics_team_statistics___info_-_sortable_stats_team_pers_park.csv` | `OVERLAPPING_VIEW` | 57.8 | `HOLD_DUPLICATE_OR_OVERLAP` | do not register a subset view without a distinct curated consumer |
| `abl_statistics_team_statistics___info_-_sortable_stats_pitching_stats.csv` | `abl_statistics_team_statistics___info_-_sortable_stats_pitching_1.csv` | `OVERLAPPING_VIEW` | 55.9 | `HOLD_DUPLICATE_OR_OVERLAP` | do not register a subset view without a distinct curated consumer |
| `abl_statistics_team_statistics___info_-_sortable_stats_staff.csv` | `abl_statistics_team_statistics___info_-_sortable_stats_abl_staff.csv` | `OVERLAPPING_VIEW` | 72.8 | `HOLD_DUPLICATE_OR_OVERLAP` | do not register a subset view without a distinct curated consumer |

## Unknown or manual review

| Staging | Existing match | Type | Confidence | Recommendation | Warning |
|---|---|---|---:|---|---|
| `abl_statistics_team_statistics___info_-_sortable_stats_abl_staff.csv` | `abl_statistics_team_statistics___info_-_sortable_stats_abl_staff.csv` | `EXACT_FILENAME_SCHEMA_CHANGED` | 80.4 | `HOLD_UNKNOWN_REVIEW` | schema contraction/change; existing-only columns: GM_ID|MA_ID|BN_ID|PC_ID|HC_ID|SC_ID|TT_ID|OWN_ID|1BC_ID|3BC_ID |
| `abl_statistics_team_statistics___info_-_sortable_stats_batting_stats.csv` | `abl_statistics_team_statistics___info_-_sortable_stats_batting_stats.csv` | `EXACT_FILENAME_SCHEMA_CHANGED` | 85.4 | `HOLD_UNKNOWN_REVIEW` | schema contraction/change; existing-only columns: CS|WAR|BatR|wSB|UBR|BsR |

## Recommended promotion plan

1. Do not promote while `abl_staff.csv` and `batting_stats.csv` fail schema compatibility.
2. Treat the eight exact schema matches as replacement candidates only after combined dry-run validation.
3. Hold the seven focused team views as overlaps unless column review proves unique governed value.
4. Re-export the full ABL staff report and full team batting-stats report with the governed columns.
5. Reconcile those corrected files, then prepare a combined dry-run promotion manifest.

## Warnings

- Physical filename presence is not a safe replacement when columns were dropped.
- The staff/manager family is still unresolved because the staged staff files are schema contractions.
- No files were promoted, moved, renamed, replaced, or deleted.
