# StatsPlus Promotion Dry Run

**No files were copied or promoted.**

| # | Family | Action | Rows | Cols | Authority | Risk |
|---|---|---|---|---|---|---|
| 1 | owner info | PROMOTE_ACCEPT_RESHAPED_SCHEMA | 24 | 17 | STATSPLUS_SPECIFIC_AFTER_PROMOTION | Promote retained traits; disable mood/objective/specific-goal signals. |
| 2 | front office/coaches | PROMOTE_AS_STATSPLUS_CURRENT | 240 | 17 | CROSSCHECK_ONLY | Promote only with field-level authority metadata; no story-engine enablement yet. |
| 3 | financials | PROMOTE_AS_STATSPLUS_CURRENT | 25 | 11 | STATSPLUS_SPECIFIC_AFTER_PROMOTION | Promote only with field-level authority metadata; no story-engine enablement yet. |
| 4 | historical fan interest | PROMOTE_ACCEPT_SCHEMA_DRIFT | 24 | 12 | STATSPLUS_SPECIFIC_AFTER_PROMOTION | Rank removed; team/year interest values remain. |
| 5 | fan data | PROMOTE_AS_STATSPLUS_CURRENT | 25 | 11 | STATSPLUS_SPECIFIC_AFTER_PROMOTION | Promote only with field-level authority metadata; no story-engine enablement yet. |
| 6 | league standings | INTENTIONALLY_EXCLUDED |  |  | RAW_OOTP_GOVERNS | Raw OOTP governs current standings. |
| 7 | playoff odds by division | PROMOTE_REORDERED_OVERLAP | 24 | 13 | STATSPLUS_SPECIFIC_AFTER_PROMOTION | Same row multiset as table 8 in a different order; model values agree with raw records. |
| 8 | playoff odds by league | PROMOTE_REORDERED_OVERLAP | 24 | 13 | STATSPLUS_SPECIFIC_AFTER_PROMOTION | Same row multiset as table 7 in a different order; preserve report identity and avoid double counting. |
| 9 | BaseRuns | PROMOTE_AS_STATSPLUS_CURRENT | 24 | 22 | STATSPLUS_SPECIFIC_AFTER_PROMOTION | Promote only with field-level authority metadata; no story-engine enablement yet. |
| 10 | ELO ratings | PROMOTE_AS_STATSPLUS_CURRENT | 24 | 11 | STATSPLUS_SPECIFIC_AFTER_PROMOTION | Promote only with field-level authority metadata; no story-engine enablement yet. |
| 11 | team WAR | PROMOTE_AS_STATSPLUS_CURRENT | 24 | 8 | STATSPLUS_SPECIFIC_AFTER_PROMOTION | Promote only with field-level authority metadata; no story-engine enablement yet. |
| 12 | injury summary | PROMOTE_AS_STATSPLUS_CURRENT | 24 | 5 | STATSPLUS_SPECIFIC_AFTER_PROMOTION | Promote only with field-level authority metadata; no story-engine enablement yet. |
| 13 | team batting by division | PROMOTE_AS_STATSPLUS_CURRENT | 25 | 23 | CROSSCHECK_ONLY | Promote only with field-level authority metadata; no story-engine enablement yet. |
| 14 | team batting by league | PROMOTE_AS_STATSPLUS_CURRENT | 25 | 23 | CROSSCHECK_ONLY | Promote only with field-level authority metadata; no story-engine enablement yet. |
| 15 | team pitching by division | PROMOTE_ACCEPT_ADDED_DETAIL | 25 | 23 | CROSSCHECK_ONLY | Adds IP, SV, and BS; overlapping pitching fields remain crosscheck-only. |
| 16 | team pitching by league | PROMOTE_ACCEPT_ADDED_DETAIL | 25 | 23 | CROSSCHECK_ONLY | Adds IP, SV, and BS; overlapping pitching fields remain crosscheck-only. |
| 17 | team fielding by division | PROMOTE_AS_STATSPLUS_CURRENT | 25 | 21 | CROSSCHECK_ONLY | Promote only with field-level authority metadata; no story-engine enablement yet. |
| 18 | team fielding by league | PROMOTE_AS_STATSPLUS_CURRENT | 25 | 21 | CROSSCHECK_ONLY | Promote only with field-level authority metadata; no story-engine enablement yet. |
| 19 | team baserunning | PROMOTE_ACCEPT_ADDED_DETAIL | 24 | 11 | STATSPLUS_SPECIFIC_AFTER_PROMOTION | Adds team UBR; use only as a labeled StatsPlus-specific enrichment after promotion. |
| 20 | player batting | PROMOTE_AS_STATSPLUS_CURRENT | 100 | 20 | CROSSCHECK_ONLY | Promote only with field-level authority metadata; no story-engine enablement yet. |
| 21 | player pitching | PROMOTE_ACCEPT_ADDED_DETAIL | 91 | 23 | CROSSCHECK_ONLY | Adds rWAR; label as StatsPlus rWAR and do not overwrite WAR or raw proof. |
| 22 | player baserunning | PROMOTE_AS_STATSPLUS_CURRENT | 100 | 9 | STATSPLUS_SPECIFIC_AFTER_PROMOTION | Promote only with field-level authority metadata; no story-engine enablement yet. |
| 23 | player fielding | PROMOTE_AS_STATSPLUS_CURRENT | 100 | 15 | CROSSCHECK_ONLY | Promote only with field-level authority metadata; no story-engine enablement yet. |
| 24 | team age | PROMOTE_ACCEPT_ADDED_DETAIL | 24 | 20 | STATSPLUS_SPECIFIC_AFTER_PROMOTION | Adds short-season and rookie-level age splits. |
| 25 | best batting game | PROMOTE_AS_STATSPLUS_CURRENT | 30 | 13 | STATSPLUS_SPECIFIC_AFTER_PROMOTION | Promote only with field-level authority metadata; no story-engine enablement yet. |
| 26 | best pitching game | PROMOTE_AS_STATSPLUS_CURRENT | 30 | 13 | STATSPLUS_SPECIFIC_AFTER_PROMOTION | Promote only with field-level authority metadata; no story-engine enablement yet. |
| 27 | Grand Tournament of Champions | INTENTIONALLY_EXCLUDED |  |  | HISTORICAL_CONTEXT | GTOC history does not change until postseason. |

## Counts

| Action | Tables |
|---|---|
| INTENTIONALLY_EXCLUDED | 2 |
| PROMOTE_ACCEPT_ADDED_DETAIL | 5 |
| PROMOTE_ACCEPT_RESHAPED_SCHEMA | 1 |
| PROMOTE_ACCEPT_SCHEMA_DRIFT | 1 |
| PROMOTE_AS_STATSPLUS_CURRENT | 16 |
| PROMOTE_REORDERED_OVERLAP | 2 |

Proposed target: `csv/statsplus/current/`. Source filenames are preserved. Promotion requires separate authorization.
