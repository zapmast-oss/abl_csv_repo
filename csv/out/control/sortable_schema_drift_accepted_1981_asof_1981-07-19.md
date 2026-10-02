# Accepted Sortable Schema Drift — 1981 as of 1981-07-19

The current 13-column captures are the forward source standard. Removed fields are not fabricated in source files.

## `abl_statistics_team_statistics___info_-_sortable_stats_abl_staff.csv`

- Old columns: 23
- New columns: 13
- Retained: `ID|Team Name|Abbr|GM|MA|BN|PC|HC|SC|TT|OWN|1BC|3BC`
- Removed: `GM_ID|MA_ID|BN_ID|PC_ID|HC_ID|SC_ID|TT_ID|OWN_ID|1BC_ID|3BC_ID`
- Added: `none`
- Downstream impact: build_star_schema.py can regenerate coach IDs from raw coaches data, but must stop rewriting the source file. Consumers that read *_ID directly need a generated compatibility view.
- Story-engine impact: Manager/staff identity remains available by name; ID-dependent manager enrichment is limited when a coach-name lookup fails.
- Field disposition: GM_ID: replaced by coaches.csv/team staff lookup; compatibility view may expose null when no ID match; MA_ID: replaced by coaches.csv/team staff lookup; compatibility view may expose null when no ID match; BN_ID: replaced by coaches.csv/team staff lookup; compatibility view may expose null when no ID match; PC_ID: replaced by coaches.csv/team staff lookup; compatibility view may expose null when no ID match; HC_ID: replaced by coaches.csv/team staff lookup; compatibility view may expose null when no ID match; SC_ID: replaced by coaches.csv/team staff lookup; compatibility view may expose null when no ID match; TT_ID: replaced by coaches.csv/team staff lookup; compatibility view may expose null when no ID match; OWN_ID: replaced by coaches.csv/team staff lookup; compatibility view may expose null when no ID match; 1BC_ID: replaced by coaches.csv/team staff lookup; compatibility view may expose null when no ID match; 3BC_ID: replaced by coaches.csv/team staff lookup; compatibility view may expose null when no ID match

## `abl_statistics_team_statistics___info_-_sortable_stats_batting_stats.csv`

- Old columns: 20
- New columns: 13
- Retained: `Team Name|G|2B|3B|HR|R|BB|SO|SB|AVG|OBP|SLG|OPS`
- Removed: `CS|SB%|WAR|BatR|wSB|UBR|BsR`
- Added: `none`
- Downstream impact: build_star_schema.py will produce a narrower team batting fact. z_abl_basepath_pressure.py cannot use this file for UBR and loses CS/SB% unless another validated source supplies them.
- Story-engine impact: Core batting totals remain available. Baserunning-value and caught-stealing enrichments must use another validated source or be disabled/null.
- Field disposition: CS: replaced by raw/team or aggregated player baserunning source when validated; otherwise unavailable; SB%: replaced by raw/team or aggregated player baserunning source when validated; otherwise unavailable; WAR: replaced by batting_xtra.csv WAR; BatR: marked unavailable; null in compatibility view; disable dependent enrichment; wSB: marked unavailable; null in compatibility view; disable dependent enrichment; UBR: marked unavailable; null in compatibility view; disable dependent enrichment; BsR: marked unavailable; null in compatibility view; disable dependent enrichment
