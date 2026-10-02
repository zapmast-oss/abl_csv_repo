# Sortable Schema Adaptation Plan — 1981 as of 1981-07-19

Official sources remain faithful to the capture. Compatibility is generated downstream; fake source columns are prohibited.

## Affected consumers

| Script | Source | Removed expectations | Required change | Limited/disabled enrichment |
|---|---|---|---|---|
| `csv/abl_scripts/build_star_schema.py` | `abl_staff.csv` | GM_ID, MA_ID, BN_ID, PC_ID, HC_ID, SC_ID, TT_ID, OWN_ID, 1BC_ID, 3BC_ID | Use attach_coach_ids against raw coaches data; write IDs only to dim_team_staff/compatibility output; remove source-file rewrite. | Disable ID-dependent staff linkage only for unmatched coach names. |
| `csv/abl_scripts/build_star_schema.py` | `batting_stats.csv` | CS, SB%, WAR, BatR, wSB, UBR, BsR | Allow narrower base table; take WAR from batting_xtra; source other fields from validated raw/curated tables or expose null. | Baserunning value and caught-stealing enrichment limited until replacement source is curated. |
| `csv/abl_scripts/z_abl_basepath_pressure.py` | `batting_stats.csv fallback` | CS, UBR | Prefer abl_team_bat_baserunning.csv or a curated raw aggregation; fail/disable the UBR component when unavailable instead of inventing zero. | Disable UBR/BsR-based basepath pressure component when no compatible source exists. |

## Compatibility-view strategy

- team_staff_compat: retain 13 captured fields and left-join coach IDs from raw coaches; null unmatched IDs.
- team_batting_compat: retain 13 captured fields, join WAR from batting_xtra, join validated CS/SB% if available, and set BatR/wSB/UBR/BsR null with availability flags.

Compatibility views belong under `csv/out/control/` during transition or `csv/out/curated/` later. They must include availability/provenance flags.
