# Final Sortable Capture Status — 1981 as of 1981-07-19

- Exact/safe replacements: **18 of 20**
- Accepted with schema drift: **2 of 20**
- Coverage: **20 of 20 governed sources**
- Promotion status: **SAFE_WITH_ACCEPTED_SCHEMA_DRIFT**

## Story and enrichment impact

- Staff names remain available; staff-ID enrichment uses raw coach lookup and is limited when names do not resolve.
- Core team batting remains available; CS/SB% and advanced baserunning value fields require another validated source or remain unavailable.
- UBR/BsR/wSB/BatR-dependent signals must be disabled or explicitly marked unavailable until curated replacements exist.
