# ABL Existing Engine Inventory

Inventory date: 2026-07-03. “Missing” means no clear, standardized implementation was found; it does not imply no one-off output exists.

| Functional engine | What exists | Missing | Fragile | Standardize |
|---|---|---|---|---|
| Raw import / OOTP source | 73 OOTP CSV exports, 20 sortable-stat exports, loader/config helpers | export manifest, freshness gate, schema drift report, immutable dated snapshots | external OneDrive fallback in `_tmp_run_all.py`; current files are overwritten | source manifest, as-of timestamp, encoding/schema validation, source-only directories |
| Star schema build | `build_star_schema.py`, SQLite `abl.db`, player/team facts and dimensions | declared table contracts, tests, lineage registry | mixed sources; one build path appears to rewrite a staff source file; season-specific naming | pure source-to-output writes, stable dimensions, primary-key/row-count tests |
| Current team snapshot | current/previous reporting facts, season backbone, reporting view | explicit `snapshot_date` and immutable snapshots | `current`/`prev` state depends on operator sequencing | dated snapshot partition plus current pointer/manifest |
| Weekly change | seed-from-games and weekly-change scripts/table | automatic cutoff discovery and comparable-snapshot validation | stale/missing `prev` can yield plausible but wrong deltas | require two dated snapshots, record cutoffs, reject mismatched seasons |
| Monday packet | standings, power, risers, fallers, show notes, wrapper | single packet manifest and editorial ranking | hard-coded 1981; wrapper does not seed previous snapshot | parameterized season/as-of/week, atomic run directory, validation summary |
| EB / Baseball Observer packet | Monday text pack, JSON player/team/WAR evidence, Act 3 pack, EB historical packs | one hierarchy-led packet contract and citations to evidence rows | manual “copy into ChatGPT” handoff; many disconnected reports | machine-readable packet plus rendered Markdown, provenance on every claim |
| Story candidates | dictionary, week-5 candidates, flashback candidates, many signal reports | unified signal evaluator across hierarchy; week-7 evaluated candidates | only some trigger logic is executable; evidence columns vary by signal | canonical candidate/evidence schemas, confidence and suppression rules |
| Story menu | week-5/week-7 CSV menus, menu build/print scripts, 1972 flashback Markdown menu | ranked cross-level menu with selection state and rationale | generated current menus land under `csv/`; dictionary/menu largely copy definitions | menus under `csv/out/story/menus/`, deterministic ranking, editorial overrides |
| Almanac | complete-looking 1972-1980 HTML sources and 31-34 CSVs per season | unified manifest/validation across all seasons | several `1972`-specific scripts coexist with `*_any`; repeated outputs may drift | one parameterized season runner and schema parity checks |
| Flashback | candidates all seasons, EB briefs/context, 1972 rendered menu, Grand Series summaries | rendered menus/packets consistently for every season; current-story linkage | split between `out/almanac` and `out/eb`; some accidental `csv/csv` outputs | historical event IDs and explicit “echo of current signal” joins |
| Manager/team/player reporting | managers, matchup/prep, identity, player/week/WAR, dozens of analytic reports | common evidence model and coverage matrix | independent thresholds/column naming; text and CSV pairs can diverge | shared loaders, keys, threshold config, structured result before rendering |
| Video outline / text outputs | Week 5 and second-half video outlines, weekly reports, forum post, 50+ text reports, pregame Markdown | repeatable generator consuming selected stories | several products look manually assembled; no candidate-to-outline lineage | outline schema, templates, citations, channel-specific renderers |

## Existing strengths

- Broad analytical coverage already exists; the main need is orchestration and contracts, not rebuilding every metric.
- The current/previous/weekly-change model is a usable foundation for movement stories.
- The almanac provides unusually deep, structured historical context across 1972-1980.
- The dictionary/menu/candidate separation is the correct conceptual start.
- Output refactoring has already concentrated most products under `csv/out/`.

## Cross-cutting gaps

1. No single run manifest connects source as-of dates, producer versions, outputs, and validation.
2. The editorial hierarchy is not encoded as a first-class ranking dimension.
3. Story “trigger logic” mixes prose and executable assumptions; thresholds are not centrally configured.
4. Candidate rows do not share a stable evidence/provenance model.
5. Current and historical engines use parallel but different candidate/menu schemas.
6. There is no clear deduplication policy when multiple metrics describe the same underlying pressure.
7. Broad `_tmp_run_all.py` orchestration is useful for discovery but unsafe as a production master runner because it runs unrelated scripts, retries CLIs, and tolerates skips.

