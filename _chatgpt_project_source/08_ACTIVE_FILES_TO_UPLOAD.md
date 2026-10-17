# Active Files to Upload

Upload Markdown where possible. The Project should receive the curated explanation plus a small number of current, human-readable evidence and deliverable files—not the repository itself.

| Priority | File | Upload directly? | Reason | Notes |
|---|---|---|---|---|
| Must upload | `_chatgpt_project_source/00_START_HERE.md` through `10_PROJECT_SOURCE_MANIFEST.md` | Yes | Curated project identity, state, data, voice, workflow, risks | Primary source pack |
| Must upload | `csv/out/story/scripts/baseball_observer_segment_stakes_first_1981_07_20_asof_1981-07-19.md` | Yes | Latest stakes-first EB voice pass | No July 20 results |
| Must upload | `csv/out/story/production/july20_stakes_first_framing_notes_1981_07_20_asof_1981-07-19.md` | Yes | Explains editorial frame and cautions | Pair with script |
| Should upload | `csv/out/story/production/eb_production_package_1981_07_20_asof_1981-07-19.md` | Yes | Full baseline lead/secondary/watch/held package | Official baseline |
| Should upload | `csv/out/story/scripts/baseball_observer_segment_1981_07_20_asof_1981-07-19.md` | Yes | Baseline safe-for-use script | Useful for comparison |
| Should upload | `csv/out/control/current_state_preflight_1981_target_1981-07-19.md` | Yes | Compact authority/cutoff proof | Prevents stale-data mistakes |
| Should upload | `csv/out/story/investigations/chicago_dallas_series_evidence_packet_1981_07_20_asof_1981-07-19.md` | Yes | Claim-level matchup evidence | Add when working Chicago–Dallas |
| Should upload | `csv/out/story/scripts/chicago_dallas_ballpark_feed_intro_1981_07_20_asof_1981-07-19.md` | Yes | Current short-form deliverable | Pair with setup |
| Should upload | `csv/out/story/production/chicago_dallas_ballpark_feed_setup_1981_07_20_asof_1981-07-19.md` | Yes | Arithmetic/history/cautions | Pair with intro |
| Optional | `csv/out/story/manifests/story_engine_run_summary_1981_07_20_asof_1981-07-19.md` | Yes | Counts, top themes, disabled signals | Compact technical context |
| Optional | `csv/out/story/manifests/story_engine_source_manifest_1981_07_20_asof_1981-07-19.md` | Yes | Used/rejected source audit | Helpful for provenance questions |
| Optional | `csv/out/control/feed3_parameterized_adapter_validation_1981_07_20_asof_1981-07-19.md` | Yes | Proves preview safety | Technical/editorial integration tasks |
| Optional | `docs/ABL_FEED3_EVIDENCE_PREVIEW_RUNBOOK.md` | Yes | Exact preview workflow and limits | Technical tasks only |
| Optional | Relevant `csv/statsplus/current/*.csv` | Selectively | Detailed current enrichment | Add one/few tables for a focused stats question |
| Do not upload | `csv/ootp_csv/game_logs.csv` and bulk `csv/ootp_csv/*.csv` | No | Hundreds of MB and excessive detail | Ask Codex to extract a focused, cutoff-safe subset |
| Do not upload | `data_work/abl.db` | No | Binary SQLite database | Export a query/result instead |
| Do not upload | `csv/out/story/enrichment/*.json` large twins | No | Large generated dumps duplicate readable summaries | Summarize or use Markdown counterpart |
| Do not upload | `csv/out/archive/`, `csv/out/almanac/` wholesale | No | Historical bulk can obscure current state | Add a specific season file only when needed |
| Do not upload | Week 5/Week 15 snapshots as current | No | Stale current-state risk | Upload only for explicitly historical/comparison tasks |
| Do not upload | `.git/`, `.env*`, logs, `__pycache__/`, `*.pyc`, temp/lock files | No | Metadata, possible private configuration, noise/binaries | Never include credentials or secrets |

## Files to summarize instead of upload

Summarize the 74 raw OOTP exports, 21 sortable extracts, 25 StatsPlus tables, 111 control artifacts, large combined Feed 3 JSON, and historical almanac. `04_DATA_DICTIONARY.md`, `02_REPO_MAP.md`, and `03_RECENT_CODEX_OUTPUTS.md` provide those summaries.

## Recommended minimum upload set

1. All 11 files in `_chatgpt_project_source/`.
2. `csv/out/story/scripts/baseball_observer_segment_stakes_first_1981_07_20_asof_1981-07-19.md`.
3. `csv/out/story/production/july20_stakes_first_framing_notes_1981_07_20_asof_1981-07-19.md`.
4. `csv/out/control/current_state_preflight_1981_target_1981-07-19.md`.

## Recommended full upload set

Use the minimum set plus the baseline EB package/script, Chicago–Dallas evidence/setup/intro, current run summary/source manifest, Feed 3 adapter validation, and Feed 3 runbook. Add specific StatsPlus CSVs only when the intended Project work needs their detailed columns.

