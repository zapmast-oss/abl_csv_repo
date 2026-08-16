# Recent Codex Outputs

**Evidence window:** eight commits ending at `dbe9355`, dated 2026-07-04 through 2026-07-05; repository file modification times; clean pre-pack worktree. Git identifies author `zapmast-oss`; “Codex” attribution is therefore an inference from the requested task context, not provable from git author metadata alone.

Codex appears to have produced a July 20 editorial package, a Chicago–Dallas direct-opportunity package, and the supporting StatsPlus Feed 3 preview/promotion pipeline. The most important human-facing outputs are the stakes-first segment, framing notes, Ballpark Feed intro/setup, baseline EB package, and claim-level evidence packet. Large JSON/CSV enrichments and raw feeds should be summarized rather than uploaded.

## Recent deliverables and changes

| File | Status | Why it matters | Safe to upload? | Notes |
|---|---|---|---|---|
| `csv/out/story/scripts/baseball_observer_segment_stakes_first_1981_07_20_asof_1981-07-19.md` | Added `292eea2` | Latest “Pressure Map” EB voice pass | Yes | StatsPlus-enriched; official order unchanged |
| `csv/out/story/production/july20_stakes_first_framing_notes_1981_07_20_asof_1981-07-19.md` | Added `292eea2` | Explains stakes-first framing and safeguards | Yes | Current human-readable production notes |
| `csv/out/story/scripts/chicago_dallas_ballpark_feed_intro_1981_07_20_asof_1981-07-19.md` | Added `41dacd8` | Short pregame read | Yes | Explicitly excludes July 20 results |
| `csv/out/story/production/chicago_dallas_ballpark_feed_setup_1981_07_20_asof_1981-07-19.md` | Added `41dacd8` | Series arithmetic and talking points | Yes | Scheduled starters unavailable |
| `csv/out/story/investigations/chicago_dallas_series_evidence_packet_1981_07_20_asof_1981-07-19.md` | Added/modified `41dacd8` | Claim-level proof for matchup framing | Yes | Prefer Markdown over large JSON twin |
| `csv/out/story/editorial/observer_story_slate_addendum_chicago_dallas_1981_07_20_asof_1981-07-19.md` | Added `41dacd8` | Adds matchup to editorial slate | Optional | Useful when editing this series |
| `csv/abl_scripts/z_abl_chicago_dallas_investigation_packet_1981.py` | Added `41dacd8` | Generates investigation package | Usually no | Upload only for code-specific help |
| `csv/abl_scripts/z_abl_story_engine_with_feed3_preview_runner.py` | Added/modified `5b8bdbb`, `5885eb7` | Parameterized safe preview wrapper | Usually no | Official artifacts protected by hashes |
| `csv/abl_scripts/z_abl_statsplus_feed3_evidence_adapter.py` | Added/refactored/modified | Creates typed StatsPlus evidence | Usually no | Technical source, not editorial context |
| `csv/abl_scripts/z_abl_statsplus_promote_current.py` | Added `88bf4b9` | Promotes staged StatsPlus sources | Usually no | Current feed governance implementation |
| `csv/abl_scripts/z_abl_statsplus_story_enrichment_preview_1981.py` | Added `88bf4b9` | Builds enrichment views/overlay | Usually no | Preview only |
| `csv/out/story/enrichment/story_evidence_with_feed3_preview_1981_07_20_asof_1981-07-19.md` | Added/modified | 68-record combined preview | Optional | Official evidence remains unchanged |
| `csv/out/control/feed3_parameterized_adapter_validation_1981_07_20_asof_1981-07-19.md` | Added `5885eb7` | PASS proof for safe preview | Yes | 15/15 candidates enriched; no rank change |
| `csv/statsplus/current/*.csv` | 25 files added `88bf4b9` | Promoted model/stat enrichment | Selectively | Small individually, but upload only those needed |
| `docs/ABL_FEED3_EVIDENCE_PREVIEW_RUNBOOK.md` | Added/modified | Documents guarded preview | Yes | Strong technical context |
| `csv/out/story/production/~$ly20_stakes_first_framing_notes_1981_07_20_asof_1981-07-19.md` | Added `b91a33f`, deleted `dbe9355` | Obsolete Word lock/temp file | No | Deletion is intentional; do not recreate/upload |
| `csv/abl_scripts/__pycache__/z_abl_chicago_dallas_investigation_packet_1981.cpython-312.pyc.1727378662304` | Added `41dacd8` | Python bytecode cache | No | Binary/generated cache; committed accidentally or operationally |

No deleted substantive deliverable was found in the eight-commit window. The only deletion was the `~$` temporary lock file.

## Upload judgment

Upload the human-readable current scripts, production notes, evidence packet, current preflight, and this source pack. Do not upload bytecode, raw OOTP dumps, archive copies, large JSON twins, or all StatsPlus tables. For a focused statistics question, add only the relevant current CSV and state that it is enrichment rather than official standings proof.

