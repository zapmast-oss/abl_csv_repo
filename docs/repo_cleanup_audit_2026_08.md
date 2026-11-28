# ABL Repository Cleanup Audit — August 2026

Audit date: 2026-08-14 (America/Los_Angeles)  
Repository: `C:\sbv_repo\abl_csv_repo`  
Branch: `add-star-schema`  
Audited HEAD: `ef4dbdb357160b58b8f6d34b18413f36b3b5d506`  
Baseline Git state: clean and synchronized with `origin/add-star-schema` before this report was created  
Authorized change in this audit: this report only

## Executive determination

**SAFE TO CLEAN: NO — not as a broad deletion or reorganization.**

It is safe to begin with a small guardrail commit, but it is not yet safe to remove or broadly move the apparent clutter. Generated outputs, production artifacts, raw historical inputs, working state, and documentation snapshots are mixed together. Several scripts depend on current paths, the only default VS Code task invokes an unsafe best-effort runner, and some outputs have no verified canonical twin. Most importantly, a generated report under `csv/ootp_csv/out/` is currently classified by the repository's registry and batch manifest as an authoritative OOTP input.

The canonical OOTP inputs are the **72 immediate `*.csv` children of `csv/ootp_csv/`**. Those files must remain byte-for-byte untouched during cleanup. The two tracked files below `csv/ootp_csv/out/` are generated reports, not OOTP exports, but they must not be removed until registry boundaries, references, hashes, and recovery are fixed.

## Audit method and limits

The live filesystem, ignored files, Git index, Git history, script source, editor task, README, repository maps, control manifests, and important output families were inspected. No pipeline was executed because the current runners write into tracked production paths. Read-only Python was run with `-B` to avoid bytecode creation.

The repository contained 1,948 tracked files. `git status --short --branch` was clean before report creation. There were no ordinary untracked files, but there is a large ignored junction at `data_raw/ootp_html` pointing outside the repository to `C:\SBV_mainC\ootp_all_reports`.

Static reference inspection cannot prove that an external scheduler, manual operator command, ChatGPT Project, forum publication step, or uncommitted local tool never uses a path. Items marked `NEEDS REVIEW` require an owner or a successful parity/recovery test before disposition.

## Current structure established from disk and Git

| Path | Files / size on disk | Git state | Observed role |
|---|---:|---|---|
| `csv/` | 1,874 files / 383.21 MiB | all tracked | Mixed input, source, derived output, documentation, and temporary material |
| `csv/ootp_csv/` | 72 immediate OOTP CSVs plus 2 nested generated reports / 299.95 MiB | tracked | Authoritative promoted OOTP input, polluted by `out/` |
| `csv/abl_scripts/` | 189 Python files plus 24 cache files / 2.73 MiB | tracked | Main processing/reporting engine plus tracked bytecode |
| `csv/abl_statistics/` | 20 sortable CSVs plus `core12_spec.md` / 1.57 MiB | tracked | Supplemental promoted input plus misplaced documentation |
| `csv/statsplus/current/` | 25 CSVs / 0.09 MiB | tracked | Promoted supplemental StatsPlus input |
| `csv/in/almanac_core/` | 585 HTML files / 46.55 MiB | tracked | Historical raw input for seasons 1972–1980 |
| `csv/out/` | 932 files / 32.23 MiB | tracked | Canonical generated/control/story destination, with mixed retention needs |
| `csv/abl_csv/` | 6 CSVs / 0.03 MiB | tracked | Legacy/convenience derived tables used by current scripts |
| `csv/csv/` | 3 files | tracked | Accidental doubled-`csv` output/documentation tree |
| `csv/docs/` | 7 files | tracked | Older editorial schemas, templates, and notes |
| root `out/` | 13 files / 12.77 MiB | tracked | Six legacy reports plus seven intentional Pittsburgh Express publication/media assets |
| root `scripts/` | 5 Python files | tracked | Story-menu drivers and two unrelated utilities |
| root `docs/` | 28 files / 0.23 MiB before this report | tracked | Repository governance, design, generated CLI snapshots, and runbooks |
| `data_work/abl.db` | 28 KiB | tracked | Generated/working SQLite database read by manager/prep tools |
| `logs/` | 3 files | tracked | One failed broad-run log and two input-copy logs with machine-specific paths |
| `_chatgpt_project_source/` | 11 files / 0.05 MiB | tracked | Point-in-time July 7 ChatGPT upload/handoff pack |
| `data_raw/ootp_html` | 34,792 files / 4.942 GiB | ignored junction | External raw HTML/almanac asset store used by scripts |

The ignored junction contains nine full almanac ZIPs, an additional 1972 league-only ZIP, 33,000+ HTML assets, and an unrelated-looking 13.9 MiB `Microsoft.VisualStudio.Services.VSIXPackage`. The junction and its target are outside the repository cleanup boundary and must not be recursively moved or deleted.

## Primary findings

1. **Input contamination affects authority metadata.** `csv/ootp_csv/out/csv_out/z_ABL_Manager_Tendencies.csv` is generated, yet `abl_source_registry.*` and `abl_batch_manifest_current.*` classify it as `ootp_csv` / `system_of_record_extract`. This is why documents and manifests report 73 raw files while disk has 72 immediate OOTP exports. The catalog-to-registry classification uses a path-prefix rule instead of enforcing immediate-child input boundaries.
2. **The default automation is not production-safe.** `.vscode/tasks.json` runs `csv/_tmp_run_all.py`. That runner discovers every `z_abl_*.py`, runs from `csv/`, passes an OOTP path as a generic `--base`, retries scripts without `--base`, skips required arguments, and still runs unrelated producers. Its committed log ends with 23 failed scripts. This combination explains `csv/csv/` and `csv/ootp_csv/out/` sprawl.
3. **“Current” control data is stale or ambiguously parameterized.** The promoted OOTP `games.csv` now contains league-200 completed regular-season games through 1981-07-24 (1,165 regular-season games), while the newest checked-in preflight is named for 1981-07-23 and `abl_batch_manifest_current` still labels the batch as 1981-07-19. `z_abl_current_state_preflight_validate.py` hard-codes July 23 and accepts later dates without filtering to the named cutoff; the batch-manifest builder hard-codes July 19.
4. **Generated material is almost universally tracked.** All 932 `csv/out` files, all 13 root `out` files, `data_work/abl.db`, all logs, 24 bytecode/cache files, and temporary diagnostic files are tracked. Some are important publication or historical records; others are reproducible. There is no enforced retention policy separating them.
5. **The source tree lacks a stable automation contract.** There is no `pyproject.toml`, dependency lock/requirements file, CI workflow, or test suite beyond `99_test_pandas.py` and `test_abl_team_helper.py`. Local imports assume `csv/abl_scripts` is placed on `sys.path` by direct script invocation. Moving scripts now would break imports and subprocess paths.
6. **One builder mutates an input.** `build_star_schema.py` writes a comment block and regenerated CSV content back to `csv/abl_statistics/...abl_staff.csv`. A source-to-derived build should not modify a promoted input.
7. **Documentation describes several different repository eras.** README names branch `refactor-output-audit`, has mojibake, calls `Prompt Eng` and `fantbbexpert` empty directories although they are non-empty extensionless files, and omits newer `control`, `story`, and StatsPlus architecture. The July repository map says 165 scripts; disk has 189 Python sources in `csv/abl_scripts`. The July 7 ChatGPT source pack explicitly describes an older cutoff and commit.

## Disposition audit: top-level paths and files

Disposition meanings: `KEEP` = retain in place for now; `MOVE` = preserve but relocate after reference updates; `ARCHIVE` = preserve as a dated record outside active paths; `IGNORE / DO NOT TRACK` = may remain locally but should not be routine Git content; `DELETE CANDIDATE` = removal only after the listed proof; `NEEDS REVIEW` = owner/lineage decision required.

| Path | What it is / references and writers | Duplicate or break risk | Disposition |
|---|---|---|---|
| `csv/` | Functioning mixed workspace used by almost every script. | A large rename would break hard-coded paths, imports, docs, tasks, and operator commands. | **KEEP** as the main workspace; simplify internally in stages. |
| `out/` | Legacy generated reports plus deliberate Pittsburgh Express HTML/PNG publication assets. Legacy defaults such as `out/csv_out/...` write here when run from repo root. | Only `z_ABL_Week_Miner.txt` is byte-identical to its `csv/out` twin; other legacy reports have no exact twin. Publication files are not duplicates and were intentionally added July 29. | **MOVE** legacy reports to a dated quarantine/archive after comparison; **MOVE** Pittsburgh assets to `csv/out/story/publication/pittsburgh_express/` or another Showrunner-approved publication path. Do not delete wholesale. |
| `scripts/` | Five tracked sources. Three are prototype story-menu tools; `extract_team_report_items.py` writes into `csv/out/text_out`; `clean_collage.py` is image utility code. | Story tools overlap the newer `csv/abl_scripts/z_abl_story_*` family but are still the only writers/readers for root-level story prototypes. | **KEEP / NEEDS REVIEW**. Do not merge script roots until entry points and parity are defined. |
| `docs/` | Primary repository documentation; includes governance, runbooks, plans, and four overlapping generated CLI snapshots. | Several documents are dated snapshots; CLI docs duplicate one another and captured absolute traceback paths. | **KEEP**, then separate maintained docs from generated/reference snapshots; archive superseded CLI snapshots. |
| `data_raw/` | Contains an ignored external junction, not an empty directory. Seven current scripts reference `data_raw/ootp_html` or its ZIPs/HTML. | Recursive cleanup could affect `C:\SBV_mainC\ootp_all_reports` (4.942 GiB). Full HTML and ZIPs duplicate forms of the same source but serve different readers. | **KEEP; IGNORE / DO NOT TRACK**. Treat the junction as external infrastructure. Review the stray VSIX only outside this repo plan. |
| `data_work/abl.db` | Small generated SQLite working database. At least manager/prep tools and `_tmp_run_all.py` use or test it. | Removing it can disable prep generation; rebuild parity is unproven. It duplicates portions of `csv/out/star_schema`. | **NEEDS REVIEW**, then **IGNORE / DO NOT TRACK** only after a deterministic rebuild and schema/content validation exist. |
| `logs/` | `_tmp_run_all.log` plus preview/update Robocopy logs. The input-update logs are audit evidence but expose local OneDrive paths. | No script reads these logs. The broad-run log documents failure, not a production success. | Existing input-copy evidence: **ARCHIVE** or distill to a control manifest. Routine logs: **IGNORE / DO NOT TRACK**. `_tmp_run_all.log`: **DELETE CANDIDATE** after its failure evidence is retained in this audit/history. |
| `_chatgpt_project_source/` | Eleven-file curated upload pack tied to commit `dbe9355`, July 7, and as-of July 19. No code references it. | Valuable editorial/handoff history, but its “current” language is now stale and can override newer facts. | **ARCHIVE / MOVE** to `docs/archive/chatgpt_project_source_2026-07-07/`, or regenerate if it remains the live handoff. Showrunner decision required. |
| `_run_eb_regular_season_1972.py` | Hard-coded 1972 orchestration source at root. | Overlaps parameterized `csv/abl_scripts/_run_eb_regular_season_any.py`; no active caller found. Paths are hard-coded to current layout. | **ARCHIVE / NEEDS REVIEW** after a 1972 output parity test against the parameterized runner. |
| `Prompt Eng` | Tracked extensionless prompt-engineering note, not a directory; no active references. | Unrelated to ABL production and README misdescribes it. | **NEEDS REVIEW**: move to a separate prompt archive or delete only with owner approval. |
| `fantbbexpert` | Tracked extensionless fantasy-baseball prompt, not a directory; no active references. | Unrelated to ABL production and README misdescribes it. | **NEEDS REVIEW**: move outside this repo or archive; do not infer deletion from age. |
| `temp.txt` | Old 8 KiB partial version of `abl_week_miner.py`; no active reference. | Current miner has 1,115 additional lines and three removed lines relative to it. | **DELETE CANDIDATE** after recording the diff/commit; Git history already preserves it. |
| `tmp_preseason_bytes.txt` | 105 KiB decimal byte dump; no active reference. | Diagnostic material, not a source or readable production artifact. | **DELETE CANDIDATE** after confirming it is not an external recovery aid. |
| `VALIDATION_RUNBOOK.md` | Valid historical EB regeneration runbook at root. | Documentation is split between root and `docs/`. | **MOVE** to `docs/runbooks/`; update links. |
| `tools/run_flashback_menu_1972.ps1` | Tracked helper invoking a user-specific Python 3.12 executable. | Breaks on other machines and duplicates a direct Python CLI command. | **KEEP / NEEDS REVIEW**; make interpreter portable or replace with documented command before considering deletion. |
| `.vscode/` | Tracked keybinding and default task. | Task launches unsafe `_tmp_run_all.py`; keybinding makes accidental broad execution easy. | **KEEP**, but first cleanup commit must remove it as the default build task or point it at a validated runner. |
| `.agents/` | Empty local directory; no tracked files observed. | Environment/tool metadata, not project source. | **IGNORE / DO NOT TRACK**. |

## Disposition audit: `csv/` and important nested paths

| Path | Role, tracking, references/writers, and duplication | Break risk and disposition |
|---|---|---|
| `csv/ootp_csv/*.csv` | **Authoritative promoted input**: 72 immediate tracked CSVs, 299.94 MiB. Read by most current-state and reporting scripts. Updated August 11. | **KEEP. PROTECTED.** Never move, rename, edit, or untrack in cleanup. Hash before every stage and compare afterward. |
| `csv/ootp_csv/out/` | Two tracked generated manager-tendencies reports. No writer uses this exact literal; the path arises when a legacy `--base` output default is given the OOTP directory. CSV twin under `csv/out/csv_out` is exact; text file has no exact twin. | It contaminates source registry/manifest authority. **MOVE to quarantine, then DELETE CANDIDATE**, but only after registry boundary fix, text comparison, and canonical-output regeneration. Do not touch the 72 sibling input CSVs. |
| `csv/abl_scripts/` | 189 tracked Python sources. Main ETL, validation, report, almanac, story, and production engine. Local imports and subprocess calls assume this path. | **KEEP**. Consolidate only after dependency manifest, import tests, entry-point map, and parity tests exist. |
| `csv/abl_scripts/__pycache__/` | 24 tracked CPython 3.12 cache artifacts, including 22 `.pyc` files and two suffixed crash/temporary bytecode files. | Reproducible and interpreter-specific. **IGNORE / DO NOT TRACK; DELETE CANDIDATE** in the first hygiene cleanup after ignore rules are added. |
| `csv/abl_statistics/*.csv` | 20 tracked promoted sortable-stat input files used by at least 19 scripts. A pre-promotion copy exists under `csv/out/archive`. | **KEEP** as supplemental input. Define capture date/authority. Do not delete archived copies until retention is decided. |
| `csv/abl_statistics/core12_spec.md` | Unreferenced documentation co-located with inputs. | **MOVE** to `docs/specs/`; no code reference found, but update human links. |
| `csv/statsplus/current/` | 25 tracked promoted supplemental files used by nine story/StatsPlus scripts. | **KEEP**. It is input, not a replacement for raw OOTP authority. Require dated promotion manifest on refresh. |
| `csv/in/almanac_core/` | 585 tracked historical raw HTML files, exactly 65 per season for 1972–1980. Seven scripts reference `csv/in`; it drives derived almanac outputs. | **KEEP** for the smallest safe change. A future rename to a clearer input path is high-churn and not justified now. Decide whether ZIPs or extracted core HTML are the long-term archival authority before untracking either. |
| `csv/abl_csv/` | Six tracked small convenience/derived CSVs. `abl_config.py` exposes this as `ANALYTICS_CSV_ROOT`; current EB/show-note scripts refer to it. Similar roles also exist in `csv/out/star_schema` and `csv/out/csv_out`. | **NEEDS REVIEW / KEEP** until each file has producer, cutoff, and consumer lineage. Later move reproducible tables to `csv/out`, or reclassify genuinely curated tables as config/input. |
| `csv/config/` | One tracked all-star venues configuration CSV. | **KEEP**. This is the clearest current configuration zone. |
| `csv/docs/` | Seven tracked older schemas/templates/operator notes; mostly no code dependency. Root docs serve the same role. | **MOVE** into categorized root `docs/` in a link-fixing commit. Preserve editorial templates. |
| `csv/csv/docs/eb_game_pack_schema.md` | Accidental doubled path; byte-identical to `csv/docs/eb_game_pack_schema.md`. | **DELETE CANDIDATE** after canonical doc is moved/verified. |
| `csv/csv/out/almanac/1972/eb_schedule_context_1972_league200.md` | Accidental doubled path; byte-identical to canonical `csv/out/almanac/1972` file. | **DELETE CANDIDATE** after output-path guard test. |
| `csv/csv/out/almanac/1972/eb_player_context_1972_league200.md` | Accidental doubled path; a canonical same-named file exists but content is not byte-identical. | **ARCHIVE / NEEDS REVIEW**. Diff content and choose the valid production version before removal. |
| `csv/_tmp_run_all.py` | Broad best-effort runner and current default editor task. It makes output dirs on import, has a user-specific OneDrive fallback, runs all `z_abl_*` scripts, retries incompatible CLIs, and tolerates skips. | **REPLACE, then DELETE CANDIDATE**. Do not run during cleanup. First disable it as the default task; replace with an explicit manifest-driven runner that fails closed. |
| `csv/story_dictionary.csv` | Tracked curated prototype configuration read by root story tools. | **KEEP / MOVE** to `csv/config/story_dictionary.csv` only with code/doc updates. |
| `csv/story_candidates_1981_week_05.csv` | Tracked generated early-season candidate snapshot. New engine outputs candidates under `csv/out/story/candidates`. It is explicitly excluded by current story scripts. | **ARCHIVE / MOVE** to a dated legacy story archive; preserve as historical example. |
| `csv/story_menu_1981_week_05.csv`, `csv/story_menu_1981_week_07.csv` | Tracked generated/selected early-season menus; the two files are byte-identical. Root tools still default to this location pattern. | **ARCHIVE / MOVE** after root story tools write/read canonical `csv/out/story/menus`. Do not delete merely because old. |
| `csv/week_game_ids.npy` | Tracked binary working artifact; no active source reference found. | **NEEDS REVIEW**, then **DELETE CANDIDATE / IGNORE** if no external manual workflow consumes it. |
| `csv/.gitattributes` | Second attributes file in addition to root. | **NEEDS REVIEW**; inspect intended scope before consolidation. No current LFS rule exists in either observed file. |

## Disposition audit: `csv/out/`

`csv/out/` should remain the single canonical generated-output root. The cleanup problem is not the existence of this directory; it is mixed retention, duplicate destinations, and the absence of run-scoped output contracts.

| Path | Current content and role | Disposition |
|---|---|---|
| `csv/out/control/` | 114 generated registries, manifests, preflights, promotion/reconciliation reports, and feature flags. Many are triplicated CSV/JSON/Markdown representations. Some are essential audit evidence, but “current” names and dates have drifted. | **KEEP / NEEDS REVIEW**. Retain validated run manifests and human decisions; make “current” a regenerated pointer or explicit dated run. Archive superseded controls by run, not loose filenames. |
| `csv/out/story/` | 82 candidate, menu, editorial, enrichment, investigation, packet, production, and script files. This mixes reproducible machine products with human-edited/publication-ready material. | **KEEP** publication and editorial decisions. **IGNORE / DO NOT TRACK** reproducible machine twins only after producer manifests and deterministic tests exist. Add a `publication/` subfolder for final assets. |
| `csv/out/star_schema/` | 35 derived facts/dimensions and Monday tables. Several checked-in files are explicitly documented as stale early snapshots. | **NEEDS REVIEW** now; eventual **IGNORE / DO NOT TRACK** if fully reproducible. Preserve dated frozen releases separately. Consumers currently depend on files being present. |
| `csv/out/csv_out/` | 47 analytic report tables plus one season subfolder. Similar destination to `star_schema`; includes stale manager tendencies. | **KEEP** until producer/consumer map and cutoff metadata exist; later retain only published/frozen extracts. |
| `csv/out/text_out/` | 89 text/Markdown prep and report files, including 33 prep files. Frequently read by pack builders. | **KEEP** active/published handoff outputs; make routine reports run-scoped and untracked after regeneration is proven. |
| `csv/out/almanac/` | 340 derived historical files across 1972–1980 (37 per season except 44 for 1972). Built from tracked core HTML and/or ignored ZIPs. | **ARCHIVE / NEEDS REVIEW**. Preserve as a frozen historical data product unless all-season deterministic rebuild parity is demonstrated. |
| `csv/out/eb/` | 166 historical EB packs and Grand Series HTML extracts. EB regular-season packs and schedule contexts also appear under `almanac`, often byte-identical. | **CONSOLIDATE / ARCHIVE** after choosing whether `almanac/<season>/publication` or `eb/` owns rendered historical products. Do not delete unique HTML/Markdown without a manifest. |
| `csv/out/archive/` | 37 frozen 1981 outputs and pre-promotion copies of sortable/StatsPlus input. Several are exact copies of live paths. | **KEEP / ARCHIVE** until retention period and recovery source are decided. Duplication may be intentional rollback protection. |
| `csv/out/statsplus_repair/` | Four generated repair artifacts. | **ARCHIVE / NEEDS REVIEW** with the promotion run that produced them. |
| `csv/out/docs/abl_data_catalog.csv` | Generated 483-row catalog used as an input by `z_abl_source_registry_build.py`. It includes duplicates and misplaced outputs. | **KEEP** until a deterministic catalog generator is committed. Then move generated documentation metadata under control/run output and prevent circular “output as source” authority. |
| Loose files at `csv/out/` | Reports, inventories, forum/video/story files, plus `.tmp_league_report_abl_1981_w15.md`. | **MOVE** into story/report subfolders. The `.tmp` file is byte-identical to the final league report and is a **DELETE CANDIDATE**. |

## Hard-coded paths and output-sprawl causes

| Source | Finding | Required correction before cleanup |
|---|---|---|
| `csv/_tmp_run_all.py` | Hard-coded user OneDrive OOTP path; `cwd=csv`; passes OOTP root as generic base; indiscriminate glob runner. | Retire or replace with explicit steps, repo-root `cwd`, separate `--input-root` and `--output-root`, strict CLI contracts, fail-fast behavior, and a run manifest. |
| `.vscode/tasks.json` / keybinding | Default build task invokes `_tmp_run_all.py`. | Disable as default immediately or point to a validated, non-destructive preflight. |
| `tools/run_flashback_menu_1972.ps1` | Hard-coded `C:\Users\earld\...\python.exe`. | Use `python`/configured interpreter and resolve repo root from script location. |
| `z_abl_sortable_capture_requirements.py`, `z_abl_sortable_schema_drift_promote.py`, and two reconciliation scripts | Hard-coded user OneDrive paths. | Require explicit CLI/config paths; no personal default in production. |
| StatsPlus legacy/promotion scripts | Hard-coded `E:\BACKUP\BACKUP 2\ABL 1981` and a user staging path. | Convert legacy baseline and staging roots to required arguments or documented environment variables. |
| `z_abl_current_state_preflight_validate.py` | Hard-coded season/date/output stem; counts all completed games, including dates later than target, and accepts `latest >= target`. | Parameterize season/as-of/newsroom; filter `date <= as_of`; require output name and evidence to agree. |
| `z_abl_batch_manifest_build.py` | Hard-coded July 19 batch labels and promotion report. | Derive/require batch ID and as-of date; reject stale “current” labels. |
| `build_star_schema.py` | Writes back into promoted `abl_statistics` staff input. | Write normalized staff data only to derived output; add a source immutability test. |
| `abl_week_miner.py` | Uses `Path.cwd()` and writes `out/csv_out` and `out/text_out`; running from root creates root `out`. | Resolve repo root from `__file__` or require output root. Prefer the canonical `z_abl_week_miner.py` after parity. |
| Legacy report scripts such as manager tendencies, platoon assassins, and power surge/outages | Default `--base=.` plus `out/...` beneath the base. Root runs create root `out`; `_tmp` can create output beneath OOTP input. | Split input and output arguments; refuse any output path under a protected input root. |
| 1972 EB scripts using `Path("csv/out/...")` | Relative to current working directory. `_tmp` retries them from `csv`, producing `csv/csv/out`. | Resolve repository root from `__file__`; add cwd-independence tests. |
| README and validation runbook | Hard-code obsolete branch and/or local repository path. | Use repository-relative commands and actual branch architecture. |

## Duplicate and overlapping scripts requiring parity review

These are not deletion recommendations. They are consolidation candidates only after same-input/same-output comparisons:

- `abl_week_miner.py` versus `z_abl_week_miner.py`; the former is cwd-sensitive and creates root output, while the latter targets `csv/out` but produces a different surface.
- `temp.txt` versus the historical/current `abl_week_miner.py`; `temp.txt` is an old partial copy.
- `abl_manager_matchup.py` versus `abl_managers_matchup.py`.
- `99_make_standings_snapshot.py` versus `99_rebuild_standings_snapshot.py`.
- `z_abl_1980_season_backbone.py` versus parameterized `z_abl_season_backbone.py`.
- Root `_run_eb_regular_season_1972.py` and multiple `*_1972.py` producers versus `_run_eb_regular_season_any.py` and `*_any.py` producers.
- Root story-menu scripts versus `z_abl_story_menu_1981_week5.py`, `z_abl_story_signal_weekly_1981.py`, and `z_abl_story_engine_current_1981.py`.
- `csv/out/almanac/<season>/eb_regular_season_pack_*` versus `csv/out/eb/eb_regular_season_pack_*` (exact duplicates for 1972–1980).
- Clean/enriched almanac series summaries that are currently exact twins; confirm whether enrichment is intentionally a no-op before consolidating.

Git contains 87 exact duplicate blob groups involving 538 paths (451 excess paths). Most are repeated empty/alike historical almanac pages and should remain raw historical evidence; duplicate hash alone is not grounds for deletion.

## Documentation drift

- README is tied to `refactor-output-audit`, while Git is on `add-star-schema`.
- README contains encoding corruption and an unclosed setup code fence.
- README and repository maps call `data_raw` empty; it is a 4.942 GiB external junction.
- README calls `Prompt Eng/` and `fantbbexpert/` empty directories; both are non-empty files.
- README's output map predates `control`, `story`, `statsplus_repair`, and the current production packets.
- `docs/ABL_REPO_MAP.md` (July 3) reports 73 raw exports and 165 scripts; disk has 72 immediate raw CSVs plus one misclassified generated CSV, and 189 Python sources.
- `docs/ABL_DATA_CATALOG.md` and the generated catalog perpetuate the 73-file count because they include the nested manager-tendencies output.
- `_chatgpt_project_source` is internally well-labeled as a July 7 snapshot, but many files use “current” language and an as-of July 19 state that is no longer current.
- `abl_batch_manifest_current` says July 19 while canonical input reaches July 24; the newest preflight says July 23. “Current” cannot be trusted without regeneration.
- `cli_options_all.md` and `cli_options_csv_abl_scripts.md` substantially duplicate one another and include captured absolute traceback paths. Treat them as generated snapshots, not maintained authority.
- `VALIDATION_RUNBOOK.md` belongs with root runbooks; `csv/docs` and `core12_spec.md` should join categorized root documentation.

## Proposed canonical repository map

This map deliberately preserves the functioning `csv` architecture and avoids a wholesale package redesign.

```text
abl_csv_repo/
├── README.md
├── .gitignore
├── .env.example
├── csv/
│   ├── ootp_csv/                 # PROTECTED: exactly immediate promoted OOTP CSV inputs
│   ├── abl_statistics/           # promoted sortable-stat inputs only
│   ├── statsplus/current/        # promoted supplemental inputs only
│   ├── in/almanac_core/          # tracked historical core inputs (retain for now)
│   ├── config/                   # curated configuration, including story dictionary
│   ├── abl_csv/                  # transitional derived/curated tables; resolve lineage
│   ├── abl_scripts/              # processing engine; one source location for now
│   └── out/                      # the only repository output root
│       ├── control/              # run manifests, validation, promotion records
│       ├── star_schema/          # derived analytical tables
│       ├── csv_out/              # derived report tables
│       ├── text_out/             # rendered/prep reports
│       ├── story/
│       │   ├── candidates/
│       │   ├── menus/
│       │   ├── editorial/
│       │   ├── production/
│       │   └── publication/      # final HTML/images/published handoff assets
│       ├── almanac/              # historical derived data, organized by season/run
│       └── archive/              # immutable, manifested rollback/frozen releases
├── scripts/                      # supported operator entry points only
├── tools/                        # portable helper tooling
├── docs/
│   ├── architecture/
│   ├── governance/
│   ├── runbooks/
│   ├── schemas/
│   ├── templates/
│   └── archive/                  # dated repo maps and ChatGPT source packs
├── data_raw/
│   └── ootp_html -> external junction, ignored and documented
├── data_work/                    # rebuildable working databases, ignored
└── logs/                         # runtime logs, ignored; durable evidence goes to control/archive
```

There should be no root `out/`, no `csv/csv/`, no `csv/ootp_csv/out/`, no generated story files loose under `csv/`, and no runtime cache/log/temp files tracked. This is a target state, not authorization to perform those removals now.

## Small, reversible cleanup sequence

### Stage 0 — Freeze the baseline (no cleanup)

Create a machine-readable baseline containing HEAD, `git ls-files`, canonical input SHA-256/size list for exactly `csv/ootp_csv/*.csv`, ignored-junction target, and hashes for every candidate being moved. Tag or branch the pre-clean state. Record the current output tree and the currently observed latest completed league-200 regular-season date.

Validation:

- Exactly 72 immediate canonical OOTP CSV hashes match the pre-audit capture.
- `git diff -- csv/ootp_csv/*.csv` is empty.
- Worktree is clean after committing the baseline manifest.
- Recovery drill can restore a sampled generated, historical, and production file from Git/archive.

### Stage 1 — Protect inputs and stop new sprawl

This is the **recommended first cleanup commit**: `chore(repo): protect canonical inputs and block output sprawl`.

- Change catalog/registry logic so only immediate `csv/ootp_csv/*.csv` files can be classified as OOTP system-of-record input; reject any directory or output below that root.
- Add a path-policy check that fails on `csv/ootp_csv/out`, `csv/csv`, root report output, output writes inside any input root, or mutation of promoted inputs.
- Parameterize preflight/manifest dates and make output labels match filtered evidence.
- Remove `_tmp_run_all.py` from the default VS Code build task (do not delete it yet); point the task to a read-only preflight or leave no default runner until replacement exists.
- Expand `.gitignore` for `__pycache__/`, `*.py[cod]`, suffixed bytecode, `.tmp*`, routine `*.log`, local databases, and known scratch names. Existing tracked items remain for later commits.

Validation:

- Canonical 72-file hash manifest is unchanged.
- Registry/manifest reports 72 raw OOTP files and classifies/rejects nested manager tendencies as generated pollution.
- A test using a temporary directory proves report writers cannot place output under OOTP, sortable, StatsPlus, or historical input roots.
- VS Code default task no longer launches the broad writer.
- Git diff contains only guardrail/config/test changes and the worktree is clean after commit.

### Stage 2 — Remove only reproducible ephemeral artifacts

Untrack the 24 cache files, `csv/out/.tmp_league_report_abl_1981_w15.md`, and—after their diffs are recorded—`temp.txt` and `tmp_preseason_bytes.txt`. Decide separately on `week_game_ids.npy`. Do not touch logs, databases, historical HTML, production images, or non-temp outputs in this commit.

Validation:

- Python sources parse with `ast.parse`; targeted local-import smoke tests run with `PYTHONDONTWRITEBYTECODE=1` and `PYTHONPATH=csv/abl_scripts`.
- The `.tmp` file's final twin hash is present.
- No ignored cache/temp files appear in `git status` after a smoke import.
- Canonical input hashes remain unchanged and worktree is clean after commit.

### Stage 3 — Replace unsafe orchestration and normalize output roots

Introduce one explicit, manifest-driven runner for the supported production slice. Separate input roots from output root; resolve repository paths independent of cwd; require season/as-of arguments; fail fast; prohibit source mutation; capture producer/version/input hashes/output list. Update portable tools and the editor task. Keep legacy runners available but clearly deprecated until parity passes.

Validation:

- Run in a temporary clone/worktree or with an isolated temporary output root.
- Compare expected output schemas, row counts, key uniqueness, and selected golden hashes/semantic checks to the pre-clean baseline.
- Snapshot the filesystem before/after and prove all writes are confined to the declared temporary/canonical output root.
- Run from both repository root and another cwd; output paths must be identical.
- Inputs, including sortable and StatsPlus files, remain byte-identical; imports succeed; Git stays clean.

### Stage 4 — Quarantine and reconcile misplaced outputs

In separate commits by family:

1. Reconcile `csv/ootp_csv/out` into a dated quarantine, regenerate canonical manager tendencies, then remove the nested output only after the registry remains clean.
2. Reconcile `csv/csv` files; delete only exact twins and archive/diff the unique player-context variant.
3. Move root legacy reports to dated archive and Pittsburgh Express assets to the approved publication folder.
4. Move loose `csv/story_*` snapshots into a dated legacy story archive and update root story tools to canonical output paths.

Validation for every family:

- Before/after manifest lists source path, destination, hash, producer, and retention reason.
- All code, task, config, README, and docs references to the old path are either updated or explicitly historical.
- Regeneration creates no old directory.
- Publication HTML opens with all asset references resolved; story/EB consumers read the new path.
- Canonical input hashes are unchanged; `git status --porcelain` is empty after commit.

### Stage 5 — Define generated-output retention and untrack only proven rebuildables

Adopt a retention matrix: promoted inputs; working derived; validation/control evidence; human editorial decisions; final publication; historical frozen release. Keep the last three categories when they are not reproducible or are records of judgment/publication. Stop tracking routine star-schema/report/cache/log/database products only after deterministic rebuilds and consumer bootstrapping exist.

Validation:

- Fresh clone plus documented external inputs can rebuild every untracked required artifact.
- Database schema/table counts and derived CSV schemas/keys match baseline.
- Historical all-season validator passes for 1972–1980.
- A publication/historical manifest proves no human-edited or published asset was lost.
- Clean clone ends with no unexpected untracked output outside ignored declared paths.

### Stage 6 — Consolidate scripts only after parity

Test each duplicate/parameterized pair on the same inputs and compare semantic output. Deprecate one implementation per commit, update callers, then archive/remove only after at least one routine production cycle succeeds.

Validation:

- CLI/help contract and import smoke tests pass.
- Grep/reference scan finds no active caller of removed scripts.
- Output schemas, row counts, keys, and editorial render checks match.
- Cwd-independence and protected-input tests pass.

### Stage 7 — Repair documentation and handoff layers

Replace README with the actual branch-agnostic architecture and supported commands; merge `csv/docs`, root runbook, and spec into categorized docs; label dated docs as snapshots; regenerate or archive the ChatGPT source pack; make one CLI reference canonical.

Validation:

- Every documented path exists or is clearly labeled historical/example.
- Branch, script count, raw file count, current cutoff, and junction description match live checks.
- Commands execute in a clean temporary clone without user-specific paths.
- Link/reference checker reports no broken local links.

### Stage 8 — Resolve unrelated and external material

Only after Showrunner decisions, move/archive `Prompt Eng`, `fantbbexpert`, and any unrelated publication experiments. Review the VSIX inside the external junction with the owner of `C:\SBV_mainC`; it is not part of repository cleanup.

Validation:

- Owner approval and archive destination are recorded.
- No ABL script/config/task references the material.
- External junction target is unchanged.

## Required validation gates for every cleanup commit

1. **Authoritative OOTP input untouched:** compare SHA-256 and size for the 72 immediate `csv/ootp_csv/*.csv`; reject additions below `csv/ootp_csv/` that are not approved immediate CSV exports.
2. **Automation still works:** run the smallest supported manifest-driven slice in an isolated output root and run relevant validators. Never use `_tmp_run_all.py` as proof.
3. **Imports and paths still work:** AST-parse all Python, run targeted tests/imports without bytecode, scan source/tasks/docs for old paths, and run entry points from multiple cwd values.
4. **Outputs go only where intended:** compare full filesystem snapshots before/after and require every write to be under the declared output root; inputs must remain read-only by hash.
5. **Git worktree clean:** after each commit, `git status --porcelain` must be empty; ignored output inspection must show only declared patterns.
6. **No historical/production loss:** use a move manifest, exact hashes, Git tag/branch, dated archive, sampled restore, and human review of publication/editorial files.

## Five biggest sources of clutter or confusion

1. The indiscriminate `_tmp_run_all.py` plus cwd/base ambiguity, which creates multiple output roots and masks failures.
2. The 932-file fully tracked `csv/out` tree with no retention boundary between rebuildable data, control evidence, editorial decisions, and publication.
3. Multiple output destinations: root `out`, `csv/out`, `csv/csv/out`, and `csv/ootp_csv/out`, plus loose generated files under `csv`.
4. 189 flat Python scripts with overlapping hard-coded, year-specific, and parameterized implementations, minimal tests, and no dependency/entry-point contract.
5. Stale and duplicated documentation/handoff snapshots that use “current” for different cutoffs and misdescribe the live filesystem.

## Showrunner decisions required

1. Which generated artifacts are durable records: control manifests, story candidates/evidence, human editorial decisions, final publication, historical almanac products, and/or all of the above?
2. Should Pittsburgh Express HTML/PNG assets remain versioned as final publication, and is `csv/out/story/publication/` the approved home?
3. Is `_chatgpt_project_source` a live publication handoff that must be regenerated, or a July 7 snapshot to archive?
4. Are `Prompt Eng` and `fantbbexpert` personal assets to preserve elsewhere, or intentionally part of ABL?
5. Should historical authority live in full external almanac ZIPs, tracked `csv/in/almanac_core` extracts, or both, and what is the retention/recovery policy?
6. May `data_work/abl.db` and routine logs become local ignored artifacts once rebuild/audit evidence is proven?
7. Should early Week 5/7 story menus and candidates be retained as versioned examples or only in a dated archive?
8. Which runner is the supported routine ABL production entry point after `_tmp_run_all.py` is retired?

## Final recommendation

Do not start by deleting old outputs or flattening directories. Start by enforcing the authority boundary around the 72 immediate OOTP CSVs, parameterizing the stale control layer, disabling the unsafe default runner, and adding output-path tests. Once new sprawl is impossible, remove only caches and exact temporary twins, then reconcile each misplaced output family with hashes and a reversible commit.

**SAFE TO CLEAN: NO for bulk cleanup; YES only for the Stage 1 guardrail commit described above.**
