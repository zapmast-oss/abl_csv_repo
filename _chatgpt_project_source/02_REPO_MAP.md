# Repository Map

**Last verified:** 2026-07-07. Paths are repository-relative.

## Top-level tree

```text
abl_csv_repo/
├── csv/                 # main ABL workspace
│   ├── ootp_csv/        # raw OOTP exports
│   ├── abl_statistics/  # sortable-stat extracts
│   ├── statsplus/       # promoted StatsPlus feed
│   ├── abl_csv/         # smaller summary inputs
│   ├── abl_scripts/     # ETL, reporting, story, validation scripts
│   ├── docs/            # legacy/templates inside csv workspace
│   └── out/             # generated reports, story products, controls, archives
├── docs/                # governance, schemas, plans, runbooks
├── data_work/           # working SQLite database
├── scripts/             # top-level story-menu drivers
├── data_raw/            # reserved; empty when inspected
├── logs/                # orchestration logs
├── out/                 # older/small root-level outputs
├── tools/               # helper tooling
└── _chatgpt_project_source/  # curated upload pack
```

## Path guide

| Path | Type | Purpose | Current / Historical / Ignore | Notes |
|---|---|---|---|---|
| `README.md` | Documentation | High-level setup and legacy map | Useful but partly stale | Names an old branch; use git for branch truth |
| `docs/` | Source documentation | Governance, schemas, plans, intake and runbooks | Current guidance plus older phase docs | Prefer dated/current-state docs when they conflict |
| `csv/ootp_csv/` | Raw source | OOTP system-of-record exports | Current authority when cutoff-filterable | 74 files, about 313 MB; do not upload wholesale |
| `csv/abl_statistics/` | Source extract | Sortable statistics and supplemental reports | Supplemental | 21 files; cannot alone prove cutoff |
| `csv/statsplus/current/` | Promoted source | 25 StatsPlus tables | Current enrichment preview | Tables 6 and 27 intentionally absent |
| `csv/abl_csv/` | Source/derived summaries | Small matchup, trend, week-miner inputs | Mixed/stale risk | Verify dates before use |
| `csv/abl_scripts/` | Application source | ETL, validation, reporting, story-engine scripts | Active code | Includes generated `__pycache__`; ignore caches |
| `csv/out/control/` | Generated control | Registries, manifests, preflights, promotion/validation reports | Important current controls plus history | Read date-specific current files first |
| `csv/out/story/` | Generated editorial output | Candidates, evidence, menus, packets, scripts, production | Highest editorial relevance | Use July 20/as-of-July-19 names for current work |
| `csv/out/star_schema/` | Generated analytical model | Facts/dimensions and Monday tables | Mixed; several early snapshots stale | Do not assume “star schema” means current |
| `csv/out/almanac/` and `csv/out/eb/` | Generated historical | 1972–1980 history and EB packs | Historical | Context only, explicitly label season |
| `csv/out/text_out/`, `csv/out/csv_out/` | Generated reports | Many analytical text/CSV reports | Mixed and often stale | Summarize selectively; validate coverage |
| `csv/out/archive/` | Archive | Pre-promotion and seasonal checkpoints | Historical/ignore normally | Do not upload wholesale |
| `data_work/abl.db` | Working data | SQLite star-schema database | Derived working state | Binary; do not upload directly |
| `scripts/` | Application source | Story trigger/menu entry points | Legacy/active mix | Week-numbered examples require date checks |
| `logs/`, `__pycache__/`, `.git/`, `.vscode/` | Operational | Logs, cache, history, editor config | Ignore for Project Source | Not editorial source material |
| `temp.txt`, `tmp_preseason_bytes.txt` | Temporary | Scratch/diagnostic content | Ignore | Not curated source |

## Naming conventions

- `1981_07_20_asof_1981-07-19` means newsroom date July 20 with facts cut off after July 19.
- `week_05`, `w05`, `week_15`, and `w15` are operational snapshots, not automatically current.
- Parallel `.csv`, `.json`, and `.md` files usually represent machine-readable, structured, and human-readable forms of the same artifact.
- `z_abl_*` generally identifies reporting/analysis pipeline scripts or outputs.
- `current` is meaningful only when backed by a manifest/preflight; a filename alone is insufficient.

## Priority for ChatGPT

Care most about this source pack, current preflight/control files, July 20 story production/scripts/evidence, and governing documentation. Usually ignore caches, logs, temporary files, binaries, archives, raw bulk exports, and old week-numbered derivatives unless a historical comparison is explicitly requested.

