# ABL Repository Cleanup Stage 0 Baseline

Baseline date: 2026-08-14 (America/Los_Angeles)  
Repository: `C:\sbv_repo\abl_csv_repo`  
Stage: **0 — Freeze the baseline**  
Result: **PASS**

## Scope and non-mutation statement

This baseline was produced with read-only Git, filesystem, hashing, and CSV inspection commands. The production pipeline was not run. `csv/_tmp_run_all.py` was not run. No existing file was edited, regenerated, moved, deleted, renamed, staged, committed, or pushed. The only filesystem additions are the four dated Stage 0 files listed in this report.

The pre-Stage-0 worktree was clean. After baseline creation, Git shows only the four new untracked Stage 0 documentation/manifest files. There are no staged changes, no unstaged modifications to tracked files, and no other untracked files.

## Git baseline

| Field | Verified value |
|---|---|
| Branch | `add-star-schema` |
| HEAD | `767b59522ad8b48056449b2100b935c7ff4d0bc4` |
| HEAD commit date | `2026-08-14T21:41:47-07:00` |
| HEAD subject | `feat: Add ABL Repository Cleanup Audit for August 2026` |
| Upstream | `origin/add-star-schema` |
| Local alignment with recorded upstream ref | `+0 / -0` (ahead 0, behind 0) |
| Staged files before baseline creation | none |
| Unstaged tracked files before baseline creation | none |
| Untracked files before baseline creation | none |
| Tracked files | 1,949 |

Origin alignment is against the existing local `origin/add-star-schema` remote-tracking ref. No network fetch was performed because Stage 0 was limited to baseline documentation and was not authorized to mutate Git metadata or retrieve remote state.

## Baseline artifacts

| File | Purpose | Rows | Bytes | SHA-256 |
|---|---|---:|---:|---|
| `docs/repo_cleanup_stage0_tracked_inventory_2026_08_14.csv` | Every tracked path with worktree size, Git index mode, index blob OID, and stage | 1,949 | 236,937 | `c52a7a7f1dfc01e14d5eb10b27a6015402c8ec2bbd7cfa38d7336f7d3b0fdc20` |
| `docs/repo_cleanup_stage0_ootp_sha256_2026_08_14.csv` | Exactly the 72 immediate authoritative OOTP CSVs, with filename, size, and SHA-256 | 72 | 7,164 | `7b5fd3af6e8bb87764207e0f20d88abd6f09f10807377cd5545dd640b45ace3f` |
| `docs/repo_cleanup_stage0_recovery_manifest_2026_08_14.csv` | File-level hashes for important candidates plus aggregate family recovery records | 115 | 17,723 | `8f9febd9de4143be27aed7f4f630da916ee3783b18c1366f87450d6e505e9591` |
| `docs/repo_cleanup_stage0_baseline_2026_08_14.md` | Human-readable baseline, method, tree, and verification result | n/a | self | not self-hashed |

The recovery manifest's family digest is SHA-256 over UTF-8/LF records sorted by repository-relative path, with each record formatted as `path|size_bytes|file_sha256\n`. File records contain direct file SHA-256 values.

## Protected authoritative OOTP input

Protected scope is **only** the immediate files matched by:

```text
csv/ootp_csv/*.csv
```

The inventory was non-recursive. It explicitly excludes `csv/ootp_csv/out/` and all nested content.

| Check | Result |
|---|---|
| Immediate authoritative CSV count | **72** |
| Required count | **72** |
| Total protected bytes | **314,512,853** |
| Name/size/SHA-256 verification after manifest creation | **PASS — 72/72** |
| Protected record-set aggregate SHA-256 | `9e384a8fb6dcd0c1d5074279198037c4461acb8f37ce7ff26dcde2c8d212458a` |
| Nested directory observed but excluded | `csv/ootp_csv/out/` |

No authoritative OOTP file changed. The protected manifest is the required comparison source for every later cleanup stage.

## Current ABL game-state baseline

The values below were read directly from `csv/ootp_csv/games.csv` using rows where:

- `league_id == 200`
- `game_type == 0`
- `played == 1`

| Measure | Verified value |
|---|---:|
| Completed 1981 ABL regular-season games | **1,165** |
| Earliest completed regular-season date | `1981-04-06` |
| Latest completed regular-season date | **`1981-07-24`** |

No output was generated from these values.

## Repository tree baseline

This is the pre-Stage-0 summary from disk and Git. `.git` internals and the external junction target were not recursively inventoried in repository totals.

| Top-level path | Type | Files | Bytes | Tracked files | Baseline role |
|---|---|---:|---:|---:|---|
| `.agents/` | directory | 0 | 0 | 0 | local metadata placeholder |
| `.vscode/` | directory | 2 | 864 | 2 | editor task/keybinding |
| `_chatgpt_project_source/` | directory | 11 | 52,901 | 11 | dated handoff/source pack |
| `_run_eb_regular_season_1972.py` | file | 1 | 7,109 | 1 | legacy fixed-season runner |
| `csv/` | directory | 1,874 | 401,827,306 | 1,874 | input, scripts, configuration, and output workspace |
| `data_raw/` | directory containing ignored junction | repository count 0 | external | 0 | external HTML/almanac access |
| `data_work/` | directory | 1 | 28,672 | 1 | working SQLite database |
| `docs/` | directory before Stage 0 additions | 29 | 282,859 | 29 | repository documentation |
| `logs/` | directory | 3 | 37,531 | 3 | run and input-copy logs |
| `out/` | directory | 13 | 13,387,037 | 13 | misplaced legacy and publication output |
| `scripts/` | directory | 5 | 28,146 | 5 | root operator/story tools |
| `tools/` | directory | 1 | 156 | 1 | PowerShell helper |

Root tracked files also include `.env.example`, `.gitattributes`, `.gitignore`, `README.md`, `Prompt Eng`, `fantbbexpert`, `temp.txt`, `tmp_preseason_bytes.txt`, and `VALIDATION_RUNBOOK.md`.

Important `csv/` zones at baseline:

| Path | Files | Approximate bytes / MiB | Role |
|---|---:|---:|---|
| `csv/ootp_csv/` | 74 total: 72 immediate CSVs plus 2 nested outputs | 299.95 MiB | protected OOTP input plus misplaced nested output |
| `csv/abl_scripts/` | 213 total: 189 `.py` plus 24 cache files | 2.73 MiB | processing/reporting engine and cache artifacts |
| `csv/abl_statistics/` | 21 | 1.57 MiB | 20 supplemental inputs plus one spec |
| `csv/statsplus/current/` | 25 | 0.09 MiB | promoted supplemental input |
| `csv/in/almanac_core/` | 585 | 46.55 MiB | tracked historical HTML input |
| `csv/abl_csv/` | 6 | 0.03 MiB | convenience/derived tables |
| `csv/docs/` | 7 | 0.03 MiB | nested documentation |
| `csv/csv/` | 3 | 0.01 MiB | accidental doubled path |
| `csv/out/` | 932 | 32.23 MiB | canonical output root with mixed retention classes |

## Ignored junction baseline

| Field | Verified value |
|---|---|
| Repository path | `C:\sbv_repo\abl_csv_repo\data_raw\ootp_html` |
| Filesystem type | directory junction / reparse point |
| External target | `C:\SBV_mainC\ootp_all_reports` |
| Git state | ignored; not tracked |
| Previously measured external content | 34,792 files / approximately 4.942 GiB |

The junction was inspected but not traversed for hashing in Stage 0. It and its external target were not modified.

## Questionable or misplaced paths frozen for recovery

The following audit candidates existed at baseline and are represented by direct file hashes and/or aggregate family digests in the recovery manifest:

- root `out/` — 13 files, including legacy reports and Pittsburgh Express HTML/PNG publication assets;
- `csv/ootp_csv/out/` — 2 generated manager-tendencies files nested under protected input;
- `csv/csv/` — 3 accidental doubled-path files;
- `csv/abl_scripts/__pycache__/` — 24 tracked cache/bytecode files;
- root `scripts/` and `csv/abl_scripts/` remain in place; specific legacy runners are individually recorded;
- root `docs/`, nested `csv/docs/`, `VALIDATION_RUNBOOK.md`, and `csv/abl_statistics/core12_spec.md`;
- `csv/in/almanac_core/`, `csv/abl_csv/`, `csv/abl_statistics/`, and `csv/statsplus/current/`;
- `data_work/abl.db`, `logs/`, and `_chatgpt_project_source/`;
- `csv/_tmp_run_all.py` (recorded, not executed);
- loose `csv/story_*` files and `csv/week_game_ids.npy`;
- `temp.txt`, `tmp_preseason_bytes.txt`, `Prompt Eng`, and `fantbbexpert`;
- loose files directly under `csv/out/`, including the tracked `.tmp` report;
- all important `csv/out` families: `control`, `story`, `star_schema`, `csv_out`, `text_out`, `almanac`, `eb`, `archive`, `statsplus_repair`, and generated `docs`.

The recovery manifest is descriptive. It does not authorize Stage 1 or any later disposition.

## Stage 0 verification

| Requirement | Result |
|---|---|
| Correct branch and HEAD recorded | PASS |
| Local upstream alignment recorded | PASS |
| Staged, unstaged, and untracked pre-state recorded | PASS |
| All 1,949 tracked files inventoried | PASS |
| Exactly 72 immediate authoritative OOTP CSVs inventoried | PASS |
| Nested `csv/ootp_csv/out` excluded from protected manifest | PASS |
| Protected names, sizes, and SHA-256 values rechecked | PASS — 72/72 |
| Current tree and questionable paths recorded | PASS |
| Junction and external target recorded | PASS |
| Latest completed ABL date and game count recorded | PASS |
| Recovery hashes recorded for candidate files/families | PASS |
| Production pipeline executed | NO |
| `csv/_tmp_run_all.py` executed | NO |
| Production output regenerated | NO |
| Existing files moved or deleted | NO |
| Existing tracked files modified | NO |

## Stage conclusion

**STAGE 0: PASS**

The baseline is sufficient to measure and reverse later cleanup work. Stage 1 may begin only as a separate assignment and must use the protected-input manifest as a hard gate.

**SAFE TO BEGIN STAGE 1: YES**
