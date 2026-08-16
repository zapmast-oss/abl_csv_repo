# ChatGPT Project Source Manifest

**Generated:** 2026-07-07 (America/Los_Angeles)  
**Repository basis:** branch `add-star-schema`, commit `dbe9355110bb36cc16b91d9fdf11f52a31317c76` (`dbe9355`).  
**Baseline status:** worktree was clean before `_chatgpt_project_source/` was created. The source-pack files are the only intended new working-tree content from this task.  
**Current editorial basis:** newsroom date 1981-07-20; completed-game cutoff 1981-07-19.

## Inventory

Lengths are approximate and were measured before this manifest was added.

| File | Purpose | Approximate length | Essential? | Warning |
|---|---|---:|---|---|
| `00_START_HERE.md` | Project identity, active context, priorities, reading order | 57 lines / 505 words | Yes | Read first |
| `01_CURRENT_STATE.md` | Current cutoff, deliverables, readiness, stale exclusions, next step | 61 lines / 544 words | Yes | “Safe for EB use” is not proof of publication |
| `02_REPO_MAP.md` | Repository structure, path roles, naming, ignore guidance | 61 lines / 633 words | Yes | README contains stale branch language |
| `03_RECENT_CODEX_OUTPUTS.md` | Recent commit/output evidence and upload safety | 35 lines / 520 words | Yes | Codex authorship is inferred, not proven by git author name |
| `04_DATA_DICTIONARY.md` | Source/report families, authority, key fields and baseball terms | 63 lines / 1,082 words | Yes | StatsPlus and sortable data are supplemental |
| `05_STORY_ENGINE.md` | Editorial hierarchy and data-to-story method | 72 lines / 618 words | Yes | Tournament implications require verified rules |
| `06_STYLE_AND_VOICE_GUIDE.md` | Naming, EB voice, formats, factual restraint, examples | 69 lines / 712 words | Yes | Preserve **Real. Fictional. ⚾.** exactly |
| `07_PRODUCTION_RHYTHM.md` | End-to-end production workflow and cadence | 35 lines / 639 words | Yes | Publication steps are not proven complete |
| `08_ACTIVE_FILES_TO_UPLOAD.md` | Minimum/full upload lists and exclusions | 43 lines / 529 words | Yes | Do not upload raw/binary/bulk artifacts |
| `09_OPEN_QUESTIONS_AND_RISKS.md` | Unknowns, contradictions, stale-data and inference risks | 29 lines / 641 words | Yes | Postseason rules are narrowed; clinch/field claims still need arithmetic |
| `10_PROJECT_SOURCE_MANIFEST.md` | Inventory, provenance, and usage instructions | About 55 lines | Yes | Regenerate after material repo/current-state changes |
| `docs/ABL_POSTSEASON_RULES.md` | Versioned ABL postseason qualification and DCS seeding authority | New rules authority | Yes for tournament claims | Qualification known; 1981 change is matchup protection only |

## How to use this source pack

These files are meant to be uploaded to ChatGPT Project Source so ChatGPT can understand the ABL/SBV repository without receiving the whole repository. Upload all 11 pack files. Then add only the small, current editorial deliverables recommended in `08_ACTIVE_FILES_TO_UPLOAD.md`.

When asking ChatGPT for work:

1. State whether the task is current July 20 coverage or an explicitly historical task.
2. Name the desired format (Observer, Ballpark Feed, Daily 411, forum, article, or video).
3. Supply a newer preflight/evidence packet if the cutoff has advanced.
4. Require source labels for official facts versus model enrichment.
5. Ask ChatGPT to identify missing evidence instead of filling gaps.

## Inspection basis

The pack was produced after inspecting top-level structure, README and documentation, Markdown outputs, raw/generated data families, scripts/configuration context, git status and eight recent commits, recently modified files, current-state preflight, source/run manifests, production scripts, Chicago–Dallas evidence, StatsPlus current-source status, and Feed 3 validation/runbook material.

## Pack-wide warnings

- Current means **as of games completed July 19, 1981**, not the host date and not July 20 results.
- Week 5 and the July 12 Week 15 snapshot are historical/superseded for current work.
- Raw OOTP is authoritative for completed games and standings; StatsPlus preview does not change official ranking.
- Postseason qualification is known by legacy ABL continuity: three division winners plus one wild card per conference. The 1981 change affects DCS seeding/matchups only; use `docs/ABL_POSTSEASON_RULES.md` and do not infer clinches, eliminations, or official fields without standings arithmetic.
- Secrets were not copied. `.env.example`, raw bulk data, binary data, caches, logs, temporary files, and account information are excluded.
