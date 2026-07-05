# ABL Feed 3 Evidence Preview Runbook

## Purpose

Feed 3 evidence preview appends typed StatsPlus evidence to a completed story-engine run. It writes only preview artifacts under `csv/out/story/enrichment/` and validation artifacts under `csv/out/control/`.

The governing rules are:

- StatsPlus enhances. It does not replace.
- StatsPlus annotates before it ranks.
- Raw OOTP remains authoritative for completed games, scores, logs, standings proof, and current-state dates.

## What preview does

- Reads already-generated candidates and evidence.
- Reads promoted Feed 3 files and the signal dictionary.
- Creates typed, source-labeled evidence records.
- Creates a combined evidence preview without replacing official evidence.
- Hashes protected official artifacts before and after execution.
- Reports feature-flag state and validation results.

## What preview does not do

- It does not run source capture or the main story engine.
- It does not overwrite candidates, evidence, menu, editorial board, slate, production package, or segment script.
- It does not adjust candidate scores or order.
- It does not create candidates.
- It does not automate owner/front-office traits, best-game discovery, or historical fan interpretation.
- It does not treat StatsPlus standings as official.

## Feature flags and defaults

| Flag | Default | Effect |
|---|---:|---|
| `enable_feed3_evidence_preview` | `true` | Runs the append-only adapter. |
| `enable_feed3_rank_adjustment` | `false` | Must remain false in preview mode. |
| `enable_feed3_new_candidates` | `false` | Must remain false in preview mode. |
| `enable_owner_front_office_signals` | `false` | Keeps traits manual/reference-only. |
| `enable_best_game_discovery_signals` | `false` | Keeps ranked-game tables out of automated evidence/ranking. |
| `enable_historical_fan_interpretation` | `false` | Keeps historical interpretation editorial. |

Ranking changes remain disabled because Feed 3 models and context have not been calibrated against the story hierarchy, freshness, duplication, and editorial outcomes across multiple newsroom runs. Annotation can be audited without changing editorial order.

## Validate the July 20 run

From the repository root:

```powershell
python csv/abl_scripts/z_abl_story_engine_with_feed3_preview_runner.py
```

The wrapper reads `csv/out/control/story_engine_feature_flags_1981_07_20_asof_1981-07-19.json`, verifies protected files, invokes only the Feed 3 adapter, and writes `feed3_preview_flag_validation_1981_07_20_asof_1981-07-19.*` under `csv/out/control/`.

A passing validation requires:

- preview enabled;
- evidence append and combined preview present;
- all 15 candidates enriched;
- ranking effects equal `none`;
- zero new candidates;
- protected official hashes unchanged;
- all reference-only automation flags false.

## Validate a future newsroom date

1. Complete and verify raw OOTP, sortable, and Feed 3 source capture for the new cutoff.
2. Run the official story engine without Feed 3 rank adjustment.
3. Create a date-specific feature-flag file with the same safe defaults.
4. Update or parameterize the adapter inputs for the new newsroom/as-of pair.
5. Run the wrapper in preview mode.
6. Compare evidence counts, candidate coverage, authority cautions, duplicate signals, and protected-file hashes.
7. Review editorial usefulness before integrating the wrapper into recurring cadence.

## Before enabling Feed 3 ranking changes

Review at least:

- multiple newsroom-date backtests;
- scale and normalization across odds, ELO, BaseRuns, WAR, injuries, fan, and financial signals;
- double-counting between related model fields;
- interaction with the story hierarchy and existing score;
- cutoff compatibility and missing-data behavior;
- false-positive, sensationalism, and editorial override rates;
- explicit rollback and feature-flag behavior.

Ranking changes require separate authorization and a versioned scoring specification.

## Before enabling Feed 3 new candidates

Define and test:

- candidate schemas and deduplication against raw-derived candidates;
- minimum evidence and authority thresholds;
- model-only versus game-proven candidate labels;
- stable team/player identity linkage;
- handling for reference-only signals;
- editorial review and hold rules;
- regression tests proving existing candidates are not displaced silently.

New-candidate creation requires separate authorization. It must never infer games, standings, motives, collapse, or destiny from Feed 3.
