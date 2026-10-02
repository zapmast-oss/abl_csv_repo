# ABL Batch Manifest and Current-State Preflight Runbook

## Purpose

The batch manifest inventories the current raw OOTP and promoted sortable capture with paths, authority, sizes, timestamps, shapes, hashes, current-state permissions, driver coverage, promotion status, and accepted drift. It does not change source files or the authoritative catalog.

The preflight validator answers one question: can the approved raw driver set prove a coherent ABL state through the target completed-game date? Sortable reports support the validated state but cannot establish its date.

## Commands

```powershell
python csv/abl_scripts/z_abl_batch_manifest_build.py
python csv/abl_scripts/z_abl_current_state_preflight_validate.py
```

## Files read

- `csv/out/control/abl_source_registry.csv`
- `csv/out/control/sortable_promotion_verification_1981_asof_1981-07-19.csv`
- `csv/ootp_csv/*.csv` registered as raw sources
- `csv/abl_statistics/*.csv` registered as sortable sources
- Current-state drivers: `games.csv`, `games_score.csv`, `game_logs.csv`
- `teams.csv` only to label and verify the 24-team universe

Historical almanac, generated reports, candidates, menus, and packets are excluded from current-state proof.

## Files written

- `csv/out/control/abl_batch_manifest_current.{csv,md,json}`
- `csv/out/control/current_state_preflight_1981_target_1981-07-19.{csv,md,json}`

## Verdicts

- `READY_FOR_CURRENT_RUN`: drivers reach the target, all 24 teams are represented, per-team counts have at most a one-game schedule spread, and score/log game-ID coverage is complete. Story work may resume using the validated cutoff and documented enrichment limits.
- `NOT_READY_MISSING_TARGET_DATE`: latest completed game precedes the target. Refresh raw OOTP exports and rerun both tools.
- `NOT_READY_INCONSISTENT_GAME_COUNTS`: target date exists, but team coverage/counts or driver reconciliation is suspicious. Investigate incomplete exports, postponed games, league/game-type filtering, and game-ID gaps before rerunning.
- `NOT_READY_NO_CURRENT_STATE_DRIVER`: one or more approved driver files are missing/unreadable. Restore a coherent raw export batch.
- `NOT_READY_NO_DATE_DETECTED`: drivers exist but no reliable completed ABL regular-season date can be derived. Validate schema, completion flags, league ID, and game type.

## Date rules

For this run, `newsroom_date` is July 20, 1981 and `target_as_of_date` is July 19, 1981. The validator does not require July 20 games. Future scheduled games never advance the completed-game cutoff.

## After READY

Freeze/reference the manifest and preflight outputs in the next run manifest. Rebuild curated current tables from the validated raw cutoff and promoted sortable support. Respect accepted schema drift and keep unavailable enrichments disabled. Story generation may then resume in a separate authorized task.

## After NOT_READY

Do not patch results manually and do not substitute generated reports. Correct or recapture the authoritative source family named by the verdict, create a new batch/checksum state, and rerun manifest plus preflight.

