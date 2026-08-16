# ABL Current-State Preflight

## Purpose

This preflight must pass before any current story-engine run. It separates the editorial target from what the captured data can actually prove.

## Present target

- `newsroom_date`: `1981-07-20`
- `target_as_of_date`: `1981-07-19`
- expected last games played: `1981-07-19`

These are target values, not established repo facts. Do not call the repository current through July 19 until a new raw OOTP capture proves the date and reconciles completed games per team.

## Required proof

1. Register every input in `csv/out/control/abl_source_registry.csv`.
2. Reference immutable raw and supplemental batch manifests.
3. From raw OOTP game/schedule/result data, detect the maximum completed league-200 regular-season game date.
4. Reconcile `games.csv`, `games_score.csv`, and `game_logs.csv` results and completion state.
5. Calculate completed games per team and reject uneven or unexpected coverage unless a documented schedule condition explains it.
6. Rebuild standings from completed games and compare raw `team_record.csv` as a validation check, not as sole date proof.
7. Tie sortable-stat files to their capture batch and compare player/team universes and game counts where present.
8. Promote validated results to curated current tables and write a curated-run manifest.
9. Require the story-engine run to cite registry source IDs and batch IDs.

## Authority boundary

Current-state proof comes from raw OOTP game, schedule, and result data. Sortable stats enrich the current state but do not prove the current date. Almanac data is historical. Generated outputs and story artifacts cannot establish or advance the cutoff.

## Date distinction

`newsroom_date` is when the newsroom is operating. `as_of_date` is the latest completed-game evidence admitted to analysis. The newsroom can be July 20 while the validated baseball cutoff is earlier. Scheduled July 20 games do not make July 20 the result cutoff.

## Fail-closed conditions

Do not run current story generation when the latest raw game date is below the target, result sources disagree, games/team coverage is inconsistent, a required batch manifest is absent, or a dependent supplemental report cannot be tied to the validated batch. Do not create manual result overlays.

