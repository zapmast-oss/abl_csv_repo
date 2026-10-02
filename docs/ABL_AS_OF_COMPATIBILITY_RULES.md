# ABL As-Of Compatibility Rules

## Active state

- Season: `1981`
- Coverage label: `1981_week_15`
- As-of date: `1981-07-12`
- Expected completed regular-season games per ABL team: `89`
- Current authority: raw, date-filterable OOTP exports

## Definition

A source is **as-of compatible** when every performance fact it contributes represents information available on or before the active cutoff, its season and league match the run, and any team-game coverage agrees with the expected 89-game state. A source may extend beyond the cutoff only when the signal explicitly concerns a future scheduled game; those rows must be unplayed and must be labeled as scheduled.

Compatibility is evaluated per source and per intended use. A valid file is not automatically valid for the active run.

## Source selection

1. Prefer raw OOTP sources with dates and filter completed league-200 regular-season games through the cutoff.
2. Recompute standings, trailing-window movement, run balance, and matchup context from those rows.
3. Use coordinated raw cumulative player totals only for the active season, league, major-league level, total split, and minimum workload.
4. Use dimensions only for identity and structure.
5. Use historical data only as explicitly labeled context.
6. Require every used path to exist in the authoritative catalog. The story engine reads but never rewrites that catalog.
7. Record both used and rejected alternatives in the source manifest.

## Why filenames are insufficient

Names such as `current`, `weekly`, or `1981` do not encode a reliable cutoff. This repository contains 32-game, 62-game, and 89-game products simultaneously. A file called `fact_manager_scorecard_1981_current.csv`, for example, still carries a 32-game standings state. Compatibility must come from row-level dates, wins plus losses, explicit game counts, coordinated-export lineage, or a declared static/historical role.

## Preserved early-season snapshots

Preserved Week 5 and other early-season candidates, menus, standings, weekly changes, and player facts remain valid archival artifacts. They are assigned `excluded_stale_snapshot` or `excluded_wrong_games_count` for the active Week 15 run. They must not contribute candidates, evidence, ranking, or packet prose. Their mention in a manifest is an exclusion record, not active use.

## Static dimensions

Team, division, and player identity files may be `compatible_static` when used only for names, abbreviations, IDs, league membership, and division membership. Static compatibility does not authorize ratings, morale, injury, strategy, or other time-varying fields from the same physical file without a separate cutoff assessment.

## Historical almanac files

Prior-season champion and almanac files may be `compatible_historical` when the candidate is explicitly a historical echo. Historical rows must be paired with current evidence, labeled by season, and never treated as a forecast or substitute for current standings.

## Stale derivative reports

Derivative reports must prove the same cutoff as the active run. Reports at another game count are `excluded_wrong_games_count`; reports known to be preserved early snapshots without sufficient date metadata are `excluded_stale_snapshot`; otherwise undated performance reports are `excluded_missing_date`. Recomputing from raw data is preferred to guessing a derivative's state.

## Manager, division, and rotation reports

The available manager tendencies, division leverage, and rotation stability reports stop at 62 games, while the manager scorecard is tied to a 32-game standings snapshot. They are `disabled_incompatible`. Management, manager-tendency, division-leverage, and rotation-stability signals remain disabled until these datasets are rebuilt and validated at 89 games through July 12, 1981.

## Disable instead of guess

A signal must be disabled when its required source is missing, undated and time-varying, from the wrong season/league, at the wrong game count, internally inconsistent, or dependent on unverified rules or status. The engine must not interpolate manager behavior, assume an injury/probable starter, borrow a stale ranking, or translate a historical result into a current claim. Missing evidence reduces output; it does not license invention.

## Status vocabulary

- `active_current`: current facts aligned to the active cutoff.
- `compatible_static`: structural identity data safe for the declared join.
- `compatible_historical`: explicitly historical context.
- `excluded_stale_snapshot`: preserved snapshot from an earlier state.
- `excluded_wrong_games_count`: detected coverage differs from 89 games.
- `excluded_missing_date`: time-varying source cannot prove a cutoff.
- `disabled_incompatible`: dependent signal family is disabled until compatible data exists.

Every run writes CSV, JSON, and Markdown source manifests under `csv/out/story/manifests/`. A mismatch in the authoritative raw games-per-team check stops candidate generation.

