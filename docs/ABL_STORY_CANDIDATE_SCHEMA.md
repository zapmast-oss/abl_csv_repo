# ABL Story Candidate Schema — Sprint 1

Version: `1.0-sprint1`

A candidate is a factual, ranked story possibility. It is not a finished claim and does not predict an outcome. One row represents one signal about one primary subject at one as-of date.

## Required columns

| Column | Type | Rule |
|---|---|---|
| `candidate_id` | string | Stable within the run; `<week_label>__<signal_type>__<subject>` |
| `run_id` | string | Identifier shared by all Sprint 1 outputs |
| `week_label` | string | Current test case is `1981_week_15` |
| `as_of_date` | ISO date | Data cutoff; current default is `1981-07-12` |
| `season` | integer | `1981` for this engine |
| `hierarchy_level` | enum | `tournament`, `standings`, `race`, `fans`, `management`, `players`, `team`, `game` |
| `signal_type` | string | Detector name such as `division_race` or `ace_performance` |
| `subject_type` | string | `division`, `team`, `player`, `manager`, `matchup`, or `historical_link` |
| `subject_id` | string | Source-system ID or deterministic composite key |
| `subject_name` | string | Human-readable primary subject |
| `related_subjects` | pipe-delimited string | Other teams/players central to the signal |
| `headline_factual` | string | Neutral statement of the observed condition |
| `stakes` | string | Why the next week/game matters; no invented emotion or outcome |
| `evidence_summary` | string | Short quantified explanation supporting the candidate |
| `source_files` | pipe-delimited paths | Every CSV directly supporting the row; required and repository-relative |
| `signal_score` | decimal 0–100 | Deterministic editorial ranking score |
| `confidence` | enum | `high`, `medium`, or `low`, based on source fit and sample |
| `status` | enum | Sprint 1 emits `candidate`; later workflow may use `selected`, `held`, `rejected` |

## Invariants

- `source_files` and `evidence_summary` must be non-empty.
- Each source path must exist and appear in the authoritative data catalog.
- Numbers in headlines, stakes, and summaries must be recoverable from evidence rows.
- Current-week candidates may not use a source whose detectable coverage extends beyond the as-of date unless the candidate explicitly concerns a future scheduled game.
- Hierarchy controls editorial ordering, but evidence quality can suppress a higher-level candidate.
- The engine describes pressure and stakes; the game remains unresolved.

## Sprint 1 output

`csv/out/story/candidates/story_candidates_1981_week_15.csv`
