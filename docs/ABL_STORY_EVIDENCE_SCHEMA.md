# ABL Story Evidence Schema — Sprint 1

Version: `1.0-sprint1`

Evidence is stored separately from candidates so detectors can emit different metrics without changing the candidate schema. One row represents one source-backed observation.

## Required columns

| Column | Type | Rule |
|---|---|---|
| `evidence_id` | string | Unique within the run |
| `candidate_id` | string | Foreign key to the candidate file |
| `metric` | string | Machine-readable metric name |
| `value` | string/decimal | Observed value as represented by the source |
| `comparison` | string | Baseline, rank, opponent, prior value, or threshold context |
| `source_file` | path | Repository-relative CSV path listed in the catalog |
| `source_row_key` | string | Reproducible key such as `team_abbr=LV` |
| `as_of_date` | ISO date | Evidence cutoff |
| `notes` | string | Interpretation limits or calculation description |

## Invariants

- Every candidate has at least one evidence row.
- Every evidence row resolves to exactly one candidate.
- `source_file` must exist and appear in `csv/out/docs/abl_data_catalog.csv`.
- Derived comparisons must name their calculation in `notes`.
- Future schedule rows are allowed only for matchup/game candidates and must be marked as scheduled, not completed.

## Sprint 1 output

`csv/out/story/candidates/story_evidence_1981_week_15.csv`
