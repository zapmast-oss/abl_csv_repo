# ABL Batch Manifest Design

## Purpose

A batch is an immutable description of a controlled capture or generation event. Typical batches are a raw OOTP CSV export, a sortable-stat report capture, or an almanac generation. The manifest establishes which files belong together; it does not make an unvalidated batch current.

## Required fields

| Field | Meaning |
|---|---|
| `batch_id` | Stable identifier such as `ootp_csv_19810719T230000` |
| `source_family` | `ootp_csv`, `sortable_stats`, or `historical_almanac` |
| `capture_datetime` | Actual extraction/capture timestamp with timezone |
| `intended_as_of_date` | Baseball cutoff the operator expects; subject to validation |
| `newsroom_date` | Editorial operating date; may be later than the cutoff |
| `file_count` | Number of files in the batch |
| `files_included` | Ordered file list or child-manifest reference |
| `row_count_summary` | Per-file counts and total count |
| `checksum_or_file_size_summary` | Prefer SHA-256 per file; size is a minimum fallback |
| `validation_status` | `captured`, `validating`, `valid`, `invalid`, or `quarantined` |
| `latest_game_date_detected` | Derived only from authoritative completed raw games |
| `games_per_team_detected` | Distribution/range, not merely one expected value |
| `notes` | Operator notes and exceptions; never substitute for validation |

## Recommended child file record

Use one child row per file: `batch_id`, `source_id`, `file_path`, `size_bytes`, `sha256`, `row_count`, `schema_hash`, `capture_datetime`, `validation_status`, `validation_errors`.

## Validation and promotion

- Raw batch: reconcile league, game type, completion status, scores, dates, and per-team game counts.
- Sortable batch: require all expected reports, consistent team/player universes, schema signatures, and a declared relationship to a validated raw batch.
- Almanac batch: validate season/league identity, completeness, and source lineage.
- Promotion writes a curated-layer run manifest referencing the source batch IDs and validation reports.
- Files are never edited in place to make a batch pass. A corrected capture receives a new batch ID.

## Present target

The newsroom target is `1981-07-20`, with intended target as-of `1981-07-19`. Those values remain targets until a new raw OOTP batch proves that the latest completed games are July 19 and reconciles games per team.

