# ABL Sortable-Stats Reconciliation Runbook

## Why captures enter staging

OOTP report exports are supplemental captures, not manually maintained tables. A new capture can change filenames, presets, columns, row universes, and report scope. Staging preserves both the official source set and the captured files while those differences are measured. Nothing is overwritten until a reviewed promotion plan and batch manifest exist.

## Why filenames are not authoritative

An exact filename can hide a schema change. A renamed file can preserve the same report. Two differently named presets can expose overlapping subsets of one broader report. Reconciliation therefore prioritizes normalized headers, header containment/Jaccard similarity, column counts, inferred subject area, grain, and row counts. Normalized filename similarity contributes confidence but cannot decide promotion alone.

## Reconciliation command

```powershell
python csv/abl_scripts/z_abl_sortable_stats_reconcile_capture.py
```

Optional `--existing` and `--staging` arguments allow another capture to be assessed without changing the script.

The script reads both folders and writes only to `csv/out/control/`. It does not promote, rename, move, delete, or replace source files.

## Recommendation meanings

- `PROMOTE_REPLACE_EXISTING`: proposed successor to a known source, normally requiring an exact filename/schema match or a high-confidence renamed structural match.
- `PROMOTE_AS_NEW_SOURCE`: distinct recognized sortable report that may deserve a new registry entry and curated contract.
- `HOLD_DUPLICATE_OR_OVERLAP`: substantially duplicates or subsets another report; retain in staging until consumer need and precedence are decided.
- `HOLD_UNKNOWN_REVIEW`: schema change, unclear subject/grain, or missing replacement requires human review.
- `IGNORE_NOT_ABL_SORTABLE_STATS`: not recognized as a valid ABL sortable-stat CSV.

Recommendations are proposals, not file operations.

## Promotion prerequisites

Before promotion:

1. Create a capture batch ID, capture timestamp, intended as-of date, file list, sizes, and SHA-256 checksums.
2. Validate every file's header, row count, duplicate grain, numeric parsing, and player/team join coverage.
3. Link the sortable batch to a raw OOTP batch that independently proves the current game cutoff.
4. Review exact-name schema changes and all renamed matches column by column.
5. Resolve overlap precedence: canonical broad report, required subset view, or rejected duplicate.
6. Confirm that missing official files were captured elsewhere; absence from a player-statistics folder never means retirement.
7. Update the source registry and intake contract for approved new sources.
8. Produce a dry-run promotion manifest and request explicit promotion authorization.

## ABL data-entry rhythm

```text
OOTP export/capture
    -> staging
    -> reconciliation
    -> capture batch manifest and checksums
    -> validation against raw current-state batch
    -> reviewed promotion plan
    -> explicit promotion
    -> registry/catalog refresh
    -> curated-layer rebuild
```

This is controlled capture and promotion, not manual row entry. When a capture is partial or ambiguous, keep it staged and disable dependent current analysis rather than guessing.

