# ABL StatsPlus Intake Contract

> **StatsPlus enhances. It does not replace.**

## Purpose

StatsPlus is ABL Feed 3. It supplies enrichment, projections, model outputs, fan and financial context, and Deep Dive texture. It does not replace raw OOTP proof or promoted sortable-stat enrichment.

Feed 3 is intentionally expandable. Advanced batting, advanced pitching, standardized metrics, matchup matrices, or other supplemental tables may be added later, but every new family must pass staged intake, profiling, classification, conflict review, and separate promotion. Existing Feed 3 authority never transfers automatically to a newly added table or field.

Every StatsPlus capture enters staging, is profiled and reconciled, and receives an explicit promotion decision before it can become a current source.

## Source architecture

The proposed structure is:

```text
csv/statsplus/
├── legacy/     # immutable legacy baseline after separately authorized intake
├── staging/    # dated fresh captures awaiting reconciliation
└── current/    # explicitly promoted StatsPlus sources

csv/out/control/  # manifests, profiles, reconciliations, and decisions
csv/out/archive/  # pre-promotion generated archives
docs/             # contracts, authority rules, and runbooks
```

This contract does not authorize creating, copying, moving, or populating those source directories.

## Required capture metadata

A staged capture must have:

- season;
- target as-of date;
- capture timestamp;
- source location or StatsPlus report identity where available;
- a bounded staging root containing only the intended capture;
- a file manifest with relative paths, sizes, modified timestamps, checksums, readability, row counts, column counts, and headers;
- an operator note for incomplete or manually assembled captures.

The staging directory must never be `csv/statsplus/current/`, a raw OOTP folder, a promoted sortable folder, or an output folder.

## Table identity

Filenames are evidence, not identity. OOTP and StatsPlus exports can change names while preserving content, or reuse names while changing schemas.

Table identity is inferred from all of:

- filename and normalized filename tokens;
- column headers and normalized header set;
- column count;
- row count and expected population;
- inferred grain;
- key columns;
- sampled content;
- checksum and duplicate relationships;
- similarity to the governed 27-table legacy baseline.

A renamed table requires structural agreement. A filename match with material schema drift must be reported as drift, not accepted silently.

## Legacy baseline

`E:\BACKUP\BACKUP 2\ABL 1981` is the read-only legacy baseline. It is coherent but stale, approximately 19 games into 1981. It establishes prior table families and schemas only.

Legacy values must not be promoted as July 19 current-state values. The two Deep Dive TXT files are historical design/reference material, not factual authority. The six division staff files lack stable staff IDs.

## Intake workflow

1. Place a fresh capture in a dedicated staging folder.
2. Run `z_abl_statsplus_fresh_capture_profile.py` with the staging path, season, and as-of date.
3. Review the manifest for unreadable, unexpected, duplicate, or out-of-scope files.
4. Review legacy reconciliation for renamed tables, missing tables, new tables, schema drift, and duplicate sort views.
5. Review the authority crosscheck against raw OOTP and promoted sortable stats.
6. Confirm that current-like tables match the target cutoff. A schema match does not prove freshness.
7. Decide promotion table by table and field family by field family.
8. Archive the pre-promotion manifest and reconciliation.
9. Promote only under a separate explicit instruction.

## Feed boundaries

- Raw OOTP proves completed games, scores, game logs, schedule state, and current-state cutoff.
- Promoted sortable stats support current player, team, staff-name, financial, park, and related enrichment within their governed fields.
- StatsPlus supplies StatsPlus-specific values and explicitly labeled model/context fields.
- StatsPlus standings are a crosscheck unless separately authorized.
- Playoff odds, ELO, BaseRuns, and StatsPlus WAR are model/enrichment values, not standings facts or game proof.

StatsPlus may enrich a fact proved elsewhere. It must not overwrite, backfill, or silently contradict raw OOTP proof.

## Conflict policy

Conflicts are recorded with both values, both sources, cutoff information, and the applicable authority rule. They are never silently resolved.

- If StatsPlus conflicts with raw OOTP on a completed game, score, schedule state, or record derived from completed games, raw OOTP governs and StatsPlus is held.
- If StatsPlus-specific model values differ from prior StatsPlus values, the fresh value may govern only after cutoff and schema validation.
- If sortable and StatsPlus values overlap, use the field-level authority rule; otherwise hold the field for review.
- Missing dates, ambiguous team/player identity, summary rows, or incompatible grains block promotion for the affected table.

## Promotion gate

A table is safe to promote only when:

- its capture and target dates are documented;
- it is readable and checksummed;
- table family, grain, and key columns are established;
- team/player identities are valid for the ABL population;
- schema drift is accepted or mapped;
- duplicates and summary rows are resolved;
- current-state fields pass raw-OOTP crosschecks;
- overlapping enrichment passes sortable-stat crosschecks;
- model semantics are documented;
- missing expected tables are disclosed;
- a specific promotion recommendation is approved.

Promotion is never implied by successful profiling.
