# ABL StatsPlus Legacy Baseline Notes

## Recommended architecture

```text
csv/statsplus/legacy/   # immutable baseline copies only after explicit authorization
csv/statsplus/staging/  # dated, unpromoted captures
csv/statsplus/current/  # reconciled and explicitly promoted tables
csv/out/control/        # manifests, profiles, reconciliations, decisions
csv/out/archive/        # pre-promotion generated archives
docs/                   # intake contracts and authority rules
```

Do not create or populate these source folders until explicitly authorized.

## Governance

- Preserve the inspected external folder unchanged as the legacy baseline.
- Give each staged capture a season, as-of date, export timestamp, report identity, row/column counts, and SHA-256.
- Compare every new table against the 27-table baseline, six staff schemas, raw OOTP authority, and promoted sortable stats.
- Detect summary rows, HTML remnants, renamed columns, duplicate sort views, missing IDs, and model-field semantic changes.
- Label odds, ELO, BaseRuns, and WAR as enrichment, never game proof.
- Keep Deep Dive 25 prompts/narrative separate from factual source tables.
- Require a validated staff-ID/name bridge before enabling manager tendencies.
- Require same-cutoff validation before enabling StatsPlus baserunning or injury signals.

## Next Codex task

Create `docs/ABL_STATSPLUS_INTAKE_CONTRACT.md` and a read-only staging manifest/reconciliation runner. Inventory a fresh capture, compare it table-by-table with this baseline and current authorities, and produce promotion recommendations without promoting files.
