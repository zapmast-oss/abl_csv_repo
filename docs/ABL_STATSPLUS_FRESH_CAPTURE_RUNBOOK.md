# ABL StatsPlus Fresh-Capture Runbook

## How to stage a fresh capture

1. Choose the intended season and as-of date.
2. Confirm raw OOTP current-state preflight has passed for that cutoff.
3. Create a dedicated external or future `csv/statsplus/staging/` capture folder. Do not use `current/`.
4. Export the complete StatsPlus batch into that one folder, retaining report names and any capture notes.
5. Do not mix legacy files, story outputs, manually edited tables, or prior captures into the folder.

The proposed repo source architecture is `csv/statsplus/legacy/`, `staging/`, and `current/`, but this runbook does not authorize creating or populating them.

## Why the legacy baseline is preserved

The legacy 27-table capture and six staff division tables document prior schemas, table populations, duplicate sort views, and Deep Dive 25 design intent. Their early-1981 values are stale. Preserve them unchanged so reconciliation remains reproducible.

## Run profiling

From the repo root:

```powershell
python csv/abl_scripts/z_abl_statsplus_fresh_capture_profile.py `
  "C:\path\to\fresh_statsplus_staging" `
  --season 1981 `
  --as-of-date 1981-07-19
```

The script is read-only with respect to the staging folder. It writes generated reports under `csv/out/control/`.

## Review order

1. **Manifest:** Confirm file count, extensions, readability, hashes, sizes, timestamps, rows, columns, and headers.
2. **Fresh profile:** Review inferred family, grain, value, authority class, duplicates, and warnings.
3. **Legacy reconciliation:** Review exact/renamed matches, missing legacy families, new tables, schema drift, and duplicate/overlap tables.
4. **Authority crosscheck:** Confirm which fields are StatsPlus-specific, crosscheck-only, or held.
5. **Story enrichment plan:** Review which sources could enable or strengthen signals and which remain disabled.

Filename equality is not enough. A table match requires structural/content evidence. A renamed table can be accepted when its headers, grain, keys, and content population establish identity.

## Conflict handling

- Raw OOTP governs completed games, scores, logs, played state, and current records.
- Promoted sortable stats govern overlapping current enrichment within their accepted schemas.
- StatsPlus governs only validated StatsPlus-specific values after promotion.
- Record both sides of any conflict. Do not edit a source file to force agreement.
- Unknown cutoff, unsafe identity, or unresolved schema drift means hold—not guess.

## High-value StatsPlus sources

- playoff odds and remaining-schedule simulations;
- ELO and recent ELO movement;
- BaseRuns expected performance;
- team WAR;
- injury burden;
- team/player baserunning, including wSB where validated;
- historical/current fan interest and fan data;
- owner, financial, and organization context;
- best batting/pitching game rankings;
- Grand Tournament of Champions history.

All model fields must be identified as model output. Injury, staff, fan, and financial signals require compatible dates and safe identity linkage.

## Promotion gate

Promotion is safe only after:

- capture/as-of metadata is documented;
- all files are checksummed and readable;
- expected tables and omissions are known;
- families, grains, keys, and identities are validated;
- schema drift and summary rows are resolved;
- duplicate views receive a keep/hold decision;
- raw and sortable authority conflicts are resolved by rule;
- model semantics and field provenance are documented;
- promotion recommendations receive explicit approval.

## After profiling

1. Correct capture omissions in a new staging batch; do not patch exports manually.
2. Document accepted schema adaptations.
3. Create a promotion dry run with exact source/destination mappings and checksums.
4. Request separate authorization to promote.
5. After promotion, update source registries and the authoritative catalog only under explicit instruction.
6. Run story-engine integration tests only after source promotion and compatibility checks.

## Do not do before promotion

- Do not copy files into `csv/statsplus/current/`.
- Do not overwrite legacy, raw OOTP, or sortable sources.
- Do not use fresh StatsPlus values in current story outputs.
- Do not enable manager tendencies from name-only staff data.
- Do not treat standings, odds, ELO, BaseRuns, or WAR as interchangeable facts.
- Do not silently discard conflicts, missing tables, duplicate views, or schema drift.
