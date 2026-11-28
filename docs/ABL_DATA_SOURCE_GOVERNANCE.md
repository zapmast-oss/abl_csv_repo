# ABL Data Source Governance

## Doctrine

**ABL data entry is not manual row entry; it is controlled capture, validation, and promotion of OOTP-derived source extracts into a curated baseball intelligence layer.**

The objective is reproducibility. A claim about current baseball state must be traceable to captured files, a batch, validation results, a detected cutoff, and curated tables. Editorial products consume that layer; they do not establish it.

## Data architecture

### Raw OOTP CSV exports

`csv/ootp_csv/` is the system-of-record extract from the ABL saved game. Its dated game, schedule, and result tables are the only source family allowed to prove the latest completed game date, results, and games per team. Other raw files—players, teams, rosters, contracts, ratings, and history—support the same capture when tied to its batch.

### Sortable-stat report extracts

`csv/abl_statistics/` contains supplemental OOTP report captures. The family mixes volatile season statistics with slower-moving ratings, indicative attributes, staff data, finances, personality, and park context. These reports enrich a current state already proved by raw games. They cannot independently advance the league cutoff because they generally lack reliable capture dates.

### Generated repository outputs

`csv/out/`, root `out/`, and legacy derived zones contain rebuildable products. They must carry lineage and as-of metadata before downstream use. They are never source of truth, even when their filename says `current`.

### Historical almanac

`csv/out/almanac/` is historical context organized by season. It supports flashbacks, precedents, champions, and historical echoes. It cannot prove the current league state.

### Editorial intelligence

Story dictionaries, menus, candidate prototypes, configuration CSVs, and documentation encode editorial taxonomy, rules, and presentation. They are intelligence about how to observe baseball, not source baseball data.

## Processing model

```text
source capture
    -> source registry
    -> batch manifest
    -> validation
    -> curated current tables
    -> story engine outputs
```

1. **Source capture:** copy/export a coherent OOTP source family without hand-editing rows.
2. **Source registry:** classify authority, subject, volatility, validation, joins, and allowed uses.
3. **Batch manifest:** bind captured files to time, intended cutoff, checksums/sizes, and validation status.
4. **Validation:** prove schemas, coverage, keys, joins, dates, and games-per-team consistency.
5. **Curated layer:** normalize compatible sources into dated, tested contracts.
6. **Story outputs:** consume curated data and retain evidence lineage.

## Promotion gates

A source is promoted only when its path is registered, its file is present in a capture batch, validation passes, and its authority permits the proposed use. Raw games establish current state. Supplemental reports inherit—but never extend—the validated batch cutoff. Derived output cannot be promoted back into source authority.

If a source is missing, stale, undated, or inconsistent, the dependent metric is disabled. Governance should reduce output rather than fill gaps with manual overlays or assumptions.

