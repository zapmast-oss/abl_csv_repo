# StatsPlus / Deep Dive 25 Legacy Baseline Report

## Verdict

`E:\BACKUP\BACKUP 2\ABL 1981` is a coherent prior capture and should be preserved unchanged as a legacy baseline. It is stale for July 19 current-state use: standings examples are roughly 19 games into 1981 and no exact as-of date is encoded.

## Counts

| Measure | Count |
|---|---|
| Total Files | 36 |
| Csv Files | 34 |
| Txt Files | 2 |
| Folders | 1 |
| Likely Statsplus Tables | 27 |
| Coach Staff Division Files | 6 |
| Deep Dive 25 Txt Files | 2 |

- All 27 numbered tables: **present**.
- All six coach/staff division files: **present**.
- Deep Dive 25 TXT files: **2 present**.
- Extra CSV: `Phoenix_Firebirds_Season_History.csv` (derived historical context).

## Coverage

The capture covers owners, staff, finances, fan data/history, standings, playoff odds, BaseRuns, ELO, team WAR, injuries, team/player batting, pitching, fielding and baserunning, team age, best games, and Grand Tournament history.

Highest story-engine value lies in labeled model enrichment (odds/ELO/BaseRuns/WAR), injury and baserunning context, fan/owner pressure, best-game discovery, and tournament history. The legacy values themselves are not current proof.

## Limits and duplicates

 Team batting 13/14 and pitching 15/16 are byte-identical pairs. Fielding 17/18 contain identical row multisets in different order. The six division staff files exactly reconstruct the 240-row master, but they lack stable IDs and cannot resolve current staff linkage alone. TXT files are generated design/reference material.

## Recommendation

Preserve all 36 files as the baseline. It is safe to proceed to a new StatsPlus staging/reconciliation task, but promotion must remain separate. Compare new schemas, identities, grains, cutoff dates, hashes, duplicate views, and model semantics against this baseline and current governed OOTP/sortable sources.
