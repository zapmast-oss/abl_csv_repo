# StatsPlus Gamehistory 1978 Postseason Repair Validation

- Repair CSV: `csv/out/statsplus_repair/statsplus_gamehistory_1978_postseason_repair.csv`
- Unresolved-fields CSV: `csv/out/statsplus_repair/statsplus_gamehistory_1978_postseason_unresolved_fields.csv`
- Primary source: `data_raw/ootp_html/almanac_1978.zip` targeted `box_scores/game_box_<id>.html` pages; no bulk extraction performed.
- Mapping sources: `csv/ootp_csv/teams.csv`, `csv/ootp_csv/players.csv`.
- DH setting cross-check: `csv/ootp_csv/sub_leagues.csv` has `designated_hitter=1` for league 200 subleagues, and all 41 targeted 1978 postseason box scores contain DH lineup entries for both teams.

## Dave-Confirmed Field Rules

- `game_type=3` for playoff games.
- `cup=0`.
- Line-score stat orientation is `0 = away` and `1 = home`: `runs0/hits0/errors0` are away-team values; `runs1/hits1/errors1` are home-team values.
- `save_pitcher=0` when no save was credited.

## Row Counts

- Total rows: 41
- DCS: 23
- CCS: 11
- GS: 7

## Audit Summary

- Rows with stat-orientation mismatches before correction: 41
- Rows that matched the old home/away orientation before correction: 41
- `save_pitcher` blank/null count before correction: 41
- Credited saves found in 1978 postseason source boxes: 0
- `save_pitcher=0` count after correction for games with no save: 41

## Validation Results

- PASS: no duplicate `game_id` values.
- PASS: all rows use `league_id = 200`.
- PASS: all rows use `played = 1`.
- PASS: all rows use `game_type = 3`, confirmed by Dave from StatsPlus for playoff games.
- PASS: all rows use `cup = 0`, confirmed by Dave from StatsPlus.
- PASS: all rows use `dh = 1`, verified from 1978 ABL postseason box-score lineups with DH entries for both teams in every game.
- PASS: all 41 rows follow Dave-confirmed stat orientation: `0 = away`, `1 = home`.
- PASS: all no-save games use `save_pitcher=0`.
- PASS: all home/away team IDs exist in `teams.csv`.
- PASS: all populated pitcher/starter IDs exist in `players.csv`.
- PASS: Grand Series scores match the provided summary.
- PASS: Miami defeats Houston in the Grand Series, 4 games to 3.
- PASS: series counts are DCS 23, CCS 11, Grand Series 7.

## Confirmed Fields

- `game_type=3`: confirmed by Dave from StatsPlus for playoff games.
- `cup=0`: confirmed by Dave from StatsPlus.
- `dh=1`: verified from OOTP sources. `csv/ootp_csv/sub_leagues.csv` has `designated_hitter=1` for league 200 subleagues, and every targeted 1978 postseason OOTP box score in `data_raw/ootp_html/almanac_1978.zip` includes a DH lineup row for both teams.
- `runs0/hits0/errors0`: away-team values, confirmed by Dave and audited against OOTP line-score tables.
- `runs1/hits1/errors1`: home-team values, confirmed by Dave and audited against OOTP line-score tables.
- `save_pitcher=0`: used for all 41 games because no credited saves were found in the source pitching decision markers.

## DH Verification Examples

| StatsPlus game_id | Date | Away batting DH row | Home batting DH row |
|---:|---|---|---|
| 1978009280 | 1978-10-03 | B. Rambow DH | E. Griffith DH |
| 1978009336 | 1978-10-03 | R. Thompson DH | A. Rosado DH |
| 1978009307 | 1978-10-04 | B. Rambow DH | E. Griffith DH |
| 1978009337 | 1978-10-04 | R. Thompson DH | A. Rosado DH |
| 1978009343 | 1978-10-04 | R. Chacon DH | B. Rose DH |

## Grand Series Cross-Check

| Date | Away | Away runs | Home | Home runs | OOTP game | StatsPlus game_id |
|---|---:|---:|---:|---:|---:|---:|
| 1978-10-23 | 1 | 6 | 23 | 7 | 9375 | 1978009375 |
| 1978-10-24 | 1 | 5 | 23 | 2 | 9376 | 1978009376 |
| 1978-10-26 | 23 | 7 | 1 | 4 | 9386 | 1978009386 |
| 1978-10-27 | 23 | 4 | 1 | 3 | 9387 | 1978009387 |
| 1978-10-28 | 23 | 3 | 1 | 4 | 9388 | 1978009388 |
| 1978-10-30 | 1 | 2 | 23 | 1 | 9389 | 1978009389 |
| 1978-10-31 | 1 | 6 | 23 | 4 | 9390 | 1978009390 |

## Game List

| Series | Date | Away | Away runs | Home | Home runs | Inn | OOTP game | StatsPlus game_id |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DCS | 1978-10-03 | 2 | 13 | 7 | 3 | 9 | 9280 | 1978009280 |
| DCS | 1978-10-03 | 11 | 6 | 1 | 4 | 9 | 9336 | 1978009336 |
| DCS | 1978-10-04 | 2 | 5 | 7 | 1 | 9 | 9307 | 1978009307 |
| DCS | 1978-10-04 | 11 | 2 | 1 | 4 | 9 | 9337 | 1978009337 |
| DCS | 1978-10-04 | 15 | 1 | 20 | 9 | 9 | 9343 | 1978009343 |
| DCS | 1978-10-04 | 23 | 3 | 18 | 0 | 9 | 9350 | 1978009350 |
| DCS | 1978-10-05 | 15 | 8 | 20 | 3 | 9 | 9344 | 1978009344 |
| DCS | 1978-10-05 | 23 | 0 | 18 | 1 | 9 | 9351 | 1978009351 |
| DCS | 1978-10-06 | 7 | 0 | 2 | 1 | 9 | 9308 | 1978009308 |
| DCS | 1978-10-06 | 1 | 7 | 11 | 6 | 9 | 9338 | 1978009338 |
| DCS | 1978-10-07 | 7 | 2 | 2 | 4 | 9 | 9332 | 1978009332 |
| DCS | 1978-10-07 | 1 | 6 | 11 | 5 | 10 | 9339 | 1978009339 |
| DCS | 1978-10-07 | 20 | 0 | 15 | 2 | 9 | 9345 | 1978009345 |
| DCS | 1978-10-07 | 18 | 2 | 23 | 4 | 9 | 9352 | 1978009352 |
| DCS | 1978-10-08 | 1 | 1 | 11 | 3 | 9 | 9340 | 1978009340 |
| DCS | 1978-10-08 | 20 | 1 | 15 | 3 | 9 | 9346 | 1978009346 |
| DCS | 1978-10-08 | 18 | 4 | 23 | 5 | 9 | 9353 | 1978009353 |
| DCS | 1978-10-09 | 20 | 9 | 15 | 4 | 9 | 9347 | 1978009347 |
| DCS | 1978-10-09 | 18 | 6 | 23 | 1 | 9 | 9354 | 1978009354 |
| DCS | 1978-10-10 | 11 | 0 | 1 | 2 | 9 | 9341 | 1978009341 |
| DCS | 1978-10-11 | 15 | 2 | 20 | 3 | 9 | 9348 | 1978009348 |
| DCS | 1978-10-11 | 23 | 5 | 18 | 1 | 9 | 9355 | 1978009355 |
| DCS | 1978-10-12 | 15 | 5 | 20 | 1 | 9 | 9349 | 1978009349 |
| CCS | 1978-10-13 | 2 | 4 | 1 | 6 | 9 | 9342 | 1978009342 |
| CCS | 1978-10-14 | 2 | 5 | 1 | 4 | 9 | 9356 | 1978009356 |
| CCS | 1978-10-14 | 15 | 0 | 23 | 2 | 9 | 9382 | 1978009382 |
| CCS | 1978-10-15 | 15 | 1 | 23 | 3 | 9 | 9383 | 1978009383 |
| CCS | 1978-10-16 | 1 | 2 | 2 | 5 | 9 | 9377 | 1978009377 |
| CCS | 1978-10-17 | 1 | 5 | 2 | 2 | 9 | 9378 | 1978009378 |
| CCS | 1978-10-17 | 23 | 7 | 15 | 3 | 9 | 9384 | 1978009384 |
| CCS | 1978-10-18 | 1 | 3 | 2 | 4 | 9 | 9379 | 1978009379 |
| CCS | 1978-10-18 | 23 | 3 | 15 | 2 | 9 | 9385 | 1978009385 |
| CCS | 1978-10-20 | 2 | 4 | 1 | 5 | 9 | 9380 | 1978009380 |
| CCS | 1978-10-21 | 2 | 1 | 1 | 2 | 9 | 9381 | 1978009381 |
| GS | 1978-10-23 | 1 | 6 | 23 | 7 | 13 | 9375 | 1978009375 |
| GS | 1978-10-24 | 1 | 5 | 23 | 2 | 9 | 9376 | 1978009376 |
| GS | 1978-10-26 | 23 | 7 | 1 | 4 | 9 | 9386 | 1978009386 |
| GS | 1978-10-27 | 23 | 4 | 1 | 3 | 11 | 9387 | 1978009387 |
| GS | 1978-10-28 | 23 | 3 | 1 | 4 | 9 | 9388 | 1978009388 |
| GS | 1978-10-30 | 1 | 2 | 23 | 1 | 9 | 9389 | 1978009389 |
| GS | 1978-10-31 | 1 | 6 | 23 | 4 | 9 | 9390 | 1978009390 |
