# ABL Current-State Preflight — Target 1981-07-23

- Newsroom date: `1981-07-23`
- Target as-of date: `1981-07-23`
- Earliest completed game: `1981-04-06`
- Latest completed game: `1981-07-23`
- Completed ABL regular-season games: **1152**
- Games per team: **95–97**
- All 24 teams represented: **YES**
- Verdict: **READY_FOR_CURRENT_RUN**
- Story work safe to resume: **YES**

The cutoff is partial-day 1981-07-23: only games marked completed count. Scheduled or unplayed games on that date do not advance team records.

## Current-state drivers used

- `csv/ootp_csv/games.csv`
- `csv/ootp_csv/games_score.csv`
- `csv/ootp_csv/game_logs.csv`

## Games per team

| Team ID | Team | Games |
|---:|---|---:|
| 1 | Miami Hurricanes | 96 |
| 2 | Phoenix Firebirds | 96 |
| 3 | Atlanta Kings | 96 |
| 4 | Dallas Rustlers | 97 |
| 5 | Detroit Dukes | 96 |
| 6 | Charlotte Colonels | 96 |
| 7 | San Diego Seraphs | 96 |
| 8 | Tampa Bay Storm | 96 |
| 9 | Los Angeles Cobras | 96 |
| 10 | San Francisco Warriors | 96 |
| 11 | Minneapolis Blizzard | 96 |
| 12 | Chicago Fire | 97 |
| 13 | Denver Rocketeers | 96 |
| 14 | Cincinnati Cougars | 96 |
| 15 | New York Aces | 96 |
| 16 | Nashville Blues | 96 |
| 17 | Las Vegas Gamblers | 96 |
| 18 | Portland Lumberjacks | 96 |
| 19 | St. Louis Stallions | 95 |
| 20 | Boston Patriots | 96 |
| 21 | Pittsburgh Express | 96 |
| 22 | Philadelphia Fury | 96 |
| 23 | Houston Mavericks | 95 |
| 24 | Seattle Comets | 96 |

## Sortable support

All **20** promoted sortable CSVs are available as supplemental support. They do not prove the date cutoff.

## Accepted schema drift

- Staff report: 13-column forward schema; IDs supplied by downstream coach lookup where possible.
- Team batting report: 13-column forward schema; WAR comes from batting-extra and removed baserunning fields remain limited.

## Limited or disabled enrichment

- Unresolved staff IDs remain null.
- CS and SB% need another validated source.
- BatR, wSB, UBR, and BsR enrichments remain disabled.

## Stale derivatives excluded from current-state proof

- `csv/out/star_schema/monday_1981_standings_by_division.csv (32-game snapshot)`
- `csv/out/star_schema/fact_team_reporting_1981_weekly_change.csv (32-game snapshot)`
- `csv/out/star_schema/fact_player_batting.csv (preserved early snapshot)`
- `csv/out/star_schema/fact_player_pitching.csv (preserved early snapshot)`
- `csv/out/csv_out/z_ABL_Manager_Tendencies.csv (62-game snapshot)`
- `csv/out/csv_out/z_ABL_Division_Leverage.csv (62-game snapshot)`
- `csv/out/csv_out/z_ABL_Rotation_Stability.csv (62-game snapshot)`

## Final verdict

**READY_FOR_CURRENT_RUN**

The raw driver set reaches 1981-07-23, represents all teams, reconciles computed G-W-L to team_record.csv, and reconciles completed game IDs across score and log files.
