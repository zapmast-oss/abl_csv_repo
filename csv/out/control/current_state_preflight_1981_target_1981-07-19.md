# ABL Current-State Preflight — Target 1981-07-19

- Newsroom date: `1981-07-20`
- Target as-of date: `1981-07-19`
- Earliest completed game: `1981-04-06`
- Latest completed game: `1981-07-19`
- Completed ABL regular-season games: **1115**
- Games per team: **92–93**
- All 24 teams represented: **YES**
- Verdict: **READY_FOR_CURRENT_RUN**
- Story work safe to resume: **YES**

July 20 is the newsroom date. July 19 is the completed-game cutoff; missing July 20 games are not an error.

## Current-state drivers used

- `csv/ootp_csv/games.csv`
- `csv/ootp_csv/games_score.csv`
- `csv/ootp_csv/game_logs.csv`

## Games per team

| Team ID | Team | Games |
|---:|---|---:|
| 1 | Miami Hurricanes | 93 |
| 2 | Phoenix Firebirds | 93 |
| 3 | Atlanta Kings | 93 |
| 4 | Dallas Rustlers | 93 |
| 5 | Detroit Dukes | 93 |
| 6 | Charlotte Colonels | 93 |
| 7 | San Diego Seraphs | 93 |
| 8 | Tampa Bay Storm | 93 |
| 9 | Los Angeles Cobras | 93 |
| 10 | San Francisco Warriors | 93 |
| 11 | Minneapolis Blizzard | 93 |
| 12 | Chicago Fire | 93 |
| 13 | Denver Rocketeers | 93 |
| 14 | Cincinnati Cougars | 93 |
| 15 | New York Aces | 93 |
| 16 | Nashville Blues | 93 |
| 17 | Las Vegas Gamblers | 93 |
| 18 | Portland Lumberjacks | 93 |
| 19 | St. Louis Stallions | 92 |
| 20 | Boston Patriots | 93 |
| 21 | Pittsburgh Express | 93 |
| 22 | Philadelphia Fury | 93 |
| 23 | Houston Mavericks | 92 |
| 24 | Seattle Comets | 93 |

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

The raw driver set reaches July 19, represents all teams, has a normal one-game schedule spread, and reconciles completed game IDs across score and log files.
