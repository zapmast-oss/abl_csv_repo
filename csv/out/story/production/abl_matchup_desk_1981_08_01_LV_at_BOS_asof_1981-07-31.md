# ABL DESK - LAS VEGAS AT BOSTON

**Game date:** Saturday, August 1, 1981  
**Data cutoff:** Completed ABL regular-season games through July 31, 1981  
**Status:** PREGAME - scheduled game is unplayed in the export

## Here's the Skinny

The Las Vegas Gamblers visit the Boston Patriots for the middle game of a three-game set. Boston took Friday's opener, 4-2, after Las Vegas went 2-1 against Cincinnati. Boston split its preceding four games, taking one of three from St. Louis before opening this series with a win.

Both clubs lead the ABC division races: Las Vegas is 60-44 atop the West, and Boston is 54-50 atop the East. The Gamblers are two games ahead of Seattle and four ahead of Denver. Boston is six ahead of Philadelphia. Las Vegas has the stronger overall record, but both clubs enter with a division lead to protect. Boston leads the season series 5-2.

**Central question:** Can Boston build on Friday's opener and keep its Eastern Division cushion, or will Las Vegas answer and protect its lead over Seattle and Denver?

## Game Card

| Item | Detail |
|---|---|
| Matchup | Las Vegas Gamblers (away) at Boston Patriots (home) |
| Scheduled game ID | 1255 |
| Scheduled time | 7:05 p.m. (`games.csv` time 1905) |
| Ballpark | Patriots Fields, Boston, Massachusetts; listed capacity 26,000 |
| Series status | Boston leads 1-0 after winning July 31, 4-2 |
| Next scheduled meeting | August 2, game ID 1267, also at Boston |
| Game status at cutoff | Scheduled, unplayed |
| Starting pitchers | **NOT VERIFIED** - both starter fields for game 1255 are zero |

The export lists 12 ABL regular-season games on August 1, and all 12 are unplayed. The game record contains no result, lineup, or assigned starting pitchers. Friday's starters do not establish who is scheduled for Saturday.

## Where They Stand

| Club | Record | Division position | Wild-card context |
|---|---:|---|---|
| Las Vegas Gamblers | 60-44 | ABC West, 1st; 2 GB of Seattle | Not eligible while leading its division |
| Boston Patriots | 54-50 | ABC East, 1st; 6 GB ahead of Philadelphia | Not eligible while leading its division |

Houston leads the ABC Central at 61-42 and is excluded from the wild-card order as its division leader. Among the non-leaders, Seattle (58-46) is first and Denver (56-48) second. Boston and Las Vegas are playing to hold their division positions; neither is currently in the wild-card ordering while leading its division.

## Recent Form and Series History

Las Vegas's last five completed games: a 4-11 loss at Portland on July 26; a 9-4 win, 2-3 loss, and 5-4 win against Cincinnati from July 27-29; then Friday's 2-4 loss at Boston. The Gamblers are 2-3 over that span and have a one-game losing streak.

Boston's last five: a 1-4 loss at Pittsburgh on July 26; a 5-2 win, 0-9 loss, and 2-6 loss against St. Louis from July 28-30; then Friday's 4-2 win over Las Vegas. The Patriots are 2-3 over that span and have a one-game winning streak.

Boston leads the 1981 head-to-head series 5-2:

| Dates | Site | Result |
|---|---|---|
| May 8-10 | Las Vegas | Boston took 2 of 3 |
| June 8-10 | Boston | Boston took 2 of 3 |
| July 31 | Boston | Patriots won 4-2 (game 1243) |

Friday's `games.csv` entry records Oscar Herrera (player ID 8695) and Jorge Galvez (player ID 1) as the starters. These were Friday's pitchers only; game 1255 has no starter assignments in the export.

## Team Snapshot Through July 31

| Measure | Las Vegas | Boston |
|---|---:|---:|
| Record | 60-44 | 54-50 |
| Runs scored | 436 (4.19 per game) | 411 (3.95 per game) |
| Batting average | .255 | .247 |
| On-base percentage | .334 | .325 |
| Slugging percentage | .363 | .366 |
| OPS | .697 | .691 |
| Home runs | 59 | 78 |
| Team ERA | 3.31 | 3.68 |

Las Vegas has scored 25 more runs and has the lower team ERA in the season-to-date export. Boston has hit 19 more home runs and has a slightly higher slugging percentage. These are descriptive season totals, not a projection for this game.

## Players to Watch

Lines below are aggregated from completed 1981 regular-season player-game records through July 31. Batters listed had at least 200 plate appearances; pitcher lines are selected from the clubs' higher innings totals.

**Las Vegas**

- Jon Martinez (player ID 5019): .295, 7 HR, 42 RBI, .827 OPS.
- Mike Rabel (player ID 923): .273, 14 HR, 60 RBI, .797 OPS.
- Oscar Herrera (player ID 8695): 7-10, 166 IP, 3.58 ERA, 132 K.
- Bobby Rodriguez (player ID 8003): 11-4, 155 IP, 2.61 ERA, 135 K.

**Boston**

- Juan Perez (player ID 160): .272, 21 HR, 58 RBI, .867 OPS.
- Danny Salas (player ID 6731): .268, 13 HR, 34 RBI, .819 OPS.
- Jorge Galvez (player ID 1): 10-5, 155 2/3 IP, 3.06 ERA, 86 K.
- Victor Martinez (player ID 4829): 9-6, 146 1/3 IP, 3.26 ERA, 115 K.

These are performance context only. They do not establish who will start, appear, or be available on August 1.

## Franchise Desk Notes

- **Las Vegas Gamblers:** ABC West leader; field manager Matt Petrie; general manager Roberto Choto; owner Terry Vinson. Official home park: Gamblers Park.
- **Boston Patriots:** ABC East leader; field manager Bill Hurff; general manager Bobby Ramsey; owner Alex Mendez. Official home park: Patriots Fields.
- Staff names, assignments, and park identities are from the repository's current team/staff snapshot. They do not establish a manager's private strategy or a club's motivation.

## Production Guardrails

- Do not name an August 1 starting pitcher until a dated, game-keyed source confirms the assignment.
- Do not describe either club as a wild-card team while it leads its division; both currently hold division leads.
- Do not describe this as a series tied at one game apiece: Boston won the July 31 opener.
- The season series is 5-2 for Boston through July 31.
- "Real-Time Sim" refers to letting the day's games play simultaneously; it is not a label for simulating this individual game.

## Sources and Reconciliation

- `csv/ootp_csv/leagues.csv` - league calendar and current date context.
- `csv/ootp_csv/teams.csv`, `sub_leagues.csv`, `divisions.csv`, `cities.csv`, and `parks.csv` - team identities, ABC alignment, division names, locations, and home parks.
- `csv/ootp_csv/games.csv` - game 1255 scheduled August 1 (away team 17, home team 20, unplayed, no starters); game 1243 July 31 (Las Vegas 2, Boston 4); and prior head-to-head results. The game export's `runs0` field is the away score and `runs1` is the home score.
- `csv/ootp_csv/team_record.csv` - Las Vegas 60-44 and Boston 54-50 at the cutoff; division and wild-card comparisons use major-league teams only.
- `csv/ootp_csv/team_batting_stats.csv` and `team_pitching_stats.csv` - club season lines through the export cutoff.
- `csv/ootp_csv/players_game_batting.csv` and `players_game_pitching_stats.csv` - player totals aggregated from completed regular-season games through July 31; `players.csv` supplies names and IDs.
- `csv/ootp_csv/team_roster_staff.csv` and `coaches.csv` - owner, general manager, and field-manager assignments.

Research desk only. No game was simulated and no OOTP or source CSV data was changed.
