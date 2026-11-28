# ABL DESK — SAN FRANCISCO AT CHICAGO

**Game date:** Saturday, August 1, 1981  
**Data cutoff:** Completed ABL regular-season games through July 31, 1981  
**Status:** PREGAME — August 1 game not yet played in the export

## Here's the Skinny

San Francisco arrives at Chicago after splitting its immediate stretch in opposite directions: the Warriors were swept in three games at Miami, then opened this Chicago series with a 6–5 win on July 31. The August 1 game is the middle game of the three-game Chicago set; the finale is scheduled for August 2.

The Warriors are 58–46 and lead the NBC West. Chicago is 55–49, third in the NBC Central and two games behind Detroit. The Fire are also one game behind Dallas for the NBC wild-card lead, tied with Miami for second on that board. San Francisco leads the teams' season series 5–2. The standings make this a useful test for Chicago's wild-card chase, while San Francisco is defending a division lead.

**Central question:** Can Chicago answer Friday's one-run loss and stay close to the NBC wild-card lead, or can San Francisco take the series and re-establish momentum after its Miami sweep?

## Game Card

| Item | Detail |
|---|---|
| Matchup | San Francisco Warriors (away) at Chicago Fire (home) |
| Scheduled game ID | 1253 |
| Scheduled time | 7:05 p.m. (`games.csv` time 1905) |
| Ballpark | Chicago Grounds, Chicago, Illinois; listed capacity 18,500 |
| Series status | San Francisco leads 1–0 after winning July 31, 6–5 |
| Next scheduled meeting | August 2, game ID 1265, also at Chicago |
| Game status at cutoff | Scheduled, unplayed |
| Starting pitchers | **NOT VERIFIED** — both starter fields for game 1253 are zero |

The league export shows all 12 August 1 ABL games scheduled and unplayed at the cutoff. Game 1253 has no recorded result, lineup, or assigned starter. Rotation order alone is not enough to name the starters for this game.

## Where They Stand

| Club | Record | Division position | Wild-card position |
|---|---:|---|---|
| San Francisco Warriors | 58–46 | NBC West, 1st | Not wild-card eligible while leading its division |
| Chicago Fire | 55–49 | NBC Central, 3rd; 2 GB of Detroit | Tied with Miami for 2nd; 1 GB of Dallas |

The current NBC wild-card order among non-division leaders is Dallas (56–48), Miami and Chicago (both 55–49), then Phoenix (53–51). A single result can change the order, but this game does not decide a berth.

## Recent Form and Series History

San Francisco's last five completed games: a 4–0 win over Los Angeles on July 26, three losses at Miami on July 27–29, then Friday's 6–5 win at Chicago. The Warriors enter with a one-game winning streak.

Chicago's last five: an 8–6 win at Minneapolis on July 26; a 1–2 series at Phoenix on July 28–30; then Friday's 5–6 loss at home to San Francisco. The Fire enter with a one-game losing streak.

San Francisco leads the 1981 head-to-head 5–2:

| Dates | Site | Result |
|---|---|---|
| May 8–10 | San Francisco | Warriors took 2 of 3 |
| June 8–10 | Chicago | Warriors took 2 of 3 |
| July 31 | Chicago | Warriors won 6–5 (game 1241) |

The July 31 opener was a one-run game. Its `games.csv` record lists Antonio Pineda (player ID 10049) and Heriberto Jimenez (player ID 8125) as the starters. Those are Friday's starters only; the August 1 starter fields remain blank.

## Team Snapshot Through July 31

| Measure | San Francisco | Chicago |
|---|---:|---:|
| Runs scored | 483 (4.64 per game) | 448 (4.31 per game) |
| Batting average | .257 | .264 |
| On-base percentage | .348 | .334 |
| Slugging percentage | .406 | .395 |
| OPS | .753 | .729 |
| Home runs | 110 | 80 |
| Team ERA | 3.78 | 3.77 |

San Francisco has the stronger run production and power profile; Chicago has the higher batting average. Their exported team ERAs are effectively even. These are season-to-date summaries, not projections for this game's outcome.

## Players to Watch

Lines below are aggregated from completed 1981 regular-season player-game records through July 31, not from an older sortable-stats snapshot.

**San Francisco**

- Miguel Collado: .321, 11 HR, 67 RBI, .916 OPS.
- Bobby Jackson: .292, 18 HR, 58 RBI, .910 OPS.
- Victor Quintero: 24 HR and 84 RBI; .902 OPS.
- Antonio Pineda: 13–6, 170⅔ IP, 3.01 ERA, 123 K.

**Chicago**

- Sal Gamez: .304, 15 HR, 70 RBI, .890 OPS.
- Miguel Morales: 17 HR and 53 RBI; .883 OPS.
- Alex O'Donell: .312, 11 HR, 55 RBI, .867 OPS.
- John Jury: 10–7, 150 IP, 2.40 ERA, 89 K.

These are performance context only. They do not establish who will start, appear, or be available on August 1.

## Franchise Desk Notes

- **San Francisco Warriors:** NBC West leader; field manager Tyler Trowbridge. Official home park: San Francisco Grounds.
- **Chicago Fire:** NBC Central third place; field manager Matt Mead. Official home park: Chicago Grounds.
- The franchise roles and park identities are cross-checked against the repository's August 1, 1981 team/staff snapshot. They do not establish a manager's private strategy or a club's motivation.

## Production Guardrails

- Do not name an August 1 starting pitcher until a dated/game-keyed source confirms the assignment.
- Do not call Chicago the current wild-card leader; Dallas leads, while Chicago is tied with Miami one game back.
- Do not call the season series close: San Francisco is 5–2 through July 31.
- Do not treat the Warriors' recent division lead as a clinch or guaranteed postseason berth.
- “Real-Time Sim” refers to running the day's games simultaneously; it is not a label for simulating this individual game.

## Sources and Reconciliation

- `csv/ootp_csv/leagues.csv` — ABL current date is August 1, 1981; trade deadline and league calendar context.
- `csv/ootp_csv/teams.csv`, `sub_leagues.csv`, `divisions.csv`, `cities.csv`, `states.csv`, and `parks.csv` — team identities, NBC alignment, locations, Chicago Grounds, and capacity.
- `csv/ootp_csv/games.csv` — game 1253 scheduled August 1 (away team 10, home team 12, unplayed, no starters); game 1241 July 31 (SF 6, CHI 5), and the prior head-to-head schedule/results.
- `csv/ootp_csv/team_record.csv` — San Francisco 58–46 and Chicago 55–49 at the cutoff. All 24 major-league W–L records reconcile to completed 1981 regular-season game results when `runs0` is treated as the away club's score and `runs1` as the home club's score.
- `csv/ootp_csv/team_batting_stats.csv` and `team_pitching_stats.csv` — club season lines through the export cutoff.
- `csv/ootp_csv/players_game_batting.csv` and `players_game_pitching_stats.csv` — player performance totals from completed regular-season games through July 31; `players.csv` supplies player names and IDs.
- `csv/ootp_csv/team_roster_staff.csv` and `coaches.csv` — field-manager assignments and IDs.

Research desk only. No game was simulated and no OOTP or source CSV data was changed.
