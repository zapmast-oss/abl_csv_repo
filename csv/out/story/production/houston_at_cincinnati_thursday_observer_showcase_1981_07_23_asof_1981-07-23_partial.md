# Houston at Cincinnati — Thursday Observer Showcase Production Packet

**Showcase date:** Thursday, July 23, 1981  
**Official cutoff:** Completed games through Chicago 15, Dallas 10, game 1155, on July 23  
**Showcase status:** Unplayed (`games.csv` game 1161, `played=0`)  
**Site:** Cougars Park, Cincinnati

## Editorial spine

First-place Houston (54-41) enters Cincinnati with the Cougars at 50-46. The direct stakes are clean: Cincinnati can take another game out of the ABC Central margin; Houston can answer after Cincinnati won Wednesday’s game 7–1.

The game is the story. Treat every number below as pregame context and never as evidence of a Thursday result.

## Stakes board

| Club | Record | Runs | Run diff. | Last 10 |
|---|---:|---:|---:|---:|
| Houston Mavericks | 54-41 | 392-358 | +34 | 4-6 |
| Cincinnati Cougars | 50-46 | 434-382 | +52 | 6-4 |

- ABC Central margin entering the Showcase: **4.5 games**.
- Season series through Wednesday: **Houston 9, Cincinnati 4**.
- Current four-game series: **Cincinnati 2, Houston 1**. Houston won Monday; Cincinnati won Tuesday and Wednesday.
- Wednesday, July 22: Cincinnati 7, Houston 1. That result is included; Thursday is not.

## Confirmed starters

- **Houston: Valentin Geffroy** — confirmed by the commissioner/user.
- **Cincinnati: Chris Collette** — confirmed by the commissioner/user.
- The names also match each club’s `starter_0` assignment in `projected_starting_pitchers.csv`. Game 1161 itself has not yet populated its starter-ID fields, so the confirmation source is documented separately.

### Valentin Geffroy — Houston

- Age 27; throws L; 7 years of listed experience; currently injured: **no**.
- 1981 ABL line through the cutoff: **5 G, 0 GS, 1-0, 7.1 IP, 1.23 ERA, 0.95 WHIP, 5 K, 2 BB, 0 HR allowed**.
- Current pitch ratings: **fastball 40, curveball 55, forkball 50, changeup 35**.
- Overall OOTP ratings: stuff 45, movement 35, control 50, stamina 65, hold 80.
- Raw velocity-band code: 6. The export does not provide a trustworthy MPH label, so do not translate this code into an MPH range on air.

### Chris Collette — Cincinnati

- Age 23; throws R; 9 years of listed experience; currently injured: **no**.
- 1981 ABL line through the cutoff: **12 G, 9 GS, 8-2, 62.2 IP, 1.72 ERA, 1.24 WHIP, 40 K, 29 BB, 3 HR allowed**.
- Current pitch ratings: **fastball 50, curveball 70, changeup 80**.
- Overall OOTP ratings: stuff 55, movement 55, control 45, stamina 35, hold 65.
- Raw velocity-band code: 9. The export does not provide a trustworthy MPH label, so do not translate this code into an MPH range on air.

## Managers and field staff

| Club | Manager | Bench coach | Pitching coach | Hitting coach | General manager |
|---|---|---|---|---|---|
| Houston Mavericks | Carlos Sanchez | Sam Downs | Rodolfo Luna | Ray Gallo | Larry Bell |
| Cincinnati Cougars | Chris Allison | Scott Willett | Jon Baugher | Jared Teitelbaum | Phil Trammel |

Manager and staff IDs come directly from `team_roster_staff.csv` and resolve through `coaches.csv`. No managerial tendencies are inferred.

## Bats to frame

### Houston Mavericks

| Player | AVG | HR | RBI | PA |
|---|---:|---:|---:|---:|
| Brent Keyser | 0.282 | 9 | 43 | 354 |
| Alex Ramirez | 0.275 | 8 | 36 | 321 |
| Albert Chavez | 0.295 | 7 | 35 | 284 |
| Apostolos Georghiou | 0.279 | 7 | 34 | 314 |

### Cincinnati Cougars

| Player | AVG | HR | RBI | PA |
|---|---:|---:|---:|---:|
| Scott Reis | 0.272 | 13 | 63 | 398 |
| Esteban Martinez | 0.315 | 9 | 56 | 415 |
| Joe Rogers | 0.294 | 8 | 54 | 348 |
| Victor Torres | 0.250 | 4 | 37 | 423 |

## Verified honors and franchise context

- **Scott Reis** has a 1981 ABL All-Star entry in raw OOTP `players_awards.csv` (award ID 9, dated July 12). He may be identified as a 1981 All-Star.
- Cincinnati won the ABC Central in 1980 at **97-65** and reached the playoffs under manager Chris Allison.
- Houston reached the playoffs in each season from **1975 through 1980**, won its division four times in that span, and has a `won_playoffs=1` entry for 1979 under manager Carlos Sanchez.
- Use the history as franchise context, not as evidence of how either clubhouse feels about Thursday.

## Observer open

Baseball matters because the stakes are incredibly high. Houston brings a 54-41 record and the ABC Central lead into Cincinnati. The Cougars are 50-46, and after Cincinnati took Wednesday’s game 7–1, Thursday gives Houston a direct chance to answer before the series moves on. The records set the pressure. This unplayed game supplies the verdict.

## Segment beats

1. Establish the cutoff: Dallas–Chicago is final; Houston–Cincinnati is not.
2. Put the ABC Central margin and current records on screen.
3. Recap Wednesday only as series context: Cincinnati 7, Houston 1.
4. Introduce the confirmed starters and distinguish user confirmation from the still-empty game-row starter IDs.
5. Move to the clubs’ leading power/run-production bats.
6. Close on direct opportunity: Cincinnati can narrow the race; Houston can protect its lead.

## Guardrails and source ledger

- Say **ABL**, never ABLE.
- Do not state, imply, simulate, or backfill a Thursday result.
- Raw OOTP is official for score, schedule, record, standings, roster, and game status.
- StatsPlus is enrichment only and is not used here to override any official fact.
- `games.csv`: cutoff, schedule, game status, records, run totals, recent form, and season series.
- `games_score.csv` and `game_logs.csv`: complete cross-coverage for all completed games in preflight.
- `players_game_batting.csv` and `players.csv`: player lines and names.
- Starter confirmation: commissioner/user confirmation; `projected_starting_pitchers.csv` independently matches both names.
- `players_game_pitching_stats.csv`, `players_pitching.csv`, and `players.csv`: starter lines, biographies, arsenals, and ratings.
- `team_roster_staff.csv` and `coaches.csv`: current manager and staff assignments.
- `teams.csv` and `parks.csv`: identities and venue.

**Packet status: SAFE FOR PREGAME PRODUCTION — SHOWCASE RESULT EXCLUDED**
