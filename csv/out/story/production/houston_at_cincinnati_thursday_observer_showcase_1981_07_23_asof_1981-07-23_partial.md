# Houston at Cincinnati — Thursday Observer Showcase Production Packet

**Showcase date:** Thursday, July 23, 1981  
**Official cutoff:** Completed games through Chicago 15, Dallas 10, game 1155, on July 23  
**Showcase status:** Unplayed (`games.csv` game 1161, `played=0`)  
**Site:** Cougars Park, Cincinnati

## Editorial spine

First-place Houston (54-41) enters Cincinnati with the Cougars at 50-46. The direct stakes are clean: Cincinnati can take another game out of the ABC Central margin; Houston can answer after Cincinnati won Wednesday’s opener 7–1.

The game is the story. Treat every number below as pregame context and never as evidence of a Thursday result.

## Stakes board

| Club | Record | Runs | Run diff. | Last 10 |
|---|---:|---:|---:|---:|
| Houston Mavericks | 54-41 | 392-358 | +34 | 4-6 |
| Cincinnati Cougars | 50-46 | 434-382 | +52 | 6-4 |

- ABC Central margin entering the Showcase: **4.5 games**.
- Season series through Wednesday: **Houston 9, Cincinnati 4**.
- Wednesday, July 22: Cincinnati 7, Houston 1. That result is included; Thursday is not.

## Probable-pitcher handling

- Houston projected rotation slot: **Valentin Geffroy**.
- Cincinnati projected rotation slot: **Chris Collette**.
- These names come from `projected_starting_pitchers.csv`; label them projected, not confirmed, because game 1161 has starter IDs set to 0.

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

## Observer open

Baseball matters because the stakes are incredibly high. Houston brings a 54-41 record and the ABC Central lead into Cincinnati. The Cougars are 50-46, and after Cincinnati took Wednesday’s game 7–1, Thursday gives Houston a direct chance to answer before the series moves on. The records set the pressure. This unplayed game supplies the verdict.

## Segment beats

1. Establish the cutoff: Dallas–Chicago is final; Houston–Cincinnati is not.
2. Put the ABC Central margin and current records on screen.
3. Recap Wednesday only as series context: Cincinnati 7, Houston 1.
4. Introduce projected starters with the explicit “projected” label.
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
- `projected_starting_pitchers.csv`: projected rotation slot only.
- `teams.csv` and `parks.csv`: identities and venue.

**Packet status: SAFE FOR PREGAME PRODUCTION — SHOWCASE RESULT EXCLUDED**
