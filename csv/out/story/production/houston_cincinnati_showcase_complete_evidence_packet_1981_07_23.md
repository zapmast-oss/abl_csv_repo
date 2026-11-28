# Houston–Cincinnati Showcase Evidence Packet

## 1. Verification status and cutoff

- Game: Houston Mavericks at Cincinnati Cougars, Thursday, July 23, 1981, Cougars Park; scheduled game ID **1161**.
- Cutoff: every completed ABL regular-season game through game 1155, **Chicago 15, Dallas 10**, on July 23.
- Game 1161 remains `played=0`, with both score fields at zero. It is excluded from every result and statistic below.
- All 1,152 completed game IDs reconcile across `games.csv`, `games_score.csv`, and `game_logs.csv`; computed G-W-L reconciles to `team_record.csv` for all 24 clubs.
- Raw OOTP is authoritative. No StatsPlus fact is used in this packet.

Sources: `games.csv`, `games_score.csv`, `game_logs.csv`, `team_record.csv`, `teams.csv`, and `current_state_preflight_1981_target_1981-07-23.json`.

## 2. Game and stakes board

| ABC Central | W-L | GB | Remaining | RS-RA | Diff. | Last 10 | Streak |
|---|---:|---:|---:|---:|---:|---:|---:|
| Houston | 54-41 | — | 67 | 392-358 | +34 | 4-6 | L2 |
| Cincinnati | 50-46 | 4.5 | 66 | 434-382 | +52 | 6-4 | W2 |
| Nashville | 50-46 | 4.5 | 66 | — | — | — | W4 |
| St. Louis | 45-50 | 9.0 | 67 | — | — | — | L4 |

- Houston road record: **24-23**. Cincinnati home record: **34-16**.
- Since the July 14 All-Star Game: Houston **2-4**; Cincinnati **4-3**.
- If Cincinnati wins Thursday, the margin becomes **3.5 games**. If Houston wins, it becomes **5.5 games**.
- Cincinnati can win the four-game series 3-1; Houston can salvage a 2-2 split.

Sources: `games.csv`, `team_record.csv`, `team_batting_stats.csv`, `team_pitching_stats.csv`, `leagues.csv`.

## 3. Current series

| Date | Result | Winning pitcher | Losing pitcher | Save | Attendance |
|---|---|---|---|---|---:|
| July 20 | Houston 5, Cincinnati 2 | Nate Higgins | Darius Jones | Tony Chavarria | 20,818 |
| July 21 | Cincinnati 2, Houston 1 | Gil Medina | Agustin Tapia | None | 20,789 |
| July 22 | Cincinnati 7, Houston 1 | Ricardo Hernandez | Eskil Pedersen | None | 20,962 |

- Series: Cincinnati 2-1. Season series: Houston 9-4.
- July 20 starters: Higgins 7⅓ IP, 7 H, 2 R/1 ER, 5 BB, 1 K; Jones 5 IP, 8 H, 5 ER, 6 BB, 3 K. Alex Ramirez homered.
- July 21 starters: Dylan Wilson 8 IP, 1 H, 0 R, 4 BB, 6 K; Alex Perez 8 IP, 6 H, 1 ER, 4 BB, 6 K. Ricky Castillo homered.
- July 22 starters: Pedersen 5⅓ IP, 5 H, 6 ER, 3 BB, 3 K; Hernandez 9 IP, 4 H, 1 ER, 1 BB, 7 K. Jordan Pedroza homered.
- Scott Reis: 0-for-3 with two runs and a walk Monday; 1-for-4 with an RBI and run Tuesday; 1-for-2 with two RBI Wednesday.
- Recent bullpen use: Chavarria 30 pitches Monday; Antonio Scott 43 Monday; Ryan Hines 14 Monday; Tapia 23 Tuesday; Medina 14 Tuesday; Miguel Salgado 33 Wednesday.

Sources: `games.csv`, `players_game_pitching_stats.csv`, `players_game_batting.csv`, `players.csv`.

## 4. Starting pitchers

The starters are confirmed by the commissioner/user and independently match `projected_starting_pitchers.csv`. Game 1161 has not yet populated its starter-ID fields.

| Pitcher | Age | B/T | 1981 line | Arsenal, current/potential |
|---|---:|---|---|---|
| Valentin Geffroy, Houston | 27 | R/L | 5 G, 0 GS, 1-0, 2 SV, 7⅓ IP, 1.23 ERA, 0.95 WHIP, 5 K, 2 BB, 0 HR | Fastball 40/40; curve 55/55; forkball 50/50; changeup 35/35 |
| Chris Collette, Cincinnati | 23 | S/R | 12 G, 9 GS, 8-2, 62⅔ IP, 1.72 ERA, 1.24 WHIP, 40 K, 29 BB, 3 HR | Fastball 50/50; curve 70/80; changeup 80/85 |

- Geffroy: born January 17, 1954; 6-foot-6, 201 pounds; seven listed experience years; not injured. Ratings: Stuff 45, Movement 35, Control 50, Stamina 65, Hold 80, ground/fly 44, arm-slot code 3.
- Collette: born August 1, 1957; 6-foot-1, 175 pounds; nine listed experience years; not injured. Ratings: Stuff 55, Movement 55, Control 45, Stamina 35, Hold 65, ground/fly 51, arm-slot code 3.
- Collette's “nine years” is the export's `experience` value, not nine adult ABL seasons; his draft data says Cincinnati selected him in the first round in 1975. The export does not define how pre-ABL experience is counted.
- Velocity codes 6 and 9 could not be translated safely: the CSV supplies no code-to-MPH dictionary, and no repository reference defines it. They are intentionally not presented as MPH.
- Geffroy's 1981 ABL work consists of five relief appearances, last on April 25. Collette last started July 18: 6⅔ IP, 5 H, 4 R/3 ER, 3 BB, 6 K, 107 pitches.

Sources: `players.csv`, `players_pitching.csv`, `players_game_pitching_stats.csv`, `projected_starting_pitchers.csv`, commissioner/user confirmation.

## 5. Expected lineups and availability

The export does not contain a confirmed Thursday batting order. The most recent game participants are available in `players_game_batting.csv`, but row order is not an official batting-order field; presenting it as a lineup would be guesswork. No expected lineup is asserted.

- Neither confirmed starter is marked injured.
- The synchronized roster and injury fields were searched in `team_roster.csv`, `players.csv`, and `players_roster_status.csv`; a complete on-air availability table still requires resolving roster-list codes and fatigue into display meanings.
- Recent bullpen workloads are listed in Section 3. No pitcher is declared unavailable solely from pitch count.

## 6. Featured Houston players

| Player | AVG | HR | RBI | Why he matters |
|---|---:|---:|---:|---|
| Brent Keyser | .282 | 9 | 43 | Team-leading featured HR/RBI line |
| Alex Ramirez | .275 | 8 | 36 | Homered and drove in two Monday |
| Albert Chavez | .295 | 7 | 35 | Highest average among Houston's featured bats |
| Apostolos Georghiou | .279 | 7 | 34 | Additional middle-order production |

Sources: completed-game aggregation from `players_game_batting.csv`; identities from `players.csv`.

## 7. Featured Cincinnati players

| Player | AVG | HR | RBI | Why he matters |
|---|---:|---:|---:|---|
| Scott Reis | .272 | 13 | 63 | Team HR/RBI leader in the featured set; 1981 All-Star |
| Esteban Martinez | .315 | 9 | 56 | Highest featured batting average |
| Joe Rogers | .294 | 8 | 54 | Secondary average and run-production threat |
| Jordan Pedroza | — | — | — | Homered and drove in two Wednesday |

Sources: `players_game_batting.csv`, `players_awards.csv`, `players.csv`.

## 8. Scott Reis career box

- Scott Reis, R-E-I-S, has spent every ABL season from 1972 through the cutoff with Cincinnati.
- Career through cutoff: **1,491 G, 6,620 PA, 1,548 H, 300 doubles, 49 triples, 249 HR, 1,031 R, 944 RBI, 189 SB, 902 BB, .282 AVG, approximately .393 OBP, 81.4 WAR**.
- Five MVP awards (award ID 5, first place): **1972, 1977, 1978, 1979, 1980**. He finished second in 1976.
- Ten All-Star selections (award ID 9): **1972-1981**, every season of ABL play to date.
- Nine position awards at position 8 (award ID 11): **1972-1980**. The export does not provide the award-name dictionary, so the packet does not rename award ID 11.
- 1981: .272, 13 HR, 63 RBI, 61 R, 14 SB, 41 BB, 3.25 WAR in 93 games.
- “Face of the Action Baseball League” is editorial language, not an OOTP field. It is factually supportable by five MVPs, ten straight All-Star selections, 81.4 WAR, 249 homers, and 1,548 hits, but must remain labeled an editorial description.
- Personality/popularity raw values exist, but their display-label dictionaries are absent; no psychological interpretation is supplied.

Sources: `players_career_batting_stats.csv`, `players_awards.csv`, `players.csv`.

## 9. Managers and managerial records

| Manager | Club | Current tenure found | Regular-season record through cutoff | Playoff years | Division firsts | Titles flag |
|---|---|---:|---:|---:|---:|---:|
| Carlos Sanchez | Houston | 1972-present | 832-722 | 1975-80 (6) | 1975, 1976, 1978, 1979 (4) | 1979 |
| Chris Allison | Cincinnati | 1976-present | 507-399 | 1976, 1977, 1979, 1980 (4) | 1977, 1980 (2) | 0 |

- 1981 records are Houston 54-41 and Cincinnati 50-46.
- Staff mapping: Houston—Sanchez, bench coach Sam Downs, pitching coach Rodolfo Luna, hitting coach Ray Gallo. Cincinnati—Allison, bench coach Scott Willett, pitching coach Jon Baugher, hitting coach Jared Teitelbaum.
- Official tendency fields exist in `coaches.csv`, but a complete code-to-display translation was not found. No tendency is converted into prose.
- Complete almanac schedules and box scores give Sanchez a **56-39** postseason record through 1980 and Allison a **14-16** record. Sanchez is **16-10** against Allison, winning all four series.

Sources: `coaches.csv`, `team_roster_staff.csv`, `team_history.csv`, `team_history_record.csv`, `team_record.csv`.

## 10. Franchise postseason comparison

| Category | Houston | Cincinnati |
|---|---:|---:|
| Playoff appearances | 6 | 4 |
| Division titles | 4 | 2 |
| Postseason series won | 11 | 1 |
| Postseason series lost | 5 | 4 |
| Postseason game record | 56-39 | 14-16 |
| Conference titles | 4 | 0 |
| Grand Series appearances | 4 | 0 |
| Grand Series championships | 1 (1979) | 0 |

The complete 1972-80 almanac postseason schedules, bracket progression and box scores resolve these totals. Houston's paths: 1975 DCS win/CCS loss; 1976 DCS win/CCS loss; 1977 DCS and CCS wins/GCS loss; 1978 DCS and CCS wins/GCS loss; 1979 DCS, CCS and GCS wins; 1980 DCS and CCS wins/GCS loss. Cincinnati under Allison: 1976 DCS loss; 1977 DCS win/CCS loss; 1979 DCS loss; 1980 DCS loss.

## 11. Houston–Cincinnati postseason history

The complete almanac HTML verifies the conclusion: **Cincinnati reached the postseason four times before 1981—1976, 1977, 1979 and 1980—and Houston ended Cincinnati's season every time.** This is established from postseason schedules/bracket progression and game-specific recaps, linescores and box-score tables—not from the phase-less CSV.

Starters are listed away/home. “After” is Houston's series standing after the game.

### 1976 DCS — Houston won 4-3

| Date; ID | Site; final | Decision; starters | Major offense; Scott Reis | After |
|---|---|---|---|---|
| Oct. 6; 9339 | Houston Stadium; HOU 2-1 | W Scott Szell; L Rick Satchell; Satchell/Szell | Warren Ufena two-run HR; Reis 0-2, 2 BB | HOU 1-0 |
| Oct. 7; 9340 | Houston Stadium; CIN 10-5 | W Netz Altman; L Mark Saunders; Altman/Saunders | Reis 3-4, two doubles, three-run HR | Tied 1-1 |
| Oct. 9; 9341 | Cougars Park; HOU 4-3 | W Nate Higgins; L Jacob Sutherland; Higgins/Nate Souza | Bill Parlier go-ahead two-run HR; Reis 1-3, three-run HR, BB | HOU 2-1 |
| Oct. 10; 9342 | Cougars Park; CIN 3-2 (11) | W Nicolá Dallaglio; L Juan Pena; Santos Peneda/David Bain | Gilberto Perez walk-off HR; Reis 2-5, double | Tied 2-2 |
| Oct. 11; 9343 | Cougars Park; HOU 7-1 | W Scott Szell; L Rick Satchell; Szell/Satchell | Szell 8.1 IP, four hits; Reis 0-3, BB | HOU 3-2 |
| Oct. 13; 9344 | Houston Stadium; CIN 10-3 (10) | W Netz Altman; L Mark Saunders; Altman/Saunders | Altman 10 complete innings; Reis 2-5, solo HR, BB | Tied 3-3 |
| Oct. 14; 9345 | Houston Stadium; HOU 7-0 | W Brian Manchester; L Nate Souza; Souza/Manchester | Manchester shutout; Reis 0-3, BB | **HOU 4-3** |

Game 9345 eliminated Cincinnati. Houston next lost the CCS to New York, 4-3.

### 1977 CCS — Houston won 4-2

| Date; ID | Site; final | Decision; starters | Major offense; Scott Reis | After |
|---|---|---|---|---|
| Oct. 12; 9372 | Cougars Park; HOU 7-5 (11) | W Ryan Hines; L David Bain; Nate Higgins/Alex Perez | Ufena go-ahead two-run single; Reis 2-6, HR, 3 RBI | HOU 1-0 |
| Oct. 13; 9373 | Cougars Park; CIN 6-4 | W Rick Satchell; L Jose Martinez; Martinez/Satchell | Josh Young 3-4, 2 RBI; Reis 1-4, double, RBI, BB | Tied 1-1 |
| Oct. 15; 9374 | Houston Stadium; HOU 6-4 | W Scott Szell; L Jorge Nino; Nino/Szell | Brent Keyser HR, 3 RBI; Reis 0-2, BB | HOU 2-1 |
| Oct. 16; 9375 | Houston Stadium; HOU 8-6 | W Kyle Chambers; L Miguel Ayala; Paul Conlon/Mark Saunders | Reis 2-4, double, 2 RBI | HOU 3-1 |
| Oct. 17; 9376 | Houston Stadium; CIN 9-3 | W Alex Perez; L Nate Higgins; Perez/Higgins | Jay Raymond HR, 3 RBI; Reis 1-4, double | HOU 3-2 |
| Oct. 19; 9377 | Cougars Park; HOU 4-3 | W Ryan Hines; L Jorge Nino; Jose Martinez/Rick Satchell | Preston Strate was series MVP; Reis 0-5 | **HOU 4-2** |

Game 9377 eliminated Cincinnati. Houston next lost the GCS to Dallas, 4-3.

### 1979 DCS — Houston won 4-3

| Date; ID | Site; final | Decision; starters | Major offense; Scott Reis | After |
|---|---|---|---|---|
| Oct. 2; 9338 | Houston Stadium; HOU 4-2 | W Scott Szell; L Gil Medina; Medina/Szell | Apostolos Georghiou three-run HR and double; Reis DNP | HOU 1-0 |
| Oct. 3; 9339 | Houston Stadium; CIN 5-0 | W Alex Perez; L Nate Higgins; Perez/Higgins | Perez complete-game five-hit shutout; Reis DNP | Tied 1-1 |
| Oct. 5; 9340 | Cougars Park; CIN 3-2 | W Jorge Nino; L Kyle Chambers; Chambers/Rick Satchell | Ethan Harrison walk-off RBI single; Reis DNP | CIN 2-1 |
| Oct. 6; 9341 | Cougars Park; CIN 6-4 | W Isaiah Simmons; L Valentin Geffroy; Geffroy/Simmons | Casey Bracht HR; Reis DNP | CIN 3-1 |
| Oct. 7; 9342 | Cougars Park; HOU 9-7 | W Scott Szell; L Gil Medina; Szell/Medina | Brent Keyser three-run HR; Reis 1-4, RBI, BB | CIN 3-2 |
| Oct. 9; 9343 | Houston Stadium; HOU 6-4 | W Anthony Jackson; L Jorge Nino; Alex Perez/Nate Higgins | Alex Ramirez walk-off three-run HR; Reis 1-5, double, RBI | Tied 3-3 |
| Oct. 10; 9344 | Houston Stadium; HOU 4-1 | W Kyle Chambers; L Rick Satchell; Satchell/Chambers | Series MVP Ramirez had 8 RBI; Reis 2-3, triple, RBI, BB | **HOU 4-3** |

Game 9344 eliminated Cincinnati. Houston next beat Seattle 4-3 in the CCS and Phoenix 4-1 in the GCS to win the ABL championship.

### 1980 DCS — Houston won 4-2

| Date; ID | Site; final | Decision; starters | Major offense; Scott Reis | After |
|---|---|---|---|---|
| Oct. 8; 9361 | Cougars Park; HOU 9-6 | W Nate Higgins; L Alex Perez; Higgins/Perez | Alex Ramirez 3-5, double, two runs; Reis 1-4, double, RBI, BB | HOU 1-0 |
| Oct. 9; 9362 | Cougars Park; CIN 3-1 | W Gil Medina; L Scott Szell; Szell/Medina | Reis 1-4, HR, 2 RBI | Tied 1-1 |
| Oct. 11; 9363 | Houston Stadium; HOU 7-0 | W Kyle Chambers; L Rick Satchell; Satchell/Chambers | Chambers complete-game four-hit shutout; Reis 1-3, BB, SB | HOU 2-1 |
| Oct. 12; 9364 | Houston Stadium; CIN 12-2 | W Isaiah Simmons; L Dylan Wilson; Simmons/Wilson | Jose Hernandez 4-6, HR, double, 5 RBI; Reis 2-3, RBI, 3 BB | Tied 2-2 |
| Oct. 13; 9365 | Houston Stadium; HOU 6-5 | W Andrew Williamson; L Jordan Allison; Alex Perez/Nate Higgins | Jeff Thome go-ahead two-run single; Reis 3-4, double, two-run HR | HOU 3-2 |
| Oct. 15; 9366 | Cougars Park; HOU 5-4 | W Scott Szell; L Arthur Blin; Szell/Gil Medina | Warren Ufena series MVP; Reis 2-4, two runs | **HOU 4-2** |

Game 9366 eliminated Cincinnati. Houston next beat Las Vegas 4-1 in the CCS, then lost the GCS to Detroit, 4-3.

### Verified aggregates

- Houston vs. Cincinnati: **16-10 in postseason games and 4-0 in series**.
- Sanchez vs. Allison: **16-10**, all in these four series. Manager history verifies Sanchez for Houston and Allison for Cincinnati beginning in 1976.
- Reis vs. Houston: **22 games, 84 AB, 19 R, 28 H, 9 doubles, 1 triple, 6 HR, 22 RBI, 15 BB, 6 K, 1 SB; .333/.434/.679 (1.113 OPS)**. He did not appear in the first four games of the 1979 series.
- Cincinnati postseason under Allison: **14-16**, one series won and four lost.
- Houston postseason under Sanchez: **56-39**, 11 series won and five lost.

## 12. Cougars Park and game atmosphere

- Cougars Park; capacity **21,000**; `turf=0` and park `type=0`. No export dictionary was found to translate those two codes safely, so surface and roof status remain unresolved rather than inferred.
- Dimensions, left to right: **316, 339, 387, 418, 389, 346, 327 feet**.
- Raw park factors: AVG .9962, doubles 1.0928, triples 1.1258, HR 1.0018.
- July climate row: 76°F typical temperature, rain code 3, wind 9. These are park climate settings, not a verified game forecast.
- Attendance in the series: 20,818; 20,789; 20,962—each near the 21,000 capacity.
- Game time is present as an OOTP numeric time field, but weather/roof and day/night display labels were not safely resolved.

Sources: `parks.csv`, `games.csv`, `teams.csv`.

## 13. Questions to watch

1. Does Cincinnati turn a 2-1 series lead into a series win, or does Houston salvage the split?
2. Does the ABC Central margin move to 3.5 games or 5.5 games?
3. How does confirmed starter Geffroy handle a start after only five 1981 ABL relief appearances?
4. Can Collette extend a 1.72 ERA line against the division leader?
5. Does Reis add to a résumé already containing five MVPs and ten All-Star selections?
6. Does Cincinnati's +52 run differential continue to outperform Houston's +34 in this series?
7. Can Houston's 9-4 season-series advantage withstand Cincinnati's current two-game response?
8. Which recently used bullpen arms are available after the workloads listed in Section 3?

## 14. Postgame and Baseball Matters possibilities

- Division race: Cincinnati closes to 3.5, or Houston restores the margin to 5.5.
- Series verdict: Cincinnati wins 3-1, or Houston earns a 2-2 split.
- Starter contrast: Geffroy's unusual relief-to-start assignment versus Collette's established 1981 starting line.
- Reis: another game in the current season of a five-time MVP and ten-time All-Star.
- Baseball Matters: the difference between record (Houston) and run differential/recent form (Cincinnati).
- Baseball Matters: direct games convert standings pressure into exact, measurable movement.

Pregame frame: Houston owns first place, a 4½-game lead, and a 9-4 season-series advantage. Cincinnati owns the stronger run differential, the better post-break record, and a 2-1 lead in this four-game set. Confirmed starters Valentin Geffroy and Chris Collette bring sharply different 1981 usage profiles to a game that will move the division margin by exactly one game.

## 15. Source ledger

- Current official state: `games.csv`, `games_score.csv`, `game_logs.csv`, `team_record.csv`, `team_batting_stats.csv`, `team_pitching_stats.csv`.
- Players and game lines: `players.csv`, `players_game_batting.csv`, `players_game_pitching_stats.csv`, `players_career_batting_stats.csv`, `players_pitching.csv`.
- Personnel: `team_roster_staff.csv`, `coaches.csv`, `team_roster.csv`, `players_roster_status.csv`.
- History and awards: `players_awards.csv`, `team_history.csv`, `team_history_record.csv`, and the complete authoritative `data_raw/ootp_html/almanac_1972.zip` through `almanac_1980.zip`, including postseason schedules, bracket/series pages, and individual `box_scores/game_box_*.html` files.
- Park: `parks.csv`, `teams.csv`.
- Supplemental StatsPlus: not used.

## 16. Unresolved facts

- Complete expected batting orders, defensive lineup, bench hierarchy, and closer hierarchy: no confirmed Thursday lineup field and unresolved roster-list display codes.
- Velocity codes 6 and 9 to MPH: no mapping dictionary found in the export or repository.
- Pronunciations, birthplace display names, and nationality display labels: ID tables exist, but a verified production translation was not completed.
- Pitcher home/road, platoon, opponent-career, and postseason splits: not safely separated from available aggregate/split rows.
- Complete Reis career rankings and records pursued remain unresolved; his postseason meetings with Houston are resolved in Section 11.
- Manager awards and tendency display meanings remain unresolved; postseason W-L and head-to-head records are resolved in Sections 9-11.
- Current-game weather and confirmed attendance: not populated for unplayed game 1161.

**COMPLETE FOR THE VERIFIED PRE-1981 HOUSTON–CINCINNATI POSTSEASON HISTORY**
