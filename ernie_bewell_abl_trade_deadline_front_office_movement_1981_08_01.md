# ERNIE BEWELL — ABL FRONT-OFFICE MOVEMENT EVIDENCE PACKET

**Reporting window:** 1980–81 offseason through the OOTP trade deadline  
**Deadline snapshot:** August 1, 1981; latest completed game date July 31, 1981  
**Purpose:** Reporter’s notebook and assignment desk; not the finished article  
**Editorial question:** What are the teams doing?

## 1. Scope and verified cutoff

### Season boundary and transaction window

- The 1980 regular season is reconstructed as April 6 through October 5. The historical game-level almanac has 1,944 club-vs-club games in that interval after excluding two All-Star games; those results reproduce all 24 major-league clubs’ 162-game records in team_history_record.csv.
- The 1980 postseason continued after the regular season. The latest ABL game in the historical game file is November 1, 1980: game 1980_1101_7, Detroit 8 at Houston 3. team_history.csv marks Detroit as having won the playoffs. Thus November 1 is the best-supported end date for the completed 1980 ABL campaign; November 2 is the inferred beginning of the ensuing offseason.
- The game-level 1980 file is a generated almanac product rather than a contemporaneous raw game export. The regular-season date window and final-game date are therefore **STRONG INFERENCE**, cross-checked to the raw historical record tables; they are not presented as a raw-export timestamp.
- The first meaningful major-league transaction notice after the season ended is November 10, 1980, message #18750: New York’s Pedro Marquez agreed to remain for three more years. The first structured trade is November 11, #18759.
- OOTP’s ABL league row (league_id 200 in leagues.csv) has trading_deadline=1, trade_deadline_date=1981-7-31, and current_date=1981-8-1. The configured trade deadline is July 31, not August 1. The trade log’s latest row is July 31; four trades are dated that day. No type-1 trade or type-2 signing/contract message is dated August 1.
- The authoritative 1981 game schedule has 1,944 regular-season slots. It contains 1,247 completed games through July 31. All 24 team records in team_record.csv reproduce from completed regular-season game rows (24/24); July 31 was 12 scheduled / 12 played. August 1 was 12 scheduled / 0 played. The entering baseball state is therefore July 31, with the league calendar one day later.
- Twenty-two teams had 104 decisions; Houston and St. Louis had 103 each. This is the schedule state, not evidence of missing results: there are no unplayed regular-season slots dated on or before July 31.

### Completeness judgment

**Complete through the configured trade deadline for structured trades: YES.** trade_history.csv contains 66 trades dated November 11–July 31, and the matching OOTP transaction messages support the transaction identities. All four July 31 trades are in both sources. The raw messages file itself reaches August 1.

**Complete for every possible transaction class: NOT PROVEN.** There is no separate exhaustive transaction-history export for waivers, releases, Rule 5 selections, or every roster assignment. messages.csv contains dated transaction notices, and players_roster_status.csv is a current snapshot, but that is not a substitute for a complete historical transaction register. No waiver/release transaction was found in the reviewed notices; that absence is not proof that none occurred. Accordingly, this packet treats 66 trades and 46 message_type=2 contract/signing notices as the verifiable transaction universe, and labels the broader absence of other transaction classes as unknown.

The August 1 scope is still valid: OOTP says the deadline had already passed on July 31, the input is dated August 1, and no eligible trade notice is missing from that configured deadline day. The desk does not claim a trade could be made on August 1.

## 2. Source map

### Primary current sources

- csv/ootp_csv/leagues.csv — ABL season start, current date, configured deadline, schedule size, draft dates, postseason configuration.
- csv/ootp_csv/trade_history.csv — structured trade rows, team sides, player IDs, cash/pick/IAFA side arrays.
- csv/ootp_csv/messages.csv — transaction notice text, message IDs, dates, contemporaneous OOTP descriptions; message type is retained as exported and is not treated as a full transaction taxonomy.
- csv/ootp_csv/teams.csv, sub_leagues.csv, divisions.csv — club identity, conference, division.
- csv/ootp_csv/games.csv, games_score.csv, game_logs.csv — 1981 schedule/results, game identity, current-state reconstruction.
- csv/ootp_csv/team_record.csv — current team-record snapshot checked against completed game results.
- csv/ootp_csv/team_batting_stats.csv and team_pitching_stats.csv — current 1981 team performance through the snapshot; used as descriptive context, not as evidence of what a GM knew earlier.
- csv/ootp_csv/team_roster_staff.csv and coaches.csv — current GM/owner links and current exported personnel fields.
- csv/ootp_csv/players.csv, players_roster_status.csv, players_contract.csv, players_salary_history.csv, players_contract_extension.csv — current player identity, roster/contract state, and draft information. These do not form an event-by-event historical roster log.
- csv/ootp_csv/league_playoffs.csv — configured postseason rounds and wild-card structure.
- docs/ABL_POSTSEASON_RULES.md — repository authority for qualification and the 1981-and-forward same-division DCS protection.
- csv/ootp_csv/team_history_record.csv and team_history.csv — historical records and playoff outcome.

### Historical and control sources

- csv/out/almanac/1980/games_1980_league200.csv — generated historical game-level schedule/results used to reconstruct the 1980 regular-season and final postseason dates.
- csv/out/story/production/abl_trade_deadline_report_1981_08_01_asof_1981-07-31.md — companion trade report, including the checked transaction ledger, at-trade ages/positions, major-move performance breakdowns, and seven-pass deadline certification.
- csv/out/story/production/abl_trade_deadline_ledger_1981_08_01_asof_1981-07-31.csv — 182 player-transfer rows, IDs, transaction teams, and current-roster exceptions.
- docs/ABL_DATA_CATALOG.md and docs/ABL_SOURCE_AUTHORITY_RULES.md — source tier and freshness guidance.
- Official OOTP personnel guidance (OOTP Wiki, “Personnel Ratings and Tendencies”): https://wiki.ootpdevelopments.com/index.php?title=OOTP_Baseball:Important_Game_Concepts/The_Personnel_Model/Personnel_Ratings_and_Tendencies
- Official OOTP 23 ownership manual: https://manuals.ootpdevelopments.com/index.php?man=ootp23&page=ownership

The OOTP personnel/ownership pages explain game concepts, but are not a schema dictionary for this OOTP 26 CSV export. No repository field dictionary was found that maps the raw numeric GM/owner codes to journalist-facing categories. The online manuals are therefore context only; raw values below are not translated into labels.

## 3. Executive baseball summary

- The transaction record is unusually busy before the season and again near the deadline: 36 of 66 trades occurred before the April 6 opener; only three more occurred from opening day through June 30; 27 occurred in July. July 16–31 accounts for 26 trades; four fell on July 31.
- Every one of the 24 ABL clubs appears in at least one trade. Houston led with 12 trades; Dallas and Miami had eight each; Chicago, San Diego, and Seattle had seven each. Los Angeles was quietest with one. Charlotte and Philadelphia had three each.
- The 66 trades contain 182 player-transfer rows representing 179 distinct players. Three IDs appear twice: Jerry Wilson #681, Vinny Martinez #8329, and Rodolfo Gonzalez #9886. Houston sent the most player rows (21); Miami sent 14; Houston received 14 and Dallas received 12.
- There were 46 OOTP message_type=2 contract/signing notices in the defined offseason-to-deadline window. Some are re-signings/extensions, not acquisitions. The messages show substantial offseason contract activity—especially Atlanta’s seven notices, St. Louis’s five, and four each for Phoenix, Nashville, and Seattle—followed by four such notices in-season.
- The trade market did not reduce cleanly to buyers and sellers. San Diego (39–65 at cutoff) moved a veteran starter to division-leading Detroit and a 17-save reliever to division-leading San Francisco; Minneapolis dealt starter Chris Jenkins to leading Houston on July 31; but Houston was simultaneously the league’s most active club by trades while sending more player rows out than it received.
- Current standings leave six division leaders and two provisional wild-card leaders: Charlotte, Detroit, and San Francisco in the NBC; Boston, Houston, and Las Vegas in the ABC; Dallas and Seattle lead the non-division-winner races. Miami and Chicago are each one game behind Dallas; Denver is two behind Seattle.
- The roster evidence supports specific transaction descriptions. It does not prove private intent, negotiations, owner instructions, or that a club was “all-in,” rebuilding, or protecting a lead.

## 4. League transaction timeline

| Period | Verifiable movement | Reporting note |
|---|---:|---|
| November 2–April 5, offseason/preseason | 36 trades; 42 type-2 contract/signing notices | The clubs were 0–0 in the new season. There was no standings context to import into these decisions. |
| April 6–May 31, opening/early season | No completed trade; three type-2 notices in May | Quiet in trades, but not a blank transaction period. |
| June | Three trades: June 11, 18, 25 | First in-season trade was June 11, #19397. |
| July 1–15, All-Star period | One trade, July 2 #19578; All-Star date July 14 | No trade July 3–15. |
| July 16–31 | 26 trades; one additional type-2 contract notice July 17 | Trade activity accelerates sharply after the All-Star date. |
| July 31 | Four trades: #19901, #19904, #19907, #19908 | This is the configured trade-deadline date. |
| August 1 | No trade or signing/extension notice; 12 games scheduled, none played | OOTP current date is Aug. 1; deadline field is July 31. |

### In-season standings at the transaction date

Because transaction records have dates but no time-of-day, the standings below are reconstructed through the previous completed game date. Same-day results are excluded rather than guessed to precede or follow a transaction. “GB” is the club’s division-leader gap; ties are not broken for this desk. All 36 earlier trades occurred before opening day, when both clubs were 0–0.

| Trade date / ID | Club A before trade | Club B before trade |
|---|---|---|
| Jun 11 #19397 | Cincinnati 33–27, 1 GB | Boston 30–30, 0 GB |
| Jun 18 #19446 | Denver 37–29, 1 GB | New York 31–35, 2 GB |
| Jun 25 #19497 | Houston 42–31, 0 GB | Atlanta 28–45, 17 GB |
| Jul 2 #19578 | Cincinnati 40–40, 7 GB | Minneapolis 34–46, 10 GB |
| Jul 16 #19729 | Nashville 45–44, 7 GB | New York 41–48, 4 GB |
| Jul 23 #19793 | Denver 51–45, 5 GB | Atlanta 42–54, 16 GB |
| Jul 24 #19798 | Tampa Bay 43–54, 16 GB | Dallas 52–45, 0 GB |
| Jul 24 #19799 | Dallas 52–45, 0 GB | Pittsburgh 40–57, 12 GB |
| Jul 24 #19800 | San Diego 36–61, 18 GB | Charlotte 59–38, 0 GB |
| Jul 25 #19806 | San Francisco 55–43, 0 GB | San Diego 37–61, 18 GB |
| Jul 25 #19807 | New York 45–53, 7 GB | Chicago 52–46, 1 GB |
| Jul 26 #19814 | New York 45–54, 7 GB | Detroit 54–45, 0 GB |
| Jul 27 #19849 | Dallas 53–47, 1 GB | San Diego 38–62, 19 GB |
| Jul 27 #19850 | Chicago 53–47, 1 GB | Tampa Bay 44–56, 16 GB |
| Jul 27 #19851 | Boston 52–48, 0 GB | Chicago 53–47, 1 GB |
| Jul 27 #19852 | St. Louis 46–53, 11 GB | Miami 52–48, 8 GB |
| Jul 27 #19853 | Tampa Bay 44–56, 16 GB | Houston 57–42, 0 GB |
| Jul 27 #19854 | Las Vegas 58–42, 0 GB | Portland 38–62, 20 GB |
| Jul 28 #19865 | Houston 57–42, 0 GB | Phoenix 51–49, 5 GB |
| Jul 28 #19866 | St. Louis 46–53, 11 GB | Seattle 55–46, 4 GB |
| Jul 28 #19867 | San Diego 38–62, 18.5 GB | Detroit 55–46, 0 GB |
| Jul 28 #19868 | Miami 53–48, 7 GB | Pittsburgh 43–58, 9.5 GB |
| Jul 29 #19877 | Miami 54–48, 7 GB | Phoenix 52–49, 4.5 GB |
| Jul 29 #19878 | Las Vegas 59–43, 0 GB | Pittsburgh 43–59, 10.5 GB |
| Jul 29 #19879 | Houston 58–42, 0 GB | San Diego 39–62, 17.5 GB |
| Jul 30 #19889 | St. Louis 47–54, 12 GB | Houston 59–42, 0 GB |
| Jul 31 #19901 | Boston 53–50, 0 GB | Chicago 55–48, 1 GB |
| Jul 31 #19904 | Pittsburgh 43–60, 10 GB | Houston 60–42, 0 GB |
| Jul 31 #19907 | Atlanta 46–57, 15 GB | Chicago 55–48, 1 GB |
| Jul 31 #19908 | Minneapolis 45–58, 11 GB | Houston 60–42, 0 GB |

### Complete structured trade inventory

The table below lists all 66 structured trades and every player ID in the two trade sides. It is an identity/movement index; player names are joined to the current player master. For at-trade position and age, full OOTP message descriptions, post-trade use, cash/retention notes, and performance context, use the companion verified trade report and ledger listed above. Current roster positions must not silently replace at-trade positions.

| Date | ID | Side 0 sent | Side 1 sent | Other structured term |
|---|---:|---|---|---|
| 1980-11-11 | #18759 | Minneapolis: Jose Rodriguez #4644; Justin Reed #10732 | Portland: Jesus Lopez #8176 |  |
| 1980-11-27 | #18791 | Minneapolis: Aires Azeredo #9290; Jesus Perez #7431; Eddy Cruz #8481 | Dallas: Bobby Kinney #2640; Elias Garza #7953 |  |
| 1980-11-30 | #18792 | Las Vegas: Edwin Garcia #10731 | Denver: Edgar Velo #2874 |  |
| 1980-12-01 | #18798 | Pittsburgh: Emilio Vazquez #7879; Kiyoharu Tamura #6960 | Nashville: Bob Newman #997 |  |
| 1980-12-05 | #18801 | Atlanta: Orlando Hernandez #8133 | Seattle: Mike Hawkins #1984 |  |
| 1980-12-06 | #18803 | Miami: Jose Trujillo #4517; Danny Munoz #6301 | Seattle: Oscar Romero #5770 |  |
| 1980-12-06 | #18804 | San Francisco: Dale O'Connell #6404 | Denver: Warren Callery #1133; Oscar Brumley #9455 |  |
| 1980-12-06 | #18805 | Dallas: Joe Turner #438 | Boston: Bryce Rose #292; Marcos Henriquez #8452 |  |
| 1980-12-07 | #18807 | Nashville: Jorge Galvez #1 | Boston: Juan Grant #10086 |  |
| 1980-12-10 | #18818 | Detroit: Victor Cerda #9589 | New York: Marcos Real #577 |  |
| 1980-12-13 | #18824 | Nashville: Bill Borden #1987 | Miami: Marco Figueroa #3624; Fernando Gomez #9939; Vinny Martinez #8329; Justin Campbell #9369 |  |
| 1980-12-14 | #18825 | Cincinnati: Oscar Rebeles #6293; Pedro Camacho #6819; Julio Rodriguez #8385 | Philadelphia: Luis Mata #4703 |  |
| 1980-12-14 | #18826 | San Diego: Ben Sabine #5882; Elijah Glenn #6330 | Houston: Serge Mazet #8344; Isaiah Hurdle #6267 |  |
| 1980-12-15 | #18835 | Atlanta: Mario Amador #9081 | Philadelphia: Ricky Velez #2284; Ismael Hernandez #7567 |  |
| 1980-12-15 | #18836 | Las Vegas: Manny Madrigal #8544 | St. Louis: Juan Navarro #866; Willie Simental #7083 |  |
| 1980-12-16 | #18839 | Cincinnati: Pascal Magana #7232 | Nashville: Darius Jones #5895 |  |
| 1980-12-16 | #18840 | Houston: Amari Baggs #6976 | Seattle: Albert Chavez #4686 |  |
| 1980-12-17 | #18841 | Charlotte: Alejandro Ordaz #5436 | Miami: Josh Maras #955 |  |
| 1980-12-17 | #18842 | Detroit: Fernando Labrado #8804; Noah Burks #10089 | San Diego: Mike Dominguez #2614 |  |
| 1980-12-18 | #18843 | Miami: Frank Duarte #9615; Yoshihiro Murakami #9299 | Dallas: Gary McMinn #2876 |  |
| 1980-12-19 | #18846 | Charlotte: David Jenkins #8961 | Seattle: Cameron Jones #4776 |  |
| 1980-12-22 | #18855 | Los Angeles: Luis Prieto #6223; Miguel Campos #8994 | Miami: Luis Davis #7015 |  |
| 1980-12-25 | #18857 | Tampa Bay: Matt Awtry #2639; Oscar Lopez #8336 | Portland: Hector Lespier #7926 |  |
| 1980-12-26 | #18858 | Portland: Mike Stanley #2430; Joe Davis #7674 | Philadelphia: Nick Jones #7585 |  |
| 1980-12-31 | #18869 | Atlanta: Elias Ruela #6567 | Dallas: Tony Resendiz #4827 |  |
| 1981-01-02 | #18870 | Seattle: Trevon Washington #5361; Georber Alvarez #6092 | Las Vegas: Danny Sanchez #4811 |  |
| 1981-01-03 | #18873 | Chicago: Danny Brown #6321 | San Francisco: Jerry Wilson #681; Dave Jackson #7776 |  |
| 1981-01-11 | #18884 | New York: Martin Martinez #4422 | Detroit: Jonathan Heberling #9444; Felix Jaquez #10153; Carlos Escobar #8568 |  |
| 1981-01-14 | #18891 | Houston: Ricky Correa #5487 | San Francisco: Nick Case #3133 |  |
| 1981-01-22 | #18901 | Phoenix: Bobby Perez #9888 | Seattle: Jose Contreras #677 |  |
| 1981-01-29 | #18909 | St. Louis: Miguel Salgado #4611 | Houston: Jaden Antunez #6875 |  |
| 1981-02-12 | #18924 | Phoenix: Jordan Pedroza #7393 | Cincinnati: Jorge Nino #4409 |  |
| 1981-02-14 | #18925 | Houston: Jayden Martin #8256; Juan Santos #7757; Tony Rodriquez #9288; Jeremiah Matthews #7545 | Minneapolis: Jesus Vasquez #4495 |  |
| 1981-02-21 | #18931 | Dallas: Corey Nelson #7640 | Nashville: Bill Rambow #841 |  |
| 1981-03-04 | #18945 | Tampa Bay: Willie Leyba #6705 | Chicago: Dave Guadalupe #8805 |  |
| 1981-03-06 | #18948 | Portland: Noah White #8837 | Boston: Renault Buso #2816 |  |
| 1981-06-11 | #19397 | Cincinnati: Jose Canas #7646 | Boston: Jason Brown #1834 |  |
| 1981-06-18 | #19446 | Denver: Ramon Pazmino #9436 | New York: Tony Oubre #949; Willie Vega #7820 |  |
| 1981-06-25 | #19497 | Houston: Ernesto Durazo #7173; Orlando Cervini #7230 | Atlanta: Mike McQuaid #416 |  |
| 1981-07-02 | #19578 | Cincinnati: Alex Rodriguez #7177; Jorge Moya #9169 | Minneapolis: Lee Byron #3989 |  |
| 1981-07-16 | #19729 | Nashville: David Mitchell #6796; Vinny Martinez #8329 | New York: Devon John #4921 |  |
| 1981-07-23 | #19793 | Denver: Benito Rodriguez #11244; Rodolfo Gonzalez #9886 | Atlanta: Mason Chaisson #1540 |  |
| 1981-07-24 | #19798 | Tampa Bay: Brian Manchester #1656 | Dallas: Jonathan Funches #6782 |  |
| 1981-07-24 | #19799 | Dallas: Ricky Perez #8879; Marcos Guerrero #6438 | Pittsburgh: Doug Farmer #2859 |  |
| 1981-07-24 | #19800 | San Diego: Yuan-zhang Tseng #2162 | Charlotte: Jairo Reyna #6327; Tony Chavez #9619 |  |
| 1981-07-25 | #19806 | San Francisco: Omari Benjamin #5204 | San Diego: Victor Hernandez #504 |  |
| 1981-07-25 | #19807 | New York: Juan Mederos #454 | Chicago: David Acosta #7107; Alex Castellanos #9305 |  |
| 1981-07-26 | #19814 | New York: Chris Allen #536 | Detroit: Hector Wilson #10151; Bobby Ramirez #8373 |  |
| 1981-07-27 | #19849 | Dallas: Josh Brown #9507 | San Diego: Eric Strickland #1737 |  |
| 1981-07-27 | #19850 | Chicago: Tony Espinal #3086; Luis Robles #10110 | Tampa Bay: Ed Montoya #2039 |  |
| 1981-07-27 | #19851 | Boston: Mike Roede #323; Desidério Chora #9908 | Chicago: Victor Hernandez #6732 |  |
| 1981-07-27 | #19852 | St. Louis: Joe Puerta #671 | Miami: Daniel Smith #6345 |  |
| 1981-07-27 | #19853 | Tampa Bay: Jeff Mitchell #2790 | Houston: Jose Carranza #9320; Matt Johnson #9625 |  |
| 1981-07-27 | #19854 | Las Vegas: Manny Moran #7215; Luis Gortarez #8610 | Portland: Nate Souza #1688 |  |
| 1981-07-28 | #19865 | Houston: Alex Monge #5242 | Phoenix: Kerry Darling #679; Willie Saunders #9015 |  |
| 1981-07-28 | #19866 | St. Louis: Harry Benjamin #5944 | Seattle: Isaiah Gibson #6307 |  |
| 1981-07-28 | #19867 | San Diego: Kennie Murdock #4963 | Detroit: Bob Morales #9196 |  |
| 1981-07-28 | #19868 | Miami: Joe Good #3032; Quadir May #9550 | Pittsburgh: Daniel Vega #8655 | cash from side 0=144000 |
| 1981-07-29 | #19877 | Miami: Omar Garcia #1184 | Phoenix: Addie de Ruijter #1849 |  |
| 1981-07-29 | #19878 | Las Vegas: Carlos Beltran #9832; Marcelino Bacardi #8525 | Pittsburgh: Bill Hall #535 |  |
| 1981-07-29 | #19879 | Houston: Tony Contreras #7860 | San Diego: Jerry Wilson #681 |  |
| 1981-07-30 | #19889 | St. Louis: Chris Sullivan #1547 | Houston: Omar Han #6286; Antonio Gama #8529 |  |
| 1981-07-31 | #19901 | Boston: Justin Bowen #854; Javy Madrigal #10507 | Chicago: Miguel Martinez #4347 |  |
| 1981-07-31 | #19904 | Pittsburgh: Ricky Bautista #1990 | Houston: Tim Mathis #7676; Tyler Hayes #8767 |  |
| 1981-07-31 | #19907 | Atlanta: Bill Torres #8235; Rodolfo Gonzalez #9886 | Chicago: Jaden Pierce #5785 |  |
| 1981-07-31 | #19908 | Minneapolis: Chris Jenkins #6159 | Houston: Jonathan Serrano #7643; Alush Demachi #6369 |  |

The structured rows contain one explicit cash payment: $144,000 from Miami’s side in #19868. The companion report records 21 OOTP messages mentioning partial contract retention without a stated amount. No draft-pick or IAFA-cap asset rows are present in the 66 structured trades. The trade_history summary string is not treated as complete when it approaches its 255-character limit; the player side arrays and message IDs control.

### Contract/signing notices

This table contains all 46 message_type=2 entries from November 2 through August 1. It is not a claim that all 46 are external acquisitions: several are extensions or re-signings. Contract amounts/terms below should be taken from the specific OOTP body only after checking the companion report or message row. OOTP-generated commentary about fans, negotiations, player reputation, or motives is not independently reported as fact.

| Date | Message | Club | Player ID | OOTP subject |
|---|---:|---|---|---|
| 1980-11-10 | #18750 | New York | Pedro Marquez #704 | New York, Veteran Marquez Sign Deal  |
| 1980-11-19 | #18780 | Denver | Boboy Nierva #122 | Denver, Nierva Agree to Extension Deal |
| 1980-11-21 | #18781 | Chicago | Shane Cobb #196 | Cobb, Chicago Sign 4-Year Pact  |
| 1980-12-04 | #18799 | Atlanta | Eric Gray #1054 | 2B Eric Gray Ends Speculation; Signs with Atlanta |
| 1980-12-05 | #18800 | Detroit | Ethan Harrison #607 | Right Fielder Ethan Harrison Lands in Detroit |
| 1980-12-06 | #18802 | Cincinnati | Tony Jones #4381 | RF Tony Jones, Cincinnati Agree to Pact |
| 1980-12-07 | #18806 | Tampa Bay | Nate Smeltz #1005 | Storm Win Sweepstakes for 2B Smeltz |
| 1980-12-08 | #18808 | Detroit | Luis Rios #649 | Rios Going Nowhere, Re-Signs |
| 1980-12-08 | #18809 | Atlanta | Jordan Santistevan #4545 | 1B Jordan Santistevan Signs Contract with Atlanta |
| 1980-12-09 | #18815 | Atlanta | Luis Rosales #387 | 3B Rosales Inks $1,000,000 Annual Pact with Atlanta |
| 1980-12-10 | #18816 | Houston | Brent Keyser #19 | Veteran Keyser Signs with Mavericks |
| 1980-12-10 | #18817 | Atlanta | Steve Farber #1360 | RP Farber Takes Atlanta's 3-Year Offer |
| 1980-12-11 | #18819 | Atlanta | Netz Altman #1309 | SP Altman Lands with the Kings |
| 1980-12-12 | #18820 | St. Louis | Victor Campos #785 | St. Louis Negotiates 5-Year Pact with 2B Campos |
| 1980-12-12 | #18821 | Philadelphia | Jose Cespedes #4940 | Philadelphia Inks Cespedes to $4,640,000 Deal |
| 1980-12-13 | #18823 | Atlanta | Sergio Duran #11341 | Atlanta Inks Star RP to $572,000 Deal |
| 1980-12-15 | #18827 | Atlanta | Mike McQuaid #416 | RP McQuaid Signs on with Atlanta |
| 1980-12-15 | #18828 | Seattle | Jesse Street #1490 | RP Jesse Street Signs Contract with Seatlle |
| 1980-12-15 | #18829 | Nashville | Joe Berger #1788 | Closer Joe Berger Lands in Nashville |
| 1980-12-22 | #18847 | Seattle | Nelson Majano #1262 | RP Majano Agrees to Seatlle's Offer |
| 1980-12-22 | #18848 | Detroit | Brett Helvie #1857 | New Deal for Helvie in Detroit |
| 1980-12-25 | #18856 | Las Vegas | Kyle Marks #521 | Veteran Marks, Las Vegas Complete Deal |
| 1980-12-28 | #18859 | Phoenix | Kerry Darling #679 | Firebirds Negotiate $3,060,000 Deal with Darling |
| 1980-12-30 | #18866 | Charlotte | Dustin Meredith #286 | Charlotte, LF Meredith Reach Contract Agreement |
| 1980-12-30 | #18867 | San Francisco | Chad Averitt #2249 | 3B Chad Averitt Settles on San Francisco |
| 1980-12-30 | #18868 | San Francisco | Kyle Chambers #5867 | SP Kyle Chambers Inks Deal with San Francisco |
| 1981-01-03 | #18871 | St. Louis | Alex Lander #779 | $2,160,000 Brings Alex Lander to St. Louis |
| 1981-01-03 | #18872 | Nashville | Jacob Sutherland #1537 | RP Sutherland, Nashville Reach Accord |
| 1981-01-07 | #18881 | Nashville | Frank Larson #38 | Larson Signs $244,000 Pact with Nashville |
| 1981-01-09 | #18882 | Seattle | Antonio Romo #26 | SP Romo Excited to Play for Seatlle |
| 1981-01-09 | #18883 | Seattle | Eduardo Gonzales #952 | RP Gonzales Selects Seatlle's $258,000 Deal |
| 1981-01-15 | #18892 | Phoenix | Joe Rogers #129 | RP Rogers Picks Phoenix's $1,300,000 Offer |
| 1981-01-15 | #18893 | Nashville | Adam Rapoport #853 | RP Rapoport, Nashville Announce Agreement |
| 1981-01-20 | #18900 | Dallas | Ryan Millich #486 | RP Millich Lands $264,000 Deal with Dallas |
| 1981-01-29 | #18908 | Chicago | Andrew Flugence #5316 | Fire Sign RP Andrew Flugence |
| 1981-01-30 | #18910 | St. Louis | Andrew Williamson #7 | RP Williamson Inks $610,000 Annual Pact with St. Louis |
| 1981-01-31 | #18912 | Phoenix | Eric Strickland #1737 | Phoenix, Strickland Hash out Details of Contract |
| 1981-02-01 | #18913 | St. Louis | Joe Puerta #671 | RP Puerta Signs New Pact with St. Louis |
| 1981-03-04 | #18944 | Detroit | Bill Hackman #1000 | RP Hackman Signs with Detroit |
| 1981-03-06 | #18946 | Las Vegas | Jon Dominguez #1217 | C Dominguez Reaches Agreement with Gamblers |
| 1981-03-17 | #18962 | Las Vegas | Eddie Rodriguez #4522 | Done Deal! 1B Rodriguez Signs with Gamblers |
| 1981-03-25 | #18969 | Charlotte | Juan Lomeli #999 | RP Lomeli to Colonels |
| 1981-05-08 | #19161 | Phoenix | Nick Perez #2025 | Extension Keeps Perez in Phoenix |
| 1981-05-13 | #19198 | Dallas | Alfredo Noriega #716 | Dallas Announces Signing of RP Noriega |
| 1981-05-16 | #19212 | St. Louis | Andrew Williamson #7 | Williamson Extends Deal with St. Louis |
| 1981-07-17 | #19735 | Miami | Logan Ellsworth #249 | RF Ellsworth Reaches Agreement with Hurricanes |

## 5. Offseason: how the 1981 clubs were built

The offseason ran from the day after the November 1 final ABL game through the April 5 eve of the 1981 opener. The first meaningful major-league notice was New York’s three-year retention of 35-year-old Pedro Marquez (#18750, November 10); the first trade followed a day later. A separate November 9 retirement notice named the Boise Majestics, outside the 24 ABL major-league teams; it is not counted as one of these clubs’ major-league additions or losses.

The offseason was a two-track market: 36 trades and 42 type-2 contract/signing notices. Multi-year examples include Tampa Bay’s Nate Smeltz (seven years, $10.34 million in the OOTP notice), Atlanta’s Jordan Santistevan (eight years, $10.88 million), Cincinnati’s Tony Jones (six years at $1.1 million per year), Detroit’s Brett Helvie (five years, $7.58 million), and Phoenix’s Kerry Darling (three years, $3.06 million). These are notice-reported terms, not independent salary-history events; amounts do not by themselves establish intent or future value.

Atlanta leads the contract-notice count with seven, including Eric Gray, Jordan Santistevan, Luis Rosales, Steve Farber, Netz Altman, Sergio Duran, and Mike McQuaid. The set spans infielders and pitchers rather than one single role. St. Louis had five contract notices, including Victor Campos and Alex Lander plus Andrew Williamson’s two separate deal events (the January notice and May extension). Phoenix, Nashville, and Seattle had four each. A contract count is not a count of new players: renewals and extensions are included.

The late-November/December market included many established players changing clubs and young/minor-league players in return packages. The transaction table and companion expanded ledger are the authority for every two-way package. Without timestamped negotiation records, message prose about “sweeps,” “voids,” pressure, or an intended role is not used as evidence of what a GM believed.

## 6. In-season movement before the All-Star break

No completed trade is recorded from Opening Day (April 6) through May. Three May type-2 contract notices are present: Nick Perez’s Phoenix extension (#19161, May 8), Alfredo Noriega’s Dallas signing (#19198, May 13), and Andrew Williamson’s St. Louis extension (#19212, May 16). A fourth in-season contract notice came July 17, when Miami signed Logan Ellsworth (#19735).

Three trades followed in June: Cincinnati–Boston (#19397, June 11), Denver–New York (#19446, June 18), and Houston–Atlanta (#19497, June 25). The June 18 deal brought Tony Oubre to Denver, with minor-league 2B Willie Vega, while Ramon Pazmino went to New York. The companion performance audit separates the June 18 game from later-date starts because transaction timestamps are absent. Denver went from 37–29, one game off its division lead before the move, to 56–48 by the August 1 snapshot and second in the ABC wild-card order.

Cincinnati–Minneapolis traded July 2 (#19578), before the July 14 All-Star date. The next completed trade was July 16. Do not use August 1 standings to imply what a June decision-maker knew: the transaction-date table above carries the last completed standings before each dated trade.

## 7. Post-All-Star market and deadline week

The post-All-Star sequence concentrates the season’s movement: 26 trades from July 16 through July 31. July 27 was the busiest single day with six trades. The July 24–31 span alone had 24 trades. Activity did not come only from division leaders; clubs well outside the race also moved players, and several clubs made moves in both directions.

The deadline-day package contains four trades:

- #19901: Boston sent RHP Justin Bowen and minor-league 3B Javy Madrigal to Chicago for 2B Miguel Martinez.
- #19904: Pittsburgh sent RHP Ricky Bautista to Houston for minor-league RF Tim Mathis and RHP Tyler Hayes.
- #19907: Atlanta sent minor-league 2B/CF Bill Torres and LHP Rodolfo Gonzalez to Chicago for minor-league LF Jaden Pierce.
- #19908: Minneapolis sent LHP Chris Jenkins to Houston for minor-league LFs Jonathan Serrano and Alush Demachi.

The July 25–31 moves with clearest present-race context include San Diego’s July 25 deal sending reliever Victor Hernandez #504 to NBC West leader San Francisco for catcher Omari Benjamin; Detroit’s July 28 acquisition of starter Kennie Murdock from San Diego for minor-league RHP Bob Morales; and the July 31 Jenkins deal from 45–58 Minneapolis to 60–42 Houston (both clubs’ records through July 30, immediately before the transaction date). Boston acquired 1B Victor Hernandez #6732 from Chicago July 27, then 2B Miguel Martinez from Chicago July 31. These moves are roster facts; they do not prove the clubs’ private assessments or expected postseason odds.

### August 1 deadline day

OOTP’s date field says the trade deadline was July 31. August 1 is a post-deadline snapshot date, not the deal date. There are no trade_history rows or type-1/type-2 transaction messages dated August 1. The 12 games on that date are scheduled but unplayed, so this desk reports neither their results nor any downstream game story. No transaction is imputed from a rumor/trade-talk message.

## 8. Team-by-team reporting index

| Club | GM | Owner | Record / division position | Tournament position at cutoff | Trades; player rows in/out | Type-2 contract notices | Direction / confidence |
|---|---|---|---|---|---:|---:|---|
| Miami | Daniel Baker | Danny Leon | 55–49; NBC East 2nd | One game behind NBC WC leader Dallas | 8; 9 / 14 | 1 | Mixed active movement; WATCH |
| Phoenix | Bryan Somers | Mike Dutcher | 53–51; NBC West 2nd | Five games behind SF division lead; outside WC lead | 4; 4 / 5 | 4 | Pitching/contract depth; WATCH |
| Atlanta | Hector Naranjo | Tyler Carter | 46–58; NBC East 3rd/4th record tie | Outside tournament places | 6; 9 / 7 | 7 | High offseason churn; PROVEN activity, direction unclear |
| Dallas | Jim Griffith | Tommy Mauldin | 56–48; NBC Central 2nd | Provisional NBC wild-card leader | 8; 12 / 10 | 2 | Active contender; direction mixed |
| Detroit | Jesus Martinez | David Figueiredo | 57–47; NBC Central 1st | Division leader | 5; 5 / 9 | 4 | Added established starter; SUGGESTIVE present help |
| Charlotte | Jose Ruiz | Adam Butler | 62–42; NBC East 1st | Division leader; best record | 3; 3 / 4 | 2 | Relatively quiet while leading; WATCH |
| San Diego | Matt Johnson | Danny Tocornal | 39–65; NBC West 4th | Outside tournament places | 7; 10 / 8 | 0 | Two veteran pitchers moved to leaders; SUGGESTIVE reallocation |
| Tampa Bay | Angel Gonzalez | Jaylin Reed | 46–58; NBC East 3rd/4th record tie | Outside tournament places | 5; 7 / 6 | 1 | Mixed movement; WATCH |
| Los Angeles | Gary Jones | Tony Cruz | 51–53; NBC West 3rd | Outside WC lead | 1; 1 / 2 | 0 | Quietest trade record; WATCH |
| San Francisco | Luis Martinez | Jon Teeple | 58–46; NBC West 1st | Division leader | 4; 5 / 5 | 2 | Added late reliever; SUGGESTIVE targeted depth |
| Minneapolis | Ken Sears | Chad Purchase | 46–58; NBC Central 4th | Outside tournament places | 5; 11 / 8 | 0 | Moved Jenkins for two minor-league OFs; SUGGESTIVE future-asset exchange |
| Chicago | Mike Hatton | Willie Roy | 55–49; NBC Central 3rd | One game behind NBC WC leader | 7; 11 / 9 | 2 | Late hitter/closer position exchange; WATCH |
| Denver | Kevin Bradley | Ken Ellinger | 56–48; ABC West 3rd | Two games behind ABC WC leader Seattle | 4; 5 / 6 | 1 | Added Oubre for rotation; SUGGESTIVE |
| Cincinnati | Phil Trammel | Ji-hu Ju | 53–51; ABC Central 2nd/3rd tie | Tied record behind division/WC leaders | 5; 5 / 8 | 1 | Mixed asset movement; WATCH |
| New York | Jeremy Burk | Alberto Dominguez | 47–57; ABC East 3rd | Outside tournament places | 6; 11 / 7 | 1 | High in/out volume; direction unclear |
| Nashville | Justin Gomez | Chris Smith | 53–51; ABC Central 2nd/3rd tie | Tied record outside WC lead | 6; 10 / 7 | 4 | Pitching-heavy contract activity; WATCH |
| Las Vegas | Roberto Choto | Terry Vinson | 60–44; ABC West 1st | Division leader | 5; 7 / 7 | 3 | Active while leading; WATCH |
| Portland | Ben Hayter | Ben Criswell | 41–63; ABC West 4th | Outside tournament places | 5; 8 / 6 | 0 | Two-way activity, no settled direction; WATCH |
| St. Louis | David Ward | J.D. Sommer | 48–55; ABC Central 4th (103 G) | Outside tournament places | 5; 6 / 6 | 5 | Contract retention plus balanced trading; WATCH |
| Boston | Bobby Ramsey | Alex Mendez | 54–50; ABC East 1st | Division leader | 6; 6 / 9 | 0 | Late infielder additions and reliever departure; SUGGESTIVE balance |
| Pittsburgh | Nick Heller | Jared Brown | 44–60; ABC East 4th | Outside tournament places | 5; 9 / 6 | 0 | Active despite record; direction unclear |
| Philadelphia | Jon Ramos | Gabe Eckard | 48–56; ABC East 2nd | Outside WC lead | 3; 6 / 4 | 1 | Offseason contract commitment, quieter in-season; WATCH |
| Houston | Larry Bell | Xavier Ferguson | 61–42; ABC Central 1st (103 G) | Division leader; best winning percentage | 12; 14 / 21 | 1 | Most active, net player-row outflow plus Jenkins in; PROVEN activity, intent mixed |
| Seattle | David Suarez | Tony Manning | 58–46; ABC West 2nd | Provisional ABC wild-card leader | 7; 8 / 8 | 4 | Pitching signings plus WC chase; SUGGESTIVE |

Notes: “player rows in/out” counts transfer rows, not unique players or trade value. Positions are from team_record.csv; a tied record is stated as tied regardless of the position field’s ordering. Tournament slots are provisional standings context under three division winners plus one wild card per conference, not a tiebreak ruling.

## 9. ERNIE BEWELL HANDOFF — ARTICLE PREPARATION

### The league in one page

The league’s transaction story is concentration, not one uniform strategy. Trades are heavy before the first pitch (36 before April 6), nearly absent during the early season, and compressed into July (27). All 24 clubs traded, but the scale differs sharply: Houston made 12; Dallas and Miami eight; Chicago, San Diego, and Seattle seven; Los Angeles one. Atlanta led contract/signing notices with seven. The two measures should not be collapsed: an extension is not an acquisition, a transaction row is not a unique player, and roster count is not importance.

The market itself has no clean buyer/seller divide. Several leaders made targeted transactions: Detroit added established starter Kennie Murdock; San Francisco added 17-save reliever Victor Hernandez; Boston brought in two established hitters from Chicago. At the same time, San Diego sent those pitchers to leaders while 39–65, and Minneapolis sent a 9–6 starter to division-leading Houston for two minor-league outfielders. Houston, already leading the ABC Central, was simultaneously the most trade-active club and sent 21 player rows out against 14 in. A single “deadline winner” narrative would flatten the evidence.

The evidence points to roles—starting rotation, relief, infield/outfield depth—more securely than it points to motives. The OOTP profile export provides numeric fields but no local scale key; the owner export gives identities but not patience/fiscal-priority profiles; no owner instruction or private negotiation record is present. Standings and performance put each transaction in context, but must be date-correct. The writing desk can say what changed, when, and where each club stood. It should frame strategic meaning as a question or cautious inference, not as a quotation or private motive.

### The 24 clubs in journalist language

**Miami — 75–150 word reporting note.** Miami was 55–49, second in the NBC East and one game behind Dallas for the provisional wild-card lead. Daniel Baker is the current GM and Danny Leon the current owner; the export gives no owner-specific personality evidence. Miami was involved in eight trades, sending 14 player rows and receiving nine, plus one contract notice: the July 17 signing of RF Logan Ellsworth. On July 28 it sent Joe Good and minor-league Quadir May to Pittsburgh for Daniel Vega, with $144,000 cash from Miami’s side. The activity runs in both directions while the club remains in the race. It supports “active and mixed,” not a firm buyer/seller label or a claim about Baker’s motive.

**Phoenix — 75–150 word reporting note.** Phoenix finished the cutoff at 53–51, second in the NBC West, five games behind San Francisco and outside the current wild-card lead. Bryan Somers is the current GM; Mike Dutcher is the owner. Four trades moved four player rows in and five out. Four type-2 notices include a three-year Kerry Darling deal, a January signing by Joe Rogers, Eric Strickland’s contract, and Nick Perez’s extension. Three of those named notices concern pitchers/relievers, so pitching depth is a documented offseason theme in the contract file; it is not proof that the GM believed the staff was deficient. Phoenix’s record leaves the club in pursuit, but the evidence does not establish whether these deals were intended to solve a specific 1981 weakness.

**Atlanta — 75–150 word reporting note.** Atlanta was 46–58, tied on record with Tampa Bay and outside the NBC tournament positions. Hector Naranjo is the current GM; Tyler Carter is the owner, with no verified ownership traits or instructions in the export. Atlanta made six trades (nine player rows in, seven out) and led all clubs with seven contract/signing notices. Those notices include 2B Eric Gray, 1B Jordan Santistevan, 3B Luis Rosales, and pitchers Steve Farber, Netz Altman, Sergio Duran, and Mike McQuaid. That is proven transaction volume and a position-spanning offseason build; Atlanta’s low cutoff OPS (.677) is later outcome context, not proof of what the front office knew when it acted. The resulting direction is still mixed rather than a proven rebuild or push.

**Dallas — 75–150 word reporting note.** Dallas stood 56–48, second in the NBC Central and the provisional NBC wild-card leader. Jim Griffith is the current GM; Tommy Mauldin the owner. The club participated in eight trades, bringing in 12 player rows and sending ten, plus two contract notices, including Ryan Millich and Alfredo Noriega. On July 24 Dallas received starter Jonathan Funches from Tampa Bay in exchange for Brian Manchester; it also moved players with Pittsburgh that day. The transactions show sustained roster movement while Dallas occupied the wild-card lead, not a one-way purchase of present help. Its cushion was narrow: Miami and Chicago were each one game behind at the snapshot. Whether the club was reinforcing depth, changing roles, or making a broader roster bet remains a reporting question.

**Detroit — 75–150 word reporting note.** Detroit led the NBC Central at 57–47. Jesus Martinez is the current GM; David Figueiredo is the owner, with owner motive unverified. Detroit made five trades, receiving five player rows and sending nine, and had four contract notices, including Ethan Harrison, Luis Rios’s re-signing, and Brett Helvie’s multi-year deal. The clearest deadline addition was 30-year-old RHP Kennie Murdock from San Diego on July 28 for minor-league RHP Bob Morales. Before the deal, Murdock had a 10–6 record, 2.43 ERA, and 137 innings; he made one later-date Detroit start by July 31. The transaction is consistent with adding rotation support to a division leader. It does not establish that Detroit considered itself one player away or that the move changed its postseason odds.

**Charlotte — 75–150 word reporting note.** Charlotte had the league’s best record, 62–42, and led the NBC East. Jose Ruiz is the current GM; Adam Butler the owner. The Colonels made three trades (three player rows in, four out) and had two contract notices, Dustin Meredith and Juan Lomeli. Charlotte’s 1981 pitching ERA was the best in the league at 3.25 through the cutoff. It did not make one of the four July 31 trades. This is a relatively quiet transaction record attached to a leading club with strong pitching results. That can support a question about confidence in the roster, but silence alone cannot prove “standing pat” was a deliberate strategy, nor can the export tell us whether the club pursued deals that did not happen.

**San Diego — 75–150 word reporting note.** San Diego was 39–65, fourth in the NBC West, and had the league’s lowest team OPS (.658) at the cutoff. Matt Johnson is current GM; Danny Tocornal owner. The Seraphs made seven trades, with ten player rows received and eight sent, and no type-2 contract notices. The most visible sequence sent 33-year-old reliever Victor Hernandez (#504, 17 saves through July 24) to division-leading San Francisco, then starter Kennie Murdock to division-leading Detroit. San Diego received catcher Omari Benjamin and minor-league RHP Bob Morales. The clubs receiving veteran arms were in the race; San Diego was far back. The evidence suggests a present-for-future exchange in these two deals, but does not establish a franchise-wide rebuild or the GM’s private plan.

**Tampa Bay — 75–150 word reporting note.** Tampa Bay was 46–58, tied on record with Atlanta in the NBC East, and outside the tournament field. Angel Gonzalez is the current GM; Jaylin Reed owner. The club made five trades (seven player rows in, six out) and one type-2 contract event, Nate Smeltz’s reported seven-year agreement. Tampa Bay’s team OPS (.742) ranked among the league’s best by August 1, while its 4.28 ERA was among the highest. That is a current snapshot, not evidence available when the offseason contract was signed. Tampa Bay later sent starter Brian Manchester to wild-card-leading Dallas for Jonathan Funches. The roster movement and results point in different directions; there is not enough evidence to call the club a buyer, seller, or team with a settled 1981 plan.

**Los Angeles — 75–150 word reporting note.** Los Angeles was 51–53, third in the NBC West, outside the current wild-card lead. Gary Jones is the current GM; Tony Cruz is the owner. The Cobras had the league’s smallest trade count: one trade, one player row in and two out, with no type-2 contract notice in the window. The single listed deal was the December 22 exchange sending Luis Prieto and minor-league Miguel Campos to Miami for Luis Davis. Los Angeles’s 4.08 team ERA was toward the high end at the cutoff, but no late external pitching acquisition is in the verified trade/signing record. A quiet transaction log is an observable fact; whether it reflects confidence, budget, missed opportunities, or internal development is unknown.

**San Francisco — 75–150 word reporting note.** San Francisco led the NBC West at 58–46. Luis Martinez is the current GM; Jon Teeple the current owner. Four trades moved five player rows in and five out; two contract notices involved Chad Averitt and Kyle Chambers. San Francisco’s .753 team OPS led the league through July 31. Its July 25 transaction sent catcher Omari Benjamin to San Diego for veteran reliever Victor Hernandez (#504), who had 17 saves and a 2.78 ERA before the trade date. The move placed a specific leverage arm with the division leader, while the available log shows no post-trade San Francisco appearance by cutoff. A targeted bullpen addition is a fair description; a claim about strategy or expected role needs the club’s roster usage after August 1.

**Minneapolis — 75–150 word reporting note.** Minneapolis was 46–58, fourth in the NBC Central and 11 games behind the division lead through July 31. Ken Sears is the current GM; Chad Purchase the owner. Five trades produced 11 player rows in and eight out, with no type-2 contract notices. The clearest late decision came July 31: Minneapolis sent 27-year-old left-handed starter Chris Jenkins (9–6, 3.48 ERA over 98⅓ innings through the cutoff) to division-leading Houston for minor-league outfielders Jonathan Serrano and Alush Demachi. That exchange moved established present rotation innings for two younger assets, a specific future-leaning transaction. Minneapolis also acquired players elsewhere in the window, so it is not evidence by itself of an organization-wide teardown.

**Chicago — 75–150 word reporting note.** Chicago was 55–49, third in the NBC Central and one game behind Dallas in the provisional wild-card standings; Miami was tied with Chicago at 55–49. Mike Hatton is the current GM; Willie Roy owner. Seven trades brought in 11 player rows and sent nine; two contract notices include Shane Cobb’s re-signing and Andrew Flugence’s signing. Late July moved established players both ways: Chicago sent 1B Victor Hernandez (#6732) to division-leading Boston on July 27, then on July 31 received 2B Miguel Martinez from Boston while sending RHP Justin Bowen and minor-league 3B Javy Madrigal. Chicago’s deadline exchange therefore included both an incoming second baseman and an outgoing reliever, alongside minor-league pieces. These are role changes, not a simple net buyer/seller signal.

**Denver — 75–150 word reporting note.** Denver stood 56–48, third in the ABC West and two games behind Seattle’s provisional wild-card lead. Kevin Bradley is the current GM; Ken Ellinger the owner. The Rocketeers made four trades, with five player rows in and six out, and one contract notice for veteran Boboy Nierva. On June 18 Denver acquired Tony Oubre from New York, sending minor-league RHP Ramon Pazmino; the deal also included 2B Willie Vega from New York. Oubre’s later-date starts after the trade date totaled 53⅓ innings with a 3.21 ERA through July 31; the same-day start is kept separate because no transaction time is exported. The acquisition is a documented rotation addition as Denver chased the wild card. It is not proof of the club’s expected postseason odds.

**Cincinnati — 75–150 word reporting note.** Cincinnati was 53–51, tied on record with Nashville in the ABC Central but behind division-leading Houston. Phil Trammel is the current GM; Ji-hu Ju the owner. The Cougars made five trades, taking in five player rows and sending eight, plus one type-2 notice for RF Tony Jones’s six-year agreement. Cincinnati’s 1980 record was 97–65 and it won its division, but that historical success does not establish 1981 urgency. The July 2 deal with Minneapolis was the only trade in the first half of July. By the cutoff, Cincinnati remained in a tied second-place record group rather than the wild-card lead. A cautious read is a club with activity but no unmistakable deadline-day statement; its private assessment of the tie is not in the records.

**New York — 75–150 word reporting note.** New York was 47–57, third in the ABC East and outside the postseason positions. Jeremy Burk is the current GM; Alberto Dominguez the owner. The Aces made six trades (11 player rows in, seven out) and had one type-2 contract notice: Pedro Marquez’s three-year retention at the start of the offseason. New York’s season included the June trade sending Tony Oubre and Willie Vega to Denver for Ramon Pazmino, then a July 16 exchange that sent minor-league LF David Mitchell and RHP Vinny Martinez to Nashville for Devon John. That sequence changed the rotation and prospect mix, but the transaction record alone cannot establish whether the moves were a shift toward the future. The club was fourth in the division by wins at cutoff; no owner instruction or GM appointment history is available.

**Nashville — 75–150 word reporting note.** Nashville was 53–51, tied with Cincinnati on record in the ABC Central, behind Houston and outside the current wild-card lead. Justin Gomez is current GM; Chris Smith owner. The Blues made six trades, receiving ten player rows and sending seven, and recorded four contract notices, chiefly pitchers: closer Joe Berger, Jacob Sutherland, Frank Larson, and Adam Rapoport. These transactions show an offseason concentration in pitching contracts, without proving that relief was the organization’s perceived weakness. The July 16 trade sent minor-league OF David Mitchell and RHP Vinny Martinez to New York for LHP Devon John. At the cutoff Nashville shared a 53–51 record with Cincinnati. Report the tie as a tie; do not invent a tiebreak or imply the club had lost a qualification decision.

**Las Vegas — 75–150 word reporting note.** Las Vegas led the ABC West at 60–44. Roberto Choto is the current GM; Terry Vinson owner. The Gamblers made five trades with seven player rows in and seven out, and had three type-2 contract notices: Kyle Marks, catcher Jon Dominguez, and 1B Eddie Rodriguez. Their 3.31 ERA ranked among the league’s best through the cutoff; they had a two-game division cushion over Seattle. The July 27 trade with Portland brought in Nate Souza, while a July 28 deal sent Kerry Darling and Willie Saunders to Phoenix for Alex Monge. This is active roster movement by a leader, not a clean one-way reinforcement pattern. There is no verified owner profile or transaction note that states why Las Vegas chose these particular exchanges.

**Portland — 75–150 word reporting note.** Portland was 41–63, fourth in the ABC West. Ben Hayter is the current GM; Ben Criswell the owner. The Lumberjacks made five trades (eight player rows in, six out) but had no type-2 contract notices. Their clearest late-July move sent veteran Nate Souza to Las Vegas in exchange for minor-league players Manny Moran and Luis Gortarez. The club also acquired and moved players in other packages, so the activity is not a straightforward mass sale. Portland’s 4.53 ERA was the league’s highest through the cutoff; that makes pitching a relevant roster question, but the late trade sent away a pitcher rather than adding one. It is not enough to conclude the club was rebuilding or to infer its budget or ownership priorities.

**St. Louis — 75–150 word reporting note.** St. Louis was 48–55 through 103 games, fourth in the ABC Central and well behind Houston. David Ward is current GM; J.D. Sommer owner. Five trades balanced at six player rows in and six out. Five contract notices included Victor Campos, Alex Lander, Joe Puerta, and two separate Andrew Williamson deal events. The latter were a January contract notice and a May extension; they are not two acquisitions. July 27 St. Louis sent Puerta to Miami for Daniel Smith; July 30 it sent Chris Sullivan to Houston for Omar Han and Antonio Gama. The numbers show roster movement and contract retention, but not a single clear direction. A 103-game record is not directly comparable to a 104-game team’s raw win total without noting games played.

**Boston — 75–150 word reporting note.** Boston led the ABC East at 54–50. Bobby Ramsey is current GM; Alex Mendez the owner. Six trades involved six player rows in and nine out; the message file has no type-2 contract notice for Boston in the window. On July 27 Boston acquired 1B Victor Hernandez (#6732) from Chicago for RHP Mike Roede and minor-league 1B Desidério Chora. On July 31 it sent 34-year-old reliever Justin Bowen and minor-league 3B Javy Madrigal to Chicago for 2B Miguel Martinez. Through July 31 Hernandez had 13 PA after the trade date; no post-date Martinez/Bowen appearance is recorded. The sequence looks like a late infield/bat adjustment coupled with a relief departure, but “all-in” is unsupported.

**Pittsburgh — 75–150 word reporting note.** Pittsburgh was 44–60, fourth in the ABC East. Nick Heller is current GM; Jared Brown owner. The Express made five trades, with nine player rows in and six out, and no type-2 contract notices. On July 28 Pittsburgh received Daniel Vega from Miami in a trade that included $144,000 cash from Miami; on July 31 it sent 32-year-old RHP Ricky Bautista to Houston for minor-league RF Tim Mathis and RHP Tyler Hayes. Pittsburgh’s late record was far from the division lead, but it did not simply move only veterans: both outgoing and incoming packages include different age/roster classes. The team’s 4.42 ERA was among the highest at cutoff; no verified transaction record establishes a broader rebuild, or that the pitching figure drove the July decisions.

**Philadelphia — 75–150 word reporting note.** Philadelphia was 48–56, second in the ABC East but outside the wild-card lead. Jon Ramos is current GM; Gabe Eckard owner. Three trades brought in six player rows and sent four; one contract notice records Luis Cespedes’s $4.64 million agreement. The early offseason trade with Cincinnati sent 30-year-old 3B Luis Mata to Cincinnati for three pitchers/young players; the signing and trade together show meaningful early roster construction. The in-season trade count remained low, and Philadelphia did not appear in the July 31 deadline set. That combination can support a “built early, then quiet” reporting question, not an assertion that management was satisfied or had no opportunities. No evidence links the owner to the contract decision.

**Houston — 75–150 word reporting note.** Houston led the ABC Central at 61–42 through 103 games, the league’s best winning percentage. Larry Bell is current GM; Xavier Ferguson owner. The Mavericks made 12 trades—most in the league—with 14 player rows received and 21 sent, plus one contract notice for Brent Keyser’s extension. Houston’s deadline day added Pittsburgh RHP Ricky Bautista and Minneapolis starter Chris Jenkins, while sending minor-league players in return. Earlier deals moved other players in and out, so the headline is not merely “buyer”: Houston was both a division leader adding established pitching and the league’s highest-volume club with net player-row outflow. Its 3.41 ERA ranked third-best. The records establish activity and race position, not why the GM chose this mix or whether ownership directed it.

**Seattle — 75–150 word reporting note.** Seattle was 58–46, second in the ABC West and the provisional wild-card leader. David Suarez is current GM; Tony Manning owner. Seven trades balanced at eight player rows in and eight out. Four contract notices involved pitchers: Jesse Street, Nelson Majano, Antonio Romo, and Eduardo Gonzales. In July Seattle was involved in trades for Isaiah Gibson and received Harry Benjamin from St. Louis; its deadline-day file shows no trade. The club’s position put it in a tournament place without a division lead. A pitching-heavy set of contract notices alongside a wild-card position is a useful question about depth construction, but the source does not establish a specific diagnosed weakness or what the GM knew when he signed each player.

### GM character cards — verified fields vs observed behavior

The GM identity is the current team_roster_staff.csv general_manager link to coaches.csv. Age and experience below are raw current fields; “experience” is not silently converted to franchise tenure. OOTP’s personnel guidance says trading-strategy ratings relate to trade frequency, aggressiveness, loyalty, and age preference, while roster/valuation ratings relate to building a roster. The export provides numeric values, but no local key defines their scale, labels, or mapping. The last column is deliberately the same caution for every GM.

| GM / Club | Verified age / experience fields | Verified raw OOTP profile tuple: trade_aggressiveness / trade_frequency / player_loyalty / favor_veterans_to_prospects / trade_preference / value_this_year / value_last_year / value_two_years | Observed transaction behavior / notable tendency in this window | Profile consistency / possible contradiction / journalistic caution |
|---|---|---|---|---|
| Daniel Baker / Miami | 53 / 20 | -6 / 1 / 3 / -5 / -1 / 9 / 2 / 2 | 8 trades; 9 in, 14 out; 1 type-2 notice | Cannot classify numeric direction; current snapshot, not tenure history |
| Bryan Somers / Phoenix | 42 / 10 | 0 / -1 / 1 / -5 / -1 / 12 / 2 / 1 | 4 trades; 4 in, 5 out; 4 notices | Cannot classify numeric direction; current snapshot, not tenure history |
| Hector Naranjo / Atlanta | 47 / 11 | 6 / -1 / 4 / 0 / 0 / 8 / 4 / 1 | 6 trades; 9 in, 7 out; 7 notices | Cannot classify numeric direction; current snapshot, not tenure history |
| Jim Griffith / Dallas | 52 / 10 | 4 / 0 / 0 / -1 / 0 / 9 / 4 / 1 | 8 trades; 12 in, 10 out; 2 notices | Cannot classify numeric direction; current snapshot, not tenure history |
| Jesus Martinez / Detroit | 50 / 15 | 1 / -1 / 3 / -1 / 0 / 8 / 4 / 2 | 5 trades; 5 in, 9 out; 4 notices | Cannot classify numeric direction; current snapshot, not tenure history |
| Jose Ruiz / Charlotte | 44 / 6 | 0 / 1 / 3 / 0 / 0 / 11 / 2 / 1 | 3 trades; 3 in, 4 out; 2 notices | Cannot classify numeric direction; current snapshot, not tenure history |
| Matt Johnson / San Diego | 62 / 20 | -5 / -1 / 2 / -6 / -1 / 11 / 2 / 0 | 7 trades; 10 in, 8 out; 0 notices | Cannot classify numeric direction; current snapshot, not tenure history |
| Angel Gonzalez / Tampa Bay | 47 / 12 | -2 / 1 / 1 / 5 / 0 / 14 / 1 / 1 | 5 trades; 7 in, 6 out; 1 notice | Cannot classify numeric direction; current snapshot, not tenure history |
| Gary Jones / Los Angeles | 45 / 13 | -2 / 1 / 2 / -5 / -1 / 8 / 3 / 2 | 1 trade; 1 in, 2 out; 0 notices | Cannot classify numeric direction; current snapshot, not tenure history |
| Luis Martinez / San Francisco | 45 / 5 | 7 / -1 / 0 / 1 / 0 / 11 / 3 / 0 | 4 trades; 5 in, 5 out; 2 notices | Cannot classify numeric direction; current snapshot, not tenure history |
| Ken Sears / Minneapolis | 47 / 6 | 5 / 0 / 3 / 2 / 0 / 11 / 3 / 1 | 5 trades; 11 in, 8 out; 0 notices | Cannot classify numeric direction; current snapshot, not tenure history |
| Mike Hatton / Chicago | 62 / 28 | 8 / 0 / 2 / 1 / 0 / 11 / 3 / 1 | 7 trades; 11 in, 9 out; 2 notices | Cannot classify numeric direction; current snapshot, not tenure history |
| Kevin Bradley / Denver | 47 / 5 | 2 / 0 / 0 / -4 / -1 / 10 / 3 / 2 | 4 trades; 5 in, 6 out; 1 notice | Cannot classify numeric direction; current snapshot, not tenure history |
| Phil Trammel / Cincinnati | 49 / 6 | -2 / 1 / 1 / 0 / 0 / 9 / 3 / 3 | 5 trades; 5 in, 8 out; 1 notice | Cannot classify numeric direction; current snapshot, not tenure history |
| Jeremy Burk / New York | 43 / 10 | -2 / -1 / 3 / -1 / 0 / 9 / 4 / 0 | 6 trades; 11 in, 7 out; 1 notice | Cannot classify numeric direction; current snapshot, not tenure history |
| Justin Gomez / Nashville | 60 / 24 | 7 / 1 / 2 / -5 / -1 / 10 / 2 / 1 | 6 trades; 10 in, 7 out; 4 notices | Cannot classify numeric direction; current snapshot, not tenure history |
| Roberto Choto / Las Vegas | 55 / 17 | 0 / 0 / 0 / -4 / -1 / 9 / 4 / 1 | 5 trades; 7 in, 7 out; 3 notices | Cannot classify numeric direction; current snapshot, not tenure history |
| Ben Hayter / Portland | 48 / 12 | 2 / 0 / 2 / 4 / 0 / 9 / 4 / 1 | 5 trades; 8 in, 6 out; 0 notices | Cannot classify numeric direction; current snapshot, not tenure history |
| David Ward / St. Louis | 49 / 6 | 5 / 1 / 3 / 4 / 0 / 9 / 4 / 3 | 5 trades; 6 in, 6 out; 5 notices | Cannot classify numeric direction; current snapshot, not tenure history |
| Bobby Ramsey / Boston | 54 / 14 | 0 / 0 / 3 / 1 / 0 / 10 / 2 / 2 | 6 trades; 6 in, 9 out; 0 notices | Cannot classify numeric direction; current snapshot, not tenure history |
| Nick Heller / Pittsburgh | 58 / 22 | 4 / 1 / 2 / 1 / 0 / 9 / 3 / 2 | 5 trades; 9 in, 6 out; 0 notices | Cannot classify numeric direction; current snapshot, not tenure history |
| Jon Ramos / Philadelphia | 47 / 10 | 6 / 1 / 2 / -2 / 0 / 9 / 3 / 2 | 3 trades; 6 in, 4 out; 1 notice | Cannot classify numeric direction; current snapshot, not tenure history |
| Larry Bell / Houston | 45 / 10 | 2 / 0 / 1 / 1 / 0 / 8 / 4 / 1 | 12 trades; 14 in, 21 out; 1 notice | Cannot classify numeric direction; current snapshot, not tenure history |
| David Suarez / Seattle | 45 / 1 | -3 / 0 / 4 / -1 / 0 / 10 / 3 / 2 | 7 trades; 8 in, 8 out; 4 notices | Cannot classify numeric direction; current snapshot, not tenure history |

No GM tenure timeline or prior-club history was established from the available raw export. The names above are current at the August 1 snapshot. The file’s experience value is shown, not redefined as years with that club. The observed behavior column is the only journalist-usable tendency established here; because no local scale key explains the numeric profile values, consistency or contradiction between the profile and behavior cannot be classified for any GM. No GM quote is present.

### Ownership character cards — what is actually available

The current staff link verifies the owner name. A team-level owner_expectation number exists in team_financials.csv, but the local repository does not provide the numeric code legend. OOTP’s ownership manual describes patience, fiscal personality, expectation, and owner performance opinion as distinct concepts; those fields are not available here as a verified owner profile. Generic coaches.csv fields are not relabeled as owner traits. The “relationship” column is therefore unknown for every club.

| Club | Owner (current staff link) | Raw owner_expectation value (uninterpreted) | Verified traits / financial posture / patience / involvement | Relationship to observed transactions; safe claim |
|---|---|---:|---|---|
| Miami | Danny Leon | 4 | Owner-specific patience/fiscal/involvement fields unavailable | No causal link established |
| Phoenix | Mike Dutcher | 5 | Owner-specific fields unavailable | No causal link established |
| Atlanta | Tyler Carter | 2 | Owner-specific fields unavailable | No causal link established |
| Dallas | Tommy Mauldin | 3 | Owner-specific fields unavailable | No causal link established |
| Detroit | David Figueiredo | 5 | Owner-specific fields unavailable | No causal link established |
| Charlotte | Adam Butler | 4 | Owner-specific fields unavailable | No causal link established |
| San Diego | Danny Tocornal | 0 | Owner-specific fields unavailable | No causal link established |
| Tampa Bay | Jaylin Reed | 4 | Owner-specific fields unavailable | No causal link established |
| Los Angeles | Tony Cruz | 2 | Owner-specific fields unavailable | No causal link established |
| San Francisco | Jon Teeple | 3 | Owner-specific fields unavailable | No causal link established |
| Minneapolis | Chad Purchase | 0 | Owner-specific fields unavailable | No causal link established |
| Chicago | Willie Roy | 4 | Owner-specific fields unavailable | No causal link established |
| Denver | Ken Ellinger | 3 | Owner-specific fields unavailable | No causal link established |
| Cincinnati | Ji-hu Ju | 5 | Owner-specific fields unavailable | No causal link established |
| New York | Alberto Dominguez | 1 | Owner-specific fields unavailable | No causal link established |
| Nashville | Chris Smith | 1 | Owner-specific fields unavailable | No causal link established |
| Las Vegas | Terry Vinson | 4 | Owner-specific fields unavailable | No causal link established |
| Portland | Ben Criswell | 2 | Owner-specific fields unavailable | No causal link established |
| St. Louis | J.D. Sommer | 3 | Owner-specific fields unavailable | No causal link established |
| Boston | Alex Mendez | 1 | Owner-specific fields unavailable | No causal link established |
| Pittsburgh | Jared Brown | 2 | Owner-specific fields unavailable | No causal link established |
| Philadelphia | Gabe Eckard | 0 | Owner-specific fields unavailable | No causal link established |
| Houston | Xavier Ferguson | 5 | Owner-specific fields unavailable | No causal link established |
| Seattle | Tony Manning | 2 | Owner-specific fields unavailable | No causal link established |

The exported owner_expectation value is not a verified label for winning expectations because this repository has no codebook for it. Owner-specific patience, involvement, and fiscal-personality fields are unavailable. The only safe ownership statement is the current owner identity; no owner-to-transaction relationship is established, and no owner motive or instruction can be claimed. The finance export includes team budget, payroll, attendance, market, fan-interest, and fan-loyalty fields. Those team-level snapshots can support a separate financial/fan context desk after field definitions and freshness are checked. They do not prove an owner instruction, spending philosophy, or fan reaction to a particular transaction. No causal ownership narrative is safe here.

### Division and tournament implications

The route toward the requested Grand Tournament of Champions begins with one of eight postseason places; the OOTP configuration names its final round the Grand Series. Qualification is three division winners per conference plus one wild card per conference (league_playoffs.csv). Under the repository’s 1981-and-forward rule in docs/ABL_POSTSEASON_RULES.md, qualification is unchanged, but division rivals cannot meet in the DCS: if seeds #1 and #4 are from the same division, #4 plays #2 and #1 plays #3. The standings below are provisional cutoff arithmetic, not an official final bracket. At the cutoff:

- NBC division leaders: Charlotte 62–42, Detroit 57–47, San Francisco 58–46. Dallas 56–48 leads the provisional NBC wild-card standings. Miami and Chicago, both 55–49, are each one game behind Dallas.
- ABC division leaders: Boston 54–50, Houston 61–42, Las Vegas 60–44. Seattle 58–46 leads the provisional ABC wild-card race; Denver 56–48 is two games back.
- Atlanta and Tampa Bay were both 46–58 in the NBC East. Cincinnati and Nashville were both 53–51 in the ABC Central. Report each pair as tied on record. The source’s position column is not a reason to invent a tiebreak outcome.
- St. Louis was 48–55 through 103 games, while Philadelphia was 48–56 through 104. The raw wins alone should not obscure the different number of games played.

No playoff probability, clinch/elimination status, or tiebreak ruling is calculated here. Any future “if the season ended today” bracket must apply the 1981 same-division protection before naming DCS matchups.

### Roster needs and transaction fit

Use team results as a check against—not a substitute for—roster evidence. Through the cutoff, San Diego had the lowest team OPS (.658); Atlanta was third-lowest (.677). Portland had the highest ERA (4.53); Pittsburgh and Tampa Bay were next-highest at 4.42 and 4.28. Charlotte owned the best ERA (3.25), Las Vegas the second-best (3.31), and Houston the third-best (3.41). San Francisco led OPS (.753). These figures are season-to-date results, not contemporaneous GM evaluations at the time of an offseason move.

Observed fit worth reporting:

- Detroit acquired an established starter while leading its division.
- San Francisco acquired a reliever with 17 pre-trade saves while leading its division.
- Denver obtained Tony Oubre during its wild-card chase; later-date starts were kept separate from same-day performance.
- Boston traded for two established infield/position-player bats from Chicago late in July, while sending reliever Justin Bowen back in the final deal.
- Chicago received Bowen, who had 18 saves before July 31, while moving Miguel Martinez to Boston.
- Houston acquired 9–6 starter Chris Jenkins on July 31 from a club 11 games behind its division leader at the previous-day snapshot.
- Minneapolis’s side of that deal was two minor-league left-fielders. That package permits a “present innings for future assets” description for this transaction only.

The data does not establish that any of those teams had no alternative, that a particular player was the only solution, or that the acquired player immediately performed as intended. OOTP player-game totals through the cutoff belong to the season across clubs; they must not be presented as receiving-club production.

### League-wide movement patterns

**Most trades:** Houston 12; Dallas and Miami 8; Chicago, San Diego, Seattle 7; Atlanta, Cincinnati, New York, Nashville, Boston 6.  
**Fewest:** Los Angeles 1; Charlotte and Philadelphia 3.  
**Most player rows out:** Houston 21, Miami 14. **Most received:** Houston 14, Dallas 12; Chicago, Minneapolis, and New York 11 each. These totals differ from unique players and from deal counts.

**Repeated trading partners:** Detroit and New York completed three trades. Detroit/San Diego, San Diego/Houston, Chicago/Boston, Tampa Bay/Chicago, St. Louis/Houston, and Minneapolis/Houston each completed two.  
**Players traded more than once:** Jerry Wilson #681 (San Francisco to Chicago in January; Houston to San Diego in July), Vinny Martinez #8329 (Miami to Nashville in December; Nashville to New York in July), and Rodolfo Gonzalez #9886 (Denver to Atlanta July 23; Atlanta to Chicago July 31). These chains should be checked against the full transaction table before describing a player’s club path.

**Conference-side participation:** ABC clubs appear on 69 trade sides; NBC clubs on 63. Division side counts are ABC Central 28, ABC East 20, ABC West 21; NBC Central 25, NBC East 22, NBC West 16. This counts each club-side appearance, so cross-division trades count once in each participating division. It is not a count of trades “belonging” to one division.

The late market involved both direct contenders and distant clubs. The two most visible present-race arms moved from 39–65 San Diego to division leaders Detroit and San Francisco. Conversely, July 31’s Minneapolis–Houston transaction paired a distant club and a leader. The records support a geography of movement; they do not support broad claims that every contender bought or every non-contender sold.

### Clubs that changed direction / stayed the course

**Transactions proven:** Every club traded at least once; no club was literally transaction-free. Activity increased sharply in July.  
**Suggestive of a change within the window:** San Diego’s pair of veteran-pitcher departures; Minneapolis’s deadline exchange of Jenkins for two minor-league outfielders; Houston’s high-volume two-way activity while leading; Boston’s late positional exchange. These observations describe the direction of specific player flows, not complete franchise plans.  
**Quiet relative to the league:** Los Angeles (one trade), Charlotte and Philadelphia (three each). The source cannot say whether they stood pat by choice or simply lacked completed deals.  
**Not supported:** “Atlanta rebuilt”; “Houston was all-in”; “Seattle’s owner demanded a wild-card berth”; “Charlotte trusted its roster”; “the market rejected a player”; any claim about an unrecorded offer, clubhouse or fan reaction.

### Front-office identity — cautious synthesis

The strongest distinction in the evidence is observable behavior, not personality. Atlanta and St. Louis combined high contract-notice counts with trade activity. Houston combined the most trades with high player-row turnover and one deadline-day acquisition. San Diego and Minneapolis each moved a visible veteran pitcher toward a contending club for a catcher/minor-league return. Detroit, San Francisco, Denver, and Boston made transactions that map to identifiable roster roles while positioned in or near tournament places. The GM raw rating fields may make useful follow-up questions, but their numerical encoding must be decoded before applying words such as “aggressive,” “prospect-minded,” or “win-now.”

### Fan-base implications

The export contains no attributable fan quotations, newspaper reaction, or attendance response tied to individual deals. Current team-financial rows do contain attendance, fan interest, fan loyalty, and market fields, but a single August 1 snapshot cannot show whether a trade changed those measures. Safe implication language is conditional: a deadline addition gives followers a visible roster change to evaluate; a quiet leader leaves its existing core to carry the race; a struggling club moving established innings for minor-league players places more attention on what those prospects become. Do not state that fans were excited, angry, or disappointed without independent evidence.

## 10. Ernie Bewell — what may be here

Each item is an assignment possibility, not a chosen article angle.

### 1. The busiest team was not a simple buyer

- **Central idea:** Houston combined the league’s highest trade count with a net outflow of player-transfer rows while leading the ABC Central.
- **Support / clubs / people:** Houston 12 trades, 14 rows in, 21 out; Larry Bell is current GM; July 31 additions included Ricky Bautista and Chris Jenkins.
- **Why now:** Houston entered the August 1 snapshot 61–42 and at the top of its division.
- **Distinctive:** It complicates “contender buys, distant club sells.”
- **What weakens it:** Transfer-row counts do not measure player value or payroll, and one returned prospect package may be more important than several rows.
- **Reporting to strengthen:** Ask how Houston described its roster needs; review player roles, salary commitments, and July usage after the cutoff.
- **Possible home:** Major standalone Ernie Bewell feature.

### 2. San Diego’s two veteran arms went to two leaders

- **Central idea:** A last-place club moved a starting pitcher and a high-save reliever to division leaders within three days.
- **Support / clubs / people:** San Diego, Detroit, San Francisco; Murdock and Victor Hernandez #504; Matt Johnson, Jesus Martinez, and Luis Martinez are current GMs.
- **Why now:** The exchange preceded the deadline and directly involved clubs in the tournament field.
- **Distinctive:** The return packages differed: a minor-league pitcher for Murdock, catcher Omari Benjamin for Hernandez.
- **What weakens it:** Two trades do not establish a full rebuild; there is no direct explanation from the front office.
- **Reporting to strengthen:** Trace what happened to Benjamin/Morales and ask what San Diego intended to do with the acquired assets.
- **Possible home:** Baseball Observer or standalone feature.

### 3. The deadline dealt in roster roles as much as names

- **Central idea:** Several late moves involved identifiable rotation, relief, and infield roles rather than a single leaguewide star auction.
- **Support / clubs / people:** Detroit/Murdock, San Francisco/Victor Hernandez #504, Denver/Oubre, Boston/Chicago’s Victor Hernandez #6732 and Miguel Martinez, Chicago/Bowen.
- **Why now:** The relevant teams were division leaders or in the wild-card race.
- **Distinctive:** The same deadline could be read through baseball construction instead of winner/loser grades.
- **What weakens it:** Role need is only partly visible; no timestamp or internal depth chart is exported.
- **Reporting to strengthen:** Compare August usage and roster position after the league advances.
- **Possible home:** Baseball Matters.

### 4. July changed the market’s tempo

- **Central idea:** A quiet early-season trade period gave way to 27 July trades, including six on July 27 and four on July 31.
- **Support / clubs / people:** All 24 clubs traded; Houston, Dallas, Miami, Chicago, San Diego, and Seattle led activity.
- **Why now:** The rise is visible before and after the July 14 All-Star date.
- **Distinctive:** It offers a leaguewide transaction chronology rather than one transaction verdict.
- **What weakens it:** Counts do not show negotiations, failed offers, or importance.
- **Reporting to strengthen:** Compare later ABL seasons or interview GMs about the calendar and trade process.
- **Possible home:** Forum, with a sidebar for dates.

### 5. A contender may have stood quieter than its record

- **Central idea:** Charlotte led the league at 62–42 yet completed only three trades; Los Angeles completed one trade despite a 51–53 record.
- **Support / clubs / people:** Charlotte, Jose Ruiz, Adam Butler; Los Angeles, Gary Jones, Tony Cruz.
- **Why now:** Both clubs’ low activity contrasts with the July surge.
- **Distinctive:** It asks what a team communicates by not making a visible late move without assuming a failed or deliberate strategy.
- **What weakens it:** The export cannot reveal offers that were made or considered.
- **Reporting to strengthen:** Ask about July priorities, health, and roster evaluations; check any later roster records.
- **Possible home:** Baseball Observer.

### 6. Two tied races, different transactions

- **Central idea:** Atlanta/Tampa Bay and Cincinnati/Nashville entered August tied on record, while their transaction patterns differed.
- **Support / clubs / people:** Atlanta 7 contract notices; Tampa Bay 1; Nashville 4; Cincinnati 1. Atlanta/TB were each 46–58; CIN/NAS each 53–51.
- **Why now:** These clubs faced the same standings line but did not make the same volume/type of moves.
- **Distinctive:** A controlled comparison without inventing tiebreak rules.
- **What weakens it:** Contract notices are not all new players, and no team messaging is supplied.
- **Reporting to strengthen:** Confirm lineup/rotation usage and interview the four front offices.
- **Possible home:** Baseball Matters.

### 7. The prospect return is not a conclusion

- **Central idea:** San Diego and Minneapolis received younger/minor-league pieces for established pitchers; what those assets mean remains open.
- **Support / clubs / people:** Murdock-to-Detroit, Jenkins-to-Houston, the two specific returns; current standings at each move.
- **Why now:** These are the clearest exchanges across competitive positions near the deadline.
- **Distinctive:** Keeps the feature focused on organizational timelines rather than assigning trade grades.
- **What weakens it:** There is no complete minor-league performance/valuation series in the verified deadline packet.
- **Reporting to strengthen:** Follow the prospects’ usage/development and ask how each selling club valued the time horizon.
- **Possible home:** Major standalone feature.

### 8. What the GM and owner files cannot yet tell us

- **Central idea:** The desk can name current GMs and owners and report raw OOTP profile values, but cannot reliably translate them into personality stories.
- **Support / clubs / people:** All 24 clubs; coaches.csv raw fields; absent owner-specific patience/fiscal/involvement/priority fields.
- **Why now:** The natural journalistic temptation is to explain transaction patterns through personality.
- **Distinctive:** It makes data limitations part of a reported question rather than presenting a caricature.
- **What weakens it:** It is a methodology angle, not necessarily the main baseball story.
- **Reporting to strengthen:** Obtain decoded OOTP profile screens, dated personnel history, and direct interviews.
- **Possible home:** Forum or reporting sidebar.

## 11. What the evidence does not support

Do not claim:

- that a trade was “won” or “lost,” or grade it by short-term performance.
- that any club was all-in, rebuilding, protecting a lead, or following an owner’s order without direct evidence.
- that a GM’s raw numeric profile is “aggressive,” “conservative,” prospect-first, or veteran-first until the scale and raw-field mapping are documented.
- that the current GM or owner necessarily held the post throughout the entire offseason; the export is a current staff snapshot without a dated appointment history.
- that all waivers, releases, Rule 5 selections, or roster assignment changes are represented by the trade table.
- that a player’s whole-season production was created by his receiving club; use game-date split lines and keep same-day samples separate.
- that generated news prose about fan mood, negotiations, player value, or private conversations is a verified quotation or independent interview.
- that any August 1 game was played, or that generated downstream ABL reports are current. No downstream reports were regenerated for this packet.

## 12. Questions worth following after the deadline

1. What role did Detroit give Murdock, and how much of his work was genuinely new Detroit innings?
2. How did San Francisco deploy Victor Hernandez #504 after July 25?
3. Did Houston use Jenkins immediately, and what was Minneapolis’s plan for Serrano and Demachi?
4. How did Boston divide playing time among Hernandez #6732 and Miguel Martinez, and how did Chicago use Bowen?
5. Did Seattle, Dallas, Miami, Denver, and Chicago make further roster moves after the August 1 snapshot?
6. What was the 1981 Rule 5 selection list, and did any selection stay on a major-league roster?
7. Why is first-rounder Tony Cossio #12045 on Denver’s active/secondary roster, and when will he appear in an ABL game? He was selected 13th overall on June 5; no major-league game appearance exists in the cutoff logs.
8. Can OOTP’s raw coach/owner attribute codes be exported with their labels and a dated personnel timeline?
9. Were transactions considered but not completed? The current raw files cannot answer.
10. Do August/September results change the teams’ competitive interpretations? That is a later reporting cutoff, not something to fold into this one.

## 13. Source and freshness notes

- The 1981 current-state evidence is the raw OOTP export dated at league current_date 1981-8-1, with completed games through July 31.
- Transaction rows have dates, not times. Pre-trade standings are “through the preceding completed game date”; performance from the transaction date is a separate bucket.
- The 1980 season-end reconstruction relies on the historical almanac game file plus raw team_history and team_history_record. It is not a direct raw 1980 game export.
- The raw OOTP file does not expose historical GM appointments or a complete non-trade transaction log.
- Current team finance/attendance/market fields are a snapshot and lack a verified before/after transaction series.
- The accompanying trade report’s seven-pass certification remains the source for full position/age/detail checks. This dossier does not regenerate or certify downstream reports as current.
- Draft context: the league file sets June 5 as the first-year draft date. Message #19362 announces the draft; players.csv identifies Denver’s Tony Cossio #12045 as 1981 round 1, overall pick 13, current ABL roster-active/secondary. His two listed game-pitching rows are with minor-league team 70, not Denver’s ABL club. The Rule 5 notice #18845 describes the event/rules but the raw export contains no complete pick-result ledger.

## 14. Verification certificate

**Build verification performed**

1. Season boundary: checked the 1980 historical game file against the 162-game historical record for all 24 ABL clubs; postseason final game and Detroit playoff flag cross-checked.
2. Deadline/cutoff: checked league current_date, trading_deadline flag/date, schedule rows, and latest completed regular-season game.
3. Transactions: counted 66 structured trade rows; compared IDs to transaction messages; checked four July 31 trades; verified no August 1 trade/contract notice.
4. Trade package: used structured side arrays, not potentially truncated summary text; all player transfer IDs are retained in the full companion ledger.
5. Current standings: reproduced team_record.csv for all 24 teams from completed games; total 1,247 decisions per side.
6. Tournament: used configured playoff structure; preserved tied records without assigning tiebreaks.
7. Profiles and ownership: verified current staff links and raw fields; did not translate unknown codes into categories.
8. Draft check: verified the current roster status and draft fields for Tony Cossio; excluded his minor-league game rows from ABL production.
9. Editorial audit: removed unsupported motive, owner, fan-reaction, quote, and winner/loser claims.

**Material corrections / additions:** No correction to the companion 66-trade count or seven-pass trade certification was required. This packet adds the full offseason-to-deadline contract-notice scope, the historical season boundary reconstruction, the current 1981 draftee note, and explicit limits on non-trade transaction completeness and GM/owner coding.

**Facts excluded as unverified:** Owner personality/financial priority labels, GM tendencies decoded from raw numeric signs, historical GM tenure, waiver/release/Rule 5 selection outcomes, unlogged negotiations/offers, and fans’ emotional reactions.

**Unresolved data limitations:** No exhaustive non-trade transaction history; no dated GM/owner personnel history; no local numeric codebook for GM/owner profile fields; no raw 1980 full-game export; no transaction timestamps.

**Full verification cycles:** One full final verification cycle after construction; source totals and critical identities cross-checked against the raw exports and the companion deadline report.

**FINAL VERDICT: VERIFIED FOR ERNIE BEWELL JOURNALIST USE**

Use only the verified facts and qualified inferences above; the final article remains Ernie Bewell’s decision.

