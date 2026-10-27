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
- csv/out/story/production/abl_trade_deadline_report_1981_08_01_asof_1981-07-31.md and abl_trade_deadline_ledger_1981_08_01_asof_1981-07-31.csv — companion cross-checks only; the essential transaction identities, sides, at-trade descriptors, and exceptions are reproduced in this packet and verified against the raw exports above.
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
| July 1–15, All-Star period | One trade, July 2 #19578; ABL All-Star Game July 14 (game 8767) | July 12 was the last regular-season date before the break and also carried the Future Stars game (game 1945) plus All-Star voting/roster announcements (messages 19669–19673); the major-league ABL All-Star Game was July 14. |
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
This is the full 66-trade audit. The source-of-truth direction is the structured `trade_history.csv` team/player side arrays; each row is cross-linked to the OOTP `messages.csv` record with the same message ID. ?Side 0 sent to Side 1? means Side 1 received those assets, and vice versa. Age, position, and minor-league labels come from the transaction summary, filled from the matching message where a long summary is truncated. A `*` marks a position disagreement between those two text fields; the position-exception table below preserves both instead of hiding the conflict.

| Date | Trade / message ID | Side 0 sent to Side 1 | Side 1 sent to Side 0 | Incoming roster class (transaction description) | Other terms |
|---|---|---|---|---|---|
| 1980-11-11 | #18759 / message #18759 | Minneapolis to Portland: Jose Rodriguez #4644 (28; minor league RHP); Justin Reed #10732 (19; minor league 3B) | Portland to Minneapolis: Jesus Lopez #8176 (24; not labeled minor CF) | Minneapolis receives 1 ABL-level/not-minor-labeled; Portland receives 2 minor-league | none in structured cash/pick/IAFA arrays |
| 1980-11-27 | #18791 / message #18791 | Minneapolis to Dallas: Aires Azeredo #9290 (20; minor league 1B); Jesus Perez #7431 (23; minor league C); Eddy Cruz #8481 (21; minor league 3B) | Dallas to Minneapolis: Bobby Kinney #2640 (29; not labeled minor LHP); Elias Garza #7953 (21; minor league 1B) | Minneapolis receives 1 ABL-level/not-minor-labeled, 1 minor-league; Dallas receives 3 minor-league | none in structured cash/pick/IAFA arrays |
| 1980-11-30 | #18792 / message #18792 | Las Vegas to Denver: Edwin Garcia #10731 (22; minor league RF) | Denver to Las Vegas: Edgar Velo #2874 (31; not labeled minor 3B) | Las Vegas receives 1 ABL-level/not-minor-labeled; Denver receives 1 minor-league | summary retention: retaining 70% |
| 1980-12-01 | #18798 / message #18798 | Pittsburgh to Nashville: Emilio Vazquez #7879 (21; minor league LF); Kiyoharu Tamura #6960 (23; minor league RHP) | Nashville to Pittsburgh: Bob Newman #997 (33; not labeled minor RHP) | Pittsburgh receives 1 ABL-level/not-minor-labeled; Nashville receives 2 minor-league | summary retention: retaining 100% |
| 1980-12-05 | #18801 / message #18801 | Atlanta to Seattle: Orlando Hernandez #8133 (21; minor league RHP) | Seattle to Atlanta: Mike Hawkins #1984 (32; not labeled minor RHP) | Atlanta receives 1 ABL-level/not-minor-labeled; Seattle receives 1 minor-league | summary retention: retaining 100% |
| 1980-12-06 | #18803 / message #18803 | Miami to Seattle: Jose Trujillo #4517 (31; not labeled minor RHP); Danny Munoz #6301 (22; minor league 3B) | Seattle to Miami: Oscar Romero #5770 (29; not labeled minor 1B) | Miami receives 1 ABL-level/not-minor-labeled; Seattle receives 1 ABL-level/not-minor-labeled, 1 minor-league | summary retention: retaining 55% |
| 1980-12-06 | #18804 / message #18804 | San Francisco to Denver: Dale O'Connell #6404 (22; minor league RF) | Denver to San Francisco: Warren Callery #1133 (31; not labeled minor RHP); Oscar Brumley #9455 (19; minor league CF) | San Francisco receives 1 ABL-level/not-minor-labeled, 1 minor-league; Denver receives 1 minor-league | none in structured cash/pick/IAFA arrays |
| 1980-12-06 | #18805 / message #18805 | Dallas to Boston: Joe Turner #438 (35; not labeled minor 1B) | Boston to Dallas: Bryce Rose #292 (35; not labeled minor RF*); Marcos Henriquez #8452 (20; minor league RF) | Dallas receives 1 ABL-level/not-minor-labeled, 1 minor-league; Boston receives 1 ABL-level/not-minor-labeled | none in structured cash/pick/IAFA arrays |
| 1980-12-07 | #18807 / message #18807 | Nashville to Boston: Jorge Galvez #1 (35; not labeled minor RHP) | Boston to Nashville: Juan Grant #10086 (24; not labeled minor CF) | Nashville receives 1 ABL-level/not-minor-labeled; Boston receives 1 ABL-level/not-minor-labeled | summary retention: retaining 30% |
| 1980-12-10 | #18818 / message #18818 | Detroit to New York: Victor Cerda #9589 (19; minor league SS) | New York to Detroit: Marcos Real #577 (34; not labeled minor RHP) | Detroit receives 1 ABL-level/not-minor-labeled; New York receives 1 minor-league | summary retention: retaining 100% |
| 1980-12-13 | #18824 / message #18824 | Nashville to Miami: Bill Borden #1987 (32; not labeled minor RHP) | Miami to Nashville: Marco Figueroa #3624 (22; minor league SS); Fernando Gomez #9939 (20; minor league SS*); Vinny Martinez #8329 (20; minor league RHP); Justin Campbell #9369 (24; minor league 1B) | Nashville receives 4 minor-league; Miami receives 1 ABL-level/not-minor-labeled | summary retention: retaining 90% |
| 1980-12-14 | #18825 / message #18825 | Cincinnati to Philadelphia: Oscar Rebeles #6293 (23; not labeled minor RHP); Pedro Camacho #6819 (22; minor league LHP); Julio Rodriguez #8385 (20; minor league RF*) | Philadelphia to Cincinnati: Luis Mata #4703 (30; not labeled minor 3B) | Cincinnati receives 1 ABL-level/not-minor-labeled; Philadelphia receives 1 ABL-level/not-minor-labeled, 2 minor-league | none in structured cash/pick/IAFA arrays |
| 1980-12-14 | #18826 / message #18826 | San Diego to Houston: Ben Sabine #5882 (28; not labeled minor RHP); Elijah Glenn #6330 (22; minor league 2B) | Houston to San Diego: Serge Mazet #8344 (19; minor league RF); Isaiah Hurdle #6267 (23; minor league C) | San Diego receives 2 minor-league; Houston receives 1 ABL-level/not-minor-labeled, 1 minor-league | summary retention: retaining 100% |
| 1980-12-15 | #18835 / message #18835 | Atlanta to Philadelphia: Mario Amador #9081 (18; minor league C) | Philadelphia to Atlanta: Ricky Velez #2284 (29; not labeled minor RHP); Ismael Hernandez #7567 (21; minor league RHP) | Atlanta receives 1 ABL-level/not-minor-labeled, 1 minor-league; Philadelphia receives 1 minor-league | summary retention: retaining 40% |
| 1980-12-15 | #18836 / message #18836 | Las Vegas to St. Louis: Manny Madrigal #8544 (20; minor league C) | St. Louis to Las Vegas: Juan Navarro #866 (30; not labeled minor RHP); Willie Simental #7083 (21; minor league RHP) | Las Vegas receives 1 ABL-level/not-minor-labeled, 1 minor-league; St. Louis receives 1 minor-league | none in structured cash/pick/IAFA arrays |
| 1980-12-16 | #18839 / message #18839 | Cincinnati to Nashville: Pascal Magana #7232 (23; not labeled minor RF) | Nashville to Cincinnati: Darius Jones #5895 (30; not labeled minor RHP) | Cincinnati receives 1 ABL-level/not-minor-labeled; Nashville receives 1 ABL-level/not-minor-labeled | summary retention: retaining 85% |
| 1980-12-16 | #18840 / message #18840 | Houston to Seattle: Amari Baggs #6976 (21; minor league RHP) | Seattle to Houston: Albert Chavez #4686 (28; not labeled minor 1B) | Houston receives 1 ABL-level/not-minor-labeled; Seattle receives 1 minor-league | summary retention: retaining 100% |
| 1980-12-17 | #18841 / message #18841 | Charlotte to Miami: Alejandro Ordaz #5436 (23; minor league SS*) | Miami to Charlotte: Josh Maras #955 (31; not labeled minor RHP) | Charlotte receives 1 ABL-level/not-minor-labeled; Miami receives 1 minor-league | summary retention: retaining 100% |
| 1980-12-17 | #18842 / message #18842 | Detroit to San Diego: Fernando Labrado #8804 (21; minor league LHP); Noah Burks #10089 (23; minor league SS*) | San Diego to Detroit: Mike Dominguez #2614 (29; not labeled minor RHP) | Detroit receives 1 ABL-level/not-minor-labeled; San Diego receives 2 minor-league | none in structured cash/pick/IAFA arrays |
| 1980-12-18 | #18843 / message #18843 | Miami to Dallas: Frank Duarte #9615 (21; minor league CF); Yoshihiro Murakami #9299 (24; minor league 3B) | Dallas to Miami: Gary McMinn #2876 (29; not labeled minor LF*) | Miami receives 1 ABL-level/not-minor-labeled; Dallas receives 2 minor-league | summary retention: retaining 100% |
| 1980-12-19 | #18846 / message #18846 | Charlotte to Seattle: David Jenkins #8961 (23; minor league LHP) | Seattle to Charlotte: Cameron Jones #4776 (30; not labeled minor LHP) | Charlotte receives 1 ABL-level/not-minor-labeled; Seattle receives 1 minor-league | summary retention: retaining 90% |
| 1980-12-22 | #18855 / message #18855 | Los Angeles to Miami: Luis Prieto #6223 (27; not labeled minor RHP); Miguel Campos #8994 (18; minor league RHP) | Miami to Los Angeles: Luis Davis #7015 (22; minor league 3B) | Los Angeles receives 1 minor-league; Miami receives 1 ABL-level/not-minor-labeled, 1 minor-league | none in structured cash/pick/IAFA arrays |
| 1980-12-25 | #18857 / message #18857 | Tampa Bay to Portland: Matt Awtry #2639 (32; not labeled minor LHP); Oscar Lopez #8336 (20; minor league LF) | Portland to Tampa Bay: Hector Lespier #7926 (22; minor league C) | Tampa Bay receives 1 minor-league; Portland receives 1 ABL-level/not-minor-labeled, 1 minor-league | summary retention: retaining 100% |
| 1980-12-26 | #18858 / message #18858 | Portland to Philadelphia: Mike Stanley #2430 (32; not labeled minor RHP); Joe Davis #7674 (21; minor league RF) | Philadelphia to Portland: Nick Jones #7585 (25; not labeled minor RF) | Portland receives 1 ABL-level/not-minor-labeled; Philadelphia receives 1 ABL-level/not-minor-labeled, 1 minor-league | summary retention: retaining 35% |
| 1980-12-31 | #18869 / message #18869 | Atlanta to Dallas: Elias Ruela #6567 (24; not labeled minor RHP) | Dallas to Atlanta: Tony Resendiz #4827 (29; not labeled minor LHP) | Atlanta receives 1 ABL-level/not-minor-labeled; Dallas receives 1 ABL-level/not-minor-labeled | summary retention: retaining 75% |
| 1981-01-02 | #18870 / message #18870 | Seattle to Las Vegas: Trevon Washington #5361 (24; minor league RHP); Georber Alvarez #6092 (23; minor league 2B) | Las Vegas to Seattle: Danny Sanchez #4811 (30; not labeled minor RHP) | Seattle receives 1 ABL-level/not-minor-labeled; Las Vegas receives 2 minor-league | none in structured cash/pick/IAFA arrays |
| 1981-01-03 | #18873 / message #18873 | Chicago to San Francisco: Danny Brown #6321 (22; minor league 1B) | San Francisco to Chicago: Jerry Wilson #681 (32; not labeled minor RHP); Dave Jackson #7776 (21; minor league RF*) | Chicago receives 1 ABL-level/not-minor-labeled, 1 minor-league; San Francisco receives 1 minor-league | summary retention: retaining 75% |
| 1981-01-11 | #18884 / message #18884 | New York to Detroit: Martin Martinez #4422 (31; not labeled minor RHP) | Detroit to New York: Jonathan Heberling #9444 (22; minor league SS); Felix Jaquez #10153 (21; minor league CF); Carlos Escobar #8568 (21; minor league 3B) | New York receives 3 minor-league; Detroit receives 1 ABL-level/not-minor-labeled | summary retention: retaining 90% |
| 1981-01-14 | #18891 / message #18891 | Houston to San Francisco: Ricky Correa #5487 (24; minor league 1B) | San Francisco to Houston: Nick Case #3133 (32; not labeled minor LHP) | Houston receives 1 ABL-level/not-minor-labeled; San Francisco receives 1 minor-league | summary retention: retaining 100% |
| 1981-01-22 | #18901 / message #18901 | Phoenix to Seattle: Bobby Perez #9888 (18; minor league 2B) | Seattle to Phoenix: Jose Contreras #677 (35; not labeled minor RHP) | Phoenix receives 1 ABL-level/not-minor-labeled; Seattle receives 1 minor-league | summary retention: retaining 60% |
| 1981-01-29 | #18909 / message #18909 | St. Louis to Houston: Miguel Salgado #4611 (29; not labeled minor RHP) | Houston to St. Louis: Jaden Antunez #6875 (22; minor league RHP) | St. Louis receives 1 minor-league; Houston receives 1 ABL-level/not-minor-labeled | summary retention: retaining 100% |
| 1981-02-12 | #18924 / message #18924 | Phoenix to Cincinnati: Jordan Pedroza #7393 (27; not labeled minor C) | Cincinnati to Phoenix: Jorge Nino #4409 (27; not labeled minor RHP) | Phoenix receives 1 ABL-level/not-minor-labeled; Cincinnati receives 1 ABL-level/not-minor-labeled | none in structured cash/pick/IAFA arrays |
| 1981-02-14 | #18925 / message #18925 | Houston to Minneapolis: Jayden Martin #8256 (20; minor league RHP); Juan Santos #7757 (20; minor league RF*); Tony Rodriquez #9288 (20; minor league C); Jeremiah Matthews #7545 (21; minor league CF) | Minneapolis to Houston: Jesus Vasquez #4495 (27; not labeled minor RF) | Houston receives 1 ABL-level/not-minor-labeled; Minneapolis receives 4 minor-league | none in structured cash/pick/IAFA arrays |
| 1981-02-21 | #18931 / message #18931 | Dallas to Nashville: Corey Nelson #7640 (21; minor league 2B) | Nashville to Dallas: Bill Rambow #841 (37; not labeled minor RF) | Dallas receives 1 ABL-level/not-minor-labeled; Nashville receives 1 minor-league | summary retention: retaining 75% |
| 1981-03-04 | #18945 / message #18945 | Tampa Bay to Chicago: Willie Leyba #6705 (29; not labeled minor LHP) | Chicago to Tampa Bay: Dave Guadalupe #8805 (24; minor league RHP) | Tampa Bay receives 1 minor-league; Chicago receives 1 ABL-level/not-minor-labeled | none in structured cash/pick/IAFA arrays |
| 1981-03-06 | #18948 / message #18948 | Portland to Boston: Noah White #8837 (23; not labeled minor SS) | Boston to Portland: Renault Buso #2816 (31; not labeled minor C) | Portland receives 1 ABL-level/not-minor-labeled; Boston receives 1 ABL-level/not-minor-labeled | summary retention: retaining 75% |
| 1981-06-11 | #19397 / message #19397 | Cincinnati to Boston: Jose Canas #7646 (21; minor league RHP) | Boston to Cincinnati: Jason Brown #1834 (34; minor league RHP) | Cincinnati receives 1 minor-league; Boston receives 1 minor-league | none in structured cash/pick/IAFA arrays |
| 1981-06-18 | #19446 / message #19446 | Denver to New York: Ramon Pazmino #9436 (23; minor league RHP) | New York to Denver: Tony Oubre #949 (30; not labeled minor RHP); Willie Vega #7820 (22; minor league SS*) | Denver receives 1 ABL-level/not-minor-labeled, 1 minor-league; New York receives 1 minor-league | none in structured cash/pick/IAFA arrays |
| 1981-06-25 | #19497 / message #19497 | Houston to Atlanta: Ernesto Durazo #7173 (23; minor league SS*); Orlando Cervini #7230 (23; minor league RHP) | Atlanta to Houston: Mike McQuaid #416 (35; not labeled minor RHP) | Houston receives 1 ABL-level/not-minor-labeled; Atlanta receives 2 minor-league | none in structured cash/pick/IAFA arrays |
| 1981-07-02 | #19578 / message #19578 | Cincinnati to Minneapolis: Alex Rodriguez #7177 (23; minor league RF*); Jorge Moya #9169 (20; minor league LHP) | Minneapolis to Cincinnati: Lee Byron #3989 (30; not labeled minor 2B*) | Cincinnati receives 1 ABL-level/not-minor-labeled; Minneapolis receives 2 minor-league | none in structured cash/pick/IAFA arrays |
| 1981-07-16 | #19729 / message #19729 | Nashville to New York: David Mitchell #6796 (24; minor league LF); Vinny Martinez #8329 (20; minor league RHP) | New York to Nashville: Devon John #4921 (30; not labeled minor LHP) | Nashville receives 1 ABL-level/not-minor-labeled; New York receives 2 minor-league | none in structured cash/pick/IAFA arrays |
| 1981-07-23 | #19793 / message #19793 | Denver to Atlanta: Benito Rodriguez #11244 (18; minor league RF*); Rodolfo Gonzalez #9886 (19; minor league LHP) | Atlanta to Denver: Mason Chaisson #1540 (33; not labeled minor RHP) | Denver receives 1 ABL-level/not-minor-labeled; Atlanta receives 2 minor-league | none in structured cash/pick/IAFA arrays |
| 1981-07-24 | #19798 / message #19798 | Tampa Bay to Dallas: Brian Manchester #1656 (33; not labeled minor RHP) | Dallas to Tampa Bay: Jonathan Funches #6782 (24; minor league LHP) | Tampa Bay receives 1 minor-league; Dallas receives 1 ABL-level/not-minor-labeled | none in structured cash/pick/IAFA arrays |
| 1981-07-24 | #19799 / message #19799 | Dallas to Pittsburgh: Ricky Perez #8879 (23; minor league RHP); Marcos Guerrero #6438 (24; minor league RHP) | Pittsburgh to Dallas: Doug Farmer #2859 (32; not labeled minor LHP) | Dallas receives 1 ABL-level/not-minor-labeled; Pittsburgh receives 2 minor-league | none in structured cash/pick/IAFA arrays |
| 1981-07-24 | #19800 / message #19800 | San Diego to Charlotte: Yuan-zhang Tseng #2162 (33; minor league LHP) | Charlotte to San Diego: Jairo Reyna #6327 (20; minor league 1B); Tony Chavez #9619 (22; minor league RHP) | San Diego receives 2 minor-league; Charlotte receives 1 minor-league | none in structured cash/pick/IAFA arrays |
| 1981-07-25 | #19806 / message #19806 | San Francisco to San Diego: Omari Benjamin #5204 (24; minor league C) | San Diego to San Francisco: Victor Hernandez #504 (33; not labeled minor RHP) | San Francisco receives 1 ABL-level/not-minor-labeled; San Diego receives 1 minor-league | none in structured cash/pick/IAFA arrays |
| 1981-07-25 | #19807 / message #19807 | New York to Chicago: Juan Mederos #454 (33; not labeled minor RHP) | Chicago to New York: David Acosta #7107 (22; minor league 1B*); Alex Castellanos #9305 (20; minor league CF) | New York receives 2 minor-league; Chicago receives 1 ABL-level/not-minor-labeled | none in structured cash/pick/IAFA arrays |
| 1981-07-26 | #19814 / message #19814 | New York to Detroit: Chris Allen #536 (40; not labeled minor RHP) | Detroit to New York: Hector Wilson #10151 (19; minor league LF); Bobby Ramirez #8373 (21; minor league RHP) | New York receives 2 minor-league; Detroit receives 1 ABL-level/not-minor-labeled | none in structured cash/pick/IAFA arrays |
| 1981-07-27 | #19849 / message #19849 | Dallas to San Diego: Josh Brown #9507 (23; minor league RHP) | San Diego to Dallas: Eric Strickland #1737 (32; not labeled minor RHP) | Dallas receives 1 ABL-level/not-minor-labeled; San Diego receives 1 minor-league | none in structured cash/pick/IAFA arrays |
| 1981-07-27 | #19850 / message #19850 | Chicago to Tampa Bay: Tony Espinal #3086 (30; not labeled minor RHP); Luis Robles #10110 (20; minor league CF) | Tampa Bay to Chicago: Ed Montoya #2039 (31; not labeled minor 2B) | Chicago receives 1 ABL-level/not-minor-labeled; Tampa Bay receives 1 ABL-level/not-minor-labeled, 1 minor-league | none in structured cash/pick/IAFA arrays |
| 1981-07-27 | #19851 / message #19851 | Boston to Chicago: Mike Roede #323 (34; not labeled minor RHP); Desid�rio Chora #9908 (19; minor league 1B) | Chicago to Boston: Victor Hernandez #6732 (26; not labeled minor 1B) | Boston receives 1 ABL-level/not-minor-labeled; Chicago receives 1 ABL-level/not-minor-labeled, 1 minor-league | none in structured cash/pick/IAFA arrays |
| 1981-07-27 | #19852 / message #19852 | St. Louis to Miami: Joe Puerta #671 (38; not labeled minor RHP) | Miami to St. Louis: Daniel Smith #6345 (23; minor league 2B) | St. Louis receives 1 minor-league; Miami receives 1 ABL-level/not-minor-labeled | none in structured cash/pick/IAFA arrays |
| 1981-07-27 | #19853 / message #19853 | Tampa Bay to Houston: Jeff Mitchell #2790 (31; not labeled minor RHP) | Houston to Tampa Bay: Jose Carranza #9320 (21; minor league 3B*); Matt Johnson #9625 (23; minor league SS*) | Tampa Bay receives 2 minor-league; Houston receives 1 ABL-level/not-minor-labeled | none in structured cash/pick/IAFA arrays |
| 1981-07-27 | #19854 / message #19854 | Las Vegas to Portland: Manny Moran #7215 (23; minor league RF); Luis Gortarez #8610 (21; minor league RHP) | Portland to Las Vegas: Nate Souza #1688 (33; not labeled minor RHP) | Las Vegas receives 1 ABL-level/not-minor-labeled; Portland receives 2 minor-league | none in structured cash/pick/IAFA arrays |
| 1981-07-28 | #19865 / message #19865 | Houston to Phoenix: Alex Monge #5242 (28; not labeled minor LF) | Phoenix to Houston: Kerry Darling #679 (35; not labeled minor RHP); Willie Saunders #9015 (19; minor league LHP) | Houston receives 1 ABL-level/not-minor-labeled, 1 minor-league; Phoenix receives 1 ABL-level/not-minor-labeled | none in structured cash/pick/IAFA arrays |
| 1981-07-28 | #19866 / message #19866 | St. Louis to Seattle: Harry Benjamin #5944 (30; not labeled minor RHP) | Seattle to St. Louis: Isaiah Gibson #6307 (23; minor league RF) | St. Louis receives 1 minor-league; Seattle receives 1 ABL-level/not-minor-labeled | none in structured cash/pick/IAFA arrays |
| 1981-07-28 | #19867 / message #19867 | San Diego to Detroit: Kennie Murdock #4963 (30; not labeled minor RHP) | Detroit to San Diego: Bob Morales #9196 (20; minor league RHP) | San Diego receives 1 minor-league; Detroit receives 1 ABL-level/not-minor-labeled | none in structured cash/pick/IAFA arrays |
| 1981-07-28 | #19868 / message #19868 | Miami to Pittsburgh: Joe Good #3032 (29; not labeled minor RHP); Quadir May #9550 (19; minor league CF*) | Pittsburgh to Miami: Daniel Vega #8655 (29; not labeled minor 3B) | Miami receives 1 ABL-level/not-minor-labeled; Pittsburgh receives 1 ABL-level/not-minor-labeled, 1 minor-league | $144,000 cash from Miami to Pittsburgh |
| 1981-07-29 | #19877 / message #19877 | Miami to Phoenix: Omar Garcia #1184 (30; not labeled minor LHP) | Phoenix to Miami: Addie de Ruijter #1849 (33; not labeled minor 2B) | Miami receives 1 ABL-level/not-minor-labeled; Phoenix receives 1 ABL-level/not-minor-labeled | none in structured cash/pick/IAFA arrays |
| 1981-07-29 | #19878 / message #19878 | Las Vegas to Pittsburgh: Carlos Beltran #9832 (20; minor league RHP); Marcelino Bacardi #8525 (21; minor league RHP) | Pittsburgh to Las Vegas: Bill Hall #535 (33; not labeled minor RHP) | Las Vegas receives 1 ABL-level/not-minor-labeled; Pittsburgh receives 2 minor-league | none in structured cash/pick/IAFA arrays |
| 1981-07-29 | #19879 / message #19879 | Houston to San Diego: Tony Contreras #7860 (23; minor league LHP) | San Diego to Houston: Jerry Wilson #681 (33; not labeled minor RHP) | Houston receives 1 ABL-level/not-minor-labeled; San Diego receives 1 minor-league | none in structured cash/pick/IAFA arrays |
| 1981-07-30 | #19889 / message #19889 | St. Louis to Houston: Chris Sullivan #1547 (32; not labeled minor 3B) | Houston to St. Louis: Omar Han #6286 (23; minor league 1B*); Antonio Gama #8529 (21; minor league RHP) | St. Louis receives 2 minor-league; Houston receives 1 ABL-level/not-minor-labeled | none in structured cash/pick/IAFA arrays |
| 1981-07-31 | #19901 / message #19901 | Boston to Chicago: Justin Bowen #854 (34; not labeled minor RHP); Javy Madrigal #10507 (19; minor league 3B) | Chicago to Boston: Miguel Martinez #4347 (27; not labeled minor 1B*) | Boston receives 1 ABL-level/not-minor-labeled; Chicago receives 1 ABL-level/not-minor-labeled, 1 minor-league | none in structured cash/pick/IAFA arrays |
| 1981-07-31 | #19904 / message #19904 | Pittsburgh to Houston: Ricky Bautista #1990 (32; not labeled minor RHP) | Houston to Pittsburgh: Tim Mathis #7676 (21; minor league CF*); Tyler Hayes #8767 (24; minor league RHP) | Pittsburgh receives 2 minor-league; Houston receives 1 ABL-level/not-minor-labeled | none in structured cash/pick/IAFA arrays |
| 1981-07-31 | #19907 / message #19907 | Atlanta to Chicago: Bill Torres #8235 (21; minor league CF*); Rodolfo Gonzalez #9886 (19; minor league LHP) | Chicago to Atlanta: Jaden Pierce #5785 (26; minor league LF) | Atlanta receives 1 minor-league; Chicago receives 2 minor-league | none in structured cash/pick/IAFA arrays |
| 1981-07-31 | #19908 / message #19908 | Minneapolis to Houston: Chris Jenkins #6159 (27; not labeled minor LHP) | Houston to Minneapolis: Jonathan Serrano #7643 (21; minor league CF*); Alush Demachi #6369 (24; minor league SS*) | Minneapolis receives 2 minor-league; Houston receives 1 ABL-level/not-minor-labeled | none in structured cash/pick/IAFA arrays |

**Position-source exceptions (23 player/transaction pairs).** In each row, `trade_history.csv` summary descriptor is shown first; the corresponding `messages.csv` body shows the competing position. The transaction arrays and message identities/directions agree; these are text-field position disagreements. Use the player identity and direction as proven, but do not assert either disputed position without an independent historical lineup/roster source.

| Trade/message ID | Player | Trade-history summary | Matched message body |
|---|---|---|---|
| #18805 | Bryce Rose #292 | RF | LF |
| #18824 | Fernando Gomez #9939 | SS | 2B |
| #18825 | Julio Rodriguez #8385 | RF | LF |
| #18841 | Alejandro Ordaz #5436 | SS | 3B |
| #18842 | Noah Burks #10089 | SS | 3B |
| #18843 | Gary McMinn #2876 | LF | RF |
| #18873 | Dave Jackson #7776 | RF | LF |
| #18925 | Juan Santos #7757 | RF | LF |
| #19446 | Willie Vega #7820 | SS | 2B |
| #19497 | Ernesto Durazo #7173 | SS | 2B |
| #19578 | Lee Byron #3989 | 2B | 1B |
| #19578 | Alex Rodriguez #7177 | RF | LF |
| #19793 | Benito Rodriguez #11244 | RF | LF |
| #19807 | David Acosta #7107 | 1B | C |
| #19853 | Jose Carranza #9320 | 3B | 1B |
| #19853 | Matt Johnson #9625 | SS | 2B |
| #19868 | Quadir May #9550 | CF | LF |
| #19889 | Omar Han #6286 | 1B | LF |
| #19901 | Miguel Martinez #4347 | 1B | 2B |
| #19904 | Tim Mathis #7676 | CF | RF |
| #19907 | Bill Torres #8235 | CF | 2B |
| #19908 | Alush Demachi #6369 | SS | LF |
| #19908 | Jonathan Serrano #7643 | CF | LF |

The 66 structured rows contain **one explicit cash transfer**: $144,000 from Miami to Pittsburgh in #19868. No draft-pick or IAFA-cap assets are present in the structured arrays. Twenty-four trade summaries carry a contract-retention percentage (14 below 100%, ten at 100%); the structured side arrays do not encode the retaining club, so the table reports only the source term and does not infer payer identity. The trade-history summary may truncate long player descriptions; the structured player-ID arrays and matching message IDs control identities and direction.

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

Three trades followed in June: Cincinnati–Boston (#19397, June 11), Denver–New York (#19446, June 18), and Houston–Atlanta (#19497, June 25). In #19446 Denver sent minor-league RHP Ramon Pazmino to New York for RHP Tony Oubre and minor-league Willie Vega; the trade summary calls Vega a SS, while the message body calls him a 2B. The game-date performance split separates Oubre’s June 18 game from later starts because transaction timestamps are absent. Denver went from 37–29, one game off its division lead before the move, to 56–48 by the August 1 snapshot and second in the ABC wild-card order.

Cincinnati–Minneapolis traded July 2 (#19578). The final regular-season slate before the break was July 12; the Future Stars game was also played July 12 (game 1945, type 8), while the ABL conference All-Star Game was July 14 (game 8767, type 4; league setting `allstar_date=1981-7-14`; OOTP game story message #19715). Regular-season play resumed July 16 (games 1069–1080). July 12’s voting results and roster announcements are not the date of the major-league ABL All-Star Game. Do not use August 1 standings to imply what a June decision-maker knew: the transaction-date table above carries the last completed standings before each dated trade.

## 7. Post-All-Star market and deadline week

The post-All-Star sequence concentrates the season’s movement: 26 trades from July 16 through July 31. July 27 was the busiest single day with six trades. The July 24–31 span alone had 24 trades. Activity did not come only from division leaders; clubs well outside the race also moved players, and several clubs made moves in both directions.

The deadline-day package contains four trades:

- #19901: Boston sent RHP Justin Bowen and minor-league 3B Javy Madrigal to Chicago for 2B Miguel Martinez.
- #19904: Pittsburgh sent RHP Ricky Bautista to Houston for minor-league Tim Mathis #7676 (CF in the trade summary, RF in the message body) and minor-league RHP Tyler Hayes.
- #19907: Atlanta sent minor-league Bill Torres #8235 (CF in the trade summary, 2B in the message body) and LHP Rodolfo Gonzalez to Chicago for minor-league LF Jaden Pierce.
- #19908: Minneapolis sent LHP Chris Jenkins to Houston for minor-league prospects Jonathan Serrano and Alush Demachi. The trade summary lists Serrano as CF and Demachi as SS; the matched message body lists both as LF, so the positions are flagged as source conflicts below rather than treated as settled.

The July 25–31 moves with clearest present-race context include San Diego’s July 25 deal sending reliever Victor Hernandez #504 to NBC West leader San Francisco for catcher Omari Benjamin; Detroit’s July 28 acquisition of starter Kennie Murdock from San Diego for minor-league RHP Bob Morales; and the July 31 Jenkins deal from 45–58 Minneapolis to 60–42 Houston (both clubs’ records through July 30, immediately before the transaction date). Boston acquired 1B Victor Hernandez #6732 from Chicago July 27, then 2B Miguel Martinez from Chicago July 31. These moves are roster facts; they do not prove the clubs’ private assessments or expected postseason odds.

### July 31 trade deadline; August 1 post-deadline pregame snapshot

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
| Minneapolis | Ken Sears | Chad Purchase | 46–58; NBC Central 4th | Outside tournament places | 5; 11 / 8 | 0 | Moved Jenkins for two minor-league prospects; SUGGESTIVE future-asset exchange |
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
| Houston | Larry Bell | Xavier Ferguson | 61–42; ABC Central 1st (103 G) | Division leader; .592 winning percentage (Charlotte league-best .596) | 12; 14 / 21 | 1 | Most active, net player-row outflow plus Jenkins in; PROVEN activity, intent mixed |
| Seattle | David Suarez | Tony Manning | 58–46; ABC West 2nd | Provisional ABC wild-card leader | 7; 8 / 8 | 4 | Pitching signings plus WC chase; SUGGESTIVE |

Notes: “player rows in/out” counts transfer rows, not unique players or trade value. Positions are from team_record.csv; a tied record is stated as tied regardless of the position field’s ordering. Tournament slots are provisional standings context under three division winners plus one wild card per conference, not a tiebreak ruling.

## 9. ERNIE BEWELL HANDOFF — ARTICLE PREPARATION

### The league in one page

The league’s transaction story is concentration, not one uniform strategy. Trades are heavy before the first pitch (36 before April 6), nearly absent during the early season, and compressed into July (27). All 24 clubs traded, but the scale differs sharply: Houston made 12; Dallas and Miami eight; Chicago, San Diego, and Seattle seven; Los Angeles one. Atlanta led contract/signing notices with seven. The two measures should not be collapsed: an extension is not an acquisition, a transaction row is not a unique player, and roster count is not importance.

The market itself has no clean buyer/seller divide. Several leaders made targeted transactions: Detroit added established starter Kennie Murdock; San Francisco added 17-save reliever Victor Hernandez; Boston and Chicago exchanged established position players and relief arms. At the same time, San Diego sent those pitchers to leaders while 39–65, and Minneapolis sent a 9–6 starter to division-leading Houston for two minor-league prospects. Houston, already leading the ABC Central, was simultaneously the most trade-active club and sent 21 player rows out against 14 in. A single “deadline winner” narrative would flatten the evidence.

The evidence points to roles—starting rotation, relief, infield/outfield depth—more securely than it points to motives. The OOTP profile export provides numeric fields but no local scale key; the owner export gives identities but not patience/fiscal-priority profiles; no owner instruction or private negotiation record is present. Standings and performance put each transaction in context, but must be date-correct. The writing desk can say what changed, when, and where each club stood. It should frame strategic meaning as a question or cautious inference, not as a quotation or private motive.

### The 24 clubs in journalist language

**Miami.** Miami was 55–49, second in the NBC East and one game behind Dallas for the provisional wild-card lead. Daniel Baker is the current GM and Danny Leon the current owner; the export gives no owner-specific personality evidence. Miami was involved in eight trades, sending 14 player rows and receiving nine, plus one contract notice: the July 17 signing of RF Logan Ellsworth. On July 28 it sent Joe Good and minor-league Quadir May to Pittsburgh for Daniel Vega, with $144,000 cash from Miami’s side. The activity runs in both directions while the club remains in the race. It supports “active and mixed,” not a firm buyer/seller label or a claim about Baker’s motive.

**Phoenix.** Phoenix finished the cutoff at 53–51, second in the NBC West, five games behind San Francisco and three games behind the provisional NBC wild-card lead. Bryan Somers is the current GM; Mike Dutcher is the owner. Four trades moved four player rows in and five out. Four type-2 notices include a three-year Kerry Darling deal, a January signing by Joe Rogers, Eric Strickland’s contract, and Nick Perez’s extension. Three named notices concern pitchers/relievers, a documented offseason contract pattern but not proof of a diagnosed weakness. On July 28 Phoenix received 35-year-old RHP Kerry Darling and minor-league LHP Willie Saunders from Houston for LF Alex Monge; on July 29 it received 33-year-old 2B Addie de Ruijter from Miami for LHP Omar Garcia. These are specific roster exchanges, not proof of intent.

**Atlanta.** Atlanta was 46–58, tied on record with Tampa Bay and outside the NBC tournament positions. Hector Naranjo is the current GM; Tyler Carter is the owner, with no verified ownership traits or instructions in the export. Atlanta made six trades (nine player rows in, seven out) and led all clubs with seven contract/signing notices. Those notices include 2B Eric Gray, 1B Jordan Santistevan, 3B Luis Rosales, and pitchers Steve Farber, Netz Altman, Sergio Duran, and Mike McQuaid. That is proven transaction volume and a position-spanning offseason build; Atlanta’s low cutoff OPS (.677) is later outcome context, not proof of what the front office knew when it acted. The resulting direction is still mixed rather than a proven rebuild or push.

**Dallas.** Dallas stood 56–48, second in the NBC Central and the provisional NBC wild-card leader. Jim Griffith is the current GM; Tommy Mauldin the owner. The club participated in eight trades, bringing in 12 player rows and sending ten, plus two contract notices, including Ryan Millich and Alfredo Noriega. On July 24 Dallas acquired 33-year-old RHP Brian Manchester from Tampa Bay and sent 24-year-old minor-league LHP Jonathan Funches to Tampa Bay (#19798). Dallas also sent two minor-league RHPs to Pittsburgh for LHP Doug Farmer that day (#19799), then acquired RHP Eric Strickland from San Diego for minor-league RHP Josh Brown on July 27 (#19849). The moves show activity while Dallas held the provisional wild-card lead, not a single-purpose buying plan.

**Detroit.** Detroit led the NBC Central at 57–47. Jesus Martinez is the current GM; David Figueiredo is the owner, with owner motive unverified. Detroit made five trades, receiving five player rows and sending nine, and had four contract notices, including Ethan Harrison, Luis Rios’s re-signing, and Brett Helvie’s multi-year deal. The clearest deadline addition was 30-year-old RHP Kennie Murdock from San Diego on July 28 for minor-league RHP Bob Morales. Before the deal, Murdock had a 10–6 record, 2.43 ERA, and 137 innings; he made one later-date Detroit start by July 31. The transaction is consistent with adding rotation support to a division leader. It does not establish that Detroit considered itself one player away or that the move changed its postseason odds.

**Charlotte.** Charlotte had the league’s best record, 62–42, and led the NBC East. Jose Ruiz is the current GM; Adam Butler the owner. The Colonels made three trades (three player rows in, four out) and had two contract notices, Dustin Meredith and Juan Lomeli. Charlotte’s 1981 pitching ERA was the best in the league at 3.25 through the cutoff. It did not make one of the four July 31 trades. This is a relatively quiet transaction record attached to a leading club with strong pitching results. That can support a question about confidence in the roster, but silence alone cannot prove “standing pat” was a deliberate strategy, nor can the export tell us whether the club pursued deals that did not happen.

**San Diego.** San Diego was 39–65, fourth in the NBC West, and had the league’s lowest team OPS (.658) at the cutoff. Matt Johnson is current GM; Danny Tocornal owner. The Seraphs made seven trades, with ten player rows received and eight sent, and no type-2 contract notices. The most visible sequence sent 33-year-old reliever Victor Hernandez (#504, 17 saves through July 24) to division-leading San Francisco, then starter Kennie Murdock to division-leading Detroit. San Diego received catcher Omari Benjamin and minor-league RHP Bob Morales. The clubs receiving veteran arms were in the race; San Diego was far back. The evidence suggests a present-for-future exchange in these two deals, but does not establish a franchise-wide rebuild or the GM’s private plan.

**Tampa Bay.** Tampa Bay was 46–58, tied on record with Atlanta in the NBC East, and outside the tournament field. Angel Gonzalez is the current GM; Jaylin Reed owner. The club made five trades (seven player rows in, six out) and one type-2 contract event, Nate Smeltz’s reported seven-year agreement. Tampa Bay’s team OPS (.742) ranked among the league’s best by August 1, while its 4.28 ERA was among the highest. That is a current snapshot, not evidence available when the offseason contract was signed. Tampa Bay later sent starter Brian Manchester to wild-card-leading Dallas for Jonathan Funches. The roster movement and results point in different directions; there is not enough evidence to call the club a buyer, seller, or team with a settled 1981 plan.

**Los Angeles.** Los Angeles was 51–53, third in the NBC West, outside the current wild-card lead. Gary Jones is the current GM; Tony Cruz is the owner. The Cobras had the league’s smallest trade count: one trade, one player row in and two out, with no type-2 contract notice in the window. The single listed deal was the December 22 exchange sending Luis Prieto and minor-league Miguel Campos to Miami for Luis Davis. Los Angeles’s 4.08 team ERA was toward the high end at the cutoff, but no late external pitching acquisition is in the verified trade/signing record. A quiet transaction log is an observable fact; whether it reflects confidence, budget, missed opportunities, or internal development is unknown.

**San Francisco.** San Francisco led the NBC West at 58–46. Luis Martinez is the current GM; Jon Teeple the current owner. Four trades moved five player rows in and five out; two contract notices involved Chad Averitt and Kyle Chambers. San Francisco’s .753 team OPS led the league through July 31. Its July 25 transaction sent catcher Omari Benjamin to San Diego for veteran reliever Victor Hernandez (#504), who had 17 saves and a 2.78 ERA before the trade date. The move placed a specific leverage arm with the division leader, while the available log shows no post-trade San Francisco appearance by cutoff. A targeted bullpen addition is a fair description; a claim about strategy or expected role needs the club’s roster usage after August 1.

**Minneapolis.** Minneapolis was 46–58, fourth in the NBC Central and 11 games behind the division lead through July 31. Ken Sears is the current GM; Chad Purchase the owner. Five trades produced 11 player rows in and eight out, with no type-2 contract notices. On July 31 Minneapolis sent 27-year-old LHP Chris Jenkins (9–6, 3.48 ERA over 98⅓ innings through the cutoff) to division-leading Houston for minor-league prospects Jonathan Serrano #7643 and Alush Demachi #6369. The summary lists them as CF and SS, while the matched message body lists both as LF; the position conflict is recorded below. The exchange moved established innings for two minor-league assets. It supports a future-facing reading of this deal only, not an organization-wide teardown.

**Chicago.** Chicago was 55–49, third in the NBC Central and one game behind Dallas in the provisional wild-card standings; Miami was tied with Chicago at 55–49. Mike Hatton is the current GM; Willie Roy owner. Seven trades brought in 11 player rows and sent nine; two contract notices include Shane Cobb’s re-signing and Andrew Flugence’s signing. Late July moved established players both ways: Chicago sent 26-year-old 1B Victor Hernandez #6732 to division-leading Boston on July 27 for RHP Mike Roede and minor-league 1B Desidério Chora (#19851). On July 31 Chicago sent Miguel Martinez #4347 to Boston for 34-year-old RHP Justin Bowen #854 and minor-league 3B Javy Madrigal #10507 (#19901); the trade summary calls Martinez a 1B, while the matched message calls him a 2B. The two exchanges moved position-player and pitching assets in both directions; they do not make Chicago a simple buyer or seller.

**Denver.** Denver stood 56–48, third in the ABC West and two games behind Seattle’s provisional wild-card lead. Kevin Bradley is the current GM; Ken Ellinger the owner. The Rocketeers made four trades, with five player rows in and six out, and one contract notice for veteran Boboy Nierva. On June 18 Denver sent minor-league RHP Ramon Pazmino to New York for RHP Tony Oubre and minor-league Willie Vega; the trade summary lists Vega as SS, while the matched message lists him as 2B (#19446). Oubre’s later-date starts totaled 53⅓ innings with a 3.21 ERA through July 31; the June 18 game is kept separate because no transaction time is exported. Acquiring Oubre added a documented rotation arm while Denver pursued the wild card, but does not prove expected odds.

**Cincinnati.** Cincinnati was 53–51, tied on record with Nashville in the ABC Central but behind division-leading Houston. Phil Trammel is the current GM; Ji-hu Ju the owner. The Cougars made five trades, taking in five player rows and sending eight, plus one type-2 notice for RF Tony Jones’s six-year agreement. Cincinnati’s 1980 record was 97–65 and it won its division, but that historical success does not establish 1981 urgency. The July 2 deal with Minneapolis was the only trade in the first half of July. By the cutoff, Cincinnati remained in a tied second-place record group rather than the wild-card lead. A cautious read is a club with activity but no unmistakable deadline-day statement; its private assessment of the tie is not in the records.

**New York.** New York was 47–57, third in the ABC East and ahead of fourth-place Pittsburgh (44–60). Jeremy Burk is the current GM; Alberto Dominguez the owner. The Aces made six trades (11 player rows in, seven out) and had one type-2 contract notice: Pedro Marquez’s three-year retention at the start of the offseason. In June New York sent RHP Tony Oubre #949 and minor-league Willie Vega #7820 (position disputed: trade summary SS, message body 2B) to Denver for minor-league RHP Ramon Pazmino #9436 (#19446). On July 16 New York received minor-league LF David Mitchell #6796 and minor-league RHP Vinny Martinez #8329 from Nashville, sending 30-year-old LHP Devon John #4921 to Nashville (#19729). These trades moved veteran pitching out and younger assets in on two dates; they do not prove a long-term plan.

**Nashville.** Nashville was 53–51, tied with Cincinnati on record in the ABC Central, behind Houston and outside the current wild-card lead. Justin Gomez is current GM; Chris Smith owner. The Blues made six trades, receiving ten player rows and sending seven, and recorded four contract notices, chiefly pitchers: closer Joe Berger, Jacob Sutherland, Frank Larson, and Adam Rapoport. These transactions show an offseason concentration in pitching contracts, without proving that relief was the organization’s perceived weakness. The July 16 trade sent minor-league LF David Mitchell and RHP Vinny Martinez to New York for LHP Devon John. At the cutoff Nashville shared a 53–51 record with Cincinnati. Report the tie as a tie; do not invent a tiebreak or imply the club had lost a qualification decision.

**Las Vegas.** Las Vegas led the ABC West at 60–44. Roberto Choto is the current GM; Terry Vinson owner. The Gamblers made five trades with seven player rows in and seven out, and had three type-2 contract notices: Kyle Marks, catcher Jon Dominguez, and 1B Eddie Rodriguez. Their 3.31 ERA ranked among the league’s best through the cutoff; they had a two-game division cushion over Seattle. On July 27 Las Vegas acquired 33-year-old RHP Nate Souza from Portland for minor-league RF Manny Moran and minor-league RHP Luis Gortarez (#19854). On July 29 it sent minor-league RHP Carlos Beltran and Marcelino Bacardi to Pittsburgh for 33-year-old RHP Bill Hall (#19878). The July 28 Alex Monge/Kerry Darling/Willie Saunders trade was Houston–Phoenix, not a Las Vegas transaction (#19865). Las Vegas was active while leading; the record does not explain why.

**Portland.** Portland was 41–63, fourth in the ABC West. Ben Hayter is the current GM; Ben Criswell the owner. The Lumberjacks made five trades (eight player rows in, six out) but had no type-2 contract notices. Their clearest late-July move sent veteran Nate Souza to Las Vegas in exchange for minor-league players Manny Moran and Luis Gortarez. The club also acquired and moved players in other packages, so the activity is not a straightforward mass sale. Portland’s 4.53 ERA was the league’s highest through the cutoff; that makes pitching a relevant roster question, but the late trade sent away a pitcher rather than adding one. It is not enough to conclude the club was rebuilding or to infer its budget or ownership priorities.

**St. Louis.** St. Louis was 48–55 through 103 games, fourth in the ABC Central and well behind Houston. David Ward is current GM; J.D. Sommer owner. Five trades balanced at six player rows in and six out. Five contract notices included Victor Campos, Alex Lander, Joe Puerta, and two separate Andrew Williamson deal events. The latter were a January contract notice and a May extension; they are not two acquisitions. July 27 St. Louis sent Puerta to Miami for Daniel Smith; July 30 it sent Chris Sullivan to Houston for Omar Han and Antonio Gama. The numbers show roster movement and contract retention, but not a single clear direction. A 103-game record is not directly comparable to a 104-game team’s raw win total without noting games played.

**Boston.** Boston led the ABC East at 54–50. Bobby Ramsey is current GM; Alex Mendez the owner. Six trades involved six player rows in and nine out; the message file has no type-2 contract notice for Boston in the window. On July 27 Boston acquired 26-year-old 1B Victor Hernandez #6732 from Chicago for RHP Mike Roede #323 and minor-league 1B Desidério Chora #9908 (#19851). On July 31 it sent 34-year-old RHP Justin Bowen #854 and minor-league 3B Javy Madrigal #10507 to Chicago for Miguel Martinez #4347 (#19901); the summary/message position fields conflict on Martinez (1B/2B), so this packet does not assign a definitive position. Through July 31 Hernandez had 13 plate appearances after the trade date; no later-date Martinez/Bowen appearance is recorded. This was a late position-player/pitcher exchange, not proof of an “all-in” plan.

**Pittsburgh.** Pittsburgh was 44–60, fourth in the ABC East. Nick Heller is current GM; Jared Brown owner. The Express made five trades, with nine player rows in and six out, and no type-2 contract notices. On July 28 Pittsburgh received Daniel Vega from Miami in a trade that included $144,000 cash from Miami; on July 31 it sent 32-year-old RHP Ricky Bautista to Houston for minor-league Tim Mathis #7676 (CF in the trade summary, RF in the message body) and minor-league RHP Tyler Hayes #8767. Pittsburgh’s late record was far from the division lead, but its packages included different age/roster classes. The team’s 4.42 ERA was among the highest at cutoff; no verified transaction record establishes a rebuild or that the pitching figure drove the July decisions.

**Philadelphia.** Philadelphia was 48–56, second in the ABC East but outside the wild-card lead. Jon Ramos is current GM; Gabe Eckard owner. Three trades brought in six player rows and sent four; one contract notice records Luis Cespedes’s $4.64 million agreement. In the Dec. 14 trade with Cincinnati (#18825), Philadelphia sent 30-year-old 3B Luis Mata #4703 for RHP Oscar Rebeles #6293, minor-league LHP Pedro Camacho #6819, and minor-league RF Julio Rodriguez #8385 (RF/LF position disagreement). The signing and trade show early roster construction, not motive. Philadelphia did not trade July 16–31. That supports a “built early, then quiet” reporting question, not an assertion management was satisfied or lacked opportunities. No evidence links the owner to the contract decision.

**Houston.** Houston led the ABC Central at 61–42 through 103 games (.592). Charlotte had the league’s best winning percentage at 62–42 (.596). Larry Bell is current GM; Xavier Ferguson owner. The Mavericks made 12 trades—the league high—with 14 player rows received and 21 sent, plus one contract notice for Brent Keyser’s extension. Houston acquired RHP Ricky Bautista from Pittsburgh and LHP Chris Jenkins from Minneapolis in two July 31 trades (#19904, #19908). On July 28 Houston sent LF Alex Monge #5242 to Phoenix for 35-year-old RHP Kerry Darling #679 and 19-year-old minor-league LHP Willie Saunders #9015 (#19865). The team was a division leader adding experienced pitching while making the league’s most deals and sending more player rows out than in. That is a mixed transaction profile, not a disclosed motive or ownership direction.

**Seattle.** Seattle was 58–46, second in the ABC West and the provisional wild-card leader. David Suarez is current GM; Tony Manning owner. Seven trades balanced at eight player rows in and eight out. Four contract notices involved pitchers: Jesse Street, Nelson Majano, Antonio Romo, and Eduardo Gonzales. On July 28 Seattle sent 23-year-old minor-league RF Isaiah Gibson #6307 to St. Louis for 30-year-old RHP Harry Benjamin #5944 (#19866); Seattle therefore added Benjamin and gave up Gibson. Seattle had no July 31 trade. The club’s position placed it in a provisional tournament slot without a division lead. Pitcher contract activity alongside a wild-card position invites a depth question, but no specific diagnosed weakness or signing-time knowledge is established.

### GM character cards — verified fields vs observed behavior

The GM identity is the current team_roster_staff.csv general_manager link to coaches.csv. Age and experience below are raw current fields; “experience” is not silently converted to franchise tenure. The export provides numeric profile values, but no local key defines their scale, labels, or mapping, so this packet does not turn them into personality claims. The last column is deliberately the same caution for every GM.

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
- Boston traded for two established position players from Chicago late in July, while sending RHP Justin Bowen back in the final deal.
- Chicago received Bowen, who had 18 saves before July 31, while moving Miguel Martinez to Boston.
- Houston acquired 9–6 starter Chris Jenkins on July 31 from a club 11 games behind its division leader at the previous-day snapshot.
- Minneapolis’s side of that deal was two minor-league prospects, Jonathan Serrano and Alush Demachi. The trade-history summary lists them as CF and SS; the matched message body lists both as LF. That package permits a “present innings for future assets” description for this transaction only, while the positions remain unresolved.

The data does not establish that any of those teams had no alternative, that a particular player was the only solution, or that the acquired player immediately performed as intended. OOTP player-game totals through the cutoff belong to the season across clubs; they must not be presented as receiving-club production.

### League-wide movement patterns

**Most trades:** Houston 12; Dallas and Miami 8; Chicago, San Diego, Seattle 7; Atlanta, Cincinnati, New York, Nashville, Boston 6.  
**Fewest:** Los Angeles 1; Charlotte and Philadelphia 3.  
**Most player rows out:** Houston 21, Miami 14. **Most received:** Houston 14, Dallas 12; Chicago, Minneapolis, and New York 11 each. These totals differ from unique players and from deal counts.

**Repeated trading partners:** Detroit and New York completed three trades. Detroit/San Diego, San Diego/Houston, Chicago/Boston, Tampa Bay/Chicago, St. Louis/Houston, and Minneapolis/Houston each completed two.  
**Players appearing in two trade rows:** Jerry Wilson #681 moved from San Francisco to Chicago in January (#18873) and appears again in a July 29 trade in which San Diego sent him to Houston (#19879); the intervening roster path is not established by these 66 structured trades. Vinny Martinez #8329 went from Miami to Nashville in December (#18824), then from Nashville to New York on July 16 (#19729). Rodolfo Gonzalez #9886 went from Denver to Atlanta July 23 (#19793), then Atlanta to Chicago July 31 (#19907). Do not imply a continuous path for Wilson or infer any unlogged transaction.

**Conference-side participation:** ABC clubs appear on 69 trade sides; NBC clubs on 63. Division side counts are ABC Central 28, ABC East 20, ABC West 21; NBC Central 25, NBC East 22, NBC West 16. This counts each club-side appearance, so cross-division trades count once in each participating division. It is not a count of trades “belonging” to one division.

The late market involved both direct contenders and distant clubs. The two most visible present-race arms moved from 39–65 San Diego to division leaders Detroit and San Francisco. Conversely, July 31’s Minneapolis–Houston transaction paired a distant club and a leader. The records support a geography of movement; they do not support broad claims that every contender bought or every non-contender sold.

### Clubs that changed direction / stayed the course

**Transactions proven:** Every club traded at least once; no club was literally transaction-free. Activity increased sharply in July.  
**Suggestive of a change within the window:** San Diego’s pair of veteran-pitcher departures; Minneapolis’s deadline exchange of Jenkins for two minor-league prospects (positions disputed in source text); Houston’s high-volume two-way activity while leading; Boston’s late position-player/pitcher exchange. These observations describe the direction of specific player flows, not complete franchise plans.  
**Quiet relative to the league:** Los Angeles (one trade), Charlotte and Philadelphia (three each). The source cannot say whether they stood pat by choice or simply lacked completed deals.  
**Not supported:** “Atlanta rebuilt”; “Houston was all-in”; “Seattle’s owner demanded a wild-card berth”; “Charlotte trusted its roster”; “the market rejected a player”; any claim about an unrecorded offer, clubhouse or fan reaction.

### Front-office identity — cautious synthesis

The strongest distinction in the evidence is observable behavior, not personality. Atlanta and St. Louis combined high contract-notice counts with trade activity. Houston combined the most trades with high player-row turnover and two deadline-day pitching acquisitions. San Diego and Minneapolis each moved a visible veteran pitcher toward a contending club for a catcher/minor-league return. Detroit, San Francisco, Denver, and Boston made transactions that map to identifiable roster roles while positioned in or near tournament places. The GM raw rating fields may make useful follow-up questions, but their numerical encoding must be decoded before applying words such as “aggressive,” “prospect-minded,” or “win-now.”

### Fan-base implications

The export contains no attributable fan quotations, newspaper reaction, or attendance response tied to individual deals. Current team-financial rows do contain attendance, fan interest, fan loyalty, and market fields, but a single August 1 snapshot cannot show whether a trade changed those measures. Safe implication language is conditional: a deadline addition gives followers a visible roster change to evaluate; a quiet leader leaves its existing core to carry the race; a struggling club moving established innings for minor-league players places more attention on what those prospects become. Do not state that fans were excited, angry, or disappointed without independent evidence.

## 10. Deadline cast — people attached to observable decisions

These are reporting subjects, not personality profiles. GM and owner names are the current August 1 staff links; the files do not establish that each person held the role for the entire reporting window. “Observed behavior” means dated transactions, not private motive. No independently decoded GM personality scale, verified owner trait, or attributable quotation is available.

| Figure | Verified position / situation | Noteworthy action and observed behavior | Why important / unanswered question |
|---|---|---|---|
| Larry Bell — Houston | Current GM; Mavericks led ABC Central at 61–42 through 103 games. | Houston made a league-high 12 trades, 14 player rows in and 21 out. Late acquisitions included Darling/Saunders (#19865), Bautista (#19904), and Jenkins (#19908). | High two-way turnover by a leader. What needs did the office identify, and how did it weigh established pitchers against players moved out? Motive is not in the export. |
| Matt Johnson — San Diego | Current GM; Seraphs were 39–65, last in NBC West. | San Diego sent Victor Hernandez to San Francisco for catcher Omari Benjamin (#19806), Murdock to Detroit for Bob Morales (#19867), and Jerry Wilson to Houston for Tony Contreras (#19879). | Two veteran arms went to leaders. What did San Diego expect from each return, and was there a broader plan? These deals alone do not prove a rebuild. |
| Jesus Martinez — Detroit | Current GM; Dukes led NBC Central at 57–47. | Acquired 30-year-old RHP Kennie Murdock from San Diego for minor-league RHP Bob Morales on July 28 (#19867). | A specific rotation addition by a division leader. What role did the club envision, and how did it deploy Murdock after this cutoff? |
| Luis Martinez — San Francisco | Current GM; Warriors led NBC West at 58–46. | Sent minor-league catcher Omari Benjamin to San Diego for 33-year-old reliever Victor Hernandez on July 25 (#19806); Hernandez had 17 saves before the trade date. | A targeted late bullpen addition. What leverage role was planned? No post-trade SF appearance is visible by August 1. |
| Ken Sears — Minneapolis | Current GM; Blizzard were 46–58, fourth in NBC Central. | Sent Chris Jenkins to Houston for prospects Jonathan Serrano and Alush Demachi on July 31 (#19908). | Established innings for minor-league assets is the clearest one-deal contrast; prospect positions conflict between source texts. No club-wide teardown can be inferred. |
| Jim Griffith — Dallas | Current GM; Rustlers were 56–48 and provisional NBC wild-card leaders. | Acquired Brian Manchester from Tampa Bay (#19798), sent two minor RHPs to Pittsburgh for Doug Farmer (#19799), and acquired Eric Strickland from San Diego (#19849). | Three July moves during a close WC contest. Did the office target particular depth/innings needs? Offers and private evaluation are unavailable. |
| Mike Hatton — Chicago | Current GM; Fire were 55–49, third in NBC Central and one game behind Dallas in the provisional WC race. | Chicago and Boston made two consecutive exchanges in both directions: #19851 and #19901. | A useful case of a contender changing roster pieces without a one-way buyer/seller story. Miguel Martinez’s position is disputed. |
| Bobby Ramsey — Boston | Current GM; Patriots led ABC East at 54–50. | Received Victor Hernandez #6732 on July 27 and Miguel Martinez on July 31 from Chicago (#19851/#19901). | A division leader exchanged position-player and pitching assets with one club in deadline week. The underlying roster plan requires reporting. |
| David Suarez — Seattle | Current GM; Comets were 58–46, second in ABC West and provisional ABC WC leaders. | Acquired RHP Harry Benjamin from St. Louis for minor-league RF Isaiah Gibson on July 28 (#19866); Seattle made no July 31 trade. | A WC leader added a veteran arm before the deadline. The transaction file does not specify intended role or why Seattle made no deadline-day deal. |

Numeric GM profile fields lack a local interpretation key; ownership fields identify current owners but provide no verified owner-to-trade relationship. Report the people and observed decisions without inventing character, motive, or tenure.

## 11. Organizational archetypes — actions, not motive labels

These overlapping groupings describe visible transaction patterns through July 31. They are not declarations of private plans.

- **Leaders with identifiable late additions:** Detroit (Murdock, #19867), San Francisco (Victor Hernandez #504, #19806), Boston (Victor Hernandez #6732 and Miguel Martinez, #19851/#19901), Houston (Darling/Saunders, Bautista, Jenkins, #19865/#19904/#19908), and Las Vegas (Nate Souza #19854; Bill Hall #19878). Additions are observable; “reinforcing” is a limited reading, not proof of intent.
- **A low-volume leader:** Charlotte led the league at 62–42 and made three trades total, last July 24 (#19800); it made no July 31 trade. Whether that reflects confidence, budget, failed talks, or another cause is unknown.
- **Wild-card leaders and challengers with different profiles:** Dallas, Miami, Chicago, Phoenix, Denver, and Seattle all made trades while in or near provisional WC positions. Their moves ranged from established pitching additions to mixed exchanges; see the race boards for particulars.
- **Established arms moved to contenders in specific deals:** San Diego sent Murdock and Victor Hernandez to division leaders; Minneapolis sent Jenkins to Houston. These player flows support a present/future contrast for those transactions only, not a franchise-wide “seller” or “rebuild” label.
- **High-volume, mixed-direction clubs:** Houston led with 12 trades; Dallas and Miami made eight; Chicago, San Diego, and Seattle made seven. Houston had 14 transfer rows in and 21 out. Counts show activity, not value or motive.
- **Front-loaded / late-quiet records:** Los Angeles (one trade, last Dec. 22), Philadelphia (three, last Dec. 26), and Cincinnati (five, last July 2) made no trade July 16–31. Denver’s last was July 23; Nashville’s July 16, so neither traded July 24–31. This is a reason to ask questions, not an explanation.
- **Other mixed or unresolved cases:** Atlanta, Tampa Bay, and Minneapolis were tied at 46–58 in NBC-related WC arithmetic; Cincinnati and Nashville tied at 53–51 in ABC Central; St. Louis had played 103 games; Portland was 41–63. Their packages differ and do not establish common intent.

## 12. Transactions Ernie should know

The current GM names are August 1 staff links, not a historical appointment timeline. All competitive records below are through the last completed game before each transaction date. The full inventory above contains all player IDs and both sides; these summaries repeat key deals for a self-contained handoff.

### MUST USE

1. **San Diego–Detroit, July 28 (#19867):** San Diego sent 30-year-old RHP Kennie Murdock (#4963) to Detroit for 20-year-old minor-league RHP Bob Morales (#9196). San Diego was 38–62 through July 27; Detroit led NBC Central at 55–46. Before the deal Murdock was 10–6, 2.43 ERA, 137 IP; he made one Detroit start by July 31. Current GMs: Matt Johnson and Jesus Martinez. **Importance:** an established starter joined a division leader. **Limit:** no proof Detroit was “one player away” or any particular valuation of Morales.
2. **San Diego–San Francisco, July 25 (#19806):** San Diego sent 33-year-old RHP Victor Hernandez (#504) to San Francisco for minor-league C Omari Benjamin (#5204). The Warriors led NBC West at 55–43; San Diego was 37–61 through July 24. Hernandez had 17 saves and a 2.78 ERA before the deal. Current GMs: Matt Johnson and Luis Martinez. **Importance:** a division leader added a proven reliever. **Limit:** pre-trade stats are not SF production; post-trade use is unknown at cutoff.
3. **Minneapolis–Houston, July 31 (#19908):** Minneapolis sent 27-year-old LHP Chris Jenkins (#6159; 9–6, 3.48 ERA, 98⅓ IP) to Houston for minor-league Jonathan Serrano (#7643) and Alush Demachi (#6369). Houston led ABC Central at 60–42 through July 30; Minneapolis was 45–58. Current GMs: Ken Sears and Larry Bell. **Importance:** established innings for two younger assets. **Limit:** positions conflict (summary CF/SS, message both LF); no valuation or general rebuild conclusion.
4. **Boston–Chicago, July 27 and 31 (#19851/#19901):** Boston first sent RHP Mike Roede (#323) and minor-league 1B Desidério Chora (#9908) to Chicago for 26-year-old 1B Victor Hernandez (#6732). Four days later Boston sent RHP Justin Bowen (#854) and minor-league 3B Javy Madrigal (#10507) to Chicago for Miguel Martinez (#4347). Through July 30, Boston led ABC East at 53–50 and Chicago was 55–48. Current GMs: Bobby Ramsey and Mike Hatton. **Importance:** consecutive two-way exchanges between contenders. **Limit:** source texts conflict on Martinez’s position (1B/2B); avoid choosing one or calling either club a simple buyer/seller.
5. **Tampa Bay–Dallas, July 24 (#19798):** Tampa sent 33-year-old RHP Brian Manchester (#1656) to Dallas for 24-year-old minor-league LHP Jonathan Funches (#6782). Dallas led provisional NBC WC at 52–45; Tampa was 43–54 through July 23. Current GMs: Angel Gonzalez and Jim Griffith. **Importance:** present/younger-player contrast in this deal. **Limit:** not a whole-season plan.
6. **Houston–Phoenix, July 28 (#19865):** Houston sent 28-year-old LF Alex Monge (#5242) to Phoenix for 35-year-old RHP Kerry Darling (#679) and 19-year-old minor-league LHP Willie Saunders (#9015). Houston led ABC Central at 57–42; Phoenix was 51–49 through July 27. Current GMs: Larry Bell and Bryan Somers. **Importance:** opposite club contexts and roster classes. **Limit:** this was Houston–Phoenix, not Las Vegas; no disclosed role/motive.
7. **Nashville–New York, July 16 (#19729):** Nashville sent minor-league LF David Mitchell (#6796) and minor-league RHP Vinny Martinez (#8329) to New York for 30-year-old LHP Devon John (#4921). Through the last completed day, Nashville was 45–44 and New York 41–48. Current GMs: Justin Gomez and Jeremy Burk. **Importance:** established arm for two minor leaguers. **Limit:** one transaction does not prove direction of either organization.

### STRONG SUPPORTING MATERIAL

- **St. Louis–Seattle, July 28 (#19866):** St. Louis sent 30-year-old RHP Harry Benjamin (#5944) to Seattle for 23-year-old minor-league RF Isaiah Gibson (#6307). Seattle was 55–46 and pursuing ABC West/WC positions; St. Louis was 46–53 through 103 games. Current GMs: David Ward and David Suarez. The trade adds an established arm to a WC leader; role is unspecified.
- **Las Vegas–Portland, July 27 (#19854):** Portland sent 33-year-old RHP Nate Souza (#1688) to Las Vegas for minor-league RF Manny Moran (#7215) and RHP Luis Gortarez (#8610). Las Vegas led its division at 58–42; Portland was 38–62 through July 26. Current GMs: Roberto Choto and Ben Hayter. This is a single present/younger-package contrast.
- **Las Vegas–Pittsburgh, July 29 (#19878):** Las Vegas sent minor RHPs Carlos Beltran (#9832) and Marcelino Bacardi (#8525) to Pittsburgh for 33-year-old RHP Bill Hall (#535). Las Vegas led ABC West at 59–43; Pittsburgh was 43–59. Current GMs: Roberto Choto and Nick Heller. Shows Pittsburgh’s late activity, not motive.
- **Tampa Bay–Houston, July 27 (#19853):** Tampa sent 31-year-old RHP Jeff Mitchell (#2790) to Houston for minor-league Jose Carranza (#9320) and Matt Johnson (#9625). Houston led its division; Tampa was 44–56. Carranza’s and Johnson’s positions conflict across the source texts (3B/1B; SS/2B). Current GMs: Angel Gonzalez and Larry Bell. Use names and level, not disputed positions.
- **Miami–Pittsburgh, July 28 (#19868):** Miami sent RHP Joe Good (#3032), minor-league Quadir May (#9550), and $144,000 cash to Pittsburgh for 3B Daniel Vega (#8655). Miami was 53–48, Pittsburgh 43–58 through July 27. Current GMs: Daniel Baker and Nick Heller. Only structured cash transfer in this window; May’s position conflicts (CF/LF).

### OPTIONAL COLOR

- **Cincinnati–Boston, June 11 (#19397):** Cincinnati sent minor RHP Jose Canas (#7646) to Boston for 34-year-old minor RHP Jason Brown (#1834). Current GMs: Phil Trammel and Bobby Ramsey. This shows depth exchange; no larger significance is established.
- **San Diego–Houston, July 29 (#19879):** San Diego sent 33-year-old RHP Jerry Wilson (#681) to Houston for minor LHP Tony Contreras (#7860). Wilson also appears in January’s San Francisco-to-Chicago trade (#18873); the intervening path is not established.
- **Dallas–Pittsburgh, July 24 (#19799):** Dallas sent minor RHPs Ricky Perez (#8879) and Marcos Guerrero (#6438) to Pittsburgh for 32-year-old LHP Doug Farmer (#2859). Adds context to Dallas’s same-day activity with Tampa Bay.

## 13. Quiet moves that matter

- **Depth, not a headline name:** Cincinnati and Boston exchanged minor-league right-handers Jose Canas and Jason Brown on June 11 (#19397). Both are explicitly minor-league in the message/trade description; deeper purpose is unknown.
- **Repeated player trail:** Jerry Wilson #681 appears in #18873 (San Francisco to Chicago) and #19879 (San Diego to Houston), with no complete intervening path in this trade window. Vinny Martinez #8329 moved Miami to Nashville (#18824), then Nashville to New York (#19729). Rodolfo Gonzalez #9886 moved Denver to Atlanta (#19793), then Atlanta to Chicago (#19907). These repeats do not prove continuous roster histories.
- **Portland depth package:** Portland sent RHP Nate Souza to Las Vegas for minor RHP Luis Gortarez and RF Manny Moran (#19854). A veteran arm went to a division leader; no reason is recorded.
- **St. Louis/Seattle:** St. Louis sent Harry Benjamin to Seattle for Isaiah Gibson (#19866). It is another specific veteran arm moving to a contender, not evidence of an overall plan.
- **Contract notices:** 46 type-2 notices appear in the window, 42 before the April 6 opener and four in-season (May 8, 13, 16; July 17). Notices include extensions/re-signings, such as two Andrew Williamson events for St. Louis; they are not all new-player acquisitions and do not exhaust every contract event.
- **Cash:** Miami sent Pittsburgh $144,000 in #19868, the only non-zero structured cash line among the 66 deals. No draft-pick or IAFA-cap asset is encoded in the trade arrays.

## 14. Clubs that did not move late

Here “late” means July 16–31, after regular-season play resumed following the break; the narrower final-week window is July 24–31. Trade-history records have dates but no time-of-day.

**No trades July 16–31:** Los Angeles made one trade in the whole window, on Dec. 22 (#18855); Cincinnati made five, last July 2 (#19578); Philadelphia made three, last Dec. 26 (#18858). At cutoff, Los Angeles was 51–53 (third NBC West), Cincinnati 53–51 (tied in ABC Central), and Philadelphia 48–56 (second ABC East). Three different records, same absence of late trades; no shared explanation is established.

**No trades July 24–31:** Denver’s last trade was July 23 (#19793); Nashville’s was July 16 (#19729). Denver was 56–48 and two games behind the provisional ABC WC leader; Nashville was 53–51, tied with Cincinnati in ABC Central. Charlotte’s last deal was July 24 (#19800), so it had late-period activity but no July 31 trade. All other ABL clubs had at least one deal July 24–31.

The absence of a trade can prompt questions about roster health, budget, prospects, or negotiations. These sources cannot establish whether a deal was offered, declined, blocked, or never considered. “Stood pat,” “missed out,” and “trusted its roster” are not proven descriptions.

## 15. Race-by-race front-office board — August 1 snapshot

All standings reflect completed regular-season games through July 31. Division-board GB is against the division leader; WC-board GB is against the current provisional wild-card leader. Houston and St. Louis had played 103 games; the other 22 clubs had 104. GM names are current August 1 staff links.

### NBC East

| Club | W–L | GB | Current GM | Relevant movement / contest context |
|---|---:|---:|---|---|
| Charlotte | 62–42 | — | Jose Ruiz | Division leader, league-best record; three trades total, last July 24 (#19800); none July 31. |
| Miami | 55–49 | 7 | Daniel Baker | Eight trades; July activity included the cash-bearing Joe Good/Quadir May-for-Daniel Vega deal (#19868) and Omar Garcia for Addie de Ruijter (#19877); also WC challenger. |
| Atlanta | 46–58 | 16 | Hector Naranjo | Tied with Tampa Bay; six trades, including July 23’s Benito Rodriguez/Rodolfo Gonzalez-for-Mason Chaisson (#19793) and July 31’s Bill Torres/Rodolfo Gonzalez-for-Jaden Pierce (#19907). |
| Tampa Bay | 46–58 | 16 | Angel Gonzalez | Tied with Atlanta; five trades, including Brian Manchester to Dallas (#19798) and Jeff Mitchell to Houston (#19853). |

The wide division gap is the cutoff arithmetic, not a tiebreak or postseason decision. Miami’s WC chase is a second track.

### NBC Central

| Club | W–L | GB | Current GM | Relevant movement / contest context |
|---|---:|---:|---|---|
| Detroit | 57–47 | — | Jesus Martinez | Division leader; acquired Kennie Murdock from San Diego for Bob Morales (#19867). |
| Dallas | 56–48 | 1 | Jim Griffith | Provisional NBC WC leader; acquired Brian Manchester, Doug Farmer, and Eric Strickland in July (#19798/#19799/#19849). |
| Chicago | 55–49 | 2 | Mike Hatton | One game behind Dallas for WC; exchanged players with Boston in #19851/#19901. |
| Minneapolis | 46–58 | 11 | Ken Sears | Sent Chris Jenkins to Houston for Serrano and Demachi on July 31 (#19908); prospect positions disputed. |

The division contest and NBC WC are separate tracks: Dallas and Chicago were second and third in their division, but also the WC leader and a one-game challenger.

### NBC West

| Club | W–L | GB | Current GM | Relevant movement / contest context |
|---|---:|---:|---|---|
| San Francisco | 58–46 | — | Luis Martinez | Division leader; acquired Victor Hernandez #504 from San Diego for Omari Benjamin (#19806). |
| Phoenix | 53–51 | 5 | Bryan Somers | Second; received Kerry Darling/Willie Saunders from Houston (#19865) and Addie de Ruijter from Miami (#19877). |
| Los Angeles | 51–53 | 7 | Gary Jones | Third; one trade in full window (#18855), none after Dec. 22. |
| San Diego | 39–65 | 19 | Matt Johnson | Fourth; sent Hernandez to SF and Murdock to Detroit (#19806/#19867), among seven trades. |

San Francisco’s division lead and Phoenix’s WC position are distinct. San Diego’s veteran-arm departures are not proof of a full rebuild.

### NBC Wild Card — provisional order

| Rank | Club | W–L | GB | Front-office context |
|---:|---|---:|---:|---|
| 1 | Dallas | 56–48 | — | Jim Griffith; three July acquisitions. |
| 2T | Miami | 55–49 | 1 | Daniel Baker; eight trades, including #19868. |
| 2T | Chicago | 55–49 | 1 | Mike Hatton; consecutive exchanges with Boston. |
| 4 | Phoenix | 53–51 | 3 | Bryan Somers; received Darling/Saunders and de Ruijter. |
| 5 | Los Angeles | 51–53 | 5 | Gary Jones; one trade total. |
| 6T | Atlanta | 46–58 | 10 | Hector Naranjo; tied on record with Tampa Bay and Minneapolis. |
| 6T | Tampa Bay | 46–58 | 10 | Angel Gonzalez; Manchester moved to Dallas. |
| 6T | Minneapolis | 46–58 | 10 | Ken Sears; Jenkins moved to Houston for two minor-league players. |
| 9 | San Diego | 39–65 | 17 | Matt Johnson; veteran arms moved to division leaders. |

These are provisional standings, not official qualification/tiebreak decisions. Division leaders Charlotte, Detroit, and San Francisco are excluded from this WC-only ordering.

### ABC East

| Club | W–L | GB | Current GM | Relevant movement / contest context |
|---|---:|---:|---|---|
| Boston | 54–50 | — | Bobby Ramsey | Division leader; acquired Victor Hernandez #6732 and Miguel Martinez from Chicago (#19851/#19901); Martinez position disputed. |
| Philadelphia | 48–56 | 6 | Jon Ramos | Second; three trades total, last Dec. 26 (#18858); none July 16–31. |
| New York | 47–57 | 7 | Jeremy Burk | Third, ahead of Pittsburgh; acquired David Mitchell and Vinny Martinez from Nashville for Devon John (#19729). |
| Pittsburgh | 44–60 | 10 | Nick Heller | Fourth; received player assets and cash from Miami (#19868), Bill Hall from Las Vegas (#19878), then sent Bautista to Houston (#19904). |

New York is third by record and wins, not fourth. Philadelphia and Pittsburgh also sit behind Seattle/Denver in provisional ABC WC arithmetic.

### ABC Central

| Club | W–L | GB | Current GM | Relevant movement / contest context |
|---|---:|---:|---|---|
| Houston | 61–42 (103 G) | — | Larry Bell | ABC Central leader at .592; Charlotte had the league-best .596 at 62–42. Houston had the league-high 12 trades; late additions included Darling/Saunders, Bautista, and Jenkins. |
| Cincinnati | 53–51 | 8.5 | Phil Trammel | Tied on record with Nashville; five trades, last July 2 (#19578); none July 16–31. |
| Nashville | 53–51 | 8.5 | Justin Gomez | Tied with Cincinnati; six trades, last July 16 (#19729); four contract notices. |
| St. Louis | 48–55 (103 G) | 13 | David Ward | Fourth; sent Harry Benjamin to Seattle (#19866) and Chris Sullivan to Houston (#19889) late in July. |

Cincinnati and Nashville are tied on record; their 8.5 GB refers to Houston, not a tiebreak. Houston and St. Louis have one fewer game than the other teams.

### ABC West

| Club | W–L | GB | Current GM | Relevant movement / contest context |
|---|---:|---:|---|---|
| Las Vegas | 60–44 | — | Roberto Choto | Division leader; acquired Nate Souza (#19854) and Bill Hall (#19878) in July. |
| Seattle | 58–46 | 2 | David Suarez | Second and provisional ABC WC leader; acquired Harry Benjamin from St. Louis for Isaiah Gibson (#19866); no July 31 trade. |
| Denver | 56–48 | 4 | Kevin Bradley | Third; acquired Tony Oubre from New York in June (#19446); last deal July 23 (#19793), none July 24–31. |
| Portland | 41–63 | 19 | Ben Hayter | Fourth; sent Nate Souza to Las Vegas for Moran/Gortarez (#19854), among five trades. |

Las Vegas’s division lead is separate from Seattle’s WC lead. Denver was two games behind Seattle in the ABC WC standings and four games behind Las Vegas in the ABC West division race.

### ABC Wild Card — provisional order

| Rank | Club | W–L | GB | Front-office context |
|---:|---|---:|---:|---|
| 1 | Seattle | 58–46 | — | David Suarez; acquired Harry Benjamin from St. Louis. |
| 2 | Denver | 56–48 | 2 | Kevin Bradley; added Tony Oubre in June. |
| 3T | Cincinnati | 53–51 | 5 | Phil Trammel; last trade July 2. |
| 3T | Nashville | 53–51 | 5 | Justin Gomez; last trade July 16. |
| 5 | St. Louis | 48–55 (103 G) | 9.5 | David Ward; last trade July 30. |
| 6 | Philadelphia | 48–56 | 10 | Jon Ramos; last trade Dec. 26. |
| 7 | New York | 47–57 | 11 | Jeremy Burk; July 16 trade with Nashville. |
| 8 | Pittsburgh | 44–60 | 14 | Nick Heller; late acquisitions from Miami/Las Vegas. |
| 9 | Portland | 41–63 | 17 | Ben Hayter; Nate Souza moved to Las Vegas. |

## 16. Grand Tournament of Champions board

The configured postseason takes three division winners and one wild card from each conference (eight teams total); its final round is the Grand Series. The repository’s 1981-and-forward rule preserves division separation in the DCS when seeds #1 and #4 share a division: #4 plays #2 and #1 plays #3. Qualification is unchanged. All standings below are provisional, and no club has clinched on this snapshot.

| Conference | Division leaders | Provisional WC leader | Nearest challengers | Observable roster construction |
|---|---|---|---|---|
| NBC | Charlotte 62–42; Detroit 57–47; San Francisco 58–46 | Dallas 56–48 | Miami/Chicago 55–49 (1 GB); Phoenix 53–51 (3 GB) | Detroit and San Francisco added veteran arms; Dallas made three July acquisitions; Miami/Chicago were active; Charlotte made no final-week trade. |
| ABC | Boston 54–50; Houston 61–42; Las Vegas 60–44 | Seattle 58–46 | Denver 56–48 (2 GB); Cincinnati/Nashville 53–51 (5 GB); St. Louis 48–55 in 103 G (9.5 GB) | Boston/Houston/Las Vegas made identifiable additions; Seattle added Benjamin; Denver added Oubre earlier; other clubs’ patterns vary. |

The tournament path provides context, not a championship prediction or trade grade. A later article must update records if the league advances.

## 17. Front-office contrasts worth reporting

1. **Houston and Charlotte:** Larry Bell’s Houston made 12 trades while leading ABC Central; Jose Ruiz’s league-best 62–42 Charlotte made three trades and none on July 31. This contrasts activity levels, not motives.
2. **Detroit and San Francisco:** Jesus Martinez acquired a 10–6 starter (Murdock, #19867); Luis Martinez acquired a reliever with 17 pre-trade saves (Victor Hernandez, #19806). Both led divisions, but the pitching roles differed.
3. **Boston and Chicago:** Bobby Ramsey and Mike Hatton made consecutive exchanges in both directions (#19851/#19901). This complicates a one-way buyer/seller story. Martinez’s position remains disputed.
4. **San Diego and Minneapolis:** Matt Johnson’s San Diego sent Murdock and Hernandez to division leaders for a minor-league pitcher and catcher (#19867/#19806); Ken Sears’s Minneapolis sent Jenkins to Houston for two minor-league prospects (#19908). The deals involved different pitching roles and return classes; motives remain unknown.
5. **Dallas and Miami:** Both made eight trades while near the NBC WC race. Dallas acquired Manchester, Farmer, and Strickland; Miami’s movement was mixed and its Joe Good/Quadir May deal included $144,000 to Pittsburgh (#19868). This is a difference in documented flows, not risk appetite.
6. **Seattle and Denver:** David Suarez’s 58–46 Seattle acquired RHP Harry Benjamin July 28 (#19866); Kevin Bradley’s 56–48 Denver acquired Tony Oubre in June (#19446) and had no July 24–31 deal. Timing and move type differ; no fixed philosophy follows.
7. **Houston and Phoenix:** The July 28 trade sent LF Alex Monge from Houston to Phoenix for Kerry Darling and Willie Saunders (#19865). The same exchange has different roster classes flowing each way; it does not show each club’s private valuation.
8. **New York and Nashville:** Jeremy Burk’s New York received minor-leaguers David Mitchell and Vinny Martinez from Justin Gomez’s Nashville for Devon John (#19729). This is a left-handed pitcher for two minor-league players in one deal, not proof of either club’s broader direction.

## 18. Verified fact box

- **Structured trades:** 66 from Nov. 11, 1980 through July 31, 1981; all matched to a contemporaneous `messages.csv` ID/date. All 24 ABL clubs traded at least once.
- **Calendar:** 36 deals before the April 6 opener; 3 in June; 27 in July; 26 from July 16–31; 24 from July 24–31; six July 27; four July 31. The configured deadline is July 31, not August 1.
- **Club transaction totals:** Houston 12; Dallas/Miami 8 each; Chicago/San Diego/Seattle 7 each; Atlanta/Cincinnati/New York/Nashville/Boston 6 each; Los Angeles 1; Charlotte/Philadelphia 3 each.
- **Player rows:** 182 transfer rows, 179 unique players; repeated IDs are Jerry Wilson #681, Vinny Martinez #8329, Rodolfo Gonzalez #9886. Summary-position classes: 92 pitcher descriptors (72 RHP, 20 LHP), 45 infield, 36 outfield, 9 catcher; 23 summary/message position disagreements are itemized, so disputed positions are not treated as settled.
- **Terms:** one nonzero structured cash transfer ($144,000 Miami to Pittsburgh, #19868); no draft-pick/IAFA-cap assets in the arrays. Twenty-four trade summaries list retention percentage (14 below 100%, ten at 100%); arrays do not encode the retaining party.
- **Contract notices:** 46 `message_type=2` rows (42 pre-opener, four in-season); this is not 46 new-player acquisitions or a complete event ledger.
- **Cutoff:** August 1 pregame snapshot; 1,247 completed regular-season games through July 31; team records reconcile 24/24 to completed game rows. August 1 has 12 scheduled, zero completed in this snapshot. Twenty-two clubs had 104 games; Houston and St. Louis had 103.
- **Leaders:** Charlotte 62–42 (NBC East), Detroit 57–47 (NBC Central), San Francisco 58–46 (NBC West), Boston 54–50 (ABC East), Houston 61–42 in 103 games (ABC Central), Las Vegas 60–44 (ABC West). Provisional WC leaders: Dallas 56–48 (NBC), Seattle 58–46 (ABC).
- **Record ties:** Atlanta/Tampa Bay 46–58 in NBC East; Cincinnati/Nashville 53–51 in ABC Central. No tiebreak ruling is asserted.

## 19. Verified baseball details and source traceability

### Game, player, roster, and contract texture

- **All-Star dates:** July 12 was the last regular-season day before the break and also had Future Stars game 1945 (type 8, teams 147/148, score 1–2) and roster/voting announcements (messages #19669–#19673). The ABL conference All-Star Game was July 14 (game 8767, type 4, teams 145/146, score 0–7; game-story message #19715; league `allstar_date=1981-7-14`). Regular season resumed July 16 (games 1069–1080). Do not collapse these distinct events.
- **Kennie Murdock #4963:** 30-year-old RHP, 10–6 with 2.43 ERA and 137 IP before July 28; moved San Diego to Detroit for Bob Morales #9196 (#19867). Detroit made one start with him by July 31. Keep same-day performance separate because the trade has no timestamp.
- **Victor Hernandez #504:** 33-year-old RHP reliever, 17 saves and 2.78 ERA before July 25; moved San Diego to San Francisco for minor catcher Omari Benjamin #5204 (#19806). Those are pre-trade figures, not receiving-club production.
- **Chris Jenkins #6159:** 27-year-old LHP, 9–6, 3.48 ERA, 98⅓ IP through July 31; Minneapolis sent him to Houston that day (#19908). No post-trade game is in the cutoff.
- **Tony Oubre #949:** New York sent him to Denver June 18 (#19446). His later-date starts totaled 53⅓ IP and 3.21 ERA through July 31; keep June 18 separate. Willie Vega #7820’s listed position conflicts (SS summary, 2B message).
- **Boston/Chicago:** Victor Hernandez #6732 was described as a 26-year-old 1B in #19851. Miguel Martinez #4347’s position conflicts in #19901. Justin Bowen #854 had 18 saves before July 31 and moved from Boston to Chicago in #19901. Whole-season player totals are not receiving-club production.
- **Draft/roster caveat:** Denver’s Tony Cossio #12045 was picked 13th overall June 5 (draft message #19362). His current roster flag is not a dated historical assignment; his two listed game-pitching rows through cutoff belong to minor-league team 70, not Denver’s ABL team. Rule 5 notice #18845 does not provide a complete selection-result ledger.
- **Ownership/fan evidence:** Current owner names are verifiable; owner expectation values remain raw/undecoded. No owner instruction or attributable fan reaction is present. Attendance/fan-interest fields are a single snapshot, not a before/after trade series.

### Exact source paths and reproducibility

All paths below are under the exact repository root `C:\sbv_repo\abl_csv_repo`. In rows with sibling basenames, subsequent filenames share the directory of the first full path in that row; the 1980 row gives every path explicitly. The full transaction inventory and the 23 position-source exceptions are reproduced above; no other repository file is required to identify the key transactions.

| Claim | Exact source path(s) | IDs / computation |
|---|---|---|
| Trade sides, player IDs, dates, cash/picks/IAFA, retention text | `C:\sbv_repo\abl_csv_repo\csv\ootp_csv\trade_history.csv`; `C:\sbv_repo\abl_csv_repo\csv\ootp_csv\messages.csv` | All 66 trade rows; matching message ID/date. Key IDs are repeated in this packet; all 66 appear in the inventory. |
| Position descriptions/exceptions | Same `trade_history.csv` summaries and matched `messages.csv` bodies | 23 player/transaction disagreements are itemized; neither conflicting position is silently selected. |
| Deadline, current date, schedule size, All-Star date | `C:\sbv_repo\abl_csv_repo\csv\ootp_csv\leagues.csv` | League ID 200: current date 1981-08-01, deadline 1981-07-31, 162-game schedule, All-Star date 1981-07-14. |
| Schedule, scores, game identity | `C:\sbv_repo\abl_csv_repo\csv\ootp_csv\games.csv`; `games_score.csv`; `game_logs.csv` | Games 1945, 8767, 1069–1080; 1,247 completed regular-season games through July 31; 12 scheduled and zero played Aug. 1. |
| Team standings / opponent records | `C:\sbv_repo\abl_csv_repo\csv\ootp_csv\team_record.csv`; `games.csv`; `games_score.csv`; team/division keys in `teams.csv`, `sub_leagues.csv`, `divisions.csv` | 24/24 team records reproduced from completed results; Houston/St. Louis each 103 games. |
| Postseason slots and DCS protection | `C:\sbv_repo\abl_csv_repo\csv\ootp_csv\league_playoffs.csv`; `C:\sbv_repo\abl_csv_repo\docs\ABL_POSTSEASON_RULES.md` | Three division winners plus one WC per conference; eight teams; 1981-forward same-division DCS protection. |
| GM/owner identities and raw values | `C:\sbv_repo\abl_csv_repo\csv\ootp_csv\team_roster_staff.csv`; `coaches.csv`; `team_financials.csv` | Current staff link only; no dated appointment history or local numeric-code key. |
| Player identities, roster/contract state, performance | `C:\sbv_repo\abl_csv_repo\csv\ootp_csv\players.csv`; `players_roster_status.csv`; `players_contract.csv`; `players_pitching.csv`; `players_batting.csv`; `players_game_pitching_stats.csv`; `players_game_batting.csv` | Player IDs appear alongside names; current roster flags do not establish historical roster assignment. |
| Contract notices and draft | `C:\sbv_repo\abl_csv_repo\csv\ootp_csv\messages.csv`; `players.csv` | Type-2 message IDs in the contract table; draft announcement #19362; Rule 5 notice #18845. |
| 1980 season-end reconstruction | `C:\sbv_repo\abl_csv_repo\csv\out\almanac\1980\games_1980_league200.csv`; `C:\sbv_repo\abl_csv_repo\csv\ootp_csv\team_history.csv`; `C:\sbv_repo\abl_csv_repo\csv\ootp_csv\team_history_record.csv` | Generated historical game file cross-checked to raw history tables; boundary/end date is **STRONG INFERENCE**, not a raw 1980 game-export timestamp. |

Companion report/ledger files were secondary cross-checks only, never the sole source for a material fact. No downstream report was regenerated or certified current.

## 20. Article angle board

Each item is an assignment possibility, not a chosen article angle.

### 1. The busiest team was not a simple buyer

- **Central idea:** Houston combined the league’s highest trade count with a net outflow of player-transfer rows while leading the ABC Central.
- **Support / clubs / people:** Houston 12 trades, 14 rows in, 21 out; Larry Bell is current GM; July 31 additions included Ricky Bautista and Chris Jenkins.
- **Why now:** Houston entered the August 1 snapshot 61–42 and at the top of its division.
- **Distinctive:** It complicates “contender buys, distant club sells.”
- **What weakens it:** Transfer-row counts do not measure player value or payroll, and one returned prospect package may be more important than several rows.
- **Reporting to strengthen:** Ask how Houston described its roster needs; review player roles, salary commitments, and July usage after the cutoff.
- **Possible opening concept:** Set Houston’s 12-deal total beside its 61–42 lead and late-July pitching sequence: Darling/Saunders on July 28, then Bautista and Jenkins in two July 31 trades; ask what its roster turnover says—and does not say.
- **Cannot safely claim:** Houston was a seller, buyer, “all-in,” or pursuing a particular plan based on raw row counts.
- **Possible home:** Major standalone Ernie Bewell feature.

### 2. San Diego’s two veteran arms went to two leaders

- **Central idea:** A last-place club moved a starting pitcher and a high-save reliever to division leaders within three days.
- **Support / clubs / people:** San Diego, Detroit, San Francisco; Murdock and Victor Hernandez #504; Matt Johnson, Jesus Martinez, and Luis Martinez are current GMs.
- **Why now:** The exchange preceded the deadline and directly involved clubs in the tournament field.
- **Distinctive:** The return packages differed: a minor-league pitcher for Murdock, catcher Omari Benjamin for Hernandez.
- **What weakens it:** Two trades do not establish a full rebuild; there is no direct explanation from the front office.
- **Reporting to strengthen:** Trace what happened to Benjamin/Morales and ask what San Diego intended to do with the acquired assets.
- **Possible opening concept:** Follow the two veteran arms from last-place San Diego to the division leaders that acquired them.
- **Cannot safely claim:** The deals prove a full rebuild, that returns were prospects of a particular value, or that Detroit/San Francisco’s postseason odds changed.
- **Possible home:** Baseball Observer or standalone feature.

### 3. The deadline dealt in roster roles as much as names

- **Central idea:** Several late moves involved identifiable rotation, relief, and infield roles rather than a single leaguewide star auction.
- **Support / clubs / people:** Detroit/Murdock, San Francisco/Victor Hernandez #504, Denver/Oubre, Boston/Chicago’s Victor Hernandez #6732 and Miguel Martinez, Chicago/Bowen.
- **Why now:** The relevant teams were division leaders or in the wild-card race.
- **Distinctive:** The same deadline could be read through baseball construction instead of winner/loser grades.
- **What weakens it:** Role need is only partly visible; no timestamp or internal depth chart is exported.
- **Reporting to strengthen:** Compare August usage and roster position after the league advances.
- **Possible opening concept:** Contrast a starter, a reliever, and two position-player exchanges as four separate roster questions.
- **Cannot safely claim:** The teams had a diagnosed need, an internal depth chart, or a shared auction strategy.
- **Possible home:** Baseball Matters.

### 4. July changed the market’s tempo

- **Central idea:** A quiet early-season trade period gave way to 27 July trades, including six on July 27 and four on July 31.
- **Support / clubs / people:** All 24 clubs traded; Houston, Dallas, Miami, Chicago, San Diego, and Seattle led activity.
- **Why now:** The rise is visible before and after the July 14 All-Star date.
- **Distinctive:** It offers a leaguewide transaction chronology rather than one transaction verdict.
- **What weakens it:** Counts do not show negotiations, failed offers, or importance.
- **Reporting to strengthen:** Compare later ABL seasons or interview GMs about the calendar and trade process.
- **Possible opening concept:** Move from 36 pre-opener deals to six on July 27 and four on deadline day, using the dated ledger as a chronology.
- **Cannot safely claim:** A trade “frenzy,” failed offers, or negotiation dynamics based only on completed deal counts.
- **Possible home:** Forum, with a sidebar for dates.

### 5. A contender may have stood quieter than its record

- **Central idea:** Charlotte led the league at 62–42 yet completed only three trades; Los Angeles completed one trade despite a 51–53 record.
- **Support / clubs / people:** Charlotte, Jose Ruiz, Adam Butler; Los Angeles, Gary Jones, Tony Cruz.
- **Why now:** Both clubs’ low activity contrasts with the July surge.
- **Distinctive:** It asks what a team communicates by not making a visible late move without assuming a failed or deliberate strategy.
- **What weakens it:** The export cannot reveal offers that were made or considered.
- **Reporting to strengthen:** Ask about July priorities, health, and roster evaluations; check any later roster records.
- **Possible opening concept:** Put the 62–42 leader’s three-trade total beside the leaguewide July surge and ask what the record does not explain.
- **Cannot safely claim:** Charlotte or Los Angeles deliberately stood pat, lacked resources, or rejected a specific deal.
- **Possible home:** Baseball Observer.

### 6. Two tied races, different transactions

- **Central idea:** Atlanta/Tampa Bay and Cincinnati/Nashville entered August tied on record, while their transaction patterns differed.
- **Support / clubs / people:** Atlanta 7 contract notices; Tampa Bay 1; Nashville 4; Cincinnati 1. Atlanta/TB were each 46–58; CIN/NAS each 53–51.
- **Why now:** These clubs faced the same standings line but did not make the same volume/type of moves.
- **Distinctive:** A controlled comparison without inventing tiebreak rules.
- **What weakens it:** Contract notices are not all new players, and no team messaging is supplied.
- **Reporting to strengthen:** Confirm lineup/rotation usage and interview the four front offices.
- **Possible opening concept:** Begin with two identical W–L lines, then show the different trade and contract-notice counts behind them.
- **Cannot safely claim:** Either tied pair had a tiebreak result or postseason consequence at the cutoff.
- **Possible home:** Baseball Matters.

### 7. The prospect return is not a conclusion

- **Central idea:** San Diego and Minneapolis received younger/minor-league pieces for established pitchers; what those assets mean remains open.
- **Support / clubs / people:** Murdock-to-Detroit, Jenkins-to-Houston, the two specific returns; current standings at each move.
- **Why now:** These are the clearest exchanges across competitive positions near the deadline.
- **Distinctive:** Keeps the feature focused on organizational timelines rather than assigning trade grades.
- **What weakens it:** There is no complete minor-league performance/valuation series in the verified deadline packet.
- **Reporting to strengthen:** Follow the prospects’ usage/development and ask how each selling club valued the time horizon.
- **Possible opening concept:** Put Jenkins’s 98⅓ innings next to the two named minor-league returns, then widen to Murdock and Hernandez’s destinations.
- **Cannot safely claim:** Any player’s future value, either return’s quality, or a franchise-wide rebuild.
- **Possible home:** Major standalone feature.

### 8. What the GM and owner files cannot yet tell us

- **Central idea:** The desk can name current GMs and owners and report raw OOTP profile values, but cannot reliably translate them into personality stories.
- **Support / clubs / people:** All 24 clubs; coaches.csv raw fields; absent owner-specific patience/fiscal/involvement/priority fields.
- **Why now:** The natural journalistic temptation is to explain transaction patterns through personality.
- **Distinctive:** It makes data limitations part of a reported question rather than presenting a caricature.
- **What weakens it:** It is a methodology angle, not necessarily the main baseball story.
- **Reporting to strengthen:** Obtain decoded OOTP profile screens, dated personnel history, and direct interviews.
- **Possible opening concept:** Pair a named executive and a documented trade with the raw profile fields whose labels the repository cannot decode.
- **Cannot safely claim:** Personality, private motivation, owner involvement, or historical tenure from an unlabeled numeric field or current staff row.
- **Possible home:** Forum or reporting sidebar.

## 21. Possible lead concepts

These are starting points for reporting, not polished opening paragraphs:

1. **Houston’s paradoxical volume:** 12 trades, an ABC Central lead, and two July 31 trades that brought in Ricky Bautista and Chris Jenkins, but 21 player rows out versus 14 in.
2. **Two San Diego arms, two division leaders:** Victor Hernandez went to San Francisco; Kennie Murdock went to Detroit; each deal had a different return.
3. **The compressed July market:** 27 July trades, six on July 27, four on July 31, following 36 pre-opener trades.
4. **A 62–42 leader with three trades:** Charlotte’s limited transaction count as a question, not an assumed strategy.
5. **A pitcher for prospects on deadline day:** Minneapolis sent Chris Jenkins to Houston for Jonathan Serrano and Alush Demachi; two source files disagree on the prospects’ positions.
6. **Two ties, two races:** Atlanta/Tampa Bay at 46–58 and Cincinnati/Nashville at 53–51, with different transaction records and no asserted tiebreak.
7. **A visible exchange rather than a winner/loser grade:** Boston and Chicago traded established players in consecutive deals, with pieces moving both ways.
8. **Named GMs, incomplete personalities:** The file can document what Larry Bell, Matt Johnson, Jesus Martinez, and others did, but raw ratings and owner codes do not establish why.

## 22. Possible article structures

- **League-wide market chronology:** Open with offseason volume, move into the quiet/early-season gap, then July’s compression and the final four deals. **Strength:** broad 24-club frame. **Risk:** can become a transaction list without race stakes or people.
- **People-first executive portraits:** Organize around Bell, Johnson, Martinez, Hatton, Ramsey, Griffith, Sears, and Suarez; tie each to a concrete deal and current race. **Strength:** keeps front-office actors central. **Risk:** no verified quotes, private motives, or historical GM tenure are available, so reporting/interviews are essential.
- **Race-by-race:** Use the eight division/WC boards, moving across both conferences. **Strength:** direct connection between decisions and postseason paths. **Risk:** repetitive if every club is given equal space; standings are provisional.
- **Deadline outward into the offseason:** Start with July 31’s four trades, trace back to July’s surge, then to the 36 offseason deals and 46 type-2 notices. **Strength:** deadline hook plus roster-building context. **Risk:** the reporting window begins in November and offseason chronology needs careful organization.
- **Contrasting organizational choices:** Pair Houston/Charlotte, Detroit/San Francisco, Boston/Chicago, Dallas/Miami, San Diego/Minneapolis, or Seattle/Denver. **Strength:** turns raw movement into controlled comparisons. **Risk:** comparisons show observable differences, not different motives.

## 23. Supported and unsupported language; quote discipline

### SUPPORTED JOURNALISTIC LANGUAGE

- “acquired,” “sent,” “traded,” “received,” “added,” and “moved” when the structured sides support the direction.
- “division leader,” “provisional wild-card leader,” “tied on record,” and exact GB only with the August 1/July 31 cutoff attached.
- “high-volume,” “mixed-direction,” “late-period quiet,” or “established player for minor-league players” when tied to a count or named deal.
- “consistent with a targeted addition,” “raises the question,” or “may suggest” when clearly labeled as interpretation rather than fact.
- “current GM/owner” for the August 1 roster snapshot; “not established” for motives, offers, or roles absent evidence.
- “minor-league player” when the trade/message text labels the player that way; “not labeled minor-league in the transaction description” rather than asserting a historical active-roster status.

### UNSUPPORTED OR OVERSTATED LANGUAGE

- “won/lost the trade,” “steal,” “bust,” “all-in,” “rebuilding,” “gave up,” “desperate,” “trusted its roster,” or “missed out” as fact or grade.
- Claims that owners demanded a result, GMs were aggressive/conservative, clubs had a diagnosed need, or a player was acquired for a specific role without a direct source.
- Fan excitement/anger, clubhouse reaction, negotiations, offers, player quotes, or insider knowledge derived from generated OOTP prose.
- Tiebreak outcomes, clinches, or final bracket labels not shown by the cutoff data.
- Receiving-club credit for whole-season player totals that include games before the trade.
- Definitive position labels for any of the 23 disputed player/transaction pairs.

**VERIFIED QUOTES AVAILABLE: NONE**

Generated transaction/news copy is not treated as an independent interview or a verified attributable quotation. Obtain and verify direct quotes separately.

## 24. Unresolved reporting questions

### ANSWERABLE WITH ADDITIONAL ANALYSIS OF EXISTING REPOSITORY DATA

- For players moved before the cutoff, what were their game-date and club splits before and after the trade, excluding the trade-date game where its time is unknown? Use the player game logs and deal dates; do not attribute prior games to the new club.
- What were the exact current contract/salary fields and roster flags for the featured players at the August 1 snapshot? These are snapshot answers, not transaction-date contract histories.
- How did each of the 24 clubs’ complete transaction rows, player-row totals, and type-2 notices distribute by month, role description, and club side? The raw tables support the computation, though notices are not exhaustive of every transaction class.
- Which players appeared in ABL games for the receiving club by July 31 after earlier-season trades, and how many such appearances/innings/PA were recorded? The date and transaction-time limitations must be preserved.
- Can current team financial, attendance, or fan-interest values be tabulated as context? Yes, as a single snapshot only; no pre/post-trade causal conclusion follows.

### NOT ANSWERABLE FROM AVAILABLE SOURCES

- What were the GMs’ private motives, what offers failed, who declined a proposal, or whether an owner ordered a move?
- What role did Murdock, Victor Hernandez #504, or Chris Jenkins receive after the July 31 cutoff? That requires later games/roster state, not this August 1 pregame export.
- What did fans, teammates, or players think? There are no independently verified attributable quotes or trade-linked reaction series.
- What are the exact personality labels for numeric GM/owner fields, and how long had each current GM/owner held the role? No local codebook or dated personnel history is supplied.
- What was the complete Rule 5 result list, waiver/release history, or historical roster-assignment path? The available files do not form exhaustive event registers.
- What did the clubs do in August/September, and did those results change the race? That is beyond this cutoff and requires a fresh data snapshot.
- Why was Tony Cossio #12045 flagged on Denver’s current roster, or what was the club’s intended use? Draft/roster fields and absence of an ABL game do not state the reason.

## 25. Source and freshness notes

- The 1981 current-state evidence is the raw OOTP export dated at league current_date 1981-8-1, with completed games through July 31.
- Transaction rows have dates, not times. Pre-trade standings are “through the preceding completed game date”; performance from the transaction date is a separate bucket.
- The 1980 season-end reconstruction relies on the historical almanac game file plus raw team_history and team_history_record. It is not a direct raw 1980 game export.
- The raw OOTP file does not expose historical GM appointments or a complete non-trade transaction log.
- Current team finance/attendance/market fields are a snapshot and lack a verified before/after transaction series.
- This packet’s 66-row appendix and position-exception table are the direct trade audit; the companion trade report and ledger were used only as secondary cross-checks, not as the sole support for any material fact. No downstream reports were regenerated or certified as current.
- Draft context: the league file sets June 5 as the first-year draft date. Message #19362 announces the draft; players.csv identifies Denver’s Tony Cossio #12045 as 1981 round 1, overall pick 13, current ABL roster-active/secondary. His two listed game-pitching rows are with minor-league team 70, not Denver’s ABL club. The Rule 5 notice #18845 describes the event/rules but the raw export contains no complete pick-result ledger.

## 26. Verification certificate

**Final post-correction verification checklist**

1. **All 66 trade directions:** compared the packet’s 66 inventory rows against `trade_history.csv` side-team/player arrays and matching `messages.csv` date/message rows. All 66 table rows are present; both club sides, all 182 player-transfer rows/IDs, and all 66 message matches reconcile. The 179 unique player IDs and three repeated IDs match the raw arrays.
2. **Transaction terms:** independently checked structured cash/pick/IAFA arrays and retention descriptors. One cash transfer ($144,000 Miami to Pittsburgh, #19868), no structured pick/IAFA transfers; retaining side is not encoded. All listed current key trades were checked again against both source arrays and matched messages.
3. **Narrative references / 24 club briefs:** checked the named deals and competitive claims in the timeline, all 24 briefs, and the added cast, archetypes, race boards, contrasts, angles, and handoff. Corrected the identified reversed/misattributed deals (#19798, #19901, #19729, #19865), New York’s division placement, the Minneapolis prospect-position wording, and the July 12/14 All-Star distinction. Directions in the inventory are source-side arrows, not inferred from prose.
4. **Positions and roles:** 23 trade-summary/message position conflicts are individually itemized. Narrative uses disputed names/levels where needed and does not present the competing positions as resolved. “Not labeled minor-league” is not asserted to mean active-roster status.
5. **Deadline and cutoff freshness:** confirmed league ID 200 current_date 1981-08-01 and trade_deadline_date 1981-07-31; 66 trade rows through the deadline, four on July 31, none August 1. The raw current-state snapshot is August 1 pregame; no downstream report is claimed current.
6. **Schedule and records:** verified 1,247 completed regular-season games through July 31; Aug. 1 12 scheduled/0 completed; reproduced all 24 `team_record.csv` lines from completed game results. Houston and St. Louis each had 103 games; other 22 had 104. Division/WC ties are preserved without invented tiebreaks.
7. **All-Star reconciliation:** verified the July 12 Future Stars game/announcements separately from July 14 ABL All-Star Game and July 16 regular-season restart; IDs, types, teams, and scores are listed in Section 19.
8. **Tournament and personnel:** checked the configured three division winners plus one WC per conference and the 1981 DCS same-division protection. GM/owner identities are current links only; numeric profile values remain uninterpreted and no owner motive or historical tenure is claimed.
9. **Research scope and editorial discipline:** no authoritative source files were changed, no downstream report was regenerated, no trade was graded, no fabricated quotation/reaction or unsupported intent is used. Exact transaction/game IDs and direct source paths are provided; companion reports are not sole support.
10. **Handoff completeness:** added the deadline cast, evidence-based organizational archetypes, selected fully described deals, quiet moves, late-quiet club audit, all eight race boards, Grand Tournament board, eight contrasts, fact box, baseball texture, strengthened eight-angle board, eight lead concepts, five article structures, supported/unsupported vocabularies, quote status, and separate answerable/unanswerable questions.

**Corrections made:** corrected Dallas/Tampa Bay #19798; Boston/Chicago #19901; Nashville/New York #19729; Houston/Phoenix #19865; New York’s third-place rank; and the July 12 Future Stars / July 14 ABL All-Star distinction. Also clarified disputed Willie Vega/Minneapolis prospect positions, Tyler Hayes’s level, Philadelphia’s Dec. 14 package (#18825), and the retention split (14 below 100%, ten at 100%; not 15/nine). Removed the unqualified statement that Minneapolis received two left-fielders. Final editorial pass standardized Detroit, San Francisco, Dallas, Chicago, Boston, and Seattle to their ABL names: Dukes, Warriors, Rustlers, Fire, Patriots, and Comets, respectively; Houston Mavericks was retained. It corrected Charlotte’s league-best .596 versus Houston’s .592; reconciled Denver’s two-game wild-card gap to Seattle and four-game division gap to Las Vegas; corrected Houston’s two July 31 trades/acquisitions; and clarified the July 31 deadline versus the August 1 pregame snapshot. No finished article, authoritative source data, or downstream generated report was created/modified.

**Limitations that remain visible:** 23 position disagreements across source descriptions; no exhaustive waiver/release/Rule 5 or historic roster event log; no dated GM/owner tenure or codebook for numeric profile fields; no transaction timestamps; 1980 season boundary relies on a generated historical game file plus raw history records; no independently verified quotations or linked fan reaction.

**Full verification cycles required:** Three. Cycle 1 identified and corrected the factual reversals, position-source conflicts, New York placement, and All-Star-date ambiguity against primary rows/messages. Cycle 2 was a post-correction source-to-inventory, message, standings, schedule, transaction-reference, and editorial handoff audit after this package’s expansion. Cycle 3 rechecked team identities against `teams.csv`, the named standings against `team_record.csv`, and Houston’s July 31 acquisitions against `trade_history.csv` and matching `messages.csv` IDs #19904 and #19908; it also searched the entire packet for the corrected labels, standings claims, deadline wording, and repeated article concepts. The limitations above remain disclosed rather than treated as pass failures.

**ERNIE BEWELL HANDOFF STATUS: ARTICLE-READY**

**FINAL VERDICT: VERIFIED FOR ERNIE BEWELL JOURNALIST USE**

The packet supports an accurate, substantial feature from the August 1 pregame snapshot without requiring the writer to reconstruct essential trades or standings; private motives, quotes, and post-cutoff results require separate reporting.

