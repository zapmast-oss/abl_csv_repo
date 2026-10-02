# Baseball Observer Packet — Newsroom 1981-07-20, Games Through 1981-07-19

> Data points where to look; the game proves the story.

## Source State

- Newsroom date: 1981-07-20
- Completed-game cutoff: 1981-07-19
- Preflight: READY_FOR_CURRENT_RUN
- July 20 results used: No
- Current proof: raw OOTP game/score/log sources
- Enrichment: promoted sortable stats
- Historical context: governed almanac only
- Stale Week-labeled derivatives: excluded

## Proven Current-State Facts

### [Standings] San Francisco Warriors leads Phoenix Firebirds by 2 game(s) in National Baseball Conference Western Division

**Stakes:** With fewer games remaining, each direct result has less schedule left to absorb it.

**Evidence:** Through July 19: San Francisco Warriors 50-43; Phoenix Firebirds 48-45.

**Sources:** csv/ootp_csv/games.csv|csv/ootp_csv/teams.csv|csv/ootp_csv/divisions.csv|csv/ootp_csv/sub_leagues.csv

### [Standings] Dallas Rustlers leads Detroit Dukes by 2 game(s) in National Baseball Conference Central Division

**Stakes:** With fewer games remaining, each direct result has less schedule left to absorb it.

**Evidence:** Through July 19: Dallas Rustlers 52-41; Detroit Dukes 50-43.

**Sources:** csv/ootp_csv/games.csv|csv/ootp_csv/teams.csv|csv/ootp_csv/divisions.csv|csv/ootp_csv/sub_leagues.csv

### [Standings] Las Vegas Gamblers leads Seattle Comets by 3 game(s) in American Baseball Conference Western Division

**Stakes:** With fewer games remaining, each direct result has less schedule left to absorb it.

**Evidence:** Through July 19: Las Vegas Gamblers 54-39; Seattle Comets 51-42.

**Sources:** csv/ootp_csv/games.csv|csv/ootp_csv/teams.csv|csv/ootp_csv/divisions.csv|csv/ootp_csv/sub_leagues.csv

### [Standings] Boston Patriots leads New York Aces by 4 game(s) in American Baseball Conference Eastern Division

**Stakes:** With fewer games remaining, each direct result has less schedule left to absorb it.

**Evidence:** Through July 19: Boston Patriots 48-45; New York Aces 44-49.

**Sources:** csv/ootp_csv/games.csv|csv/ootp_csv/teams.csv|csv/ootp_csv/divisions.csv|csv/ootp_csv/sub_leagues.csv

### [Standings] Houston Mavericks leads Cincinnati Cougars by 5 game(s) in American Baseball Conference Central Division

**Stakes:** With fewer games remaining, each direct result has less schedule left to absorb it.

**Evidence:** Through July 19: Houston Mavericks 53-39; Cincinnati Cougars 48-45.

**Sources:** csv/ootp_csv/games.csv|csv/ootp_csv/teams.csv|csv/ootp_csv/divisions.csv|csv/ootp_csv/sub_leagues.csv

### [Race] Atlanta Kings posted the week's sharpest rise

**Stakes:** The week changed immediate race pressure; the next games determine whether that movement persists.

**Evidence:** July 13-19 record: 3-1; season record 40-53.

**Sources:** csv/ootp_csv/games.csv|csv/ootp_csv/teams.csv

### [Race] Philadelphia Fury posted the week's sharpest fall

**Stakes:** The week changed immediate race pressure; the next games determine whether that movement persists.

**Evidence:** July 13-19 record: 1-3; season record 43-50.

**Sources:** csv/ootp_csv/games.csv|csv/ootp_csv/teams.csv

### [Players] Jose Coronado has built the strongest current starter value case

**Stakes:** Each remaining start carries team-race weight and the burden of sustaining an ace-level season.

**Evidence:** 4.3 WAR, 2.86 ERA, 141.7 innings, 7-7 record.

**Sources:** csv/ootp_csv/players_career_pitching_stats.csv|csv/ootp_csv/players.csv

### [Players] Manny Flores owns the strongest current position-player value line

**Stakes:** The remaining schedule tests whether that broad contribution retains league-leading weight.

**Evidence:** 4.3 WAR, 18 HR and 384 PA.

**Sources:** csv/ootp_csv/players_career_batting_stats.csv|csv/ootp_csv/players.csv

### [Team] Atlanta Kings's record and run balance remain far apart

**Stakes:** The gap is a condition to monitor, not proof that reversal is due.

**Evidence:** Record 40-53; runs 378-393; expected wins 44.7; gap -4.7.

**Sources:** csv/ootp_csv/games.csv|csv/ootp_csv/teams.csv

### [Team] Seattle Comets's record and run balance remain far apart

**Stakes:** The gap is a condition to monitor, not proof that reversal is due.

**Evidence:** Record 51-42; runs 405-405; expected wins 46.5; gap +4.5.

**Sources:** csv/ootp_csv/games.csv|csv/ootp_csv/teams.csv

### [Game] Las Vegas Gamblers at Denver Rocketeers places current standings pressure on the field

**Stakes:** This is an unplayed July 20 schedule fact. No July 20 result is assumed or required.

**Evidence:** Scheduled July 20; records 54-39 and 50-43; same division=True.

**Sources:** csv/ootp_csv/games.csv|csv/ootp_csv/teams.csv|csv/ootp_csv/divisions.csv

## Enrichment Signals

### [Fans] Detroit Dukes is drawing the league's largest listed home crowd per game

**Stakes:** Attendance gives the race public scale; it does not establish what fans feel or predict results.

**Evidence:** Sortable report attendance/game: 26 990; raw season record 50-43.

**Sources:** csv/abl_statistics/abl_statistics_team_statistics___info_-_sortable_stats_team_finan.csv|csv/ootp_csv/games.csv|csv/ootp_csv/teams.csv

### [Management Organization] Tampa Bay Storm's payroll commitment and record create an organization question

**Stakes:** Resources and results can be compared; motive, blame, and future action remain unproven.

**Evidence:** Listed payroll $10.8m; record 42-51; mode Win Now!.

**Sources:** csv/abl_statistics/abl_statistics_team_statistics___info_-_sortable_stats_team_finan.csv|csv/ootp_csv/games.csv|csv/ootp_csv/teams.csv

## Historical Echoes

### [Tournament] 1980 qualifier Charlotte Colonels remains central to the 1981 race

**Stakes:** Prior achievement gives the current race context, not a forecast.

**Evidence:** 1980 record 91-71; current record 57-36 through July 19.

**Sources:** csv/out/almanac/1980/league_champions_1980_league200.csv|csv/ootp_csv/games.csv|csv/ootp_csv/teams.csv

## Unavailable / Disabled Signals

- **manager_tendency** — Staff names available, but no cutoff-compatible tactical tendency source.
- **staff_id_complete_linkage** — Removed staff ID fields; unresolved coach-name matches must remain null.
- **caught_stealing_rate** — CS and SB% unavailable in accepted team batting schema without another validated source.
- **BatR** — Removed and unavailable.
- **wSB** — Removed and unavailable.
- **UBR** — Removed and unavailable.
- **BsR** — Removed and unavailable.

## Editorial Guardrail

These are evidence-backed places to look. They are not predetermined outcomes, motives, or emotions. July 20 schedule rows are preview context only.
