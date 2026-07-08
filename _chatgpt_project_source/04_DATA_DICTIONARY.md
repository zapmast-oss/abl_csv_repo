# Data Dictionary

**Last verified:** 2026-07-07. Authority follows repository governance: raw OOTP is the system of record for current game facts; sortable and StatsPlus sources are supplemental; generated outputs inherit their inputs and cutoff.

## Data/report families

| Data/report | Path | Format | Contains | Used for | Current? | Caveats |
|---|---|---|---|---|---|---|
| OOTP raw exports | `csv/ootp_csv/*.csv` | CSV | Games, logs, scores, teams, divisions, players, stats, history, rosters, finances | Official game/standings reconstruction and identity | Current when date-filtered | 74 files, ~313 MB; mixed historical/current rows; filter league/game type/date |
| Game drivers | `csv/ootp_csv/games.csv`, `games_score.csv`, `game_logs.csv` | CSV | Game identity/date/status/score plus inning and narrative records | Current-state preflight, standings, results | Yes through 1981-07-19 | Large; do not upload wholesale |
| Dimensions | `teams.csv`, `divisions.csv`, `sub_leagues.csv`, `players.csv` | CSV | Stable IDs, names, organizational structure | Joins and labels | Compatible/current identity | Identity is not performance proof |
| Career player totals | `players_career_batting_stats.csv`, `players_career_pitching_stats.csv` | CSV | Season/split batting and pitching totals | Player-value candidates | Current only with correct season/league/level/split | Enforce workload minimums |
| Sortable extracts | `csv/abl_statistics/*.csv` | CSV | Supplemental staff, batting, pitching, finance and other sortable reports | Enrichment and secondary evidence | 20 promoted files available | No reliable date field; cannot prove cutoff alone; schema drift exists |
| StatsPlus Feed 3 | `csv/statsplus/current/*.csv` | CSV | 25 tables: personnel, finance/fans, odds, BaseRuns, ELO, WAR, injuries, team/player batting/pitching/fielding/baserunning/age, best games | Model/stat context | Promoted for July 19 preview | Enhances, does not replace; tables 6/27 excluded; some semantic loss/drift |
| Source registry | `csv/out/control/abl_source_registry.{csv,json,md}` | Multi | Authority, role, validation, joins, permitted use | Source governance | Current control layer | Read with batch manifest |
| Batch manifest | `csv/out/control/abl_batch_manifest_current.{csv,json,md}` | Multi | Captured files, checksums, sizes, detected coverage | Coherent-run proof | Current at inspected cutoff | “Current” must be regenerated after capture |
| Preflight | `csv/out/control/current_state_preflight_1981_target_1981-07-19.*` | Multi | Cutoff, counts, team coverage, verdict, stale exclusions | Gate story work | Current | Strongest compact current-state proof |
| Story candidates | `csv/out/story/candidates/story_candidates_1981_07_20_asof_1981-07-19.*` | CSV/JSON/MD if present | Ranked candidate metadata, hierarchy, stakes, evidence references | Editorial selection | Current run | Candidate is a proposition, not publication truth |
| Story evidence | `csv/out/story/candidates/story_evidence_1981_07_20_asof_1981-07-19.*` | CSV/JSON/MD if present | One or more evidence rows per candidate | Traceability | Current run | Future schedule rows must be labeled scheduled |
| Source/run manifests | `csv/out/story/manifests/*1981_07_20_asof_1981-07-19*` | Multi | Used/rejected sources, counts, disabled signals | Audit and handoff | Current | Official run has 15 candidates and 15 official evidence rows |
| Editorial board/slate | `csv/out/story/editorial/*1981_07_20_asof_1981-07-19*` | JSON/MD | Selected, watch, hold, and editorial ordering | Editorial decisions | Current | Human judgment still applies |
| Production/scripts | `csv/out/story/production/*1981_07_20_asof_1981-07-19*`, `csv/out/story/scripts/*...*` | JSON/MD | Talking points, cautions, finished reads | Narration/post production | Current | No July 20 results |
| Feed 3 preview | `csv/out/story/enrichment/*1981_07_20_asof_1981-07-19*` | CSV/JSON/MD | Team/player enrichment, overlay, 53 appended records, 68 combined records | Editorial context | Current preview | No rank changes/new candidates; not official replacement |
| Star schema | `csv/out/star_schema/*.csv`, `data_work/abl.db` | CSV/SQLite | Derived facts/dimensions/reporting tables | Analytics | Mixed | Several 32-game snapshots explicitly stale |
| Almanac | `csv/out/almanac/1972/` … `1980/`, `csv/out/eb/` | CSV/JSON/MD | Historical seasons, champions, spotlights, schedules | Historical echoes | Historical | Never silently merge with 1981 current facts |
| Legacy analytical reports | `csv/out/text_out/`, `csv/out/csv_out/` | TXT/CSV | Standings, trends, matchups, manager and specialty analytics | Research | Mixed/unknown | Validate cutoff; named 62-game reports are stale |

## Key StatsPlus tables and fields

| Table | Key fields | Interpretation/caution |
|---|---|---|
| `07...Playoff_Odds_Div.csv`, `08...League.csv` | W, L, AvgW/L, Div %, PO %, schedule strength | Model estimates; duplicated/reordered overlap must not be double-counted |
| `09_statsplus_Base_Runs.csv` | RS/RA, expected RS/RA, W/L, Pythagorean and BaseRuns expected wins, gaps | Descriptive gap; not proof of luck or guaranteed regression |
| `10_statsplus_Elo.csv` | record, season/30-day/7-day changes, Elo | Comparative model signal; not a game result |
| `11_statsplus_Team_WAR.csv` | batter, pitcher, total WAR | Team value context; model definition should remain labeled |
| `12_statsplus_injury_sumary.csv` | injury count, DL days, dollars on DL | Availability scale; not causal proof for wins/losses |
| `13`–`19` team tables | hitting, pitching, fielding, baserunning metrics | Team profile enrichment; accepted drift for some tables |
| `20`–`23` player tables | rate/count stats, WAR/rWAR, fielding and baserunning | Player context; names are not stable IDs |
| `25`–`26` best-game tables | opponent, date, score, game score and line | Reference only in current feature flags; validate against OOTP before asserting result |

## Baseball and ABL terms

| Term | Meaning in this project |
|---|---|
| W / L / record | Wins and losses through the declared completed-game cutoff |
| GB / games behind | Difference in standings position; recompute from authoritative standings data where needed |
| Division / conference | ABL structure from OOTP `divisions.csv` and `sub_leagues.csv`; ABC and NBC appear in current output |
| Race | Current separation between leading and pursuing clubs; not a qualification claim |
| Wild card / “if season ended today” | **UNKNOWN without verified 1981 qualification rules**; ask for the rules/source |
| Trend / rise / fall | Movement over a declared window; current weekly window was July 13–19 |
| Streak | Consecutive result sequence; use only a cutoff-compatible source |
| Last 10 | Record over ten most recent completed games; verify cutoff before use |
| Pythagorean expectation | Expected record from run balance | Descriptive, not destiny or luck proof |
| BaseRuns | Run-estimation/expected-record model | Enrichment only |
| ELO | Model rating of team strength | Enrichment only |
| WAR / rWAR | Estimated wins above replacement/player value | Current value marker, not automatic award verdict |
| PA / IP / GS | Plate appearances / innings pitched / games started | Workload fields used for minimum eligibility and context |
| `newsroom_date` | Date the coverage is framed/published |
| `as_of_date` / cutoff | Latest completed game allowed in claims |
| team/player IDs | Stable OOTP identifiers where available | Prefer IDs over names; StatsPlus name joins may be ambiguous |
| status: active/current | Compatible with declared cutoff | Must be supported by validation, not filename alone |

