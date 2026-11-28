# ABL Data Catalog

Inventory date: 2026-07-03.

The authoritative file-level catalog is [`csv/out/docs/abl_data_catalog.csv`](../csv/out/docs/abl_data_catalog.csv). It contains 483 rows and these fields: `file_path`, `row_count`, `column_count`, `likely_grain`, `key_columns`, `coverage`, and `possible_story_uses`.

## Method and cautions

Each CSV was parsed with comment-line handling and encoding fallbacks. Counts exclude the header, blank rows, and leading `#` metadata lines. Grain, keys, coverage, and story use are inferred from filenames, headers, and up to 1,000 sampled records. They are discovery metadata, not enforced contracts. The catalog intentionally includes archive, legacy, duplicate, and misplaced CSVs so output-location defects remain auditable.

## Principal data zones

| Zone | Scope | Typical grain | Story value |
|---|---|---|---|
| `csv/ootp_csv/` | 73 canonical OOTP exports | game, player, team, roster assignment, transaction, historical record | source of truth for game, team, player, organization, fan, and history signals |
| `csv/abl_statistics/` | 20 sortable-stat exports | player-season and team-season/snapshot | rich rates, ratings, splits, fielding, finance, staff |
| `csv/out/star_schema/` | reusable dimensions/facts and Monday products | player, team, team snapshot, team-week, game | preferred analytical contract layer |
| `csv/out/csv_out/` | individual analytic reports | signal-specific team/player rows | direct evidence for story signals, but schemas vary |
| `csv/out/almanac/<1972-1980>/` | repeated historical dataset family | game, series, team-week/month, player-season, ranked entity | flashbacks, historical echoes, champions, rematches |
| `csv/abl_csv/` | small legacy derived tables | team/report row | older matchup, last-10, standings, week-miner inputs |
| `csv/story_*.csv` | dictionary, menus, evaluated candidates | definition or candidate/entity | prototype story engine; generated files are misplaced |

## High-value source families

- Games and schedule: `games.csv`, `game_logs.csv`, `games_score.csv`, almanac games/team schedules/series summaries.
- Standings and race: `team_record.csv`, team history records, star-schema standings/reporting/current/previous/weekly-change tables.
- Team identity: batting, pitching, fielding, catcher, Pythag, one-run, home/road, leverage, momentum, power ranking.
- Players: player master/profile, batting/pitching/fielding stats and ratings, rosters, injuries, salary history, WAR/leader/rookie/milestone products.
- Fans and management: team financials, park/personality/finance sortable tables, coaches/staff/managers and scorecards.
- Tournament/history: league history, champions, season summaries, Grand Series extracts, flashback candidates.

## Coverage observed

- Current-league reporting is labeled predominantly 1981, including Week 5, Week 7, Week 15, and all-star/Act 3 products.
- Almanac inputs and outputs cover nine seasons, 1972-1980, for league 200.
- The catalog detects season/date/league values where exposed; `not detectable` means the file/sample offered no reliable value, not that coverage is absent.

## Known schema concerns

- Story definition files use `name`, while evaluated candidates use `story_name`; candidates also introduce evidence columns ad hoc.
- Snapshot tables depend on filenames (`current`, `prev`) rather than run metadata and immutable snapshot dates.
- Team identity is represented by IDs, abbreviations, and display names across layers; a canonical team key contract is required.
- Many report CSVs are presentation-shaped rather than normalized evidence tables.
- Historical tables are consistently named by season and league, but current-season tables often hard-code `1981` in scripts and filenames.
- Duplicate archive/legacy copies can produce double counting unless consumers declare allowed source zones.

## Recommended catalog contract

Future inventory code should add `data_tier`, `producer_script`, `generated_at`, `content_hash`, `schema_hash`, `primary_key_test`, `freshness_status`, and `canonical_or_legacy`. It should fail when generated data lands outside `csv/out/`, while allowing maintained config/source files under `csv/`.

