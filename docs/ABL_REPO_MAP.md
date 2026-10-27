# ABL Repository Map

Inventory date: 2026-07-03. This is a descriptive map; no existing file was moved, renamed, or deleted.

## Operating rule

Generated artifacts belong in `csv/out/` or one of its descendants. `csv/ootp_csv/`, `csv/abl_statistics/`, and `csv/in/` are inputs. Files currently generated under `csv/`, root `out/`, or nested `csv/csv/` are documented below as migration candidates, not corrected here.

## Folder map

```text
abl_csv_repo/
|-- csv/
|   |-- ootp_csv/          # 73 raw OOTP CSV exports
|   |-- abl_statistics/    # 20 OOTP sortable-stat exports plus Core 12 spec
|   |-- abl_csv/           # small legacy/current derived report tables
|   |-- abl_scripts/       # 165 ETL, analysis, report, EB, and almanac scripts
|   |-- config/            # maintained configuration tables
|   |-- docs/              # older schemas/templates and operator notes
|   |-- in/almanac_core/   # raw HTML almanacs, seasons 1972-1980
|   |-- out/
|   |   |-- star_schema/   # reusable current-season facts, dimensions, Monday tables
|   |   |-- csv_out/       # analytic report tables
|   |   |-- text_out/      # text reports, prep, and pregame packs
|   |   |-- almanac/       # season datasets and EB/flashback products, 1972-1980
|   |   |-- eb/            # Grand Series extracts/summaries and league overview
|   |   `-- archive/       # frozen 1981 outputs and manifest
|   |-- story_*.csv        # legacy story dictionary/menu/candidates (wrong output tier for generated files)
|   `-- _tmp_run_all.py    # broad, best-effort runner
|-- scripts/               # weekly story-menu build/evaluate/print tools
|-- tools/                 # PowerShell flashback-menu helper
|-- docs/                  # repo-level operator documentation
|-- data_work/abl.db       # working SQLite database
|-- data_raw/              # reserved, currently empty
|-- out/                   # six legacy duplicate report outputs outside canonical csv/out
|-- logs/                  # orchestration log
`-- .vscode/               # editor tasks/settings
```

The full file-level CSV inventory is in `csv/out/docs/abl_data_catalog.csv`.

## Important scripts and data flow

| Script | Appears to read | Appears to write |
|---|---|---|
| `csv/abl_scripts/build_star_schema.py` | sortable files in `csv/abl_statistics/`, `ootp_csv/coaches.csv` | dimensions/facts in `csv/out/star_schema/`; also updates a staff source with comments, an unusual source mutation to remove |
| `z_abl_current_team_snapshot.py` | OOTP team/standings/stat exports and reporting backbone | `fact_team_reporting_1981_current.csv` |
| `z_abl_seed_prev_from_games_1981.py` | game history through an `--asof` date | `fact_team_reporting_1981_prev.csv` |
| `z_abl_weekly_change_1981.py` | current and previous reporting snapshots | `fact_team_reporting_1981_weekly_change.csv` |
| `z_abl_monday_packet_1981.py` | current snapshot and weekly change | Monday standings, power ranking, risers, fallers in `star_schema/` |
| `z_abl_manager_scorecard_1981.py` | standings/reporting data and manager dimension | `fact_manager_scorecard_1981_current.csv` |
| `z_abl_monday_show_notes_1981.py` | Monday packet tables | `monday_1981_show_notes.csv` |
| `z_abl_run_week_1981.py` | prior snapshot plus raw/current data via child scripts | current, change, Monday, manager, show-note tables |
| `z_abl_eb_pack_1981_monday.py` | Monday/star-schema products | `csv/out/text_out/eb_data_pack_1981_monday.txt` and JSON context packs |
| `z_abl_monday_eb_turnkey_1981.py` | invokes weekly runner and EB pack | same weekly and EB outputs; does not seed previous week |
| `z_abl_core12_weekly.py` | current team/player/game products | Core 12 weekly JSON/report products |
| `z_abl_team_identity_pack_1981.py`, `z_abl_player_of_week_1981.py`, `z_abl_war_leaders_1981.py` | star-schema and raw player/team stats | JSON EB evidence packs under `text_out/` |
| `z_abl_pregame_pack.py` | schedule, team, player, park, finance, fan data | `csv/out/text_out/pregame/*.md` |
| `scripts/build_story_menu_for_week.py` | `csv/story_dictionary.csv` | `csv/story_menu_<week>.csv` (currently violates output rule) |
| `scripts/eval_story_triggers_for_week.py` | weekly story menu plus OOTP/team data | `csv/story_candidates_<week>.csv` (currently violates output rule) |
| `scripts/print_story_menu.py` | story menu/candidates | console editorial view |
| `z_abl_story_menu_1981_week5.py` | 1981 report inputs | week-5 story material; overlaps root story tools |
| `z_abl_almanac_league200_extract_core.py` | raw season HTML in `csv/in/almanac_core/<season>/` | core season extracts under `csv/out/almanac/<season>/` |
| `z_abl_almanac_scores_pipeline.py`, `z_abl_almanac_schedule_extract.py` | almanac HTML/core extracts | games, team schedule, series summaries |
| `z_abl_almanac_standings_pipeline.py`, `z_abl_enrich_almanac_standings.py` | standings HTML and extracted games | standings and enriched standings |
| `z_abl_almanac_time_slices_run_all.py`, `_run_almanac_time_slices_any.py` | season game/standings datasets | monthly, weekly, half, momentum datasets |
| `z_abl_almanac_player_stats_extract.py`, `z_abl_almanac_player_context_extract.py`, `z_abl_almanac_player_leaderboards.py` | player almanac HTML | player batting/pitching/context/leader datasets |
| `z_abl_almanac_flashback_story_pack.py` | season almanac datasets | `flashback_story_candidates_<season>_league200.csv` |
| `z_abl_almanac_flashback_story_menu.py` | flashback candidates | `flashback_story_menu_<season>_league200.md` |
| `_run_eb_regular_season_any.py` and `z_abl_eb_*_any.py` | season almanac outputs | EB regular-season, schedule, player, all-star, and flashback Markdown |
| `z_abl_grand_series_extractor.py`, `z_abl_grand_series_summarize.py` | game/box HTML | `csv/out/eb/gs_<season>/` and Grand Series summaries |
| `csv/_tmp_run_all.py` | environment-selected OOTP base and almost every `z_abl_*.py` | many output families plus a log; retries incompatible CLIs and skips required-argument scripts |

## Existing story assets

- `csv/story_dictionary.csv`: 11-column declarative story definitions, primarily team/player facets.
- `csv/story_menu_1981_week_05.csv` and `_week_07.csv`: selected definitions; the two observed schemas match.
- `csv/story_candidates_1981_week_05.csv`: evaluated Pythag candidates with team evidence; no week-7 candidate file was found.
- `csv/out/abl_1981_act3_story_pack.md`, weekly league reports, forum report, video outlines, and `core12_1981_w05.json`.
- Dozens of reusable signal tables/text reports: division leverage, bullpen stress, home/road splits, momentum, runways, manager tendencies, rookies, milestones, matchup history, Pythag, and team identity.

## Almanac and flashback assets

`csv/in/almanac_core/1972` through `1980` contain raw league HTML. `csv/out/almanac/<season>/` contains 31-34 CSVs per season and six or more EB Markdown products. All nine seasons have games, standings/time slices, player tables, leaderboards, transactions, schedule evaluation, summaries, and flashback candidates. A rendered flashback menu is visible for 1972. Grand Series box/log extracts and summaries live separately in `csv/out/eb/`.

## Likely run sequence

1. Refresh raw OOTP exports in `csv/ootp_csv/` and sortable tables in `csv/abl_statistics/`.
2. Build/refresh star schema and manager/team dimensions.
3. Seed the prior snapshot from games for the prior cutoff date.
4. Run `z_abl_run_week_1981.py` for current snapshot, weekly change, Monday tables, manager scorecard, and show notes.
5. Run analytic signal scripts required by the editorial packet.
6. Run `z_abl_eb_pack_1981_monday.py` or the turnkey wrapper.
7. Build a week menu from the dictionary, evaluate triggers, and print/select candidates.
8. Generate article/video outline from selected, evidence-backed candidates (currently manual/one-off).

Historical flow: extract season HTML -> scores/schedule/standings -> time slices and player context -> flashback candidates -> flashback menu/EB pack -> Grand Series context.

## Duplicate, temporary, and misplaced candidates (do not delete)

- Root `out/` duplicates six reports also represented under `csv/out/`.
- `csv/csv/out/almanac/1972/` and `csv/csv/docs/` indicate an accidental doubled `csv/` path.
- Generated `csv/story_menu_*.csv` and `csv/story_candidates_*.csv` are directly under `csv/`.
- `_tmp_run_all.py`, `tmp_preseason_bytes.txt`, `temp.txt`, `logs/_tmp_run_all.log`, and `*.tmp` prep files are temporary/diagnostic candidates.
- `__pycache__/` products are present in the script tree.
- Multiple similarly named implementations exist (`abl_week_miner.py`/`z_abl_week_miner.py`; general story tools/`z_abl_story_menu_1981_week5.py`; 1972-specific and `*_any.py` EB scripts).
- Current, previous, archive, and season-freeze copies are intentional-looking but need explicit lineage and retention rules.

