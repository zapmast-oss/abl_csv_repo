# CLI Options (csv/abl_scripts)

## csv/abl_scripts/_run_almanac_time_slices_any.py
```
usage: _run_almanac_time_slices_any.py [-h] --season SEASON
                                       [--league-id LEAGUE_ID]

Run scores pipeline and enrich time slices for any season/league.

options:
  -h, --help            show this help message and exit
  --season SEASON
  --league-id LEAGUE_ID
```

## csv/abl_scripts/_run_eb_md_only_any.py
```
usage: _run_eb_md_only_any.py [-h] --season SEASON [--league-id LEAGUE_ID]

Run EB markdown-generation steps for any season/league.

options:
  -h, --help            show this help message and exit
  --season SEASON
  --league-id LEAGUE_ID

Example: python csv/abl_scripts/_run_eb_md_only_any.py --season 1980 --league-
id 200
```

## csv/abl_scripts/_run_eb_regular_season_any.py
```
usage: _run_eb_regular_season_any.py [-h] --season SEASON
                                     [--league-id LEAGUE_ID]

Run EB regular-season pack for any ABL season.

options:
  -h, --help            show this help message and exit
  --season SEASON       Season year (e.g. 1972, 1973, ...)
  --league-id LEAGUE_ID
                        League ID (default 200)
```

## csv/abl_scripts/_run_eb_regular_season_range.py
```
usage: _run_eb_regular_season_range.py [-h] --start-season START_SEASON
                                       --end-season END_SEASON
                                       [--league-id LEAGUE_ID]

Run EB pipeline over a range of seasons.

options:
  -h, --help            show this help message and exit
  --start-season START_SEASON
  --end-season END_SEASON
  --league-id LEAGUE_ID
```

## csv/abl_scripts/abl_week_miner.py
```
usage: abl_week_miner.py [-h] [--start START] [--end END]
                         [--max_rows MAX_ROWS] [--mode {weekly,sim}]

ABL weekly highlight miner

options:
  -h, --help           show this help message and exit
  --start START        Start date YYYY-MM-DD
  --end END            End date YYYY-MM-DD
  --max_rows MAX_ROWS
  --mode {weekly,sim}  weekly recap or sim-through-Saturday
```

## csv/abl_scripts/load_managers_star.py
```
usage: load_managers_star.py [-h] [--db DB] [--src SRC]

Load manager summary CSV into SQLite star schema tables.

options:
  -h, --help  show this help message and exit
  --db DB     Path to SQLite database (default data_work/abl.db)
  --src SRC   Source CSV path (default out/csv_out/abl_managers_summary.csv)
```

## csv/abl_scripts/parse_managers.py
```
usage: parse_managers.py [-h] [--index INDEX] [--leaders LEADERS]
                         [--out-csv OUT_CSV] [--out-text OUT_TEXT]

Parse ABL manager index/leaders into CSV/TXT outputs.

options:
  -h, --help           show this help message and exit
  --index INDEX        Path to managers index (Template A)
  --leaders LEADERS    Path to managers leaders (Template B)
  --out-csv OUT_CSV    Destination CSV path
  --out-text OUT_TEXT  Destination text report
```

## csv/abl_scripts/report_batter_profile_prep_from_abl_statistics.py
```
usage: report_batter_profile_prep_from_abl_statistics.py [-h] [--team TEAM]
                                                         [--away AWAY]
                                                         [--home HOME] [--all]
                                                         [--limit LIMIT]
                                                         [--out OUT_PATH]

Generate Batter Profile Prep report from ABL Statistics.

options:
  -h, --help      show this help message and exit
  --team TEAM     Single team abbreviation (e.g., CHI)
  --away AWAY     Away team abbreviation for matchup mode
  --home HOME     Home team abbreviation for matchup mode
  --all           Output league-wide master list of all active ABL batters
  --limit LIMIT   Limit batters per team
  --out OUT_PATH  Override output path
```

## csv/abl_scripts/report_broadcast_prep.py
```
usage: report_broadcast_prep.py [-h] [--db DB] [--csv CSV] [--out OUT]
                                [--topN TOPN]

Broadcast prep report for managers

options:
  -h, --help   show this help message and exit
  --db DB
  --csv CSV
  --out OUT
  --topN TOPN
```

## csv/abl_scripts/report_manager_matchup.py
```
usage: report_manager_matchup.py [-h] [--home HOME] [--away AWAY] [--db DB]
                                 [--csv CSV] [--out OUT] [--verify]

Generate manager matchup card

options:
  -h, --help   show this help message and exit
  --home HOME  Home team name/code
  --away AWAY  Away team name/code
  --db DB
  --csv CSV
  --out OUT
  --verify
```

## csv/abl_scripts/report_pitcher_arsenal_prep_from_abl_statistics.py
```
usage: report_pitcher_arsenal_prep_from_abl_statistics.py [-h] [--team TEAM]
                                                          [--away AWAY]
                                                          [--home HOME]
                                                          [--all]
                                                          [--limit LIMIT]
                                                          [--out OUT_PATH]

Generate Pitcher Arsenal Prep report from ABL Statistics.

options:
  -h, --help      show this help message and exit
  --team TEAM     Single team abbreviation (e.g., CHI)
  --away AWAY     Away team abbreviation for matchup mode
  --home HOME     Home team abbreviation for matchup mode
  --all           Output league-wide master list of all active ABL pitchers
  --limit LIMIT   Limit pitchers per team
  --out OUT_PATH  Override output path
```

## csv/abl_scripts/validate_managers.py
```
usage: validate_managers.py [-h] [--src SRC]

Validate manager summary CSV

options:
  -h, --help  show this help message and exit
  --src SRC   Path to abl_managers_summary.csv
```

## csv/abl_scripts/z_abl_30for30_1981_pythag.py
```
usage: z_abl_30for30_1981_pythag.py [-h] [--dry-run]

options:
  -h, --help  show this help message and exit
  --dry-run   Run without writing output CSV
```

## csv/abl_scripts/z_abl_almanac_champions_from_standings.py
```
usage: z_abl_almanac_champions_from_standings.py [-h] --season SEASON
                                                 [--league-id LEAGUE_ID]
                                                 [--almanac-zip ALMANAC_ZIP]
                                                 [--dim-team-park DIM_TEAM_PARK]
                                                 [--write-csv]

Extract league champions from almanac standings HTML.

options:
  -h, --help            show this help message and exit
  --season SEASON
  --league-id LEAGUE_ID
  --almanac-zip ALMANAC_ZIP
  --dim-team-park DIM_TEAM_PARK
  --write-csv           Write champions CSV to csv/out/almanac/<season>/league
                        _champions_<season>_league<league_id>.csv
```

## csv/abl_scripts/z_abl_almanac_flashback_story_menu.py
```
usage: z_abl_almanac_flashback_story_menu.py [-h] --season SEASON --league-id
                                             LEAGUE_ID
                                             [--almanac-root ALMANAC_ROOT]
                                             [--candidates CANDIDATES]
                                             [--out-path OUT_PATH]

Build Flashback story menu Markdown from story candidates.

options:
  -h, --help            show this help message and exit
  --season SEASON       Season year (e.g., 1972)
  --league-id LEAGUE_ID
                        League ID (ABL canon: 200)
  --almanac-root ALMANAC_ROOT
                        Root folder for almanac outputs (default:
                        csv/out/almanac)
  --candidates CANDIDATES
                        Optional explicit path to flashback_story_candidates
                        CSV. If not provided, built from almanac-root/season.
  --out-path OUT_PATH   Optional explicit output path for the Markdown menu.
                        If not provided, uses almanac-root/<season>/flashback_
                        story_menu_<season>_league<id>.md
```

## csv/abl_scripts/z_abl_almanac_flashback_story_pack.py
```
usage: z_abl_almanac_flashback_story_pack.py [-h] --season SEASON
                                             [--league-id LEAGUE_ID]
                                             [--almanac-root ALMANAC_ROOT]

Build Flashback story candidates from almanac tables.

options:
  -h, --help            show this help message and exit
  --season SEASON       Season year (e.g. 1972)
  --league-id LEAGUE_ID
                        League ID (default 200)
  --almanac-root ALMANAC_ROOT
                        Root folder for almanac-derived CSV outputs.
```

## csv/abl_scripts/z_abl_almanac_league200_extract_core.py
```
usage: z_abl_almanac_league200_extract_core.py [-h] --season SEASON
                                               --league-id LEAGUE_ID

Extract league core HTML (including player pages) from almanac zip.

options:
  -h, --help            show this help message and exit
  --season SEASON
  --league-id LEAGUE_ID
```

## csv/abl_scripts/z_abl_almanac_league200_manifest.py
```
usage: z_abl_almanac_league200_manifest.py [-h] --season SEASON --league-id
                                           LEAGUE_ID
                                           [--almanac-root ALMANAC_ROOT]

Build an almanac manifest for a given season/league.

options:
  -h, --help            show this help message and exit
  --season SEASON       Season year (e.g. 1972)
  --league-id LEAGUE_ID
                        League ID (e.g. 200)
  --almanac-root ALMANAC_ROOT
                        Path to almanac root (default: almanac_<season>
                        relative to repo root)
```

## csv/abl_scripts/z_abl_almanac_league_4k_summary.py
```
usage: z_abl_almanac_league_4k_summary.py [-h] --season SEASON --league-id
                                          LEAGUE_ID
                                          [--almanac-root ALMANAC_ROOT]

Build 4k conference/division summaries from league season summary.

options:
  -h, --help            show this help message and exit
  --season SEASON       Season year (e.g., 1972)
  --league-id LEAGUE_ID
                        League ID (ABL=200)
  --almanac-root ALMANAC_ROOT
                        Root of almanac outputs (default: csv/out/almanac)
```

## csv/abl_scripts/z_abl_almanac_league_season_summary.py
```
usage: z_abl_almanac_league_season_summary.py [-h] --season SEASON --league-id
                                              LEAGUE_ID

Build league-season summary from almanac core + dim_team_park.

options:
  -h, --help            show this help message and exit
  --season SEASON       Season year, e.g. 1972
  --league-id LEAGUE_ID
                        League ID, e.g. 200
```

## csv/abl_scripts/z_abl_almanac_momentum_3k_summary.py
```
usage: z_abl_almanac_momentum_3k_summary.py [-h] --season SEASON --league-id
                                            LEAGUE_ID
                                            [--almanac-root ALMANAC_ROOT]

Build 3k momentum summaries (half/month/week).

options:
  -h, --help            show this help message and exit
  --season SEASON       Season year (e.g., 1972)
  --league-id LEAGUE_ID
                        League ID (ABL=200)
  --almanac-root ALMANAC_ROOT
                        Root of almanac outputs (default: csv/out/almanac)
```

## csv/abl_scripts/z_abl_almanac_player_context_extract.py
```
usage: z_abl_almanac_player_context_extract.py [-h] --season SEASON
                                               --league-id LEAGUE_ID

Extract player context tables from almanac core HTML.

options:
  -h, --help            show this help message and exit
  --season SEASON
  --league-id LEAGUE_ID
```

## csv/abl_scripts/z_abl_almanac_player_leaderboards.py
```
usage: z_abl_almanac_player_leaderboards.py [-h] --season SEASON --league-id
                                            LEAGUE_ID
                                            [--player-profile PLAYER_PROFILE]
                                            [--team-dim TEAM_DIM]

Build player leaderboards from extracted player stats.

options:
  -h, --help            show this help message and exit
  --season SEASON
  --league-id LEAGUE_ID
  --player-profile PLAYER_PROFILE
  --team-dim TEAM_DIM
```

## csv/abl_scripts/z_abl_almanac_player_stats_extract.py
```
usage: z_abl_almanac_player_stats_extract.py [-h] --season SEASON --league-id
                                             LEAGUE_ID

Extract player batting/pitching stats from almanac player pages.

options:
  -h, --help            show this help message and exit
  --season SEASON
  --league-id LEAGUE_ID
```

## csv/abl_scripts/z_abl_almanac_schedule_extract.py
```
usage: z_abl_almanac_schedule_extract.py [-h] --season SEASON --league-id
                                         LEAGUE_ID

Extract schedule grid/evaluator into CSV.

options:
  -h, --help            show this help message and exit
  --season SEASON
  --league-id LEAGUE_ID
```

## csv/abl_scripts/z_abl_almanac_scores_pipeline.py
```
usage: z_abl_almanac_scores_pipeline.py [-h] --almanac-zip ALMANAC_ZIP
                                        --season SEASON
                                        [--league-id LEAGUE_ID]
                                        [--out-root OUT_ROOT]

Parse OOTP almanac daily scoreboard HTML into game logs, then monthly, weekly,
and series summaries.

options:
  -h, --help            show this help message and exit
  --almanac-zip ALMANAC_ZIP
                        Path to almanac_YYYY.zip
  --season SEASON       Season year (e.g. 1972).
  --league-id LEAGUE_ID
                        League ID to parse (ABL = 200). Default: 200.
  --out-root OUT_ROOT   Root output folder for parsed data (default:
                        csv/out/almanac).
```

## csv/abl_scripts/z_abl_almanac_standings_pipeline.py
```
[DEBUG] z_abl_almanac_standings_pipeline.py starting up
usage: z_abl_almanac_standings_pipeline.py [-h] [--almanac-dir ALMANAC_DIR]
                                           [--pattern PATTERN]
                                           [--league-id LEAGUE_ID]
                                           [--out-root OUT_ROOT]

Batch-parse ABL standings from OOTP almanac_YYYY.zip files.

options:
  -h, --help            show this help message and exit
  --almanac-dir ALMANAC_DIR
                        Directory containing almanac_YYYY.zip files (default:
                        current directory).
  --pattern PATTERN     Glob pattern to match almanac zip files (default:
                        almanac_*.zip).
  --league-id LEAGUE_ID
                        League ID to parse (ABL = 200). Default: 200.
  --out-root OUT_ROOT   Root output folder for parsed standings (default:
                        csv/out/almanac).
```

## csv/abl_scripts/z_abl_almanac_time_slices_enriched.py
```
usage: z_abl_almanac_time_slices_enriched.py [-h] --season SEASON
                                             [--league-id LEAGUE_ID]
                                             [--almanac-root ALMANAC_ROOT]
                                             [--dim-team-park DIM_TEAM_PARK]

Enrich almanac time-slice tables (monthly/weekly/series) with dim_team_park
info.

options:
  -h, --help            show this help message and exit
  --season SEASON       Season year (e.g. 1972).
  --league-id LEAGUE_ID
                        League ID (ABL = 200). Default: 200.
  --almanac-root ALMANAC_ROOT
                        Root folder where almanac-derived CSVs live (default:
                        csv/out/almanac).
  --dim-team-park DIM_TEAM_PARK
                        Path to dim_team_park.csv (default:
                        csv/out/star_schema/dim_team_park.csv).
```

## csv/abl_scripts/z_abl_attach_managers_to_standings.py
```
usage: z_abl_attach_managers_to_standings.py [-h] [--dry-run]

options:
  -h, --help  show this help message and exit
  --dry-run   Run without writing output CSV
```

## csv/abl_scripts/z_abl_basepath_pressure.py
```
usage: z_abl_basepath_pressure.py [-h] [--base BASE] [--batting BATTING]
                                  [--record RECORD] [--logs LOGS]
                                  [--teams TEAMS]
                                  [--user-baserunning USER_BASERUNNING]
                                  [--out OUT]

ABL Basepath Pressure report.

options:
  -h, --help            show this help message and exit
  --base BASE           Base directory.
  --batting BATTING     Override batting/baserunning totals.
  --record RECORD       Override season record file.
  --logs LOGS           Override team logs for games.
  --teams TEAMS         Override team info file.
  --user-baserunning USER_BASERUNNING
                        Override user baserunning file (default:
                        abl_team_bat_baserunning.csv).
  --out OUT             Output CSV path (default:
                        out/csv_out/z_ABL_Basepath_Pressure.csv).
```

## csv/abl_scripts/z_abl_blowout_resilience.py
```
usage: z_abl_blowout_resilience.py [-h] [--base BASE] [--logs LOGS]
                                   [--out OUT]

ABL blowout resilience report.

options:
  -h, --help   show this help message and exit
  --base BASE  Base directory for CSVs.
  --logs LOGS  Explicit path to team log CSV.
  --out OUT    Output CSV (default: out/csv_out/z_ABL_Blowout_Resilience.csv).
```

## csv/abl_scripts/z_abl_bullpen_stress_index.py
```
usage: z_abl_bullpen_stress_index.py [-h] [--base BASE] [--apps APPS]
                                     [--teamlogs TEAMLOGS] [--teams TEAMS]
                                     [--out OUT]

ABL Bullpen Stress Index.

options:
  -h, --help           show this help message and exit
  --base BASE          Base directory for CSVs.
  --apps APPS          Override pitcher appearance log.
  --teamlogs TEAMLOGS  Override team game logs.
  --teams TEAMS        Override team info file.
  --out OUT            Output CSV path.
```

## csv/abl_scripts/z_abl_catcher_battery_value.py
```
usage: z_abl_catcher_battery_value.py [-h] [--base BASE] [--fielding FIELDING]
                                      [--battery BATTERY]
                                      [--gamelogs GAMELOGS]
                                      [--lineups LINEUPS] [--teams TEAMS]
                                      [--roster ROSTER] [--out OUT]
                                      [--min_inn_c MIN_INN_C]
                                      [--min_sbcs MIN_SBCS]

ABL Catcher Battery Value report.

options:
  -h, --help            show this help message and exit
  --base BASE           Base directory.
  --fielding FIELDING   Override catcher fielding file.
  --battery BATTERY     Override pitching-by-catcher file.
  --gamelogs GAMELOGS   Override pitching gamelog file.
  --lineups LINEUPS     Override lineup/catcher of record file.
  --teams TEAMS         Override team info file.
  --roster ROSTER       Override roster file.
  --out OUT             Output CSV path.
  --min_inn_c MIN_INN_C
                        Minimum innings caught for ERA stability.
  --min_sbcs MIN_SBCS   Minimum SB+CS for CS{'option_strings': ['--min_sbcs'],
                        'dest': 'min_sbcs', 'nargs': None, 'const': None,
                        'default': 15, 'type': 'int', 'choices': None,
                        'required': False, 'help': 'Minimum SB+CS for CS%
                        stability.', 'metavar': None, 'container':
                        <argparse._ArgumentGroup object at
                        0x000001E42E3B5D90>, 'prog':
                        'z_abl_catcher_battery_value.py'}tability.
```

## csv/abl_scripts/z_abl_core12_weekly.py
```
usage: z_abl_core12_weekly.py [-h] --year YEAR --week WEEK

Generate ABL Core 12 weekly summary.

options:
  -h, --help   show this help message and exit
  --year YEAR
  --week WEEK
```

## csv/abl_scripts/z_abl_current_team_snapshot.py
```
usage: z_abl_current_team_snapshot.py [-h] [--dry-run]

options:
  -h, --help  show this help message and exit
  --dry-run   Run without writing output CSV
```

## csv/abl_scripts/z_abl_damage_with_risp.py
```
usage: z_abl_damage_with_risp.py [-h] [--base BASE] [--totals TOTALS]
                                 [--splits SPLITS] [--teams TEAMS]
                                 [--min_pa MIN_PA] [--min_pa_risp MIN_PA_RISP]
                                 [--out OUT]

ABL Damage With RISP report.

options:
  -h, --help            show this help message and exit
  --base BASE           Base directory for CSV files.
  --totals TOTALS       Override file for player batting totals.
  --splits SPLITS       Override file for situational splits.
  --teams TEAMS         Override file for team info.
  --min_pa MIN_PA       Minimum PA to qualify.
  --min_pa_risp MIN_PA_RISP
                        Minimum PA with RISP.
  --out OUT             Output CSV path.
```

## csv/abl_scripts/z_abl_division_leverage.py
```
usage: z_abl_division_leverage.py [-h] [--base BASE] [--logs LOGS]
                                  [--teams TEAMS] [--sched SCHED] [--out OUT]

ABL division leverage report.

options:
  -h, --help     show this help message and exit
  --base BASE    Base directory for CSVs.
  --logs LOGS    Explicit played logs CSV.
  --teams TEAMS  Explicit team metadata CSV.
  --sched SCHED  Explicit future schedule CSV.
  --out OUT      Output CSV (default:
                 out/csv_out/z_ABL_Division_Leverage.csv).
```

## csv/abl_scripts/z_abl_eb_all_star_brief_any.py
```
usage: z_abl_eb_all_star_brief_any.py [-h] --season SEASON
                                      [--league-id LEAGUE_ID]

EB All-Star brief generator for any season/league.

options:
  -h, --help            show this help message and exit
  --season SEASON
  --league-id LEAGUE_ID
```

## csv/abl_scripts/z_abl_eb_flashback_brief_1972.py
```
usage: z_abl_eb_flashback_brief_1972.py [-h] --season SEASON --league-id
                                        LEAGUE_ID

Build EB flashback data brief for a season/league.

options:
  -h, --help            show this help message and exit
  --season SEASON
  --league-id LEAGUE_ID
```

## csv/abl_scripts/z_abl_eb_player_context_1972.py
```
usage: z_abl_eb_player_context_1972.py [-h] [--season SEASON]
                                       [--league-id LEAGUE_ID]

EB player context brief (financials, prospects, preseason).

options:
  -h, --help            show this help message and exit
  --season SEASON
  --league-id LEAGUE_ID
```

## csv/abl_scripts/z_abl_eb_player_leaders_1972.py
```
usage: z_abl_eb_player_leaders_1972.py [-h] [--season SEASON]
                                       [--league-id LEAGUE_ID]
                                       [--player-profile PLAYER_PROFILE]
                                       [--team-dim TEAM_DIM]

Build EB player leaders brief for a season/league.

options:
  -h, --help            show this help message and exit
  --season SEASON
  --league-id LEAGUE_ID
  --player-profile PLAYER_PROFILE
  --team-dim TEAM_DIM
```

## csv/abl_scripts/z_abl_eb_player_spotlights_1972.py
```
usage: z_abl_eb_player_spotlights_1972.py [-h] [--season SEASON]
                                          [--league-id LEAGUE_ID]
                                          [--player-profile PLAYER_PROFILE]
                                          [--team-dim TEAM_DIM]

EB player spotlight brief for a season/league.

options:
  -h, --help            show this help message and exit
  --season SEASON
  --league-id LEAGUE_ID
  --player-profile PLAYER_PROFILE
  --team-dim TEAM_DIM
```

## csv/abl_scripts/z_abl_eb_regular_season_pack_any.py
```
usage: z_abl_eb_regular_season_pack_any.py [-h] --season SEASON
                                           [--league-id LEAGUE_ID]
                                           [--almanac-zip ALMANAC_ZIP]
                                           [--dim-team-park DIM_TEAM_PARK]

Build EB regular-season pack markdown for any season/league.

options:
  -h, --help            show this help message and exit
  --season SEASON
  --league-id LEAGUE_ID
  --almanac-zip ALMANAC_ZIP
  --dim-team-park DIM_TEAM_PARK
```

## csv/abl_scripts/z_abl_eb_schedule_context_1972.py
```
usage: z_abl_eb_schedule_context_1972.py [-h] [--season SEASON]
                                         [--league-id LEAGUE_ID]

EB schedule context brief.

options:
  -h, --help            show this help message and exit
  --season SEASON
  --league-id LEAGUE_ID
```

## csv/abl_scripts/z_abl_eb_schedule_context_any.py
```
usage: z_abl_eb_schedule_context_any.py [-h] --season SEASON
                                        [--league-id LEAGUE_ID]

Build EB schedule-context fragment from almanac schedule_grid HTML.

options:
  -h, --help            show this help message and exit
  --season SEASON       Season year, e.g. 1972 or 1980
  --league-id LEAGUE_ID
                        League ID (default 200 for ABL)
```

## csv/abl_scripts/z_abl_enrich_almanac_standings.py
```
usage: z_abl_enrich_almanac_standings.py [-h] [--season SEASON]
                                         [--league-id LEAGUE_ID]
                                         [--standings-root STANDINGS_ROOT]
                                         [--dim-team-park DIM_TEAM_PARK]
                                         [--out-root OUT_ROOT]

Enrich almanac standings with team_id, conference, and division from
dim_team_park.

options:
  -h, --help            show this help message and exit
  --season SEASON       Season year to process (default: 1972).
  --league-id LEAGUE_ID
                        League ID (ABL = 200). Default: 200.
  --standings-root STANDINGS_ROOT
                        Root folder where
                        standings_{season}_league{league_id}.csv lives
                        (default: csv/out/almanac).
  --dim-team-park DIM_TEAM_PARK
                        Path to dim_team_park.csv (default:
                        csv/out/star_schema/dim_team_park.csv).
  --out-root OUT_ROOT   Root folder for enriched output (default:
                        csv/out/almanac).
```

## csv/abl_scripts/z_abl_extract_abl_power_rankings.py
```
usage: z_abl_extract_abl_power_rankings.py [-h] --input-html INPUT_HTML
                                           --output-csv OUTPUT_CSV

Extract ABL (League 200) Weekly Team Power Rankings from league_200_home.html
into CSV.

options:
  -h, --help            show this help message and exit
  --input-html INPUT_HTML
                        Path to league_200_home.html (the League 200 home
                        page).
  --output-csv OUTPUT_CSV
                        Path to output CSV file for full power rankings
                        (1–24).
```

## csv/abl_scripts/z_abl_extract_league200_minimal.py
```
usage: z_abl_extract_league200_minimal.py [-h] --root-dir ROOT_DIR
                                          --output-dir OUTPUT_DIR

Extract minimal ABL (League 200) site: league_200_home.html + css/js/images.

options:
  -h, --help            show this help message and exit
  --root-dir ROOT_DIR   Path to the root of the full OOTP HTML site (the
                        'html' folder).
  --output-dir OUTPUT_DIR
                        Path to the output folder for the minimal League 200
                        site.
```

## csv/abl_scripts/z_abl_fip_vs_era_gap.py
```
usage: z_abl_fip_vs_era_gap.py [-h] [--base BASE] [--pitching PITCHING]
                               [--teams TEAMS] [--roster ROSTER] [--out OUT]
                               [--min_ip_sp MIN_IP_SP] [--min_ip_rp MIN_IP_RP]

ABL FIP vs ERA gap report.

options:
  -h, --help            show this help message and exit
  --base BASE           Base directory for CSVs.
  --pitching PITCHING   Override path for pitching totals.
  --teams TEAMS         Override path for team info.
  --roster ROSTER       Override path for roster file.
  --out OUT             Output CSV path (default inside out/csv_out).
  --min_ip_sp MIN_IP_SP
                        Minimum IP for SP/Swing qualification.
  --min_ip_rp MIN_IP_RP
                        Minimum IP for RP qualification.
```

## csv/abl_scripts/z_abl_firestarter_table.py
```
usage: z_abl_firestarter_table.py [-h] [--base BASE] [--spot SPOT]
                                  [--inning INNING] [--linescore LINESCORE]
                                  [--record RECORD] [--logs LOGS]
                                  [--teams TEAMS] [--out OUT]

ABL Firestarter Table report.

options:
  -h, --help            show this help message and exit
  --base BASE           Base directory for CSV files.
  --spot SPOT           Override lineup-spot splits file.
  --inning INNING       Override inning splits file.
  --linescore LINESCORE
                        Override linescore file.
  --record RECORD       Override season record file.
  --logs LOGS           Override team logs file.
  --teams TEAMS         Override team info file.
  --out OUT             Output CSV path (default inside out/csv_out).
```

## csv/abl_scripts/z_abl_grand_series_extractor.py
```
usage: z_abl_grand_series_extractor.py [-h] --season SEASON
                                       [--league-id LEAGUE_ID] [--start START]
                                       [--end END] [--out-dir OUT_DIR]

Extract Grand Series games from almanac HTML.

options:
  -h, --help            show this help message and exit
  --season SEASON
  --league-id LEAGUE_ID
  --start START         GS start date YYYY-MM-DD (optional; derive from grid
                        if omitted)
  --end END             GS end date YYYY-MM-DD (optional; derive from grid if
                        omitted)
  --out-dir OUT_DIR     Optional output dir to save box/log HTML for each GS
                        game
```

## csv/abl_scripts/z_abl_grand_series_summarize.py
```
usage: z_abl_grand_series_summarize.py [-h] --season SEASON
                                       [--league-id LEAGUE_ID]
                                       [--gs-dir GS_DIR] [--out-md OUT_MD]

Summarize Grand Series box scores.

options:
  -h, --help            show this help message and exit
  --season SEASON
  --league-id LEAGUE_ID
  --gs-dir GS_DIR       Directory with game_box_*.html; defaults to
                        csv/out/eb/gs_<season>.
  --out-md OUT_MD       Optional explicit output path for summary markdown.
```

## csv/abl_scripts/z_abl_grit_index.py
```
usage: z_abl_grit_index.py [-h] [--base BASE] [--linescore LINESCORE]
                           [--pbp PBP] [--teamlogs TEAMLOGS] [--teams TEAMS]
                           [--out OUT]

ABL Grit Index report.

options:
  -h, --help            show this help message and exit
  --base BASE           Base directory for CSV files.
  --linescore LINESCORE
                        Override linescore file.
  --pbp PBP             Override play-by-play file for fallback.
  --teamlogs TEAMLOGS   Override team log file (optional fallback).
  --teams TEAMS         Override team info file.
  --out OUT             Output CSV path (default inside out/csv_out).
```

## csv/abl_scripts/z_abl_ground_ball_savants.py
```
Traceback (most recent call last):
  File "c:\sbv_repo\abl_csv_repo\csv\abl_scripts\z_abl_ground_ball_savants.py", line 559, in <module>
    main()
  File "c:\sbv_repo\abl_csv_repo\csv\abl_scripts\z_abl_ground_ball_savants.py", line 365, in main
    args = parser.parse_args(list(argv) if argv is not None else None)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\earld\AppData\Local\Programs\Python\Python312\Lib\argparse.py", line 1896, in parse_args
    args, argv = self.parse_known_args(args, namespace)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\earld\AppData\Local\Programs\Python\Python312\Lib\argparse.py", line 1932, in parse_known_args
    namespace, args = self._parse_known_args(args, namespace)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\earld\AppData\Local\Programs\Python\Python312\Lib\argparse.py", line 2153, in _parse_known_args
    start_index = consume_optional(start_index)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\earld\AppData\Local\Programs\Python\Python312\Lib\argparse.py", line 2093, in consume_optional
    take_action(action, args, option_string)
  File "C:\Users\earld\AppData\Local\Programs\Python\Python312\Lib\argparse.py", line 2008, in take_action
    action(self, namespace, argument_values, option_string)
  File "C:\Users\earld\AppData\Local\Programs\Python\Python312\Lib\argparse.py", line 1144, in __call__
    parser.print_help()
  File "C:\Users\earld\AppData\Local\Programs\Python\Python312\Lib\argparse.py", line 2642, in print_help
    self._print_message(self.format_help(), file)
                        ^^^^^^^^^^^^^^^^^^
  File "C:\Users\earld\AppData\Local\Programs\Python\Python312\Lib\argparse.py", line 2626, in format_help
    return formatter.format_help()
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\earld\AppData\Local\Programs\Python\Python312\Lib\argparse.py", line 287, in format_help
    help = self._root_section.format_help()
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\earld\AppData\Local\Programs\Python\Python312\Lib\argparse.py", line 217, in format_help
    item_help = join([func(*args) for func, args in self.items])
                      ^^^^^^^^^^^
  File "C:\Users\earld\AppData\Local\Programs\Python\Python312\Lib\argparse.py", line 217, in format_help
    item_help = join([func(*args) for func, args in self.items])
                      ^^^^^^^^^^^
  File "C:\Users\earld\AppData\Local\Programs\Python\Python312\Lib\argparse.py", line 547, in _format_action
    help_text = self._expand_help(action)
                ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\earld\AppData\Local\Programs\Python\Python312\Lib\argparse.py", line 644, in _expand_help
    return self._get_help_string(action) % params
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~
ValueError: incomplete format
```

## csv/abl_scripts/z_abl_heat_check.py
```
usage: z_abl_heat_check.py [-h] [--base BASE] [--gamelogs GAMELOGS]
                           [--totals TOTALS] [--roster ROSTER] [--teams TEAMS]
                           [--min_pa_last7 MIN_PA_LAST7]
                           [--min_pa_prior7 MIN_PA_PRIOR7] [--out OUT]

ABL Heat Check report.

options:
  -h, --help            show this help message and exit
  --base BASE           Base directory for CSV files.
  --gamelogs GAMELOGS   Override batting game logs file.
  --totals TOTALS       Override season totals file.
  --roster ROSTER       Override roster/positions file.
  --teams TEAMS         Override team info file.
  --min_pa_last7 MIN_PA_LAST7
                        Minimum PA in last7 window.
  --min_pa_prior7 MIN_PA_PRIOR7
                        Minimum PA in prior7 window.
  --out OUT             Output CSV path (default inside out/csv_out).
```

## csv/abl_scripts/z_abl_high_leverage_relievers.py
```
usage: z_abl_high_leverage_relievers.py [-h] [--base BASE]
                                        [--pitching PITCHING]
                                        [--relief RELIEF] [--applogs APPLOGS]
                                        [--teams TEAMS] [--roster ROSTER]
                                        [--out OUT] [--min_ip_rp MIN_IP_RP]
                                        [--min_app MIN_APP]
                                        [--li_high LI_HIGH]

ABL High-Leverage Relievers report.

options:
  -h, --help            show this help message and exit
  --base BASE           Base directory for CSVs.
  --pitching PITCHING   Override path for pitching totals.
  --relief RELIEF       Override path for relief splits.
  --applogs APPLOGS     Override path for appearance logs.
  --teams TEAMS         Override path for team info.
  --roster ROSTER       Override path for roster file.
  --out OUT             Output CSV path (default inside out/csv_out).
  --min_ip_rp MIN_IP_RP
                        Minimum IP for RP qualification.
  --min_app MIN_APP     Minimum appearances for qualification.
  --li_high LI_HIGH     Threshold to define high-leverage entry LI.
```

## csv/abl_scripts/z_abl_home_road_splits.py
```
usage: z_abl_home_road_splits.py [-h] [--base BASE] [--season SEASON]
                                 [--logs LOGS] [--parks PARKS] [--out OUT]

ABL home/road splits report.

options:
  -h, --help       show this help message and exit
  --base BASE      Base directory for CSVs.
  --season SEASON  Explicit season totals CSV.
  --logs LOGS      Explicit per-game logs CSV.
  --parks PARKS    Explicit park factors CSV.
  --out OUT        Output CSV (default:
                   out/csv_out/z_ABL_Home_Road_Splits.csv).
```

## csv/abl_scripts/z_abl_late_inning_clutch.py
```
usage: z_abl_late_inning_clutch.py [-h] [--base BASE] [--linescore LINESCORE]
                                   [--pbp PBP] [--splits SPLITS] [--out OUT]

ABL late-inning clutch report.

options:
  -h, --help            show this help message and exit
  --base BASE           Base directory for CSVs.
  --linescore LINESCORE
                        Explicit linescore file.
  --pbp PBP             Explicit play-by-play file (not yet supported).
  --splits SPLITS       Explicit inning splits file.
  --out OUT             Output CSV (default:
                        out/csv_out/z_ABL_Late_Inning_Clutch.csv).
```

## csv/abl_scripts/z_abl_list_md_outputs.py
```
usage: z_abl_list_md_outputs.py [-h] [--year YEAR] [--week WEEK]

List Markdown outputs.

options:
  -h, --help   show this help message and exit
  --year YEAR
  --week WEEK
```

## csv/abl_scripts/z_abl_manager_scorecard_1981.py
```
usage: z_abl_manager_scorecard_1981.py [-h] [--dry-run]

options:
  -h, --help  show this help message and exit
  --dry-run   Run without writing output CSV
```

## csv/abl_scripts/z_abl_manager_tendencies.py
```
usage: z_abl_manager_tendencies.py [-h] [--base BASE] [--batting BATTING]
                                   [--baserun BASERUN] [--apps APPS]
                                   [--splits SPLITS] [--lineups LINEUPS]
                                   [--teams TEAMS] [--out OUT]

ABL Manager Tendencies.

options:
  -h, --help         show this help message and exit
  --base BASE        Base directory for CSVs.
  --batting BATTING  Override batting totals file.
  --baserun BASERUN  Override baserunning detail file.
  --apps APPS        Override pitcher appearance logs.
  --splits SPLITS    Override platoon splits file.
  --lineups LINEUPS  Override lineup file (not used if splits found).
  --teams TEAMS      Override team info file.
  --out OUT          Output CSV path (default inside out/csv_out).
```

## csv/abl_scripts/z_abl_matchup_history.py
```
usage: z_abl_matchup_history.py [-h] [--min-date MIN_DATE]
                                [--max-date MAX_DATE]

Compute ABL head-to-head matchup history from games.csv

options:
  -h, --help           show this help message and exit
  --min-date MIN_DATE  Lower bound inclusive for game date (YYYY-MM-DD)
  --max-date MAX_DATE  Upper bound inclusive for game date (YYYY-MM-DD)
```

## csv/abl_scripts/z_abl_milestones_on_the_horizon.py
```
usage: z_abl_milestones_on_the_horizon.py [-h] [--base BASE] [--within WITHIN]
                                          [--out OUT]

Highlight ABL milestones on the horizon.

options:
  -h, --help       show this help message and exit
  --base BASE      Base directory for CSVs.
  --within WITHIN  Include milestones where to-go is within this threshold.
  --out OUT        Output CSV path (default inside out/csv_out).
```

## csv/abl_scripts/z_abl_momentum_windows.py
```
usage: z_abl_momentum_windows.py [-h] [--base BASE] [--logs LOGS]
                                 [--window WINDOW] [--out OUT]

Compute ABL momentum windows (last-N vs prior-N win percentages).

options:
  -h, --help       show this help message and exit
  --base BASE      Base directory to search for logs (default: current
                   directory).
  --logs LOGS      Explicit path to a log CSV (overrides autodetect).
  --window WINDOW  Window size for comparison (default: 10).
  --out OUT        Output CSV path (default:
                   out/csv_out/z_ABL_Momentum_Windows.csv).
```

## csv/abl_scripts/z_abl_monday_packet_1981.py
```
usage: z_abl_monday_packet_1981.py [-h] [--dry-run]

options:
  -h, --help  show this help message and exit
  --dry-run   Run without writing output CSVs
```

## csv/abl_scripts/z_abl_monday_show_notes_1981.py
```
usage: z_abl_monday_show_notes_1981.py [-h] [--dry-run]

options:
  -h, --help  show this help message and exit
  --dry-run   Run without writing output CSV
```

## csv/abl_scripts/z_abl_month_glory_misery_any.py
```
usage: z_abl_month_glory_misery_any.py [-h] --season SEASON
                                       [--league-id LEAGUE_ID]
                                       [--games-csv GAMES_CSV]
                                       [--schedule-grid-html SCHEDULE_GRID_HTML]
                                       [--dim-team-park DIM_TEAM_PARK]
                                       [--min-games MIN_GAMES]
                                       [--out-md OUT_MD]

Compute Month of Glory/Misery from games CSV.

options:
  -h, --help            show this help message and exit
  --season SEASON
  --league-id LEAGUE_ID
  --games-csv GAMES_CSV
                        Path to games_{season}_league{league_id}_by_team.csv
  --schedule-grid-html SCHEDULE_GRID_HTML
                        Path to schedule grid HTML
  --dim-team-park DIM_TEAM_PARK
                        Path to dim_team_park.csv
  --min-games MIN_GAMES
  --out-md OUT_MD       Output markdown fragment path
```

## csv/abl_scripts/z_abl_one_run_record.py
```
usage: z_abl_one_run_record.py [-h] [--base BASE] [--logs LOGS] [--out OUT]

ABL one-run record report.

options:
  -h, --help   show this help message and exit
  --base BASE  Base directory for CSVs.
  --logs LOGS  Explicit path to team log CSV.
  --out OUT    Output CSV (default: out/csv_out/z_ABL_One_Run_Record.csv).
```

## csv/abl_scripts/z_abl_outfield_arms.py
```
usage: z_abl_outfield_arms.py [-h] [--base BASE] [--fielding FIELDING]
                              [--opps OPPS] [--teams TEAMS] [--roster ROSTER]
                              [--out OUT] [--min_inn MIN_INN]
                              [--min_attempts MIN_ATTEMPTS]

ABL Outfield Arms report.

options:
  -h, --help            show this help message and exit
  --base BASE           Base directory.
  --fielding FIELDING   Override fielding path.
  --opps OPPS           Override opportunity file.
  --teams TEAMS         Override team info.
  --roster ROSTER       Override roster file.
  --out OUT             Output CSV path (default inside out/csv_out).
  --min_inn MIN_INN     Minimum OF innings to rank.
  --min_attempts MIN_ATTEMPTS
                        Minimum attempts to rank no-go rate.
```

## csv/abl_scripts/z_abl_platoon_assassins.py
```
usage: z_abl_platoon_assassins.py [-h] [--base BASE] [--splits SPLITS]
                                  [--roster ROSTER] [--teams TEAMS]
                                  [--min_pa_both MIN_PA_BOTH]
                                  [--min_pa_adv MIN_PA_ADV] [--show_all]
                                  [--out OUT]

ABL Platoon Assassins report.

options:
  -h, --help            show this help message and exit
  --base BASE           Base directory.
  --splits SPLITS       Override path for splits vs hand.
  --roster ROSTER       Override path for roster/names.
  --teams TEAMS         Override path for team names.
  --min_pa_both MIN_PA_BOTH
                        Minimum PA vs each hand.
  --min_pa_adv MIN_PA_ADV
                        Minimum advantaged PA.
  --show_all            Include non-qualifiers.
  --out OUT             Output CSV path (default inside out/csv_out).
```

## csv/abl_scripts/z_abl_power_surge_outages.py
```
usage: z_abl_power_surge_outages.py [-h] [--base BASE] [--logs LOGS]
                                    [--boxes BOXES] [--games GAMES]
                                    [--teams TEAMS] [--parks PARKS]
                                    [--week-end WEEK_END] [--limit LIMIT]
                                    [--out OUT]

Track weekly HR surge/outage trends.

options:
  -h, --help           show this help message and exit
  --base BASE          Base directory for CSV exports.
  --logs LOGS          Override team game log CSV.
  --boxes BOXES        Override team box log CSV when logs missing.
  --games GAMES        Override schedule/games CSV for park info.
  --teams TEAMS        Override team info CSV.
  --parks PARKS        Override park info CSV.
  --week-end WEEK_END  Force week end date (YYYY-MM-DD).
  --limit LIMIT        Max entries per section in the text report.
  --out OUT            Output CSV path (defaults to out/csv_out/...).
```

## csv/abl_scripts/z_abl_pregame_pack.py
```
usage: z_abl_pregame_pack.py [-h] [--base BASE] [--season SEASON]
                             [--week WEEK] [--league_id LEAGUE_ID]
                             [--matchups MATCHUPS] [--date DATE]
                             [--starter STARTER] [--arsenal-top ARSENAL_TOP]
                             [--show-arsenal-count] [--bats-top BATS_TOP]

Generate ABL pregame pack.

options:
  -h, --help            show this help message and exit
  --base BASE           Repo root (optional).
  --season SEASON
  --week WEEK
  --league_id LEAGUE_ID
  --matchups MATCHUPS   Explicit matchups list, e.g., CHI@MIA,DEN@NAS
  --date DATE           Game date (YYYY-MM-DD) to derive matchups from
                        schedule/games
  --starter STARTER     Manual starter override(s), e.g., CHI=8125 or MIA=Bill
                        Borden; can repeat or use commas
  --arsenal-top ARSENAL_TOP
                        Top N pitches to display for arsenal
  --show-arsenal-count  Show pitch count when available
  --bats-top BATS_TOP   Top N key bats per team
```

## csv/abl_scripts/z_abl_prep_card_1981.py
```
usage: z_abl_prep_card_1981.py [-h] --away AWAY --home HOME [--dry-run]

options:
  -h, --help   show this help message and exit
  --away AWAY  Away team abbreviation (e.g., CHI)
  --home HOME  Home team abbreviation (e.g., MIA)
  --dry-run    Run without writing output CSV
```

## csv/abl_scripts/z_abl_preseason_hype_any.py
```
usage: z_abl_preseason_hype_any.py [-h] --season SEASON
                                   [--league-id LEAGUE_ID]
                                   [--preseason-html PRESEASON_HTML]
                                   [--output-md OUTPUT_MD]

Build EB preseason hype fragment.

options:
  -h, --help            show this help message and exit
  --season SEASON       Season year, e.g. 1980
  --league-id LEAGUE_ID
                        League ID (default 200)
  --preseason-html PRESEASON_HTML
                        Optional explicit path to preseason prediction HTML.
                        Defaults to csv/in/almanac_core/{season}/leagues/leagu
                        e_{league_id}_preseason_prediction_report.html
  --output-md OUTPUT_MD
                        Optional output markdown path override.
```

## csv/abl_scripts/z_abl_pythag_over_under.py
```
usage: z_abl_pythag_over_under.py [-h] [--base BASE] [--in INPUT_PATH]
                                  [--out OUT] [--sort {diff,division}]

ABL Pythagorean over/under report.

options:
  -h, --help            show this help message and exit
  --base BASE           Base directory to search.
  --in INPUT_PATH       Explicit input CSV.
  --out OUT             Output CSV (default:
                        out/csv_out/z_ABL_Pythag_Over_Under.csv).
  --sort {diff,division}
                        Sort by luck (diff) or by conference/division
                        (division).
```

## csv/abl_scripts/z_abl_report_ballparks.py
```
usage: z_abl_report_ballparks.py [-h] [--base BASE] [--league_id LEAGUE_ID]

Generate ABL ballparks report.

options:
  -h, --help            show this help message and exit
  --base BASE           Repo root (optional).
  --league_id LEAGUE_ID
```

## csv/abl_scripts/z_abl_report_fans_markets.py
```
usage: z_abl_report_fans_markets.py [-h] [--base BASE] [--league_id LEAGUE_ID]
                                    [--season SEASON]

Generate ABL fans/markets report.

options:
  -h, --help            show this help message and exit
  --base BASE           Repo root (optional).
  --league_id LEAGUE_ID
  --season SEASON
```

## csv/abl_scripts/z_abl_report_finances.py
```
usage: z_abl_report_finances.py [-h] [--base BASE] [--league_id LEAGUE_ID]
                                [--season SEASON]

Generate ABL finances report.

options:
  -h, --help            show this help message and exit
  --base BASE           Repo root (optional).
  --league_id LEAGUE_ID
  --season SEASON
```

## csv/abl_scripts/z_abl_rookie_watch.py
```
usage: z_abl_rookie_watch.py [-h] [--base BASE] [--roster ROSTER]
                             [--batting BATTING] [--fielding FIELDING]
                             [--pitching PITCHING] [--teams TEAMS]
                             [--parks PARKS] [--out_hit OUT_HIT]
                             [--out_pit OUT_PIT] [--min_pa MIN_PA]
                             [--min_ip MIN_IP] [--show_all]

ABL Rookie Watch report.

options:
  -h, --help           show this help message and exit
  --base BASE          Base directory.
  --roster ROSTER      Override roster file.
  --batting BATTING    Override batting totals.
  --fielding FIELDING  Override fielding totals.
  --pitching PITCHING  Override pitching totals.
  --teams TEAMS        Override team info file.
  --parks PARKS        Override park factors file.
  --out_hit OUT_HIT    Output CSV for hitters.
  --out_pit OUT_PIT    Output CSV for pitchers.
  --min_pa MIN_PA      Minimum PA for hitter ranking.
  --min_ip MIN_IP      Minimum IP for pitcher ranking.
  --show_all           Include non-qualifiers.
```

## csv/abl_scripts/z_abl_rotation_stability.py
```
usage: z_abl_rotation_stability.py [-h] [--base BASE] [--apps APPS]
                                   [--inj INJ] [--teams TEAMS] [--out OUT]

Compute ABL rotation stability metrics.

options:
  -h, --help     show this help message and exit
  --base BASE    Base directory for CSVs.
  --apps APPS    Override pitcher appearances/logs file.
  --inj INJ      Override injuries/status file.
  --teams TEAMS  Override team info file.
  --out OUT      Output CSV path.
```

## csv/abl_scripts/z_abl_run_creation_profile.py
```
usage: z_abl_run_creation_profile.py [-h] [--base BASE] [--record RECORD]
                                     [--batting BATTING] [--scoring SCORING]
                                     [--logs LOGS] [--pbp PBP] [--teams TEAMS]
                                     [--out OUT]

ABL Run Creation Profile.

options:
  -h, --help         show this help message and exit
  --base BASE        Base directory for CSVs.
  --record RECORD    Override team record file.
  --batting BATTING  Override team batting file.
  --scoring SCORING  Override scoring detail file.
  --logs LOGS        Override team log file for games fallback.
  --pbp PBP          Override play-by-play log for run detail.
  --teams TEAMS      Override team info file for names.
  --out OUT          Output CSV path.
```

## csv/abl_scripts/z_abl_run_prevention_dna.py
```
usage: z_abl_run_prevention_dna.py [-h] [--base BASE] [--fielding FIELDING]
                                   [--pitching PITCHING] [--teams TEAMS]
                                   [--out OUT]

ABL Run Prevention DNA.

options:
  -h, --help           show this help message and exit
  --base BASE          Base directory for CSVs.
  --fielding FIELDING  Override fielding CSV.
  --pitching PITCHING  Override team pitching/defense CSV.
  --teams TEAMS        Override team info CSV.
  --out OUT            Output CSV path (relative to --base).
```

## csv/abl_scripts/z_abl_runways_streak_builders.py
```
usage: z_abl_runways_streak_builders.py [-h] [--base BASE]
                                        [--schedule SCHEDULE]
                                        [--records RECORDS] [--parks PARKS]
                                        [--teams TEAMS]
                                        [--out_next10 OUT_NEXT10]
                                        [--out_summary OUT_SUMMARY]
                                        [--today TODAY] [--window WINDOW]

ABL Runways/Streak Builders.

options:
  -h, --help            show this help message and exit
  --base BASE           Base directory.
  --schedule SCHEDULE   Override schedule file.
  --records RECORDS     Override team records file.
  --parks PARKS         Override park factors.
  --teams TEAMS         Override team info.
  --out_next10 OUT_NEXT10
                        Next games CSV.
  --out_summary OUT_SUMMARY
                        Summary CSV.
  --today TODAY         Override today date (YYYY-MM-DD).
  --window WINDOW       Number of games to scan.
```

## csv/abl_scripts/z_abl_season_backbone.py
```
usage: z_abl_season_backbone.py [-h] --season SEASON [--dry-run]

options:
  -h, --help       show this help message and exit
  --season SEASON  Season year (e.g. 1980)
  --dry-run        Run without writing output CSV
```

## csv/abl_scripts/z_abl_seed_prev_from_games_1981.py
```
usage: z_abl_seed_prev_from_games_1981.py [-h] [--asof ASOF]

Seed fact_team_reporting_1981_prev.csv from the regular-season game log.

options:
  -h, --help   show this help message and exit
  --asof ASOF  Cutoff date (YYYY-MM-DD) for including games. Defaults to
               1981-05-03.
```

## csv/abl_scripts/z_abl_series_miner.py
```
usage: z_abl_series_miner.py [-h] --start-date START_DATE --end-date END_DATE

ABL Series Miner

options:
  -h, --help            show this help message and exit
  --start-date START_DATE
                        Start date YYYY-MM-DD
  --end-date END_DATE   End date YYYY-MM-DD
```

## csv/abl_scripts/z_abl_sos_last14.py
```
usage: z_abl_sos_last14.py [-h] [--base BASE] [--logs LOGS]
                           [--min_sos_games MIN_SOS_GAMES] [--out OUT]

ABL strength of schedule (last 14 days).

options:
  -h, --help            show this help message and exit
  --base BASE           Base directory for CSVs.
  --logs LOGS           Explicit per-game logs CSV.
  --min_sos_games MIN_SOS_GAMES
                        Minimum opponent games required to include in SOS
                        average (default: 3).
  --out OUT             Output CSV (default:
                        out/csv_out/z_ABL_SOS_Last14.csv).
```

## csv/abl_scripts/z_abl_system_crash_slumps.py
```
usage: z_abl_system_crash_slumps.py [-h] [--base BASE] [--gamelogs GAMELOGS]
                                    [--totals TOTALS] [--teams TEAMS]
                                    [--out_current OUT_CURRENT]
                                    [--out_history OUT_HISTORY]
                                    [--window_pa WINDOW_PA]
                                    [--delta_thresh DELTA_THRESH]
                                    [--min_pa_season MIN_PA_SEASON]
                                    [--lookback_days LOOKBACK_DAYS]

ABL System Crash Slumps.

options:
  -h, --help            show this help message and exit
  --base BASE           Base directory.
  --gamelogs GAMELOGS   Override gamelog path.
  --totals TOTALS       Override totals path.
  --teams TEAMS         Override team info path.
  --out_current OUT_CURRENT
                        Current slumps CSV.
  --out_history OUT_HISTORY
                        History CSV.
  --window_pa WINDOW_PA
                        Rolling PA target.
  --delta_thresh DELTA_THRESH
                        OPS delta threshold (<= value flags slump).
  --min_pa_season MIN_PA_SEASON
                        Minimum season PA to consider.
  --lookback_days LOOKBACK_DAYS
                        History lookback in days (<=0 for all).
```

## csv/abl_scripts/z_abl_table_setter_clearer.py
```
usage: z_abl_table_setter_clearer.py [-h] [--base BASE] [--totals TOTALS]
                                     [--splits SPLITS] [--pbp PBP]
                                     [--roster ROSTER] [--teams TEAMS]
                                     [--min_pa MIN_PA]
                                     [--min_pa_leadoff MIN_PA_LEADOFF]
                                     [--min_pa_menon MIN_PA_MENON]
                                     [--show_all] [--out OUT]

ABL Table Setter vs Clearer report.

options:
  -h, --help            show this help message and exit
  --base BASE           Base directory for CSVs.
  --totals TOTALS       Override path for batting totals.
  --splits SPLITS       Override path for situational splits.
  --pbp PBP             Override path for play-by-play / at-bat data.
  --roster ROSTER       Override path for roster file.
  --teams TEAMS         Override path for team info file.
  --min_pa MIN_PA       Minimum PA to qualify.
  --min_pa_leadoff MIN_PA_LEADOFF
                        Minimum leadoff PA to compute share.
  --min_pa_menon MIN_PA_MENON
                        Minimum men-on PA to compute share.
  --show_all            Include players below thresholds.
  --out OUT             Output CSV path.
```

## csv/abl_scripts/z_abl_team_babip_luck.py
```
usage: z_abl_team_babip_luck.py [-h] [--base BASE] [--batting BATTING]
                                [--pitching PITCHING] [--teams TEAMS]
                                [--out OUT]

Compute team BABIP luck metrics.

options:
  -h, --help           show this help message and exit
  --base BASE          Base directory for CSVs.
  --batting BATTING    Override team batting file.
  --pitching PITCHING  Override team pitching file.
  --teams TEAMS        Override team info file.
  --out OUT            Output CSV path.
```

## csv/abl_scripts/z_abl_team_reporting_view.py
```
usage: z_abl_team_reporting_view.py [-h] [--dry-run]

options:
  -h, --help  show this help message and exit
  --dry-run   Run without writing output CSV
```

## csv/abl_scripts/z_abl_team_season_backbone_1981.py
```
usage: z_abl_team_season_backbone_1981.py [-h] [--dry-run]

options:
  -h, --help  show this help message and exit
  --dry-run   Run without writing outputs
```

## csv/abl_scripts/z_abl_travel_fatigue.py
```
usage: z_abl_travel_fatigue.py [-h] [--base BASE] [--gamelogs GAMELOGS]
                               [--teams TEAMS] [--parks PARKS]
                               [--out_summary OUT_SUMMARY]
                               [--out_legs OUT_LEGS]
                               [--short_miles SHORT_MILES]
                               [--long_miles LONG_MILES]

ABL Travel Fatigue report.

options:
  -h, --help            show this help message and exit
  --base BASE           Base directory.
  --gamelogs GAMELOGS   Override game log path.
  --teams TEAMS         Override team info path.
  --parks PARKS         Override park info path.
  --out_summary OUT_SUMMARY
                        Summary CSV path.
  --out_legs OUT_LEGS   Legs CSV path.
  --short_miles SHORT_MILES
                        Short-haul threshold in miles.
  --long_miles LONG_MILES
                        Long-haul threshold.
```

## csv/abl_scripts/z_abl_validate_eb_pack_any.py
```
usage: z_abl_validate_eb_pack_any.py [-h] --season SEASON
                                     [--league-id LEAGUE_ID]

Validate EB pack against almanac sources.

options:
  -h, --help            show this help message and exit
  --season SEASON
  --league-id LEAGUE_ID
```

## csv/abl_scripts/z_abl_viz_export_1981.py
```
usage: z_abl_viz_export_1981.py [-h] [--dry-run]

options:
  -h, --help  show this help message and exit
  --dry-run   Run without writing output CSVs
```

## csv/abl_scripts/z_abl_week_miner.py
```
usage: z_abl_week_miner.py [-h] --start-date START_DATE [--end-date END_DATE]

ABL Week Miner

options:
  -h, --help            show this help message and exit
  --start-date START_DATE
                        Start date YYYY-MM-DD
  --end-date END_DATE   End date YYYY-MM-DD (inclusive)
```

## csv/abl_scripts/z_abl_weekly_change_1981.py
```
usage: z_abl_weekly_change_1981.py [-h] [--dry-run]

options:
  -h, --help  show this help message and exit
  --dry-run   Run without writing output CSV
```

## csv/abl_scripts/z_abl_weekly_league_report.py
```
usage: z_abl_weekly_league_report.py [-h] --year YEAR --week WEEK

ABL Weekly League Report

options:
  -h, --help   show this help message and exit
  --year YEAR
  --week WEEK
```

## csv/abl_scripts/z_abl_whiff_merchants.py
```
usage: z_abl_whiff_merchants.py [-h] [--base BASE] [--pitching PITCHING]
                                [--pitchdetail PITCHDETAIL] [--teams TEAMS]
                                [--roster ROSTER] [--out OUT]
                                [--min_ip_sp MIN_IP_SP]
                                [--min_ip_rp MIN_IP_RP]
                                [--min_pitches_total MIN_PITCHES_TOTAL]
                                [--min_pitches_type MIN_PITCHES_TYPE]

ABL Whiff Merchants report.

options:
  -h, --help            show this help message and exit
  --base BASE           Base directory for CSVs.
  --pitching PITCHING   Override path for pitching totals.
  --pitchdetail PITCHDETAIL
                        Override path for pitch-type summary.
  --teams TEAMS         Override path for team info.
  --roster ROSTER       Override path for roster file.
  --out OUT             Output CSV path.
  --min_ip_sp MIN_IP_SP
                        Minimum IP for SP/Swing qualification.
  --min_ip_rp MIN_IP_RP
                        Minimum IP for RP qualification.
  --min_pitches_total MIN_PITCHES_TOTAL
                        Minimum total pitches for CSW stability.
  --min_pitches_type MIN_PITCHES_TYPE
                        Minimum pitches for top pitch consideration.
```

## csv/abl_scripts/z_abl_zone_rating_spotlight.py
```
usage: z_abl_zone_rating_spotlight.py [-h] [--base BASE] [--fielding FIELDING]
                                      [--teams TEAMS] [--roster ROSTER]
                                      [--out OUT]

ABL Zone Rating Spotlight.

options:
  -h, --help           show this help message and exit
  --base BASE          Base directory.
  --fielding FIELDING  Override fielding path.
  --teams TEAMS        Override team info path.
  --roster ROSTER      Override roster path.
  --out OUT            Output CSV path.
```
