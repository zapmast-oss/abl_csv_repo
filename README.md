# abl_csv_repo

Repo of ABL data

## Overview

This repository is part of the **SimBaseballVision (SBV)** network, focusing on refining and auditing the output structure of our Action Baseball League (ABL) simulation data.

This branch, `refactor-output-audit`, is specifically for:

- Reorganizing CSV and text report outputs
- Ensuring files are structured consistently
- Making it easier to audit, validate, and feed downstream storytelling (EB, Producer, etc.)

All refactored outputs live under `csv/out`.

---

## Setup & Usage

### 1. Clone and checkout

```bash
git clone <your-repo-url> abl_csv_repo
cd abl_csv_repo
git checkout refactor-output-audit
2. Data inputs
Export league data from OOTP into csv/ootp_csv/.

(Optional) Export sortable stats into csv/abl_statistics/.

Ensure the season you care about (1981, etc.) is fully up to date in the exports.

3. Running pipelines (high level)
Star schema / core facts & dims
Use the scripts in csv/abl_scripts/ (for example, build_star_schema.py and the z_abl_* ETL scripts) to populate:

csv/out/star_schema/

data_work/abl.db (SQLite)

Text reports & EB packs
Run the analytic z_abl_*.py scripts in csv/abl_scripts/ to generate:

csv/out/text_out/ (z_ABL_*.txt reports)

Monday packs and other show-notes CSVs in csv/out/star_schema/

Story menus
Use the root scripts/ entry points with the csv/story_*.csv files to:

Evaluate story triggers

Build and print weekly story menus

(Exact command sequences can be wired into .vscode/tasks.json and _tmp_run_all.py as needed.)

Directory Structure (High-Level)
.vscode/ – VS Code tasks and keybindings for this repo

csv/ – Core workspace for all ABL data, scripts, and outputs

data_raw/ – Reserved for untouched raw snapshots (currently empty)

data_work/ – Working data, including abl.db (SQLite star schema)

logs/ – Logs for orchestration scripts (e.g., _tmp_run_all.log)

scripts/ – Top-level story menu drivers

Prompt Eng/ – Placeholder for prompt-engineering experiments

fantbbexpert/ – Placeholder for fantasy baseball experiments

Contributing
Contributions are welcome via pull requests, especially for:

Improvements to the refactor / audit process

New validation or consistency checks

Enhancements to analytics and text reports

Better documentation of pipelines and story systems

Please keep changes consistent with the existing structure under csv/out and the star-schema model in csv/out/star_schema/ and data_work/abl.db.

Appendix A – Detailed Repo Map (as of 2025-11-25_2005)
This appendix documents the current structure and intent of each major component in the repo.

1. Root
Root of the repo:

.git/ – full git history

.env.example

.gitattributes, .gitignore

.vscode/

README.md

Prompt Eng/

fantbbexpert/

csv/

data_raw/

data_work/

logs/

scripts/

2. The Core: csv/
This is the main workspace.

Inside csv/:

ootp_csv/ – raw exports from OOTP

abl_scripts/ – Python scripts (ETL, analysis, reporting)

abl_statistics/ – large “sortable stats” CSV dumps from OOTP

abl_csv/ – small summary CSVs (last-10, matchups, week miner, etc.)

out/ – all outputs: star schema, EB packs, viz tables, text reports

docs/ – markdown docs/templates (It’s Monday structure, matchup intros)

story_candidates_1981_week_05.csv

story_dictionary.csv

story_menu_1981_week_05.csv

story_menu_1981_week_07.csv

_tmp_run_all.py

week_game_ids.npy

2.1 csv/ootp_csv/ – Raw League Exports
Canonical ABL data straight from OOTP.

Typical contents include:

games.csv, game_logs.csv, games_score.csv

teams.csv, team_record.csv, team_history_*.csv

players.csv, player_batting_stats.csv, player_pitching_stats.csv, etc.

league_history_*.csv

team_financials.csv, team_roster.csv, team_roster_staff.csv

Other standard OOTP exports

Purpose:
Raw ingredients. All star-schema dimensions and facts ultimately derive from here.

2.2 csv/abl_scripts/ – Script Toolbox
This folder contains the main Python toolbox for the ABL pipelines.

Utilities (00_*.py, 99_*.py):

00_preview_csv.py, 00_prompt_wizard_aim_maap.py, etc.

99_inspect_csvs.py

99_list_teams.py

99_make_standings_snapshot.py

99_preview_chi_at_mia.py

…and similar helper scripts.

Core config / helpers:

abl_config.py

abl_team_helper.py

abl_team_code_lookup.py

abl_week_miner.py

abl_team_report.py, abl_team_summary.py

abl_player_leaders.py, abl_top_players.py

abl_power_rankings_forum.py

abl_show_notes.py

abl_weekly_forum_post.py

etc.

Star schema / backbone / ETL (selected z_abl_*.py):

z_abl_dim_ballparks.py

z_abl_dim_managers.py

z_abl_team_season_backbone_1981.py

z_abl_team_reporting_view.py

z_abl_weekly_change_1981.py

z_abl_viz_export_1981.py

z_abl_1980_season_backbone.py

build_star_schema.py (ETL helper, separate file in this folder)

Analytic “DNA” scripts (selected z_abl_*.py):

z_abl_basepath_pressure.py

z_abl_blowout_resilience.py

z_abl_bullpen_stress_index.py

z_abl_catcher_battery_value.py

z_abl_damage_with_risp.py

z_abl_division_leverage.py

z_abl_sos_last14.py

z_abl_system_crash_slumps.py

z_abl_table_setter_clearer.py

z_abl_team_babip_luck.py

z_abl_travel_fatigue.py

z_abl_whiff_merchants.py

z_abl_zone_rating_spotlight.py

…and others in the same style.

Packaging / EB / identity:

z_abl_team_identity.py

z_abl_team_identity_pack_1981.py

z_abl_team_aces_1981.py

z_abl_30for30_1981_pythag.py

z_abl_eb_pack_1981_monday.py (also referenced with a csv/ prefix in some listings)

Purpose:
This folder is the engine room. It takes ootp_csv → builds dims/facts → produces csv/out/star_schema CSVs → generates text reports and EB packs.

2.3 csv/out/ – Generated Outputs
Inside csv/out/:

abl_power_rankings.csv

abl_power_rankings_top9.csv

archive/

csv_out/

star_schema/

text_out/

2.3.1 csv/out/star_schema/ – The Brain
This is the heart of the data model.

Key dimension tables:

dim_player_profile.csv

dim_player_batting_ratings.csv

dim_player_fielding_ratings.csv

dim_player_pitching_ratings.csv

dim_team_park.csv

dim_team_staff.csv

face_of_franchise_1981.csv (popularity-based faces of each franchise)

Key fact tables:

fact_player_batting.csv

fact_player_fielding.csv

fact_player_pitching.csv

fact_schedule_1980.csv

fact_team_batting.csv

fact_team_catcher_fielding.csv

fact_team_financials.csv

fact_team_pitching.csv

fact_team_record.csv

fact_team_reporting_1981_current.csv

fact_team_reporting_1981_prev.csv

fact_team_reporting_1981_weekly_change.csv

fact_team_reporting_view.csv

fact_team_season_1981_backbone.csv

fact_team_season_1981_pythag_report.csv

fact_team_standings.csv

fact_team_standings_with_managers.csv

Monday / show outputs & visualization tables:

monday_1981_power_ranking.csv

monday_1981_risers.csv

monday_1981_fallers.csv

monday_1981_show_notes.csv

monday_1981_standings_by_division.csv

prep_1981_CHI_at_MIA.csv

team_aces_1981.csv

abl_1981_30for30_pythag_report.csv

viz_1981_pythag_bar.csv

viz_1981_power_vs_run_diff.csv

viz_1981_weekly_change_bar.csv

Purpose:
If EB, Producer, or Tech needs a trusted table, it lives here. These are the preferred sources for stories, visuals, and reporting—not the raw OOTP exports.

2.3.2 csv/out/text_out/ – EB Packs & Written Reports
This directory is full of z_ABL_*.txt text files, including (sample):

z_ABL_Deep_Dive_25.txt

z_ABL_League_Weather_Report.txt

z_ABL_Monday_Notebook.txt

z_ABL_Week_Miner.txt

z_ABL_Schedule_Heatmap.txt

z_ABL_SOS_Last14.txt

z_ABL_Team_Report.txt

z_ABL_Runways_Next10_Games.txt

z_ABL_Run_Prevention_DNA.txt

z_ABL_System_Crash_Slumps_Current.txt

…and many more.

Purpose:
Script-ready / EB-ready narrative feeds coming off the analytics. Many map directly to the z_abl_*.py scripts in csv/abl_scripts/.

2.3.3 csv/out/csv_out/ and csv/out/archive/
csv/out/csv_out/season_1981/ – currently empty; intended as a place to drop published CSV slices by season/week for consumption or archiving.

csv/out/archive/ – storage for older or frozen outputs from prior runs.

Purpose:
Organized, export-friendly CSVs and historical archives.

2.4 csv/abl_statistics/ – Sortable Stats Dumps
Contains large OOTP “Sortable Stats” exports, such as:

abl_statistics_player_splits_-_vs_left_players.csv

abl_statistics_player_splits_-_vs_right_players.csv

abl_statistics_player_splits_-_vs_teams.csv

abl_statistics_player_statistics_-_career_batting_stats.csv

abl_statistics_player_statistics_-_career_pitching_stats.csv

Multiple player_statistics_-_sortable_stats_* CSVs

Multiple team_statistics___info_-_sortable_stats_* CSVs

Purpose:
Raw/medium-processed stats for deeper work (career arcs, splits, etc.), outside the core star schema pipeline.

2.5 csv/abl_csv/ – Small Convenience CSVs
Contains:

abl_last10.csv

abl_matchup_ratings.csv

abl_sunday_matchups.csv

abl_team_bat_baserunning.csv

abl_week_miner.csv

standings_snapshot.csv

Purpose:
One-step-up summary tables for matchups, last-10 form, baserunning, etc. Useful as direct inputs to show notes, forum posts, or quick analysis.

2.6 csv/docs/ – Editorial Notes & Templates
Contents:

abl_relaunch_season10_eb.md

its_monday_video_structure.md

matchups/1981_week6_DAL_at_CHA_intro.md

templates/matchup_intro_template.md

Purpose:
Editorial / EB playbook: how to structure “It’s Monday,” relaunch posture for Season 10, and matchup intro templates.

2.7 csv/story_*.csv – Story Menu System
Files:

story_dictionary.csv

story_candidates_1981_week_05.csv

story_menu_1981_week_05.csv

story_menu_1981_week_07.csv

Purpose:

story_dictionary.csv – defines story triggers and categories

story_candidates_* – candidate storylines for a given week

story_menu_* – the curated story menu promoted for EB/Producer for that week

2.8 _tmp_run_all.py and week_game_ids.npy
_tmp_run_all.py – orchestration helper script to run a full Monday pipeline (dims/facts + analytics + EB pack).

week_game_ids.npy – NumPy file containing game IDs for a given week, used by miners or matchup prep.

Purpose:
A convenience entry point and data helper for weekly automation.

3. Outside csv: Data, Logs, Scripts
3.1 data_work/abl.db
data_work/abl.db – SQLite database.

Purpose:
Database form of the star schema (facts/dims), likely mirroring the contents of csv/out/star_schema/ for SQL-based analysis.

3.2 data_raw/
Currently present but empty.

Purpose:
Intended for untouched raw data snapshots (e.g., original exports before any processing).

3.3 logs/
Contains:

logs/_tmp_run_all.log – log output from the orchestration script.

Purpose:
Debug and audit record of pipeline runs.

3.4 Root scripts/
Contains:

build_story_menu_for_week.py

eval_story_triggers_for_week.py

print_story_menu.py

Purpose:
Top-level drivers for the story system. These work with the csv/story_*.csv files and the star schema to evaluate triggers, build, and print weekly story menus.

3.5 .vscode/, Prompt Eng/, fantbbexpert/
.vscode/ – tasks.json, keybindings.json for VS Code automation and hotkeys.

Prompt Eng/ – currently empty; reserved for prompt-engineering artifacts.

fantbbexpert/ – currently empty; reserved for fantasy baseball–related experiments.

Purpose:
Editor automation and placeholders for related SBV workstreams.