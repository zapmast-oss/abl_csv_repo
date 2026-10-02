# ABL Almanac Mining Plan – 1972 Baseline

Measure twice, cut once. This document defines how to mine the OOTP Almanac (starting with 1972) into story-ready data for ABL Flashback, ABL Legacy, and EB’s reports.

---

## 1. Objectives

**Primary goal:**  
Turn giant OOTP almanac exports (~2GB) into a small, clean set of CSVs and views that let us “fly the league” at multiple altitudes:

- **5,000 ft – League-era view**
  - Big eras, dynasties, historical context.
- **4,000 ft – Conference / division view**
  - Which halves of the league were strong or weak, by year.
- **3,000 ft – Team-season view**
  - Each team’s yearly arc: early, mid, pennant race, playoffs.
- **2,000 ft – Month / half-season view**
  - First-half vs second-half, monthly swings.
- **1,000 ft and below – Week / series view**
  - Specific turning points, streaks, and key series.

We **never** ask Codex/ChatGPT to read the whole 2GB. Instead:

> Almanac stays big on disk → parsers turn it into tidy CSVs → AI only sees small, curated tables.

---

## Turnkey EB pipeline (preferred for any season)

If you want the full EB-ready regular-season pack for a single season, run from the repo root:

```
python csv/abl_scripts/_run_eb_regular_season_any.py --season 1980 --league-id 200
```

That script now handles everything end to end: it extracts the core almanac HTML, builds league summaries, ensures time-slice score and *_enriched CSVs via `_run_almanac_time_slices_any.py`, then builds momentum layers, the flashback story pack, EB briefs, player context (players, schedule, financials), and schedule context. Use this as the default unless you are debugging individual parsers.

---

## 2. 1972 Almanac Structure (from `almanac_1972.zip`)

Top-level folder inside the zip:

- `almanac_1972/`

Within that, main groups of files:

1. **`box_scores/`** – 9,169 files  
   - Example: `game_box_1.html`, `game_box_10.html`, `game_box_100.html`, `game_box_1000.html`  
   - **Use:** Game-level details (R/H/E, innings, line scores, players, etc.) for deep dives and key-game stories.

2. **`coaches/`** – 914 files  
   - Example: `coach_1.html`, `coach_10.html`, `coach_100.html`, `coach_101.html`  
   - **Use:** Manager / coach histories, future EB manager profiles, narratives about staffs.

3. **`game_logs/`** – 9,169 files  
   - Example: `log_1.html`, `log_10.html`, `log_100.html`, `log_1000.html`  
   - **Use:** Compact game summaries with dates, opponents, scores — perfect for series/month/weekly aggregates.

4. **`history/`** – 23,719 files  
   - Example:  
     - `index.html`  
     - `league_200_0_0_leaderboards.html`  
     - `league_200_0_0_leaderboards_post.html`  
     - `league_200_0_10_leaderboards.html`  
   - **Use:** Leaderboards, historical context, awards, records.

5. **`images/`** – 7,897 files  
   - Example: `base_0_0_0.png`, `base_0_0_1.png`, `base_0_1_0.png`, `base_0_1_1.png`  
   - **Use:** Visuals for eventual web / video integration (not essential for the data pipeline).

6. **`leagues/`** – 4,559 files  
   - Example:  
     - `league_200_all_transactions_0_0.html`  
     - `league_200_available_coaches_page.html`  
     - `league_200_batting_report.html`  
     - `league_200_draft_log_0.html`  
   - **Critical files for 1972 ABL (league_id = 200):**  
     - `league_200_standings.html`  
     - `league_200_scores.html` (if present)  
     - `league_200_stats.html`, batting/pitching reports  
   - Other league IDs (201–205) are minors or other leagues. **ABL canon is league_200 only.**

7. **`players/`** – 5,592 files  
   - Example: `player_1.html`, `player_10.html`, `player_100.html`, `player_1000.html`  
   - **Use:** Individual player pages, histories, awards — good for deep EB features.

8. **`teams/`** – 44,185 files  
   - Example: `team_1.html`, `team_10.html`, `team_100.html`, `team_100_all_transactions_0_0.html`  
   - **Use:** Team-specific pages, season history, transactions, etc. Ideal for team-season profiles.

Additional root files:

- `index.html` – BNN index landing page.  
- `styles.css` – Shared CSS; not needed for parsing.

---

## 3. Altitude Layers – What We Want to Build

The almanac is the raw lake; we’re going to build “views” at different altitudes. All of these will live as CSVs inside the `csv/out/` or `csv/out/almanac/` area and will be joinable to star-schema tables (e.g. `dim_team_park`, `dim_players`, etc.).

### 3.1 5,000 ft – League-Season Summary

**Target table:** `fact_league_season_summary.csv`  
**Grain:** 1 row per (season, league_id = 200)

Columns (example):

- season, league_id  
- league_name  
- champion_team_id / champion_team_name  
- runner_up_team_id / runner_up_team_name  
- best_regular_season_record_team (W-L, pct)  
- league_run_diff_leader_team  
- average_runs_per_game, HR_per_game, etc.  
- notable bullet flags: “first expansion”, “rule change”, etc. (could be encoded or stored separately).

**Sources:**

- `almanac_1972/leagues/league_200_standings.html`  
- `almanac_1972/history/league_200_*_leaderboards*.html`  
- Possibly `teams/` and `history/` summary pages.

---

### 3.2 4,000 ft – Conference / Division Summary

**Target table:** `fact_conf_div_season_summary.csv`  
**Grain:** 1 row per (season, league_id, conference/division)

Columns (example):

- season, league_id  
- conference_name, division_name  
- team_count  
- best_record_team, worst_record_team  
- aggregate W-L, RS, RA  
- top 2–3 players by WAR in that group (can be separate tables or JSON blobs).

**Sources:**

- `league_200_standings.html` (division tables)  
- Star-schema team mapping (e.g. `dim_team_park` → conference/division).

---

### 3.3 3,000 ft – Team-Season Profiles

**Target table:** `fact_team_season_profile.csv`  
**Grain:** 1 row per (season, team_id)

Columns (example):

- season, league_id, team_id, team_abbr, team_name  
- conf/div (via `dim_team_park`)  
- W, L, PCT, GB, Pythag record, Pythag diff  
- home/away splits  
- 1-run game record, extras record  
- first-half record, second-half record  
- monthly splits: W/L and RS/RA April–September  
- playoff result (missed, lost DS, lost CS, lost GS, won GS)  
- key flag fields (e.g. “franchise-best record”, “historic collapse”).

**Sources:**

- Standings (for W/L, PCT, GB, Pythag, splits).  
- Game logs + box scores (for month and half splits).  
- Playoff boxes/history pages.

---

### 3.4 2,000 ft – Monthly / Half-Season Splits

**Target table:** `fact_team_monthly_summary.csv`  
**Grain:** 1 row per (season, team_id, calendar_month)  
Optional: `fact_team_half_summary.csv` for first/second half.

Columns (example):

- season, team_id, month (YYYY-MM)  
- W, L, RS, RA  
- streaks: longest win streak, longest losing streak in that month  
- summary strings for EB: “15–4 in June, +42 run diff”.

**Sources:**

- `game_logs/` games → per-team monthly aggregates  
- `box_scores/` if needed for extra details.

---

### 3.5 1,000 ft – Week / Series-Level Stories

**Target tables:**

- `fact_series_summary.csv` – 1 row per (season, series_id / team matchup / date range)  
- `fact_team_weekly_summary.csv` – 1 row per (season, team_id, week_index)

Columns (example):

- For series:
  - season, home_team_id, away_team_id  
  - start_date, end_date, games, series_result (e.g. Home 3–1)  
  - series_run_diff, blowouts, shutouts, walkoffs, etc.

- For weekly team:
  - season, team_id, week_start_date  
  - W, L, RS, RA  
  - status flags (e.g. “season-changing week”).

**Sources:**

- `game_logs/` (primary)  
- `box_scores/` for deeper highlights.

---

## 4. Extract-Transform-Load (ETL) Strategy

We always follow this pattern:

1. **Extract (from almanac)**  
   - Start with 1972 as the baseline: `almanac_1972.zip`.  
   - Access key HTML files via `zipfile.ZipFile` (no need to unzip to disk if we don’t want to).  
   - For each file type (e.g., `league_200_standings.html`, game logs, etc.), write *one* parser script in `csv/abl_scripts/`.

2. **Transform (in Python)**  
   - Use `pandas.read_html` for tables where possible.  
   - Normalize column names: no guessing; use what’s in the HTML (`Team`, `W`, `L`, `PCT`, `GB`, `Pyt.Rec`, `Diff`, `Home`, `Away`, `XInn`, `1Run`, `M#`, `Streak`, `Last10`).  
   - Add canonical keys: `season`, `league_id`, and eventually `team_id` via joins to existing dimensions.

3. **Load (to star schema / reporting tables)**  
   - Output goes under `csv/out/almanac/1972/` (or similar).  
   - Separate “raw extracted” CSVs from “aggregated” CSVs.  
   - Aggregation scripts build 5,000 ft / 4,000 ft / 3,000 ft tables from the raw extracts.

---

## 5. First Script: Standings Extract (1972, league_200)

**Input (inside the zip):**

- `almanac_1972/leagues/league_200_standings.html`

**Output:**

- `csv/out/almanac/1972/standings_1972_league200.csv`

**Grain:**

- 1 row per team participating in the ABL in 1972.

**Columns (initial version):**

- `season` (e.g., 1972)  
- `league_id` (200 for ABL)  
- `team_name` (e.g., Chicago Fire)  
- `wins` (`W`)  
- `losses` (`L`)  
- `pct` (`PCT`)  
- `gb` (`GB`)  
- `pyt_rec` (`Pyt.Rec`)  
- `pyt_diff` (`Diff`)  
- `home_rec` (`Home`)  
- `away_rec` (`Away`)  
- `xinn_rec` (`XInn`)  
- `one_run_rec` (`1Run`)  
- `magic_num` (`M#`)  
- `streak` (`Streak`)  
- `last10` (`Last10`)

Division and conference can be joined later using `dim_team_park` or other dimensions, so the first pass doesn’t have to parse the division headings.

---

## 6. How This Feeds Flashback and Legacy

Once the pipeline exists:

- **Flashback** episodes (starting from 1980 backward) will primarily use:
  - `fact_league_season_summary` (5,000 ft)  
  - `fact_conf_div_season_summary` (4,000 ft)  
  - `fact_team_season_profile` (3,000 ft)  
  - Optionally, `fact_team_monthly_summary` and series/weekly tables for big turning points.

- **Legacy** episodes (starting from 1972 forward) will use the exact same tables but with:
  - More emphasis on “firsts,” league formation, and early dynasties.

EB’s writing prompts will reference these tables rather than raw HTML. Codex/ChatGPT will **never read the full almanac**, only these curated CSVs or slices of them.

---

## 7. Implementation Order

1. Implement `parse_standings_from_html.py` for 1972 → validate output.  
2. Generalize standings parser to work for any season + league (e.g. 1972–1981, league_200 only).  
3. Implement game-log parser → build monthly and weekly team summaries.  
4. Implement league-history / leaderboards parser → build league-season summary tables.  
5. Wire aggregation scripts that roll everything up into the 5,000 / 4,000 / 3,000 / 2,000 / 1,000 ft tables.  
6. Connect these tables to EB’s “story candidates” and ABL Flashback / Legacy episode templates.

Measure twice, cut once. 1972 is the proving ground; once the scripts are validated there, they can be applied to the rest of the ABL timeline.

---

## Running multiple seasons (1973-1980)

To run several seasons in one pass, use the range runner from the repo root:

```
python csv/abl_scripts/_run_eb_regular_season_range.py --start-season 1973 --end-season 1980 --league-id 200
```

This loops season by season via the turnkey runner above, logging any failures and continuing through the requested span.
