# ABL Story Engine Sprint 1 Notes

## Scope

Sprint 1 implements a narrow, reproducible path from the latest date-filterable 1981 data to structured story candidates, ranked menu, evidence table, and Baseball Observer packet. The authoritative catalog at `csv/out/docs/abl_data_catalog.csv` is read-only and is used to validate every selected source path. The current cutoff is July 12, 1981: 89 completed regular-season games per ABL team, treated operationally as Week 15.

## Catalog-selected source map

| Need | Best existing CSV | Sprint 1 use |
|---|---|---|
| Standings / division race | `csv/ootp_csv/games.csv`, `teams.csv`, `divisions.csv` | Primary; records recomputed through the cutoff |
| Weekly movement | `csv/ootp_csv/games.csv` | Primary; completed games in the trailing seven-day window |
| Team strength / weakness | `csv/ootp_csv/games.csv` | Runs scored/allowed and Pythagorean gaps recomputed through the cutoff |
| Player leader stories | `csv/ootp_csv/players_career_batting_stats.csv`, `players.csv` | Current 1981 league-200 totals with playing-time qualification |
| Pitcher / ace stories | `csv/ootp_csv/players_career_pitching_stats.csv`, `players.csv` | Current workload-qualified starter WAR and calculated ERA |
| Manager tendencies | `csv/out/csv_out/z_ABL_Manager_Tendencies.csv`; `csv/out/star_schema/fact_manager_scorecard_1981_current.csv` | Identified but excluded because the available tendency snapshot is only at 62 games |
| Matchup / series stakes | `csv/ootp_csv/games.csv`, `teams.csv`, `divisions.csv` | Unplayed regular-season games in the seven days after the cutoff |
| Historical / almanac echoes | `csv/out/almanac/1980/league_champions_1980_league200.csv`; `flashback_story_candidates_1980_league200.csv` | Sprint 1 links current standings to the prior champion table |

## Run

```powershell
python csv/abl_scripts/z_abl_story_engine_sprint1_run.py --week-label 1981_week_15 --as-of 1981-07-12
```

The runner stops on any child failure. The signal generator verifies all selected input paths against the existing authoritative catalog and does not alter that catalog.

## Outputs

- `csv/out/story/candidates/story_candidates_1981_week_15.csv`
- `csv/out/story/candidates/story_evidence_1981_week_15.csv`
- `csv/out/story/menus/story_menu_1981_week_15.csv`
- `csv/out/story/packets/baseball_observer_packet_1981_week_15.json`
- `csv/out/story/packets/baseball_observer_packet_1981_week_15.md`

## Detector behavior

- Division races require the top two teams to be within four wins in standings recomputed from raw games.
- Weekly rise/fall uses completed games in the seven days ending at the cutoff.
- Team pressure uses Pythagorean win gaps calculated from cutoff-filtered runs and explicitly avoids claims of luck or inevitable regression.
- Player stories require at least 150 plate appearances.
- Ace stories require at least ten starts and 80 innings.
- Matchup candidates are explicitly unplayed scheduled games in the next seven days.
- Historical echoes join a current team to its 1980 champion-table row and never treat history as a forecast.

## Known limits and next work

- “Week 15” is an operational label supplied by the project; the data-grounded facts are the July 12 cutoff and 89 games per team.
- Wild-card and tournament arithmetic are not emitted because no verified 1981 qualification-rules table was found.
- Manager signals remain disabled until tendencies can be rebuilt through the current cutoff.
- The Sunday matchup composite is accepted as existing evidence but is not yet decomposed into independently validated components.
- Team name joins across current and almanac files work for the tested data; stable franchise/team IDs should replace name linkage.
- Sprint 2 should add as-of manifests, run-specific output directories, team profile adapters, series state, candidate deduplication, and editorial selection state.
