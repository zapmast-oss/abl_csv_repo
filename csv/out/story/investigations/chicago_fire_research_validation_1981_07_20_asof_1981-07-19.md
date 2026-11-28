# Chicago Fire Research Validation - July 20, 1981 Pregame Cutoff

- Regular-season records were recalculated from almanac game CSVs for 1972-1980 and raw `games.csv` for 1981 through July 19.
- Playoff qualification was reconciled against `team_history.csv` and `team_history_record.csv`.
- Series results were reconciled against OOTP `messages.csv`; Grand Series games were cross-checked against generated Grand Series summaries and archived HTML where parsed.
- Manager tenure was reconciled against `team_history.csv`, `team_roster_staff.csv`, and `coaches.csv`.
- Postseason games were not included in regular-season totals because historical regular-season tables came from almanac game CSVs and 1981 was filtered to `game_type=0`.
- The 1981 cutoff excludes dates after 1981-07-19.
- Contradiction found: historical almanac regular-season game CSVs do not include postseason games; postseason evidence had to come from OOTP messages and archived HTML.
- Unresolved: exact Matt Mead hire date was not found; some non-Grand-Series individual postseason game rows may remain unrecovered if archive parsing missed boxes.

One-manager/five-playoff-trips claim: PROVEN.