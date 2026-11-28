# ABL Curated Current Layer Design

The curated layer is the only intended direct data interface for the future story engine. Every table carries `season`, `as_of_date`, `source_batch_ids`, `built_at`, and validation status.

| Table | Purpose | Source families / likely files | Grain and keys | As-of requirement | Story uses |
|---|---|---|---|---|---|
| `curated_current_games` | Completed authoritative results | `ootp_csv`: `games.csv`, `games_score.csv`, `game_logs.csv` | game; `game_id` | completed rows through raw-proved cutoff; sources reconcile | results, streaks, movement, series state |
| `curated_current_standings` | Recomputed records and race position | raw games plus team/division dimensions | team-season-cutoff; `season,team_id,as_of_date` | games/team must match promoted batch | divisions, races, pressure |
| `curated_current_team_stats` | Team offense, pitching, defense | raw team stats plus sortable team reports | team-season-cutoff | report batch linked to same raw cutoff; G reconciles | identity, strengths, weaknesses |
| `curated_current_player_batting` | Current batting performance | raw career batting totals plus sortable batting stats | player-team-season/split | season/league/split filtered; batch linkage | leaders, burden, form |
| `curated_current_player_pitching` | Current pitching performance | raw career pitching totals plus sortable pitching reports | pitcher-team-season/split | workload and batch aligned to raw cutoff | aces, bullpen, workload |
| `curated_current_player_ratings` | Batting/pitching/fielding ratings | sortable rating reports; raw ratings | player-capture | explicit capture batch; no inferred game date | talent/context, matchup traits |
| `curated_current_player_indicative` | Misc and indicative attributes | sortable indicative/misc reports | player-capture | explicit capture batch and player join | role, profile, context |
| `curated_current_team_context` | Finance, park, personality, market, transactions | sortable finance/personality/park; raw team/financial files | team-capture or event | distinguish slow context from dated events | fans, organization, ballpark |
| `curated_current_staff` | Managers/coaches and validated tendencies | raw coaches/roster staff; sortable staff | team-staff-capture | staff identity current to batch; tendencies match games/team | management stories |
| `curated_current_schedule` | Future unplayed schedule | raw `games.csv` and schedule structures | scheduled game; `game_id` | rows after as-of clearly marked unplayed | matchup/series stakes |
| `curated_historical_context` | Normalized prior seasons and champions | historical almanac and raw history | historical entity/event-season | immutable season/league identity; never current authority | echoes, rematches, precedent |

## Promotion behavior

Curated builds read only valid registered batches. They fail closed on date, schema, join, or games-count disagreement. No story output, manual overlay, or prior curated table may override the current raw game batch.

