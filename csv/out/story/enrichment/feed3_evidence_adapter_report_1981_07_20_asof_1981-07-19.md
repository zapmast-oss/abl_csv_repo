# Feed 3 Evidence Adapter Report

| Feature flag | Value |
|---|---|
| enable_feed3_evidence | true |
| enable_feed3_rank_adjustment | false |
| enable_feed3_new_candidates | false |
| enable_owner_front_office_signals | false |
| enable_best_game_discovery_signals | false |
| enable_historical_fan_interpretation | false |

- Candidates enriched: **15**.
- Feed 3 evidence records: **53**.
- Evidence types used: statsplus_baseruns, statsplus_elo, statsplus_fan_interest_context, statsplus_financial_context, statsplus_injury_context, statsplus_pitcher_rwar, statsplus_playoff_odds, statsplus_team_baserunning, statsplus_team_war, statsplus_ubr.
- Evidence types held back for lack of candidate fit: statsplus_player_baserunning, statsplus_team_age_context.
- Reference-only signals: owner/front-office traits, best-game discovery tables, historical fan interpretation, dual playoff-odds views as separate ranking signals.
- Ranking changed: **No**.
- New candidates created: **No**.
- Official story artifacts untouched: **Yes**.

Recommended next task: Add this adapter behind an explicit preview flag in the master runner, validate on a later newsroom date, and retain ranking adjustment/new-candidate flags as false.
