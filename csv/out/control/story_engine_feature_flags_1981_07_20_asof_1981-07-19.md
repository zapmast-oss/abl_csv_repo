# Story Engine Feature Flags — July 20, 1981

> **StatsPlus enhances. It does not replace.**  
> **StatsPlus annotates before it ranks.**

| Feature flag | Default |
|---|---:|
| `enable_feed3_evidence_preview` | `true` |
| `enable_feed3_rank_adjustment` | `false` |
| `enable_feed3_new_candidates` | `false` |
| `enable_owner_front_office_signals` | `false` |
| `enable_best_game_discovery_signals` | `false` |
| `enable_historical_fan_interpretation` | `false` |

Feed 3 evidence preview appends typed evidence only under `csv/out/story/enrichment/`. Ranking changes and new-candidate creation are disabled by default. Owner/front-office traits, best-game discovery, historical fan interpretation, and dual odds views as ranking signals remain reference-only.

The preview does not run the July 20 story generator and does not overwrite candidates, evidence, menus, editorial products, production packages, or scripts.
