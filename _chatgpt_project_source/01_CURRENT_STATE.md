# Current State

**Last verified:** 2026-07-07. **Git:** clean worktree before this source pack; branch `add-star-schema`; HEAD `dbe9355`.

## Authoritative current window

| Item | Current fact | Evidence |
|---|---|---|
| League/world | Action Baseball League, 1981 | Current output names and preflight |
| Editorial phase | Post–All-Star-break, Act 3 / “After the Break” | User-supplied project identity; consistent with July 20 package |
| Newsroom date | 1981-07-20 | `csv/out/control/current_state_preflight_1981_target_1981-07-19.md` |
| Completed-game cutoff | 1981-07-19 | Same preflight |
| Coverage completeness | 1,115 games; 24 teams; 92–93 games/team | Same preflight |
| Preflight verdict | `READY_FOR_CURRENT_RUN` | Same preflight |
| Official authority | Raw OOTP for games/standings/current state | Source manifest and governance docs |
| StatsPlus status | 25 files promoted; preview enrichment only | Current-source status and adapter validation |

## What has just been produced

Recent commits on July 4–5, 2026 produced or refined:

- the July 20 stakes-first framing notes and Baseball Observer “Pressure Map” segment;
- a Chicago–Dallas evidence packet, story-slate addendum, production setup, and 75–85 second Ballpark Feed intro;
- a parameterized Feed 3 preview runner and validation artifacts;
- StatsPlus current promotion, team/player enrichment, signal dictionary, and combined evidence previews;
- scripts supporting the Chicago–Dallas investigation and StatsPlus preview workflow.

The latest commit removed only an obsolete Microsoft Word temporary lock file. It did not remove the real framing-notes document.

## Current deliverables

| File | State | Use |
|---|---|---|
| `csv/out/story/scripts/baseball_observer_segment_stakes_first_1981_07_20_asof_1981-07-19.md` | Current voice pass | Stakes-first EB segment; rankings unchanged |
| `csv/out/story/production/july20_stakes_first_framing_notes_1981_07_20_asof_1981-07-19.md` | Current production notes | Pressure-map framing and cautions |
| `csv/out/story/scripts/baseball_observer_segment_1981_07_20_asof_1981-07-19.md` | Current baseline | Safe-for-use Observer script |
| `csv/out/story/production/eb_production_package_1981_07_20_asof_1981-07-19.md` | Current baseline package | Lead, secondary, watch, held, and format guidance |
| `csv/out/story/scripts/chicago_dallas_ballpark_feed_intro_1981_07_20_asof_1981-07-19.md` | Current pregame script | Chicago–Dallas intro; no July 20 result |
| `csv/out/story/production/chicago_dallas_ballpark_feed_setup_1981_07_20_asof_1981-07-19.md` | Current production setup | Series arithmetic, history, cautions |
| `csv/out/story/investigations/chicago_dallas_series_evidence_packet_1981_07_20_asof_1981-07-19.md` | Current evidence | Claim-level Chicago–Dallas support |
| `csv/out/story/enrichment/story_evidence_with_feed3_preview_1981_07_20_asof_1981-07-19.md` | Current preview, not official replacement | Combined raw-derived and StatsPlus evidence |

## Ready, unfinished, and review status

**Ready:** preflight is ready; official run summary says safe for EB/Baseball Observer editorial use; baseline production package says `SAFE FOR EB USE`; Feed 3 parameterized validation passes and leaves official artifacts untouched.

**Unfinished or gated:** final human editorial selection, factual spot-check, narration/performance, video/post assembly, publishing, and promotion are not evidenced as completed. StatsPlus ranking effects, new-candidate creation, owner signals, best-game automation, and historical-fan automation remain disabled. Wild-card/tournament arithmetic remains unavailable without verified rules.

## What is not current

- `csv/story_*_1981_week_05.csv`, `csv/out/core12_1981_w05.json`, Week 5 forum/video files, and Week 5 story packets are historical examples.
- Week 15 outputs using cutoff July 12 and 89 games/team are superseded for July 20 current-state work.
- `csv/out/star_schema/monday_1981_standings_by_division.csv` and related 32-game player/team snapshots are stale.
- manager tendencies, division leverage, and rotation reports identified as 62-game snapshots are stale.
- almanac seasons 1972–1980 are history, not 1981 current state.

## Next likely step

Human-review the stakes-first Observer segment and Chicago–Dallas intro against their evidence/caution notes, choose the final read, prepare narration and video/post assets, then publish and archive the July 20 checkpoint. If new game exports arrive, create a new dated preflight rather than silently updating this package.

