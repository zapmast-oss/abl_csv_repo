# Production Rhythm

The repo documents daily, series, weekly Monday, milestone, and historical cadences. The table below combines its governance pipeline with the editorial workflow requested for SBV. Publication-platform actions are workflow expectations; the repository does not prove that the July 20 material has been published.

| Step | Input | Action | Output | Human decision needed? | Files involved |
|---:|---|---|---|---|---|
| 1 | Coherent OOTP/sortable/StatsPlus export | Capture without hand-editing rows | Raw source batch | Yes: confirm intended cutoff | `csv/ootp_csv/`, `csv/abl_statistics/`, staged StatsPlus sources |
| 2 | Captured files | Register authority, role, volatility, joins, checksums | Registry and batch manifest | Review anomalies | `csv/out/control/abl_source_registry.*`, `abl_batch_manifest_current.*` |
| 3 | Games, scores, logs, teams | Run current-state preflight for season/as-of | Ready/not-ready verdict | Stop on non-ready verdict | `current_state_preflight_*` and preflight scripts |
| 4 | Validated completed games | Recompute standings, recent window, run balance | Current standings/race evidence | Choose relevant margins | OOTP drivers; story evidence |
| 5 | Versioned postseason rules | Check wild card / “if season ended today” | Qualification picture and protected DCS matchup projection | Required | `docs/ABL_POSTSEASON_RULES.md`; clinch/elimination claims still require standings arithmetic |
| 6 | Divisions/conferences and schedule | Review division margins, direct games, unplayed matchups | Race and matchup candidates | Select news value | `divisions.csv`, `sub_leagues.csv`, `games.csv` |
| 7 | Team/player totals and historical context | Detect rise/fall, value cases, pressure, historical echoes | Candidate/evidence/menu set | Apply hierarchy and restraint | `csv/out/story/candidates/`, `menus/`, almanac sources |
| 8 | Official candidates/evidence | Build editorial board/slate; mark lead, secondary, watch, hold | Editorial plan | Yes: final selection | `csv/out/story/editorial/` |
| 9 | Optional promoted StatsPlus data | Append typed preview evidence; verify hashes and feature flags | Enrichment preview | Decide whether context is useful | `csv/out/story/enrichment/`, `csv/out/control/feed3_*` |
| 10 | Selected slate and evidence | Draft Baseball Observer, Ballpark Feed, Daily 411, forum/article/video products as applicable | Production package and scripts | Yes: format and framing | `csv/out/story/production/`, `scripts/` |
| 11 | Draft plus source evidence | Factual/editorial review: dates, arithmetic, authority, uncertainty, naming, voice | Approved script | Yes, mandatory | Production cautions, evidence packet, style guide |
| 12 | Approved script | Prepare EB narration, pacing, pronunciation, chapters | Voice/narration assets | Yes: performance | Script and production notes; media assets are outside inspected repo evidence |
| 13 | Narration and visual plan | Assemble video/post/article | Publishable media/post | Yes | Platform project files are UNKNOWN/not located |
| 14 | Finished output | Upload/publish with correct title, description, cutoff, and labels | Public SBV content | Yes, external action | Not represented as completed in repo |
| 15 | Published link/content | Promote via forum/thread/social as appropriate | Audience update | Yes | Forum template/output where applicable |
| 16 | Final outputs and source hashes | Archive/checkpoint the run and decisions | Reproducible historical checkpoint | Confirm completeness | `csv/out/archive/`, manifests, git |
| 17 | New completed games/series | Advance cutoff and repeat capture/preflight | Next daily/series package | Choose new newsroom angle | New date-specific artifacts; never overwrite meaning silently |

## Cadence details

- **Daily:** validate sources, today’s schedule, availability/transactions when verified, game-level candidates, and a pregame packet.
- **Series:** update series state, sweep/road/contender tests, rotation and matchup context.
- **Weekly Monday:** preserve prior/current snapshots, measure standings and trend movement, and build a complete Observer packet/menu.
- **Milestones:** All-Star break, trade deadline, September call-ups, Act 3, postseason/tournament.
- **Historical:** regenerate season-parameterized almanac/flashback material only when source or code changes.

For the current July 20 package, steps 1–10 are evidenced. Step 11 is partly evidenced by `SAFE FOR EB USE` statuses but final human approval is not recorded. Steps 12–16 are not proven complete.
