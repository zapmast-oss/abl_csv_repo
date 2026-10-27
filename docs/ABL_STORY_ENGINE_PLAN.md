# ABL Story Engine Plan

## Editorial contract

The engine observes baseball, not a prewritten drama. Its hierarchy is:

`Tournament -> Standings -> Race -> Fans -> Management -> Players -> Team -> Today's game`

Every output must follow: **Do not invent drama. Notice pressure. Name stakes. Let the game prove the story.** The packet may state verified conditions and consequences; it must label inference, avoid mind-reading, and leave outcomes unresolved. The core test is: **Baseball matters because the stakes couldn't be higher.**

## 1. Repo Map Engine

Build `scripts/abl_inventory_repo.py` as a read-only scanner. It should classify source/config/code/generated/archive/temp files, map imports and literal read/write paths, flag generated outputs outside `csv/out/`, detect duplicate hashes and doubled paths, and emit Markdown plus CSV/JSON manifests under `csv/out/docs/`. Do not infer deletion from duplicate detection.

Acceptance: deterministic output; ignored cache rules; producer/consumer edges for important scripts; CI failure only for new output-boundary violations.

## 2. Data Catalog Engine

Build `scripts/abl_catalog_data.py` around Python’s CSV parser with encoding/comment handling. Record path, data tier, size/hash, row/column counts, headers/types, inferred grain/keys, min/max dates, seasons/leagues, null and duplicate-key diagnostics, producer, and story-use tags. Support `--quick` sampling and `--full` validation.

Acceptance: catalogs all approved input/output zones; repeatable schema hash; family summaries for almanac seasons; errors are rows in the catalog, not silently skipped.

## 3. Story Signal Engine

Use existing facts and analytic outputs as evidence adapters. Do not put prose generation in detectors. Each detector emits candidates and evidence; a separate ranker applies hierarchy, urgency, confidence, novelty, and editorial overrides.

### Canonical candidate schema

Required fields: `candidate_id`, `run_id`, `as_of_date`, `season`, `league_id`, `hierarchy_level`, `signal_type`, `subject_type`, `subject_id`, `subject_name`, `opponent_id`, `window_start`, `window_end`, `headline_factual`, `stakes`, `trigger_summary`, `signal_score`, `confidence`, `urgency`, `status`, `source_tables`, `evidence_ids`, `historical_echo_id`, `created_at`.

Evidence is a separate long table: `evidence_id`, `candidate_id`, `metric`, `value`, `comparison`, `threshold`, `rank`, `sample_size`, `source_path`, `source_row_key`, `as_of_date`. This prevents every signal from adding incompatible columns.

### Signal catalog

| Hierarchy | Signal | Minimum evidence and guardrail |
|---|---|---|
| Tournament | tournament pressure | qualification path, games remaining, elimination/clinching arithmetic; never imply a berth without verified rules |
| Tournament | Grand Series rematch | verified prior championship matchup plus current scheduled game/series |
| Standings | division race movement | current and prior rank/games-back; require comparable snapshots |
| Race | wild-card movement | explicit wild-card rules, current cutoff, delta, games remaining |
| Race | contender test | opponent quality plus schedule window; label “test,” not destiny |
| Race | collapse warning | sustained loss of position/performance across a minimum window; avoid causal claims |
| Fans | fan pressure | attendance/interest/expectation/market and performance trend; do not invent sentiment |
| Management | manager tendency | repeated tactical pattern with opportunity denominator and sample size |
| Management | organization pressure | payroll/expectation/roster action plus standings context; separate facts from inference |
| Players | player burden | share of team value/production/workload, replacement gap, and recent use |
| Players | ace matchup | probable/confirmed starters, quality/workload/rest; downgrade if starter is unconfirmed |
| Players | September call-up implication | verified call-up/roster status plus role opportunity and race context |
| Team | team identity | multiple stable indicators (run creation/prevention, style, splits), not a one-game label |
| Team | bullpen stress | recent workload, leverage use, availability uncertainty; avoid declaring unavailable without evidence |
| Team | road split significance | home/road gap, sample sizes, upcoming road stakes |
| Game | sweep opportunity | series results plus remaining scheduled game; distinguish sweep from series win |
| Game | historical echo | structured similarity and linked historical evidence; describe echo, not prediction |

Additional existing adapters should cover one-run volatility, Pythag gaps, momentum, injuries, milestones, rookie emergence, platoon edges, defense, catcher control, travel, and schedule strength.

### Ranking and restraint

Score = hierarchy weight + stakes + time proximity + movement magnitude + evidence quality + novelty - redundancy - uncertainty. Higher hierarchy normally wins, but weak tournament context must not outrank a well-supported lower-level story. Cluster candidates sharing the same teams/event and select one lead with supporting angles. Persist suppressed candidates and reasons. No detector may emit emotional language as evidence.

## 4. Baseball Observer Packet Engine

Input: selected candidates, evidence rows, schedule, standings, rosters, and optional historical links. Output both `packet.json` and rendered `packet.md` under `csv/out/story/packets/<run_id>/`.

Packet order follows the hierarchy, then includes: what changed, stakes, evidence, uncertainties, what to watch, game proving conditions, historical echo, and source citations. Add a “do not say” block for unverified playoff rules, probable starters, injuries, motives, and fan sentiment. Existing EB JSON/text packs become adapters until retired.

## 5. Article/Video Outline Engine

Consume an editorially selected packet, never raw tables. Produce a common outline JSON and channel renderers for article, forum, show notes, and video. Sections: factual cold open, stakes thesis, evidence beats, counterevidence, game/series proving questions, historical context, and closing watch list. Every beat carries candidate/evidence IDs. Generated prose should remain an outline until a human selects the lead.

## 6. Run Cadence Engine

One parameterized runner: `python -m abl_story run --season 1981 --as-of YYYY-MM-DD --cadence daily|series|weekly --run-id ...`.

- Daily: refresh source validation, today’s schedule/probables, bullpen/load, injuries/transactions, game-level candidates and pregame packet.
- Series: series state, sweep/road/contender tests, rotation and matchup changes.
- Weekly Monday: immutable prior/current snapshots, standings/race movement, team/player trend, full Observer packet and menu.
- Milestone cadence: all-star break, trade deadline, September call-ups, Act 3, postseason/tournament.
- Historical: season-parameterized almanac build and flashback linkage; rerun only when sources/code change.

Each run writes a manifest, logs, validation report, candidates, menu, packet, and outlines beneath a run-specific directory. A failed required stage must stop downstream rendering; optional context failures are visible warnings.

## Proposed output standard (proposal only)

```text
csv/out/
|-- docs/                         # inventories, catalogs, schemas, validation
|-- story/
|   |-- candidates/               # canonical candidate/evidence tables by run
|   |-- menus/                    # ranked and editorially selected menus
|   `-- packets/                  # Observer packet JSON/Markdown by run
|-- text_out/                     # rendered article/video/show-note outputs
|-- almanac/                      # canonical historical season datasets/products
`-- archive/                      # immutable freezes plus manifests
```

Recommended naming: `<product>_s<season>_<cadence>_<asof-YYYYMMDD>_<run-id>.<ext>`. During migration, write new standardized outputs alongside existing files; do not move old products until consumers and retention rules are verified.

## Implementation roadmap

1. **Preserve and document existing working scripts.** Freeze checksums/CLI examples and record known-good 1981 and almanac outputs; make no behavioral changes.
2. **Create repo/data inventory scripts.** Automate the two inventories produced in this phase and add output-boundary/schema checks.
3. **Standardize story candidate schema.** Define versioned candidate/evidence/menu JSON Schema or data dictionary; write adapters for current week-5 and flashback candidates.
4. **Build story signal engine.** Start with standings/race movement, sweep, road split, contender test, bullpen stress, player burden, and historical echo using existing tables.
5. **Build Baseball Observer packet generator.** Render hierarchy-ordered JSON/Markdown with evidence citations, uncertainty, and proving questions.
6. **Build article/video outline generator.** Use selected packets and trace every beat to evidence.
7. **Create one master runner.** Parameterize season/as-of/cadence, emit run manifest, fail safely, and replace broad `_tmp_run_all.py` only after parity.
8. **Test with existing 1981 Week 5/Week 7 outputs.** Golden-test menus, reproduce week-5 candidates, explain week-7 gaps, and compare observer/video products.
9. **Extend to daily/series/weekly Act 3 coverage.** Add schedule-state, clinch/elimination, roster, bullpen, and September signals with verified tournament rules.
10. **Extend to almanac/flashback seasons 1972-1980.** Normalize all seasons, render missing menus, link current candidates to historical events and Grand Series history.

## Build first

Build the inventory/catalog automation and canonical candidate/evidence contracts first. Then implement a narrow vertical slice: weekly snapshot -> division-race and bullpen signals -> ranked menu -> Observer packet. This reuses working data products, tests the full editorial chain, and prevents another collection of disconnected report scripts.

