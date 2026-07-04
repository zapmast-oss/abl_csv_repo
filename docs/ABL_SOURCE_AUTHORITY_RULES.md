# ABL Source Authority Rules

## Authority hierarchy

1. `system_of_record_extract`: raw OOTP exports; dated game/result rows prove current state.
2. `supplemental_report_extract`: sortable OOTP reports; enrich a validated capture.
3. `historical_context`: almanac products; prior-season evidence only.
4. `derived_output`: rebuildable calculations and reports; never source of truth.
5. `editorial_config`: story taxonomy and selection logic; no baseball-state authority.
6. `documentation`: contracts and operating guidance; no baseball-state authority.

## Rules

- Current standings, results, games per team, and the last completed-game cutoff must derive from raw OOTP game/schedule/result data.
- `games.csv`, `games_score.csv`, and `game_logs.csv` must reconcile before a batch is promoted. A downstream standings file cannot resolve a disagreement among them.
- Sortable stats may enrich a validated current state but cannot prove the latest game date.
- Player ratings, indicative data, misc data, and scouting-style attributes must be tied to a named capture batch. Their absence of a date is not permission to call them current.
- Team finances, personality, market, and park reports provide context. They are not standings or results evidence.
- Staff and manager reports support management analysis only when their detected games coverage matches the active raw batch.
- Historical almanac data may support historical echoes, comparisons, and prior accomplishments. It cannot prove any 1981 current-state fact.
- Generated outputs—including files named `current`, candidates, packets, menus, and reports—are never source of truth. They require lineage back to registered captured sources.
- Editorial dictionaries and documentation can shape questions and presentation but cannot supply performance facts.
- A source with insufficient as-of evidence must be rejected for current-state use or limited to an explicitly static/historical role.

## Date doctrine

`newsroom_date` is the editorial operating date. `as_of_date` is the latest completed baseball evidence admitted to the run. They may differ. Future scheduled rows can support preview context but cannot advance `as_of_date`.

## Conflict resolution

When sources disagree, stop promotion and report the discrepancy. Do not average records, patch results manually, prefer a newer-looking filename, or use a generated report to override raw game evidence.

