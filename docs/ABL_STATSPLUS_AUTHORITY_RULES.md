# ABL StatsPlus Authority Rules

> **StatsPlus enhances. It does not replace.**

Feed 3 is an expandable supplemental layer. It adds context, probability, pressure, and texture while preserving raw OOTP and promoted sortable authority. Future sources—such as advanced batting, advanced pitching, standardized metrics, and matchup matrices—may enter Feed 3 only through staged intake, profiling, authority classification, and a separate explicitly authorized promotion.

## Authority order

Authority is field-specific. It is not a blanket ranking that allows one feed to replace every value from another.

### Feed 1 — Raw OOTP CSV

Raw OOTP is authoritative for:

- completed-game existence and game IDs;
- dates and played/unplayed state;
- home and road teams;
- final scores and inning/game logs;
- schedule proof;
- records and standings derived from approved completed games;
- current-state cutoff and coverage.

StatsPlus must not override these fields.

### Feed 2 — Promoted sortable statistics

Promoted sortable stats are authoritative or supporting within their governed current-capture fields for:

- player batting, pitching, fielding, ratings, and value statistics;
- team batting, pitching, fielding, park, financial, and current-history context;
- current staff names and jobs where the promoted schema supports them;
- current team/player enrichment not requiring game-level proof.

Accepted schema limitations remain in force. A third feed does not automatically enable a missing signal.

### Feed 3 — StatsPlus

After explicit promotion, StatsPlus is authoritative for StatsPlus-specific values:

- ELO and ELO movement;
- playoff odds and simulation outputs;
- BaseRuns and related expected-performance fields;
- StatsPlus team/player WAR tables when labeled as StatsPlus values;
- historical and current fan-interest tables;
- StatsPlus financial and organization-context tables;
- injury summary when no more authoritative equivalent exists and cutoff/identity are validated;
- best batting and pitching game tables as StatsPlus rankings;
- Grand Tournament of Champions history after historical reconciliation.

These values must retain source labels. Model outputs are not observed standings facts.

## Crosscheck-only StatsPlus tables

Unless separately authorized, StatsPlus standings and ordinary batting, pitching, fielding, score, schedule, and record fields are crosscheck/support only. Raw OOTP or promoted sortable sources remain authoritative for the overlapping field.

Team/player tables may still contribute genuinely StatsPlus-specific columns. Promotion can therefore be column-level rather than whole-table.

## Staff limitation

StatsPlus staff tables may crosscheck names, jobs, team abbreviations, reputation, and style fields. The legacy tables contain no stable staff IDs and do not resolve the staff-ID/name limitation.

Manager tendencies remain disabled until a fresh, same-cutoff table and a validated identity bridge support the signal. Names or descriptive labels alone do not authorize inferred motives or tendencies.

## Conflict outcomes

Use one of these outcomes:

- `RAW_OOTP_GOVERNS`: conflict concerns game/current-state proof.
- `SORTABLE_GOVERNS`: conflict concerns an overlapping governed sortable field.
- `STATSPLUS_SPECIFIC`: validated field is unique to promoted StatsPlus.
- `CROSSCHECK_AGREES`: overlapping values agree at the same cutoff.
- `CROSSCHECK_CONFLICT`: values differ and both values must be reported.
- `HOLD_CUTOFF_UNKNOWN`: freshness cannot be established.
- `HOLD_IDENTITY_AMBIGUOUS`: team, player, game, or staff linkage is unsafe.
- `HOLD_SCHEMA_DRIFT`: field meaning or schema change is unresolved.
- `DISABLED_UNAVAILABLE`: required evidence is absent.

No conflict may be silently repaired by choosing the convenient value.

## Story-engine use

StatsPlus can point toward stories involving projected race pressure, team strength, fan/organization pressure, injury burden, baserunning, standout games, and historical tournament echoes.

Editorial output must distinguish:

- observed fact;
- sortable enrichment;
- StatsPlus-specific model/context value;
- historical context;
- unavailable or disabled signal.

The governing rule remains: data points where to look; the game proves the story.
