# ABL Postseason Rules Authority

**Version:** 2026-07-08  
**Current story-engine use:** 1981 season and ABL continuity work  
**Scope:** Action Baseball League, league `200`

## Authority and evidence

This file is the repo authority for ABL postseason qualification and DCS seeding language until replaced by a more complete historical rules export.

Evidence currently available in the repo:

- `csv/ootp_csv/league_playoffs.csv` records league `200` with `num_wild_cards = 1`, `max_round = 3`, and round names `Division Championship Series`, `Conference Championship Series`, and `Grand Series`.
- `csv/ootp_csv/messages.csv` contains historical ABL playoff-begin messages for 1972-1980 listing four Division Championship Series matchups, consistent with eight total playoff teams.
- ABL continuity principle: league structure remains stable across the 1972-2022 continuity unless an explicit rules change is documented.
- Promoted 1981 rules correction: the 1981 change is a DCS seeding/matchup protection, not a postseason qualification change.

Search note: the local repo search did not locate a standalone website prose page spelling out the 1981 same-division DCS protection. If that source exists outside this repo, archive or cite it here in a future update. The qualification structure is nevertheless supported by the raw OOTP league playoff export and ABL continuity.

## Qualification rules

Postseason qualification remains the legacy ABL format:

- Each conference qualifies three division winners.
- Each conference qualifies one wild card.
- The wild card is the best remaining non-division-winner in that conference.
- This creates four playoff teams per conference and eight total postseason teams.

Story-engine status: qualification arithmetic is now allowed when it is computed from cutoff-compatible raw standings, division/conference mappings, and the rules above.

## Seeding rules before 1981

Before 1981, the DCS matchup rule is:

- Seed `#1` hosts seed `#4`.
- Seed `#4` is the wild card.
- Seed `#2` plays seed `#3`.

Story-engine status: use this only for historical seasons before 1981 unless a season-specific rules exception is documented.

## 1981-and-forward DCS seeding protection

Beginning in 1981, the qualification format does not change. The seeding/matchup protection changes:

- If seed `#1` and seed `#4` are from different divisions, the normal `#1` vs `#4` and `#2` vs `#3` DCS matchups apply.
- If seed `#1` and seed `#4` are from the same division, they do not meet in the Division Championship Series.
- In that same-division case, seed `#4` plays seed `#2`, and seed `#1` plays seed `#3`.
- Division rivals can meet in the Conference Championship Series, but not in the Division Championship Series.

Story-engine status: DCS matchup projections for 1981 and forward must apply this protection before publishing a bracket or "if the season ended today" matchup claim.

## Safe story-engine claims

The following claims are safe when supported by current, cutoff-compatible standings data:

- "Postseason qualification is known by legacy ABL continuity: three division winners plus one wild card per conference."
- "The 1981 change affects DCS seeding/matchups only: same-division clubs cannot meet in the DCS; if `#1` and `#4` are division rivals, `#4` plays `#2` and `#1` plays `#3`."
- "Raw OOTP standings and records determine the current division winners, wild-card leader, games back, and any clinch/elimination arithmetic."
- "StatsPlus playoff odds are model enrichment only and do not define the official field."

The following claims remain unsafe without additional arithmetic or source evidence:

- A club has clinched qualification.
- A club has been eliminated.
- A complete official postseason field is locked.
- A projected DCS matchup is official before applying the 1981 same-division protection.
- A StatsPlus odds percentage is equivalent to official qualification.

## Implementation notes

Any story-engine postseason module should:

1. Use raw OOTP standings/records as the system of record.
2. Identify division winners by conference.
3. Identify the best non-division-winner in each conference as the wild card.
4. Seed each conference field.
5. Apply pre-1981 or 1981-and-forward DCS matchup rules based on season.
6. Label any midseason field as "if the season ended today" or "current arithmetic," not as an official final field.
7. Keep StatsPlus odds and ELO as labeled enrichment.

