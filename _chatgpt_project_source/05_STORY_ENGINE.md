# ABL / SBV Story Engine

## Goal

Turn simulated fictional baseball data into coverage that feels native to a living baseball world. Data locates pressure; reporting names the stakes; the games retain the right to resolve the story. Fictional players and clubs are treated as real within the ABL world, while every factual claim remains traceable.

## Story-attention hierarchy

1. Stakes / tournament implications
2. Standings
3. Race
4. Fans / meaning / pressure
5. Management / organization
6. Players
7. Team identity
8. Today’s game or series

The hierarchy orders attention, but evidence quality can suppress a higher-ranked idea. Tournament implications use `docs/ABL_POSTSEASON_RULES.md`: qualification is known by ABL continuity, while the 1981 change affects DCS seeding/matchup protection only. Fan emotion, management intent, and clubhouse meaning require evidence beyond numerical proxies.

## From data to story

1. Declare newsroom date and completed-game cutoff.
2. Run current-state validation. Stop if the raw drivers are incomplete or inconsistent.
3. Rebuild standings and recent results from cutoff-compatible OOTP data.
4. Find compressed division races, direct head-to-head opportunities, movement, run-balance gaps, player value, and historical echoes.
5. State the arithmetic and consequence: what ground can be gained or lost, current wild-card/field implications when the rules file and standings support them, what the remaining schedule can answer, and what cannot yet be known.
6. Attach exact evidence and authority labels to every candidate.
7. Rank using the editorial hierarchy; separate lead, secondary, quick mention, watch, and held items.
8. Add StatsPlus/sortable context only after official facts are fixed. Keep model signals labeled.
9. Draft the appropriate format: Baseball Observer, Daily 411, Ballpark Feed, forum post, or video outline.
10. Apply caution language and human editorial review before narration/publication.

## Identifying what matters today

- Prefer small, current standings margins and direct games between clubs affected by them.
- Quantify series arithmetic without predicting outcomes.
- Treat short windows as movement, not automatically a turnaround or collapse.
- Identify player cases with adequate workload; say “current value leader/case,” not “winner.”
- Use history to add continuity and pressure, never destiny.
- Ask what additional evidence would promote a watch item into a story.

## Supported story versus speculation

| Evidence supports | Evidence does not automatically support |
|---|---|
| A two-game lead | A settled division |
| Four scheduled head-to-head games | Their outcomes |
| Attendance scale | Fan emotion or home-field causation |
| Payroll/record contrast | Blame, incompetence, or owner intent |
| Injury totals | Injury causation for performance |
| Expected-record gap | Bad luck or inevitable regression |
| Franchise postseason history | Current clubhouse motive, curse, or destiny |
| Model playoff odds | Official qualification or a single-game forecast |

Use possibility language only when it is useful and clearly labeled. Otherwise omit unsupported claims. Never manufacture quotes, private thoughts, injuries, transactions, rivalries, or atmosphere.

## Baseball Matters framing

“Baseball matters because…” is a test for consequence, not an invitation to exaggerate.

- Baseball matters because the stakes could not be higher.
- Baseball matters because this game leaves scars.
- Baseball matters because standings compress a season’s work into ground a club must defend or recover.
- Team names and standings should carry history, pressure, and meaning, not just numbers.

The current stakes-first script uses “Baseball matters because the stakes are incredibly high.” That line is supported by multiple compressed races and direct opportunities, while the script carefully avoids declaring results.

## Current example

The July 20 package leads with Dallas–Detroit as a two-game NBC Central race, then gives Chicago–Dallas a direct-opportunity frame: Chicago begins four games behind Dallas; a sweep can produce a tie, while being swept can make the gap eight. This is valid because the records, schedule, and arithmetic are explicit. It must not become “Chicago can take sole possession” or a predicted result.
