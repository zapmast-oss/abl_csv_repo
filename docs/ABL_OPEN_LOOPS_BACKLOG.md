# ABL Open Loops Backlog

This file tracks intentional future work only. It is not a defect list for the July 20 package.

## Postseason and tournament logic

- Implement postseason/wild-card/seeding logic in story-engine code using `docs/ABL_POSTSEASON_RULES.md`.
- Add automated "if season ended today" outputs only after tests cover qualification, wild-card selection, and 1981-and-forward protected DCS matchups.
- Add clinch and elimination outputs only after standings-arithmetic tests are in place.
- Add protected-DCS matchup outputs only after tests verify the rule: if `#1` and `#4` are division rivals, `#4` plays `#2` and `#1` plays `#3`.

## Rules-source archive

- Archive or cite the original ABL forum or website rule source if found.
- Update `docs/ABL_POSTSEASON_RULES.md` with that source citation when available.

## Repository hygiene

- Create a separate repo cleanup branch.
- Review committed temporary/cache artifacts, obsolete generated snapshots, and upload-exclusion guidance without changing current production outputs.

