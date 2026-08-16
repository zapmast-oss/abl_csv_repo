# Nashville at Houston — TBO Game Prep

## Pregame Lock

- Date/matchup: Friday, July 24, 1981 — Nashville Blues at Houston Mavericks.
- Required entering records from the visually verified live-OOTP controls: Nashville **51-46**; Houston **55-41**.
- Independently recomputed local-raw records: Nashville **50-46**; Houston **54-41**.
- Scheduled starters (live-OOTP control values only): Nashville — Devon John, **5-8, 4.78 ERA**; Houston — Scott Szell, **10-5, 2.94 ERA**.
- Evidence cutoff: immediately before the July 24 game; completed games through July 23 only.
- Reconciliation status: **FAIL — BLOCKED**.

The local raw OOTP export does not contain the complete July 23 ABL schedule. Of 12 ABL games scheduled for July 23, only game 1155 (Dallas at Chicago) is marked played. The other 11 are unplayed.

Two missing games directly prevent the acceptance-gate reconciliation:

- Game 1162, St. Louis at Nashville, July 23: `played=0`, no score rows, and no game-log rows. Nashville therefore remains 50-46 locally instead of the required 51-46.
- Game 1161, Houston at Cincinnati, July 23: `played=0`, no score rows, and no game-log rows. Houston therefore remains 54-41 locally instead of the required 55-41.

No result for either missing game was inferred from the control records.

## Why This Game

**BLOCKED.** Not produced because the mandatory entering-record acceptance gate failed.

## Nashville Is Hot

**BLOCKED.** Recent-form calculations were not promoted into a game packet because the local game log stops before Nashville's scheduled July 23 game.

## Stakes

**BLOCKED.** Cutoff-compatible standings and tournament arithmetic cannot be finalized from the incomplete July 23 raw state. The governing rule is three division winners plus the best remaining non-division-winner as one wild card per conference, but no current-position claim is made here.

## Starting Pitchers

The names and stat lines above are retained only as visually verified live-OOTP control values. Repository reconciliation was not continued after the critical record gate failed.

## Things Ernie Should Watch

**BLOCKED.** No broadcaster watch list was issued from an incomplete cutoff state.

## If This Happens...

**BLOCKED.** No consequence notes were issued from an incomplete cutoff state.

## Continuity

**BLOCKED.** No continuity claims were issued.

## Source Ledger

- `csv/ootp_csv/games.csv`: ABL regular-season rows filtered to `league_id=200`, `game_type=0`, `played=1`, and date on or before `1981-07-23`; wins and losses recomputed independently for team 16 (Nashville) and team 23 (Houston). The same file identifies the 12 scheduled July 23 ABL games and the 11 unplayed rows, including games 1161 and 1162.
- `csv/ootp_csv/team_record.csv`: confirms the partial local state at Nashville 50-46 and Houston 54-41; used as a cross-check, not as the independent calculation.
- `csv/ootp_csv/games_score.csv` and `csv/ootp_csv/game_logs.csv`: contain rows for played game 1155 but no rows for missing games 1161 or 1162.
- `csv/out/control/current_state_preflight_1981_target_1981-07-23.md`: explicitly labels its cutoff a partial July 23 day and reports Nashville at 96 games and Houston at 95 games.
- `docs/ABL_POSTSEASON_RULES.md`: repository authority for three division winners plus one wild card per conference.
- Live-OOTP control values supplied in the commission: Nashville 51-46; Houston 55-41; Devon John 5-8, 4.78 ERA; Scott Szell 10-5, 2.94 ERA.
- No July 24 result was read, used, or disclosed.

## Validation

1. **PASS** — No July 24 Nashville-Houston result or other future result was used or revealed.
2. **FAIL** — Nashville independently reconciles locally to 50-46, not the required 51-46; July 23 game 1162 is missing.
3. **FAIL** — Houston independently reconciles locally to 54-41, not the required 55-41; July 23 game 1161 is missing.
4. **FAIL** — Starter values are clearly marked as live-OOTP controls, but repository reconciliation was halted at the acceptance gate.
5. **FAIL** — Nashville recent-form arithmetic cannot be finalized through July 23 because game 1162 is absent.
6. **FAIL** — Postseason rules are verified, but official cutoff-compatible standings arithmetic cannot be finalized from the incomplete July 23 state.
7. **PASS** — Every claim in this blocked report is traceable, and unresolved facts are not guessed.

**FINAL STATUS: BLOCKED**

Showrunner action required: refresh/export the OOTP raw CSVs after the complete July 23 ABL schedule has been played and verify that games 1161 and 1162 carry completed game, score, and log data. Then rerun the pregame analysis and require independent reconciliation to Nashville 51-46 and Houston 55-41 before publication.
