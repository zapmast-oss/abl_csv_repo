# Experiencing Action Baseball

Open **ABL Production Control → Open EAB Website**. The local site starts at [`output/eab/dist/index.html`](../output/eab/dist/index.html) and needs only a browser. Use **Edition timeline** to reopen a past edition. A bookmarked edition keeps its own league date and cutoff.

## Opening edition

The verified league date is **August 20, 1981**, with **1,450 regular-season finals through August 19**. All 72 source files agreed with the promoted repository export. The older July baseline belongs to a prior production cycle. The two reviewed August manuscripts are in the internal Newsroom. EAB opens with the verified baseball record and beat pages for all 24 clubs.

EAB uses the existing OOTP capture and preflight, league and franchise identifiers, approved logos and dated team portraits, six division races and season-correct DCS rules, archived scores and club records, and the Newsroom’s sealed issue and review format. It also renders 1972–1980 regular-season batting and pitching using OOTP player IDs. The current player lines derive from the admitted player-game records. Attendance comes from completed home games with recorded positive attendance. [Source authority](ABL_SOURCE_AUTHORITY_RULES.md) · [Franchise reference](ABL_LEAGUE_AND_FRANCHISE_REFERENCE.md) · [Postseason rules](ABL_POSTSEASON_RULES.md)

The source pack describes the older baseline, Story Engine and editorial standards. The current Newsroom adds the Desk, 24 beat assignments and immutable reviewed issues. EAB adds a static visitor site and its local release history around those existing systems. [ABL source pack](../_chatgpt_project_source/00_START_HERE.md)

## Produce an update

1. Play and save in OOTP. Run **ABL Production Control → Run ABL Production Session** and complete the existing safety copy and CSV validation. Standard and Featured sessions offer capture immediately. Quick Play can capture when resumed. [Command Center operating guide](ABL_COMMAND_CENTER.md)
2. EAB verifies the exact capture and its cutoff, builds the baseball edition and switches the local current page. The baseball stage runs before the Desk and video steps. It works for a validated individual final, a full slate, a series or several days of play at once. If the build fails, review the EAB stage log, then resume the session or select **Run EAB Publishing → Publish latest verified update locally** after correcting the source.
3. For a story, use **Manage publication approvals and receipts** in the existing session console. Queue the exact sealed `.md` manuscript under **Editorial** with channel **EAB**, then approve that exact artifact. EAB matches the receipt’s hash and cutoff to a verified edition and rechecks the newsroom seal. Run **Publish latest verified update locally** to include newly eligible copy. Safe HTML and an available reading PDF use the approved text. Internal evidence is excluded. [Editorial review handoff](../tools/newsroom/reviews/editorial-handoff.md)

**Run EAB Publishing** also opens a separate preview, lists editions, restores a previous edition and builds the deployment ZIP. Preview files live under `output/eab/preview/`. Local attempt receipts are under `logs/eab/`. The existing Production Control, league backup and Input Data scripts retain their original file hashes. The repository copy of the session workflow from before EAB is checkpointed under `logs/eab/source-checkpoints/`.

## Verify and recover

Each edition under `output/eab/dist/releases/` carries a checksum manifest with its OOTP source fingerprint, simulated date, completed-game cutoff, edition timestamp and eligible issue versions. Shared artwork and almanac pages use content-derived paths. Each dated homepage and published issue has a permanent URL. Approved corrections receive new URLs and retain their predecessors. An identical rebuild keeps the current edition ID. The root `index.html` switches only after the required files, internal links, anchors and checksums pass validation.

For recovery, choose **Restore an earlier local edition** and enter its ID from **List local editions**. The publisher verifies that edition and switches the pointer. The timeline still lists later issues and editions. Publishing the latest verified source again selects its existing edition when the inputs have stayed the same.

Run the following from the repository with the production Python runtime. The acceptance replay uses local Git LFS snapshots and the existing PDF-capable Newsroom dependencies. Its approval receipts and failures are disposable fixtures.

```powershell
python -B tools/eab/abl_eab.py status
python -B tools/eab/abl_eab.py verify
python -B -m unittest discover -s tools/production -p 'test*.py'
python -B tools/eab/test_abl_eab.py
```

## Opening-day verification

The acceptance replay uses the verified August 18 and August 19 captures. It checks repeat builds, the cutoff change, prior editions and issue URLs, rejected and regressing input, an interruption before switching the pointer, retry and rollback, the public allowlist, all 24 team beats, game and manuscript cutoffs, and a baseball update with an ineligible issue and no new broadcast. The Command Center regression suite checks that EAB runs after validation and before a failed Desk attempt, while the protected production session can still close.

Chrome checks used desktop, tablet and phone widths. They covered the current edition, each franchise beat, division standings, score and schedule navigation, historical player statistics, search, both newsroom issue versions, and the generated PDF link. The replay inspected the PDF text against the original publishing cutoff. The opening ZIP passed its integrity check and its homepage matches the current local edition.

## Host the public edition

The deployment ZIP is a self-contained static website. **Run EAB Publishing → Build deployment ZIP** assembles the current edition, earlier admitted editions and only their allowlisted assets. At an authorized destination, extract the ZIP as a complete site version, stage it, verify the homepage, two dates and a team or article, then switch the host to the complete version in one operation. Keep prior snapshots accessible at their paths. For host-side recovery, switch back to the previous hosted version or deploy a ZIP after restoring the desired local edition.

The existing ChatGPT-hosted ABL site has restricted access. The commissioner’s existing OOTP report and StatsPlus addresses are documented in the public league guide. Their direct endpoints returned HTTP 403 during verification here, so hosting access and upload procedures are unverified. EAB is ready for a separately authorized public static destination. [ABL follow guide](https://forums.ootpdevelopments.com/showthread.php?t=366841)

## Source limits

The automatic verification gate covers the regular season. The postseason picture is current standings arithmetic, with tied races carried forward. Current StatsPlus odds are unavailable at this cutoff. The almanac explains gaps in matched game records. Fan Culture uses reported attendance and venue information. SBV links point to the commissioner’s published programming and discussions. ABL Advance opens the game calendar while an episode-specific destination is unverified. [Sim Baseball Vision](https://www.youtube.com/channel/UCd1wShl1bahaTldUtqVZVwg) · [League discussions](https://forums.ootpdevelopments.com/showthread.php?p=5049449)
