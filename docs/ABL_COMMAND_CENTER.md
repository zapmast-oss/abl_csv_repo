# ABL Production Command Center

Playing baseball advances the ABL. The session checkpoint protects the saved game first. The commissioner can add coverage, production, and release work when the playing window calls for it.

## Open and operate

From **ABL Production Control**, select **Open ABL Command Center**. It regenerates local status from the session records, verified newsroom edition, OOTP and repository exports, and existing logs, then opens the dashboard in the default browser. Reopen it or select **Refresh the Command Center** in the session console after an outside operation. Reload an already-open tab to show the regenerated status. The dashboard uses local files and needs no server.

After playing and saving in OOTP, select **Run ABL Production Session** and **Checkpoint a new session**. Choose the mode and trigger, enter the date shown in the saved league and name the playing window. Keep OOTP paused through the backup and any chosen capture or synchronization. The launcher checks the OOTP save marker, hashes the save's top-level `.dat` files, calls the existing league backup, and checks the copied bytes. The expected rolling backup is on H:.

From the repository, the same operations are available directly.

```powershell
python -B tools/production/abl_production.py open
python -B tools/production/abl_production.py preview
python -B tools/production/abl_production.py console
```

The PowerShell [menu installer](../tools/production/Install-ABLCommandCenter.ps1) adds the two entries through the existing menu's script discovery. It is safe to rerun and refuses conflicting files. For an installation preview, run `./tools/production/Install-ABLCommandCenter.ps1 -WhatIf` from PowerShell. The dashboard and the session ledger use Python 3.12. Desk generation uses the existing PDF-capable Newsroom runtime when the default Python lacks ReportLab.

## Choose a session mode

| Mode | Required protection | Optional production |
|---|---|---|
| Quick Play | Play, save, verified rolling safety copy, archive the receipt, close | Capture and Desk work appear deferred for later resumption. |
| Standard Session | Same protection | Export and validate CSV, preview or update ABL Input Data, prepare the Desk, select stories and track their release. |
| Featured Production | Same protection | Standard options plus a selected game or program, separate OBS-original and master backup, and production and release tracking. |

Play a featured game or record its pregame material at the commissioner's discretion. OBS, narration, editing, rendering and review stay with the operator. Featured offers the native video **Preview** followed by a separate **Update** choice once recordings and masters are closed.

## Production rhythm

**PLAY → SAVE → BACKUP → CAPTURE → VALIDATE → PUBLISH BASEBALL TO EAB → UPDATE THE DESK → PREPARE JOURNALISM → REVIEW → PUBLISH WHEN APPROVED → ARCHIVE → CLOSE**

After **Validate**, the session opens the next local **Experiencing Action Baseball** edition from the verified capture. This baseball update runs before the Desk and SBV steps. [EAB operations guide](EAB_OPERATIONS.md)

1. **Play and save.** Choose the game, day, series, or milestone. Let OOTP finish saving. Record the date displayed by OOTP and the window.
2. **Back up.** The existing rolling backup copies the saved directory with Robocopy `/E`. The session records the Robocopy result, destination, and source and destination comparison of all source `.dat` files. The console holds the session open if protection fails.
3. **Capture.** In OOTP, export CSV for the same saved state when reporting is wanted. Select capture in the session. Supply the last completed date and a completed game ID if one is known. Expected dates are checked against the export. A same-day post-game export that predates the save needs the completed game ID or a fresh export.
4. **Validate.** The existing Newsroom verifier inventories the registered CSVs, stages the immutable capture by SHA-256, reconciles finals, inning lines, game logs and player totals, and runs current-state preflight. A failed validation blocks CSV sync, Desk preparation and data-dependent releases. Post-day checkpoints require the cutoff day's scheduled slate to be complete.
5. **Update the Desk.** For ABL Input Data, the native Preview compares the verified capture with the repository. If there are changes, the session offers the existing Update prompt and verifies the full set afterward. The Desk can be prepared directly from the verified capture when repository sync is deferred. The Newsroom reuses or builds its matching local edition.
6. **Prepare journalism.** Review the ABL 411, franchise beats, evidence, and story slate. Select the angle and check every material claim against the cutoff. Use the existing editorial review handoff when creating a local reviewed issue.
7. **Review and publish.** Queue the exact artifact for a named channel. The commissioner verifies the facts and approves those bytes before the operator uploads. Record the final public URL after manual publication.
8. **Archive and close.** The checkpoint records protection and any verified export. Close after those records are written. Optional work can be deferred as soon as protection passes. An incomplete optional step remains visible in the session and can be resumed; its success or failure never substitutes for the protection receipt.

### Checkpoint triggers

| Trigger | Start the checkpoint when | Optional work due at this trigger |
|---|---|---|
| Post-game | At the first completed game after which the commissioner stops play or recording, once the final is recorded and OOTP has saved. | For a featured game, keep the capture and final game ID, results/recap and closed OBS original. A partially played league day is valid. |
| Post-day | After the last scheduled league game for the simulated date is resolved and the game is saved, before further play. | Export CSV, verify the completed slate and open the 411 and beat queue. If the source still lists unplayed fixtures, use post-game until a complete day is evidenced. |
| Post-series | After the last resolved fixture of the chosen series, including a documented reschedule, before playing the next series. | Capture the ending matchup and final ID when available. Recheck standings movement, sweep or split, roster and rotation changes, and queue divisional follow-ups. |
| Monday | At the first desk on a simulated Monday, before Monday play. If play skips past it, use the first next desk and record the actual cutoff. | Compare the last completed Sunday reporting window with the prior compatible weekly snapshot. Prepare the Observer and It's Monday packet after checking dates and run parameters. |
| Milestone | At the last safe save before a documented event and again after OOTP records its effects. | Cover the All-Star break, trade deadline, roster expansion, or confirmed postseason qualification. For September 1, 1981 roster expansion, capture before and after, verify actual promotions and demotions, and use the affiliate map. |
| End of season | After the final regular-season game, after the Grand Series final, and at the last save before rollover. | Freeze dated protection and reports. Validate records, playoff fixtures and awards as available before rollover. Use the approved postseason rules for DCS matchups and mark any unverified claims on hold. |

The regular-season validator is the current automatic Desk gate. Postseason game coverage, clinch/elimination arithmetic, and protected DCS matching still need a separate verified process. Use [the postseason rules](ABL_POSTSEASON_RULES.md) and official OOTP records for those decisions. At season close copy the checkpoint and dated source/report extracts into the preservation archive before moving the live season forward.

## Operating lanes

| Lane | Entry and evidence | Release or preservation step |
|---|---|---|
| OOTP HTML | Generate reports in OOTP. Inspect the saved game's `news/html/index.html` and the intended report pages, and record the date and bundle manifest in the session's HTML lane. | Queue the reviewed bundle or its checksummed upload manifest under **OOTP HTML website**. Upload through the confirmed site procedure and record the URL. |
| OOTP CSV and ABL Input Data | Export in OOTP, capture, validate, then use **Preview ABL Input Data** and **Update ABL Input Data** as needed. Session sync uses those same modes with the validated snapshot. | Statistics, star schema, and derived reports run from compatible inputs. A Git commit or push is a separate repository action. |
| StatsPlus | Review the capture and promotion rules, profile the fresh batch and reconcile authority at the same cutoff. Record local promotion evidence in the StatsPlus lane. | Use the separately confirmed external StatsPlus refresh after approval. Record its receipt in the release ledger. July 19 promotion evidence is historical for the August desk. |
| Newsroom | The verified Desk produces the 411, story slate and 24 franchise reports. Retain the date and source fingerprint with the evidence. | Use a separate commissioner release approval for the Observer, forum article or other editorial upload. |
| Video | Close OBS MKVs and rendered MP4s in the existing source folder. Run native **Preview SBV Video Backup** and **Update SBV Video Backup**. Check the originals and master copy lanes in the log. | Queue the final master separately, with approval and public link for the intended channel. |
| League and mirror | The league safety copy runs immediately after save. For a mirror audit, mount and verify H: and I:, then run **Verify H to I Mirror** and inspect the read-only results. | The legacy **Mirror H to I** uses Robocopy `/MIR`, which can delete destination-only files. Compare the proposed difference and exclusion sets before separately authorizing that command. |
| Historical research | Use the season-specific OOTP HTML, [almanac outputs](../csv/out/almanac/) and [pack validation](../VALIDATION_RUNBOOK.md). | Record the season, evidence, and validation before using history in a script or article. |

Use **Record or defer a manual lane** to attach a local evidence file or manifest and an operator note for HTML, StatsPlus enrichment, or read-only mirror verification. For a website update, build a checksummed bundle manifest covering every file to upload. Verify those file hashes against the approved manifest before transfer.

### Approval and publication receipts

Use **Manage publication approvals and receipts** in the session console after protecting the session.

1. **Queue an artifact.** Choose the release lane and channel and identify the exact file. The console records a SHA-256 fingerprint and a release ID. Editorial and StatsPlus releases require a validated OOTP cutoff.
2. **Approve an exact artifact.** The commissioner reads the artifact and source evidence, reviews the destination, and records approval. If an approved story changes materially, use a new reviewed revision and queue its new bytes. Existing local newsroom revisions use the [editorial handoff](../tools/newsroom/reviews/editorial-handoff.md).
3. **Check an approved artifact before upload.** The controller verifies the current file against the approved fingerprint. If it differs, return the revised artifact to review.
4. **Record a manual publication.** Enter the actual public URL after uploading. The controller checks the approved artifact again, prevents duplicate artifact/channel entries or duplicate URL receipts, and records the timestamp. Only the operator carries out the upload.

## Recovery and receipts

| Status | Meaning |
|---|---|
| Not started | The step has no recorded attempt in this session. |
| Running | An attempt started and has yet to write its outcome. A resumed interrupted attempt records an interruption before retrying. |
| Succeeded | The step passed its verification or has an explicitly recorded operator attestation. |
| Failed | The step has a failed receipt. Open its run log and resolve the specific failure before resuming dependent work. |
| Deferred | Work was skipped or is waiting on an earlier step. The session can close after mandatory protection and archiving succeed. |
| Requires approval | A release artifact is waiting for a commissioner decision. |

Use **Resume a session or its deferred work** and choose the session ID on the dashboard. Successful stages are reused. Recheck a failed league backup while the original save is still available. If the live save or CSV advanced, start a new session for current production and use earlier receipts for historical review. A pending CSV capture can be retried from the same save. Fixed exports receive a new fingerprint and keep the earlier attempt's evidence. The stored checkpoint is a receipt for the rolling backup at that time. For a durable historical save, preserve a separate dated safety copy with the league script's destination option.

Session receipts, output paths, approval records and attempt logs live under `logs/production_control/sessions/<session-ID>/`. The protection receipt is `checkpoint.json`. Retain these local records with archival copies. `status.js` is a generated dashboard view. Edit [the registry](../tools/production/registry.json) for scope or plan changes, then use Refresh. Registry owners name operating roles. Proposed and unconfirmed records contain their evidence and next action where available and leave unverified dates blank.

## Current and planned

| Current operation | Next planned work |
|---|---|
| ABL save and native backup, dated checkpoint receipts, and local session recovery | Test a complete OOTP restore and preserve dated milestone copies beside rolling backups. |
| Verified August CSV and newsroom edition, 24 club reports and six division desk entries | Exercise autumn continuity, September call-ups and postseason-specific verification. |
| Existing StatsPlus historical promotion and local reconciliation tools | Capture a new, date-compatible batch and confirm the StatsPlus external update interface. |
| Local Observer, ABL 411, showcase preparation, and historical programs | Test Monday and milestone packages against recorded cutoffs. Confirm the active production and publication destinations. |
| Local HTML, video preservation scripts, and read-only mirror evidence | Verify the next HTML release, media backup with H: connected, and current replica state with H: and I: connected. |

The July 20, 1981 project source pack preserves an earlier phase of the operation. Current status comes from the latest verified OOTP and newsroom receipts. Open [verification results](ABL_COMMAND_CENTER_VERIFICATION.md) for the checked source dates and open items.

## Automation matrix

| Activity | Level | Start or decision |
|---|---|---|
| Play, save, and select the window | Manual | Commissioner in OOTP |
| Checkpoint receipts and dashboard status | Automatic | Session controller or dashboard opener |
| League copy and integrity check | Assisted | Commissioner confirms save, controller runs native backup and compares the data files |
| Generate CSV, HTML and fresh StatsPlus exports | Manual | Commissioner uses the applicable export interface |
| CSV verification and local Desk generation | Automatic after capture | Controller or existing Daily Newsroom poll on a changed export |
| Repository CSV synchronization | Assisted | Existing Preview and Update with commissioner confirmation for changed files |
| StatsPlus enrichment and promotion | Assisted | Run profiling and reconciliation before approved promotion |
| Journalism selection, corrections and approval | Manual | Commissioner and editorial desk |
| Record, narrate, edit, and render | Manual | Commissioner and SBV producer |
| Video preservation and read-only mirror verification | Assisted | Operator runs Preview or verification, then reviews results |
| Destructive mirror | Manual | Separate explicit authorization after reviewing the comparison |
| External publication and link receipt | Manual | Commissioner approval followed by operator upload and receipt |
| Historical and event-based packages | Assisted | Operator chooses cutoff, extracts evidence and verifies outputs |
