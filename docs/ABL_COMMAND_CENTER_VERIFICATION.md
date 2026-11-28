# ABL Command Center verification

Checked on November 22, 2026 in Pacific time. Tests used previews, copied CSVs and disposable save data. The newsroom replay used an isolated archive under the repository's `tmp` directory.

## Current operating baseline

| Area | Verified state |
|---|---|
| OOTP date and results | League 200 reports **August 20, 1981**. The latest verified regular-season cutoff is **August 19**, with 12 finals on the cutoff day and 1,450 through the cutoff. |
| CSV source authority | The 72 registered files in the live OOTP export and `csv/ootp_csv` matched the verified newsroom fingerprint `2acae29146076f6298fcafecf8afc73fb47bfdc8e8362ea73a5c38454767d02d`. |
| Newsroom | The current edition is `1981-08-19_2acae29146076f62`. The existing status audit passed for the edition, Daily Desk and reviewed local issue. The current output has the 411, story slate, six division entries and 24 beat reports. |
| Most recent newsroom production | The existing poll receipt shows a successful build check on November 21 at 12:30 p.m. PST, then unchanged source on November 22. The prior scheduler acceptance records the five-minute check. A fresh Task Scheduler management query was denied in this environment. |
| Local OOTP HTML | The local report index is in the OOTP 26 save's `news/html/index.html`. Its file timestamp is November 17, 2026. It has a separate path and capture time from the CSV export. |
| StatsPlus | The repository carries the July 19, 1981 promotion verification and profiling tools. An August-compatible batch and the active external update procedure were unconfirmed. |
| Historical research | The repo contains dated source HTML and derived almanac material for 1972 through 1980, including season and flashback packs. The July 20, 1981 project pack is historical documentation. |
| Mirror | The November 18 6:38 p.m. PST read-only log recorded zero copy, mismatch, failed or extra entries. H: and I: were unavailable during this verification. |

Key evidence is in the [current edition pointer](../csv/out/story/newsroom/current.json), [newsroom acceptance](../csv/out/story/newsroom/acceptance/production-control.json), [live poll receipt](../logs/newsroom_control/poll-state.json), [StatsPlus promotion verification](../csv/out/control/statsplus_promotion_verification_1981_asof_1981-07-19.json), and [almanac outputs](../csv/out/almanac/).

## Live Production Control scripts

The active scripts are in `C:\Users\earld\Scripts`. The installed names below correspond to the function names in the assignment.

| Active file | Observed behavior |
|---|---|
| `ABL-Production-Control.ps1` | Discovers adjacent PowerShell files and confirms the selected operation. It offers native Preview and Update entries for CSV and video. It listed both installed Command Center launchers, and the session launcher opened its console. The original menu SHA-256 matched the pre-installation value. |
| `Backup ABL League.ps1` | Non-purging Robocopy `/E` copy of the saved league to H:. Source, destination, log file and `-NoPause` can be set. The disposable copy verified nested coverage, matching bytes and retention of a destination-only fixture. |
| `Backup SBV Videos.ps1` | Default Preview uses `/L`. Update copies top-level MKV originals and MP4 masters to their own H: destinations and records each lane. A live copy requires H:. |
| `Mirror H to I.ps1` | Uses `/MIR` against I: and can delete destination-only files. The menu exposes this as a separate legacy operation. |
| `Verify H to I Mirror.ps1` | Uses `/MIR /L` as a read-only difference check with a timestamped log. Its exclusions differ from the legacy mirror command. Compare both exclusion lists when acting on a difference. |
| `Update the Repo.ps1` | Compares the exact top-level CSV name set and content hashes. Update prompts for `UPDATE`, stages changed files, verifies the result and has rollback logic. Confirmed on disposable data. |

The two launchers were added through script discovery. The legacy PowerShell files were preserved byte for byte, so recovery of the original menu requires only removing the added launcher files.

## Test results

| Check | Result |
|---|---|
| Active source Preview | **Passed.** The native CSV Preview compared 72 files and found zero content differences. |
| Native CSV safeguards | **Passed.** Disposable Preview found one difference, cancellation preserved the prior bytes, confirmed Update replaced the one changed file, and the next Update reported zero differences. The initial sandbox run denied Windows `File.Replace` and retained the old bytes. The permission-reviewed disposable rerun succeeded. |
| Native league copy | **Passed.** The disposable source and destination hashes matched. The non-purging copy kept the destination-only fixture. |
| Standard Session replay | **Passed.** The controller verified backup, captured 72 CSVs, reconciled 1,450 finals, ran native Preview, generated an isolated Desk with 24 beat PDFs, archived and closed. An unchanged resume reused the completed stages. |
| Command Center recovery and approval tests | **Passed.** Seven tests covered Quick Play close, mandatory protection failure, CSV failure gating, changed-save rejection, interrupted attempts, Featured media failure and release approval/duplication. |
| Newsroom unit tests | **Passed.** Fourteen existing tests covered the poll, scoring corrections, slate, and Desk. |
| Launcher installation | **Passed.** Preview and repeated installation worked on a menu copy. The live menu listed both launchers and the session console started from its installed entry. |
| Dashboard data and interactions | **Passed.** All 66 registry records and their evidence links, six division groups, and 24 beat links resolved. Script checks exercised the seven sections, project search, planned-work filter, anchor navigation and the missing-status message. |

Reproduce the unit checks from the repository.

```powershell
python -B -m unittest discover -s tools/production
python -B tools/production/abl_production.py preview
python -B tools/production/abl_production.py refresh
```

## Remaining verification and manual work

- Browser visual verification ended when the computer-use tool could not verify the selected Chrome window's URL. The dashboard files, generated status data and links can be inspected locally.
- Live league and video copy still require the configured H: volume. Live read-only replica verification requires both H: and I:. A full disposable OOTP restore is planned.
- Playing and saving in OOTP, exporting CSV and HTML, recording and editing video, choosing stories, and approval and upload are operator tasks.
- HTML website upload, StatsPlus external update, publication destinations, and postseason and seasonal rollover automation need operator evidence. The current Newsroom pipeline verifies regular-season results.
