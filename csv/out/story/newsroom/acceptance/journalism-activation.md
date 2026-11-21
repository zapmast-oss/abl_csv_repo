# ABL Newsroom | Journalism activation

Commissioner's implementation record  |  Branch add-star-schema

## What ran

Production Control filed eight franchise articles, one division report and Ernie's league briefing. All 24 intelligence notebooks remain available. The 24-team slate ranks 140 investigations, including 61 eligible pitches. All reporting is internal.

All 72 live CSVs matched the repository. The league date is August 20, 1981. All 12 August 19 games were complete, bringing the regular-season count to 1,450.

## Implementation

Under csv/abl_scripts/, abl_newsroom.py extends run and reading_index to admit a separately archived journalism edition and link it to its intelligence edition. abl_newsroom_print.py extends write_report with article typography and paired reading copies.

abl_newsroom_slate.py adds build_slate and assess for ranking, story histories and changed evidence. abl_newsroom_east.py adds EastDesk and build_east for the four franchise investigations and their division report. abl_newsroom_observer.py adds prepare_observer for Ernie's cross-division assignment.

abl_newsroom_editorial.py adds prepare_editorial, reported_brief, record_pitches and audit_editorial. abl_newsroom_factcheck.py adds ClaimCheck.evaluate for source tables, identities, game status, dates, scores, standings and transactions. scripts/abl_newsroom_control.py extends current_issue_matches to check the journalism files before skipping a repeat export.

## Verification

Ten newsroom regression tests and nine Stage 1 guardrail tests passed. In csv/abl_scripts/test_stage1_guardrails.py, historical fixtures now come from their pinned snapshot and the registry test covers the original and repaired registries. The 24-club team helper also passed.

tools/newsroom/test_east.py, test_slate.py and test_editorial_integration.py cover the new workflow. The integration run admitted 12 new games across two real exports, retained earlier investigations, rejected an incomplete export and treated a byte-only change as a quiet update. A scoring change was tested on an isolated copy. Earlier editions retained their original hashes.

All 11 editorial PDFs matched their Markdown sources. All 18 rendered pages were checked visually. The live index had 82 working links. The Windows monitor completed its post-activation check successfully.

## Where to read

The current issue opens from csv/out/story/newsroom/index.html. Under journalism/1981-08-19_2acae29146076f62/, beat/ holds team articles, league/ holds division reporting and observer/ holds Ernie's briefing. story-slate.pdf gives the reading copy. slate.json holds the investigations and their history. articles.json carries claim checks and source selectors. Acceptance receipts sit alongside this report.
