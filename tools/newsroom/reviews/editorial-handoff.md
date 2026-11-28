# ABL Newsroom | Editorial handoff

Production Control and the internal reading edition

## After the commissioner exports

Production Control checks for a changed OOTP export every five minutes while the commissioner is signed in. Run ABL Newsroom Now performs the same check immediately. It verifies the source, updates the 24 notebooks, builds the review queue and writes the six-division ABL 411. A day can close with zero approved articles.

## How to file the next story

Open csv/out/story/newsroom/index.html and follow Editorial queue and follow-ups. For a continuing story, start with the prior article, its thread IDs, and the new completed game or personnel record. Read the scoring corrections before using an earlier line. If the baseball adds little, leave the item on the desk.

Write one reviewed JSON manuscript under tools/newsroom/reviews/<league date>/. The two August 20 files serve as worked examples. Match the exact export ID, league date and cutoff. Give each material claim a row selector. Identify game IDs and the sentences that depend on them in claim_map. Keep final scores separate from scheduled fixtures.

Read the finished article against the records. Record how the headline is answered, the central question, the sequence, the stakes and the specific next result. The reviewer records a decision and name. Run abl_newsroom.py --review-digest MANUSCRIPT.json and enter the result in review.copy_sha256 after the wording and evidence have been checked.

From the repository root, use the same Python interpreter as Run ABL Newsroom Now. Run python -B csv/abl_scripts/abl_newsroom.py --publish MANUSCRIPT.json. Several reviewed files may be submitted together. The command verifies them against the admitted source and files an internal issue revision. Use the same slug to replace earlier copy. Earlier revisions remain available from the reading page.

## August 20 implementation and checks

The changed production files are csv/abl_scripts/abl_newsroom.py, abl_newsroom_editorial.py, abl_newsroom_dailydesk.py, abl_newsroom_reading.py, abl_newsroom_slate.py and scripts/abl_newsroom_control.py. Source manuscripts are in tools/newsroom/reviews/1981-08-20/. The reading page, edition pointer, daily desk, reviewed copies and acceptance receipts are under csv/out/story/newsroom/.

Regression coverage lives in tools/newsroom/test_daily_desk.py, test_newsroom_control.py and test_editorial_integration.py. Reproduce the checks with python -B -m unittest discover -s tools/newsroom, python -B -m unittest discover -s csv/abl_scripts -p test_stage1_guardrails.py and python -B tools/newsroom/test_editorial_integration.py. The integration test uses its own temporary archive.
