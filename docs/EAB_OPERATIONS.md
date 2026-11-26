# Experiencing Action Baseball

Open **ABL Production Control → Run EAB Publishing → Open the public EAB website**. The homepage opens directly to the current baseball edition. The existing **Open EAB Website** entry opens the [local edition](../output/eab/dist/index.html). **Edition timeline** remains available at `#editions`. A bookmarked edition keeps its own league date and cutoff. [Public EAB website](https://zapmast-oss.github.io/experiencing-action-baseball/) · [EAB GitHub repository](https://github.com/zapmast-oss/experiencing-action-baseball)

## Opening edition

The verified league date is **August 20, 1981**, with **1,450 regular-season finals through August 19**. All 72 source files agreed with the promoted repository export. The older July baseline belongs to a prior production cycle. The internal Newsroom holds the two sealed source manuscripts. EAB serves the authorized reading copies with the verified baseball record and beat pages for all 24 clubs.

EAB uses the existing OOTP capture and preflight, league and franchise identifiers, approved logos and dated team portraits, six division races and season-correct DCS rules, archived scores and club records, and the Newsroom’s sealed issue and review format. It also renders 1972–1980 regular-season batting and pitching using OOTP player IDs. The current player lines derive from the admitted player-game records. Attendance comes from completed home games with recorded positive attendance. [Source authority](ABL_SOURCE_AUTHORITY_RULES.md) · [Franchise reference](ABL_LEAGUE_AND_FRANCHISE_REFERENCE.md) · [Postseason rules](ABL_POSTSEASON_RULES.md)

The source pack describes the older baseline, Story Engine and editorial standards. The current Newsroom adds the Desk, 24 beat assignments and immutable reviewed issues. EAB adds a static visitor site and its local release history around those existing systems. [ABL source pack](../_chatgpt_project_source/00_START_HERE.md)

## Produce an update

1. Play and save in OOTP. Run **ABL Production Control → Run ABL Production Session** and complete the existing safety copy and CSV validation. Standard and Featured sessions offer capture immediately. Quick Play can capture when resumed. [Command Center operating guide](ABL_COMMAND_CENTER.md)
2. EAB verifies the exact capture and its cutoff, builds the baseball edition and switches the local current page. The session then submits that edition to GitHub Pages and checks the public timeline and cutoff. The local build and public deployment have separate stage receipts. Validated baseball publishes independently of journalism or video. If local building fails, review the EAB stage log and resume or select **Publish latest verified edition** after correcting the source.
3. For a story, use **Manage publication approvals and receipts** in the existing session console. Queue the exact sealed `.md` manuscript under **Editorial** with channel **EAB**, then approve that exact artifact. EAB matches the receipt’s hash and cutoff to a verified edition and rechecks the newsroom seal. Select **Publish latest verified edition** to include newly eligible copy. Safe HTML and an available reading PDF use the approved text. Internal evidence is excluded. [Editorial review handoff](../tools/newsroom/reviews/editorial-handoff.md)

**Run EAB Publishing** also opens a separate local preview, checks public publication status, restores a previous edition and builds the deployment ZIP. Preview files live under `output/eab/preview/`. Local build and public deployment receipts are under `logs/eab/`. The Windows menu and launchers use the same entry points as before. Recovery copies of the changed source files are under `logs/eab/source-checkpoints/`.

The Newsroom’s `articles.json` and `files.json` seal each reviewed issue under `csv/out/story/newsroom/issues/`. Internal review and EAB channel approval are separate decisions. For an already published baseball cutoff without a protected playing session in Command Center, `approve-article` can record an explicit human EAB decision for an exact sealed manuscript. It rechecks the published source, seal and manuscript hash and stores its approval privately under `logs/eab/article-approvals/`. Run **Publish latest verified edition** after recording approval.

## Reconciled batting lines

The newsroom and EAB read the same verified player-game records. A repeated batting line can be reconciled when a sealed earlier capture shows the same unfinished game and exactly one starter line with plate appearances. The completed capture must preserve that row exactly. Its second row must match the club, player, position, split and stint, report one appearance with no start, and carry zero batting, baserunning and other event values. Both captures must contain the same plate-appearance records. The verified completed play-by-play must corroborate them in the innings already played at the earlier cutoff. Any unexplained duplicate stops validation. A unique zero-PA appearance remains in the derived records, including recorded runs and stolen bases.

Preserve the earlier sealed capture in `output/newsroom/snapshots/`. The verifier writes `player-game-reconciliation.json` alongside the preflight whenever it reconciles a duplicate. The newsroom also seals the receipt in its edition's `verification.json`. EAB records the same evidence in `logs/eab/publications.jsonl` and shows differences from native OOTP batting totals under **Players & statistics**. The private receipt identifies both capture hashes, the original row numbers and hashes, the plate-appearance records, the play-by-play references and the season comparison.

In the August 30, 1981 cutoff replay, OOTP's native player totals credit Brent Keyser with 83 appearances and Chris Sullivan with 93. The verified game records give them 82 and 92, respectively. St. Louis has 5,046 PA, 4,352 AB and 1,179 summed player starts in the derived totals, compared with 5,044 PA, 4,350 AB and 1,178 starts in OOTP's club totals. Houston has 5,106 PA, 4,477 AB, 1,179 summed starts and 793 strikeouts, compared with 5,104 PA, 4,475 AB, 1,178 starts and 792 strikeouts. All 24 club standings and the batting run and hit totals agree with the verified finals. Player pitch counts are reported separately in the private receipt. The game batting lines carry zero pitches seen, while the native player season table provides those counts. Closed seasons continue to use the historical sources.

The targeted regression test runs with `python -B tools/newsroom/test_player_game_reconciliation.py`. Add `--replay` in an isolated checkout containing the two sealed snapshots and normal repository data to exercise the newsroom and EAB builds locally.

## August 20 editorial review

- **Ernie Bewell, “San Francisco scored 18 and still couldn’t shake Phoenix.”** Verified the July 26 six-game margin, the eight-game losing streak, the 64–57 tie, San Francisco’s 18–2 win and Luke Margalit’s five RBIs. Phoenix’s 3–0 win and the four scheduled games support the article’s pre-series framing. [Sealed manuscript](../csv/out/story/newsroom/issues/1981-08-19_2acae29146076f62/d486b29935a56f9145e3/observer/phoenix-san-francisco-first-place.md)
- **ABL League Desk, “Miami needs a game before Charlotte comes to town.”** Verified Charlotte’s 4½-game lead, the July 26 finish, both clubs’ six remaining games before August 27 and the conditional standings arithmetic. [Sealed manuscript](../csv/out/story/newsroom/issues/1981-08-19_2acae29146076f62/d486b29935a56f9145e3/league/miami-needs-a-game-before-charlotte.md)

Both manuscripts carry the August 19 completed-game cutoff. The commissioner approved both sealed manuscripts for EAB. The 24 franchise notebooks are reporting material. The older automated package contains eight superseded beat drafts plus a league draft and an Observer draft. They require fresh editorial review before release. [Internal newsroom](../csv/out/story/newsroom/index.html)

## Verify and recover

Each edition under `output/eab/dist/releases/` carries a checksum manifest with its OOTP source fingerprint, simulated date, completed-game cutoff, edition timestamp and eligible issue versions. Shared artwork and almanac pages use content-derived paths. Each dated homepage and published issue has a permanent URL. Approved corrections receive new URLs and retain their predecessors. An identical rebuild keeps the current edition ID. The root `index.html` switches only after the required files, internal links, anchors and checksums pass validation.

The homepage labels the ABL baseball date and its publication revision separately. The timeline numbers revisions in publication order within each baseball date and marks the active edition as current. Newly sealed homepages also carry the publication ID from their permanent URL. A changed verified capture, newly eligible journalism or a changed site build can create a new snapshot. Repeating the same build uses its existing edition, and restoring an earlier edition keeps all revision numbers and archive URLs.

For recovery, choose **Restore an earlier local and public edition** and enter its ID. EAB verifies the selected edition and publishes its pointer through a new Git commit. The timeline still lists later issues and editions. Publishing the latest verified source again selects its existing edition when the inputs have stayed the same.

Run the following from the repository with the production Python runtime. The deployment tests use a disposable Git branch and simulated Pages responses. The original `tools/eab/test_abl_eab.py` replay uses local Git LFS snapshots and the existing PDF-capable Newsroom dependencies.

```powershell
python -B tools/eab/abl_eab.py status
python -B tools/eab/abl_eab.py pages-status
python -B tools/eab/abl_eab.py verify
python -B -m unittest discover -s tools/production -p 'test*.py'
python -B -m unittest discover -s tools/eab -p 'test_github_pages.py'
```

## Opening-day verification

The acceptance replay uses the verified August 18 and August 19 captures. It checks repeat builds, the cutoff change, prior editions and issue URLs, rejected and regressing input, an interruption before switching the pointer, retry and rollback, the public allowlist, all 24 team beats, game and manuscript cutoffs, and a baseball update with an ineligible issue and no new broadcast. The Command Center regression suite checks that EAB runs after validation and before a failed Desk attempt, while the protected production session can still close.

Chrome checks used desktop, tablet and phone widths. They covered the current edition, each franchise beat, division standings, score and schedule navigation, historical player statistics, search, both newsroom issue versions, and the generated PDF link. The replay inspected the PDF text against the original publishing cutoff. The opening ZIP passed its integrity check and its homepage matches the current local edition.

Public HTTPS checks matched the verified homepage, active manifest, box-score day and previous edition byte for byte. Public Chrome checks covered all 24 beats, the current issue, those game and statistics sections, search, artwork, Fan Culture, Sim Baseball Vision and the earlier edition. Reviewed articles and PDFs require EAB approval and were checked in the existing local replay. Defending the Shield returned HTTP 200 with the same homepage hash before and after the EAB launch.

## Publish to GitHub Pages

### Website traffic

Choose **View EAB Website Traffic** in the local Command Center to open Cloudflare Web Analytics. Create a Web Analytics site for `zapmast-oss.github.io` using manual setup. In **Manage site**, copy the public `token` value from the install snippet, then use **Run EAB Publishing → Configure EAB Website Traffic** to save it. Use **Publish latest verified edition** to activate it. The builder reads the saved beacon value from the ignored `output/eab/cloudflare-beacon-token.txt` file. The value is the public embed identifier. Cloudflare API credentials stay outside this workflow. [Cloudflare setup](https://developers.cloudflare.com/web-analytics/get-started/)

Cloudflare reports page views, visits, paths and referrers. Visits measure entrances and provide an audience proxy. They do not count distinct people. Filter by the `/experiencing-action-baseball/` path prefix, and by page paths for newsroom articles, teams, standings, players and history. The configured build includes the beacon on generated HTML. Current links to pre-analytics newsroom issues and articles use tracked reading copies. Original editions and direct links to earlier uninstrumented pages remain available. PDF files are served as documents. Reporting starts with the first successful live beacon submission after deployment. [Cloudflare metrics](https://developers.cloudflare.com/web-analytics/data-metrics/high-level-metrics/) · [Cloudflare dimensions](https://developers.cloudflare.com/web-analytics/data-metrics/dimensions/)

Check the browser Network tab for `static.cloudflareinsights.com/beacon.min.js` and a successful request to `cloudflareinsights.com/cdn-cgi/rum`. Confirm the visit appears in Cloudflare Web Analytics. An unchanged verified build with the same beacon uses its existing revision. For an unconfigured site, the builder emits no beacon.

The dedicated repository publishes `main` at its project URL over HTTPS. GitHub's built-in Pages workflow serves the static files with its managed permissions. The ABL computer prepares each release. The publisher uses the same file allowlist and checksums as the ZIP, verifies the staged Git files byte for byte, and pushes a fast-forward commit. Each commit preserves the immutable paths already on GitHub. The browser can still open earlier editions, issue revisions and article URLs. Git pushes use the existing `zapmast-oss` credential from Windows Git Credential Manager. [Repository deployments](https://github.com/zapmast-oss/experiencing-action-baseball/actions)

**Check local and public publication status** displays the active local edition, the latest deployment attempt and the actual public cutoff. A submitted commit can be pending while GitHub builds it. A connectivity or deployment failure leaves the local edition available and the session can close. Check status first, then use **Retry public deployment of the current local edition** if needed. A retry republishes the same verified tree when GitHub has the commit and Pages still serves another edition. An edition already confirmed on Pages creates no new commit. For a failed restore, repeat the restore operation. Check the session’s EAB Pages log, `logs/eab/github-pages.jsonl` and the repository deployments for details. If Windows no longer has an authorized credential, sign in again as `zapmast-oss` in Git Credential Manager. [Repository deployments](https://github.com/zapmast-oss/experiencing-action-baseball/actions)

The opening site is about 40 MB uncompressed. The second stored edition added about 12.4 MB, so another 60 editions of that size would approach the 800 MB warning threshold. Shared artwork and almanacs keep their content-derived URLs. The publisher stops before the 1 GB Pages site limit. If growth reaches the warning threshold, move older HTML editions to an authorized static archive with the same directory layout. Keep redirect pages at the original HTML URLs and retain existing PDF and media URLs on Pages. Verify moved editions and deep links before changing the publication manifest. [GitHub Pages limits](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits)

## Source limits

The automatic verification gate covers the regular season. The postseason picture is current standings arithmetic, with tied races carried forward. Current StatsPlus odds are unavailable at this cutoff. The almanac explains gaps in matched game records. Fan Culture uses reported attendance and venue information. SBV links point to the commissioner’s published programming and discussions. ABL Advance opens the game calendar while an episode-specific destination is unverified. [Sim Baseball Vision](https://www.youtube.com/channel/UCd1wShl1bahaTldUtqVZVwg) · [League discussions](https://forums.ootpdevelopments.com/showthread.php?p=5049449)
