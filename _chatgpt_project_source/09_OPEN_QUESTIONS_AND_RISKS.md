# Open Questions and Risks

**Rule:** an unresolved item is not permission to guess.

| Issue | Why it matters | Evidence | Recommended action |
|---|---|---|---|
| 1981 wild-card/tournament rules are UNKNOWN | “If season ended today,” qualification, clinch, and elimination claims may be wrong | Story-engine Sprint 1 notes say no verified rules table was found | Obtain authoritative 1981 rules and add a versioned rules source before calculating |
| July 20 results excluded | A result would contaminate a pregame/as-of-July-19 package | Preflight and every current script declare cutoff | Keep language pregame; create a new dated run after games complete |
| Week 5 and Week 15 artifacts coexist | Easy to mistake historical snapshots for current state | Week 5 files and July 12 Week 15 docs remain in repo | Use `1981_07_20_asof_1981-07-19` artifacts for current work |
| README branch is stale | Can misstate development context | README says `refactor-output-audit`; git says `add-star-schema` | Trust git; update README separately only if requested |
| Git cannot prove “Codex” authorship | Attribution may be overstated | Recent commits list `zapmast-oss` | Say “appears to have produced” or attribute to recent repository work |
| Final human approval/publication UNKNOWN | “Safe for EB use” is not proof of publishing | Generated statuses exist; no publishing record inspected | Human-review, approve, narrate, publish, and record link/status |
| Scheduled starters unavailable | Pregame copy cannot name pitching matchup safely | Chicago–Dallas setup explicitly says unavailable | Ask for a cutoff-compatible probable-starters source |
| StatsPlus is preview-only | Treating it as official can alter story ranking or create false certainty | Feature flags disable ranking/new candidates; validation shows none changed | Keep authority labels; use raw OOTP for official facts |
| StatsPlus schema drift/semantic loss | Fields may be absent, renamed, or incomparable | Current-source status lists affected tables; owner mood/objectives unavailable | Consult signal dictionary and intake reports; do not infer missing fields |
| Tables 7 and 8 overlap | Odds may be double-counted | Current-source status calls overlap reordered | Use one appropriate view per claim and label model source |
| Name-based joins | Players/teams with similar names may mismatch | Story docs recommend stable IDs; StatsPlus commonly supplies names | Verify against OOTP IDs/team context before detailed claims |
| Aggregate injuries/fan/finance | Numerical context can invite unsupported causation or emotion | Current scripts repeatedly caution against it | State scale/contrast only; request longitudinal or player-linked evidence |
| Expected-record gaps | “Luck” and “regression” language overclaims models | Production package labels them watch conditions | Use descriptive language and monitor across more series |
| Temporary/binary files were committed | Noise can enter uploads and repo history | Recent commit added a `.pyc`; lock file was later deleted | Exclude caches/lock files from uploads; consider cleanup in a separate task |
| Mojibake appears in console rendering | Smart punctuation/names may be corrupted when copied through a wrong encoding | PowerShell inspection rendered UTF-8 punctuation as `â...` | Upload original UTF-8 files; preserve José, en dashes, curly apostrophes, and ⚾ |
| SimBaseballVision spelling varies | Brand name could be inconsistent | README uses unspaced form; user specifies “Sim Baseball Vision” | Use user-approved spaced name except when quoting |
| Current state will age | “Current” becomes stale after the next export/game | Current cutoff is fixed at July 19 | Always state cutoff and regenerate manifest/preflight for later work |

## What ChatGPT should request when data is missing

Ask for the intended newsroom date and cutoff, a fresh preflight, the relevant current evidence/production file, verified tournament rules for qualification questions, a focused raw export/query for unsupported game facts, or a current probable-starters/transactions/injury source. Never request the entire repository when a narrow extract will answer the question.

