# StatsPlus Story-Signal Enablement Review

**No signals are enabled by this review.**

| Signal | Tables | Review status | Enabled now | Boundary |
|---|---|---|---|---|
| playoff odds pressure | 7\|8 | READY_AFTER_PROMOTION | no | Model probability; never standings proof. |
| ELO/team strength | 10 | READY_AFTER_PROMOTION | no | Labeled StatsPlus model value. |
| BaseRuns over/under-performance | 9 | READY_AFTER_PROMOTION | no | Expected-performance model, not observed result. |
| team WAR | 11 | READY_AFTER_PROMOTION | no | Labeled StatsPlus team value. |
| injury context | 12 | READY_AFTER_PROMOTION | no | Team summary; avoid unsupported player causation. |
| fan interest | 4\|5 | READY_AFTER_PROMOTION | no | Do not infer emotion beyond measured fields. |
| financial pressure | 3 | READY_AFTER_PROMOTION | no | Resource/results context; no blame or motive. |
| owner/front-office/personnel context | 1\|2 | READY_WITH_LIMITED_SIGNALS | no | Table 1 goals/mood absent; staff IDs remain unavailable. |
| team age/context | 24 | READY_AFTER_PROMOTION | no | Adds lower-level age detail. |
| best batting game | 25 | READY_AFTER_PROMOTION | no | StatsPlus ranking; raw game proof remains authority. |
| best pitching game | 26 | READY_AFTER_PROMOTION | no | StatsPlus ranking; raw game proof remains authority. |
| team/player baserunning | 19\|22 | READY_AFTER_PROMOTION | no | Label StatsPlus definitions. |
| UBR | 19 | READY_AFTER_PROMOTION | no | Usable as StatsPlus-specific team UBR; no silent substitution. |
| rWAR | 21 | READY_AFTER_PROMOTION | no | Usable as StatsPlus-specific rWAR; does not override WAR. |

UBR and rWAR are usable after promotion as explicitly labeled StatsPlus values. Owner/personnel context is limited by absent owner goal fields and unresolved staff IDs.
