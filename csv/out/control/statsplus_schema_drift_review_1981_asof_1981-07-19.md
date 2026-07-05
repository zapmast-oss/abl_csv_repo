# StatsPlus Schema-Drift and Overlap Review

| Table | Family | Old/new cols | Change | Recommendation | Confidence |
|---|---|---|---|---|---|
| 1 | owner info | 14/17 | semantic reshape with partial loss | ACCEPT_WITH_LIMITED_SIGNALS | 0.96 |
| 4 | historical fan interest | 12/12 | layout-only | ACCEPT_SCHEMA_DRIFT | 0.99 |
| 15 | team pitching by division | 19/23 | additive | ACCEPT_ADDED_DETAIL | 0.99 |
| 16 | team pitching by league | 19/23 | additive | ACCEPT_ADDED_DETAIL | 0.99 |
| 19 | team baserunning | 9/11 | additive | ACCEPT_ADDED_DETAIL | 0.98 |
| 21 | player pitching | 22/23 | additive | ACCEPT_ADDED_DETAIL | 0.98 |
| 24 | team age | 14/20 | additive | ACCEPT_ADDED_DETAIL | 0.99 |
| 7 | playoff odds by division | 13/13 | reordered overlap | ACCEPT_SCHEMA_DRIFT | 1.00 |
| 8 | playoff odds by league | 13/13 | reordered overlap | ACCEPT_SCHEMA_DRIFT | 1.00 |

## Specific decisions

- **Table 19 UBR:** usable after promotion as a labeled StatsPlus-specific team baserunning value. It must not silently substitute for a differently defined sortable field.
- **Table 21 rWAR:** usable after promotion as labeled StatsPlus rWAR context. It does not override WAR, raw game proof, or sortable authority.
- **Tables 7 and 8:** same row multiset, different order. Both schemas and values are safe, but their two sort/report identities must not be double counted.
- **Table 1:** retained traits are safe under a reshaped field contract; mood/objective/specific goals remain unavailable.
