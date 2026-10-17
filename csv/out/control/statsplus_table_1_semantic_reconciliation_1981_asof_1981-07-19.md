# StatsPlus Table 1 Semantic Reconciliation

**Classification:** `PARTIAL_MEANING_LOSS`  
**Recommendation:** `PROMOTE_ACCEPT_RESHAPED_SCHEMA`  
**Confidence:** 0.96

## Verdict

Table 1 retains the same owner-context family and all 24 team entities. Twenty-three owner names match; Chicago changes from Art Roy to Willie Roy, consistent with a current personnel replacement rather than a join failure.

The legacy grouped fields are substantially preserved in normalized columns. `Negotiation Tendencies` maps to `POS` and `NEG`; `Management Style` maps to `Patience`, `Spending`, and `Involvement`; `Financial Goal` maps to `Priority`. Personality, patience, and involvement mappings are stable apart from the changed Chicago owner. Several spending labels changed from legacy `Economizer` to current `Normal`, which is a value/category update rather than disappearance of the spending concept.

Actual semantic loss exists: `Owner Mood`, `Season Objective`, and `Specific Goals` have no fresh equivalents. Conference is absent but derivable from governed team data. The fresh table adds explicit position, job, league, level, and nationality fields.

Table 1 is safe to promote for retained current owner traits with a limited field contract. It is not a drop-in replacement for mood/objective/goal signals, which must remain unavailable.

## Data-element map

| Legacy element | Fresh element | Status | Evidence |
|---|---|---|---|
| Number |  | derived/removed | Legacy display ordinal has no source meaning; fresh table is keyed by team/name. |
| Owner Name | Name | retained | Owner name; 23/24 identical, with Chicago showing a current personnel change. |
| Team | TM | retained_normalized | Full team name becomes current team abbreviation; all 24 teams resolve. |
| Conference |  | removed_but_derivable | Conference is absent but can be derived from governed current team/division data. |
| Age | Age | retained | Direct field mapping. |
| Years of Experience | EXP | retained_normalized | Numeric years become formatted years. |
| Reputation | Rep | retained_normalized | Same concept; current export uses its present label scale. |
| Personality Type | Type | retained_separated | 23/24 exact; changed Chicago owner accounts for the exception. |
| Negotiation Tendencies | POS + NEG | retained_separated | Two grouped values become separate positive/negative relationship columns; 23/24 match. |
| Management Style | Patience + Spending + Involvement | retained_separated | Three grouped values become separate fields; Patience and Involvement match 24/24; Spending matches 16/24, with current/normalized category changes. |
| Financial Goal | Priority | retained_normalized | 23/24 exact; changed Chicago owner accounts for the exception. |
| Owner Mood |  | removed | No corresponding fresh field. |
| Season Objective |  | removed | No corresponding fresh field. |
| Specific Goals |  | removed | No corresponding fresh field. |
|  | Pos | added_layout | Explicit personnel-position code. |
|  | Job | added_layout | Explicit job code. |
|  | LG | added_context | Explicit league code; not a replacement for legacy conference. |
|  | Lev | added_context | Explicit competition level. |
|  | NAT | added_detail | Owner nationality. |

## Sample comparisons

| Team | Legacy/fresh owner | Legacy negotiation | Fresh POS/NEG | Legacy management | Fresh separated management |
|---|---|---|---|---|---|
| Boston Patriots | Alex Mendez / Alex Mendez | Temperamental and Personable | Temperamental / Personable | Understanding, Controlling, Hands-off | Understanding / Controlling / Hands-off |
| Chicago Fire | Art Roy / Willie Roy | Temperamental and Personable | Personable / Controlling | Demanding, Economizer, Hands-off | Demanding / Controlling / Hands-off |
| Las Vegas Gamblers | Terry Vinson / Terry Vinson | Personable and Easygoing | Personable / Easygoing | Demanding, Contolling, Hands-off | Demanding / Controlling / Hands-off |

**Risk:** downstream consumers must not infer missing mood or goals, and must not turn owner traits into motives.
