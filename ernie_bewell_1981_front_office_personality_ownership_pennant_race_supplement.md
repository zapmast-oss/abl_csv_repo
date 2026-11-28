# ERNIE BEWELL — 1981 FRONT-OFFICE PERSONALITY, OWNERSHIP & PENNANT-RACE INTELLIGENCE

## Scope and reporting standard

This is a standalone companion to `ernie_bewell_abl_trade_deadline_front_office_movement_1981_08_01.md`. It adds people, franchise history, financial context, race structure, and baseball consequences; it does not repeat or replace the companion packet’s 66-deal transaction audit. The existing packet was not modified.

The cutoff is the August 1, 1981 pregame snapshot. The league export says its current date is August 1, the configured non-waiver deadline is July 31, and the parent ABL schedule is 162 games. All game results and player lines discussed below stop at July 31. Games listed for August 1 are scheduled, unplayed games—not results.

The evidence distinguishes three things that can easily be conflated: (1) a person’s current job, (2) OOTP’s strategic tendency fields, and (3) what a club actually did. Where the source records do not provide a scale key, no personality label is inferred from a number. A “trade count” describes activity, not motive, quality, risk tolerance, or a long-term plan.

## 1. What the GM export does—and does not—say

The current executive in a job is selected using `coaches.csv.position` (the current position), corroborated by `team_roster_staff.csv.general_manager`. `coaches.csv.occupation` is a preferred/normal occupation, not proof of the job the person currently holds. The OOTP documentation explicitly allows personnel to work outside their normal occupation. This distinction matters: for example, Cincinnati’s current GM is Phil Trammel even though his preferred occupation code is different.

Official OOTP personnel documentation says GM roster-strategy and valuation ratings affect budget allocation and roster construction. Its trading-strategy description says the tendencies concern trade frequency, aggressiveness in completing a trade, loyalty to long-time players (high loyalty means greater reluctance to move a fan favorite), and preference for younger players or veterans. OOTP documentation for AI player evaluation also describes weights for current-year, previous-year, and two-years-ago statistics. What the public sources found do not provide is an OOTP 26 CSV codebook that maps these exact export columns to numeric endpoints, sign direction, user-visible labels, or percentage weights. The existence of a similar global setting is not enough to decode an individual GM’s raw integer.

| Export field | Verified conceptual meaning | Numeric direction / valid range in this export |
|---|---|---|
| `trade_aggressiveness` | Trading-strategy tendency affecting how aggressively a GM tries to complete trades. | Exact CSV mapping and scale are unverified. In these 24 records, observed values are **−6 to 8**; this is an observed range, not a validated allowable range. |
| `trade_frequency` | Trading-strategy tendency related to how frequently a GM attempts trades. | Exact CSV mapping and scale are unverified. Observed **−1 to 1**, not a validated allowable range. |
| `player_loyalty` | High loyalty is documented as reluctance to part with long-time/fan-favorite players. | Direction of the concept is documented; mapping of exported numbers to “high” and the scale are not. Observed **0 to 4**. |
| `favor_veterans_to_prospects` | A GM’s roster/trade preference between veterans and younger players/prospects. | Concept is documented, but this column’s sign, anchors, and valid range are not. Observed **−6 to 5**. |
| `trade_preference` | Individual player preference in trade/valuation decisions; related OOTP settings describe favoring veterans or prospects. | The exact CSV encoding is not documented. Observed **−1 to 0**. Do not read zero or a negative number as a label. |
| `value_this_year` | A GM valuation field. OOTP evaluation documentation describes current-year statistical weight as one input to AI player evaluation. | Likely related in concept, but the export-to-interface mapping is not established. Observed **8 to 14**. Do not call these percentages. |
| `value_last_year` | A GM valuation field; analogous OOTP documentation discusses previous-year statistics in player evaluation. | Exact mapping, direction, and scale unverified. Observed **1 to 4**. |
| `value_two_years` | A GM valuation field; analogous OOTP documentation discusses statistics from two years ago. | Exact mapping, direction, and scale unverified. Observed **0 to 3**. |

The Personnel Editor’s general 1–200 statement concerns personnel ratings; it does not establish that every signed trading slider or these GM valuation exports use that same scale. The older OOTP manual’s global evaluation weights are also not a codebook for this individual-GM export. We therefore retain the raw values below as research leads, not as decoded personality profiles.

Other potentially relevant GM fields include `value_stats`, `draft_value`, `intl_fa_value`, `develop_value`, `ratings_value`, `manager_value`, plus age, experience, contract salary, and years remaining. Across these 24 current GM slots, the first five raw fields respectively range **4–7, −8–7, −9–9, −9–8, and −9–8**; `manager_value` ranges **45–124**; experience is **1–28 years**; and current contract salary is **$57,500–$223,500** for **1–5 years**. These are observed values, not verified allowable ranges, career accomplishments, or decoded quality labels. The raw `personality` and `management_style` fields should not be presented as a GM’s human personality: OOTP’s user-facing personality/style descriptions are principally documented for field managers and bench coaches. The current GM export supplies no quotes, career biography, private goals, or verified tenure-by-club history.

An older generated `z_ABL_Manager_Tendencies.csv` is expressly excluded. Repository lineage notes identify it as a stale 62-game team-manager snapshot, not a current August 1 GM personality source. It also describes on-field manager tactics, not front-office executives.

## 2. Ownership and financial setting

The raw OOTP staff file identifies the current owner by `position=9`; each of the 24 owner names below matches the current owner ID in `team_roster_staff.csv`. The repository also contains a StatsPlus owner snapshot with literal display labels for owner type, positive/negative tendencies, patience, spending/fiscal style, involvement, and priority. Those labels provide useful owner context, but the snapshot has no demonstrated capture timestamp tied to August 1; treat it as the owner-profile capture stored in the repository, not as a dated in-game instruction. OOTP’s public ownership manual explains patience, fiscal personality, priority, and expectations as separate concepts and says owners ordinarily set budgets and can act on expectations and team performance. This is a conceptual reference, not proof of any owner’s private directive.

`team_financials.csv` contains a 24-club financial snapshot with team budget, player payroll, market, fan-interest, fan-loyalty, cash, revenue/expense, and `owner_expectation` fields. The integer in `owner_expectation` has no local codebook and is not reliably a plain-language “win now” expectation. It is shown below as an uninterpreted export code. “Budget less payroll” is only gross arithmetic; it is not cash on hand or the amount truly available to sign a player, because other expenses, commitments, owner controls, and the game’s budget calculation matter. Market, fan-interest, and fan-loyalty values are likewise retained as exported indices; the repository did not establish their scale anchors.

| Club | Current owner | Type / positive / negative labels | Patience | Spending style | Involvement | Priority | Budget | Player payroll | Budget − payroll | Market / fan interest / loyalty | `owner_expectation` code |
|---|---|---|---|---|---|---|---:|---:|---:|---|---:|
| Miami Hurricanes | Danny Leon | Normal / Personable / Normal | Tolerant | Penny-pincher | Normal | Balanced | $9.50M | $7.76M | $1.74M | 6 / 82 / 8 | 4 |
| Phoenix Firebirds | Mike Dutcher | Normal / Temperamental / Normal | Tolerant | Generous | Normal | Balanced | $15.60M | $12.54M | $3.06M | 7 / 88 / 8 | 5 |
| Atlanta Kings | Tyler Carter | Personable / Normal / Easygoing | Demanding | Charitable | Hands-off | Balanced | $13.00M | $8.34M | $4.66M | 7 / 78 / 7 | 2 |
| Dallas Rustlers | Tommy Mauldin | Normal / Personable / Normal | Tolerant | Controlling | Hands-off | Balanced | $7.80M | $7.46M | $0.34M | 5 / 71 / 7 | 3 |
| Detroit Dukes | David Figueiredo | Easygoing / Normal / Temperamental | Lenient | Controlling | Hands-off | Winning | $11.40M | $9.98M | $1.42M | 10 / 94 / 7 | 5 |
| Charlotte Colonels | Adam Butler | Normal / Easygoing / Personable | Demanding | Normal | Hands-off | Profit | $11.20M | $9.08M | $2.12M | 7 / 80 / 8 | 4 |
| San Diego Seraphs | Danny Tocornal | Normal / Controlling / Easygoing | Lenient | Generous | Hands-off | Winning | $6.60M | $4.00M | $2.60M | 3 / 53 / 3 | 0 |
| Tampa Bay Storm | Jaylin Reed | Normal / Controlling / Temperamental | Demanding | Generous | Hands-off | Balanced | $13.00M | $9.63M | $3.37M | 6 / 66 / 7 | 4 |
| Los Angeles Cobras | Tony Cruz | Normal / Personable / Easygoing | Demanding | Controlling | Hands-off | Winning | $6.30M | $5.73M | $0.57M | 4 / 54 / 4 | 2 |
| San Francisco Warriors | Jon Teeple | Temperamental / Easygoing / Controlling | Tolerant | Normal | Hands-off | Balanced | $9.50M | $6.45M | $3.05M | 5 / 66 / 6 | 3 |
| Minneapolis Blizzard | Chad Purchase | Temperamental / Temperamental / Easygoing | Understanding | Controlling | Interfering | Profit | $7.00M | $4.72M | $2.28M | 5 / 58 / 6 | 0 |
| Chicago Fire | Willie Roy | Personable / Personable / Controlling | Demanding | Controlling | Hands-off | Winning | $10.70M | $8.98M | $1.72M | 5 / 82 / 7 | 4 |
| Denver Rocketeers | Ken Ellinger | Normal / Personable / Normal | Demanding | Penny-pincher | Hands-off | Balanced | $8.20M | $7.88M | $0.08M | 6 / 72 / 7 | 3 |
| Cincinnati Cougars | Ji-hu Ju | Personable / Controlling / Temperamental | Lenient | Controlling | Hands-off | Profit | $11.90M | $11.30M | $0.60M | 10 / 95 / 8 | 5 |
| New York Aces | Alberto Dominguez | Normal / Normal / Controlling | Understanding | Normal | Hands-off | Extreme Profit! | $11.30M | $6.97M | $4.33M | 6 / 80 / 7 | 1 |
| Nashville Blues | Chris Smith | Normal / Easygoing / Temperamental | Demanding | Charitable | Meddling | Profit | $13.40M | $8.00M | $5.40M | 11 / 94 / 7 | 1 |
| Las Vegas Gamblers | Terry Vinson | Normal / Personable / Easygoing | Demanding | Controlling | Hands-off | Winning | $6.50M | $5.65M | $0.85M | 4 / 55 / 5 | 4 |
| Portland Lumberjacks | Ben Criswell | Normal / Controlling / Personable | Unmerciful | Penny-pincher | Meddling | Balanced | $6.40M | $6.12M | $0.28M | 4 / 57 / 7 | 2 |
| St. Louis Stallions | J.D. Sommer | Normal / Personable / Easygoing | Demanding | Normal | Hands-off | Profit | $9.00M | $5.83M | $3.17M | 5 / 64 / 6 | 3 |
| Boston Patriots | Alex Mendez | Easygoing / Temperamental / Personable | Understanding | Controlling | Hands-off | Winning | $10.00M | $8.57M | $1.43M | 9 / 82 / 7 | 1 |
| Pittsburgh Express | Jared Brown | Normal / Personable / Controlling | Understanding | Normal | Meddling | Balanced | $7.50M | $7.47M | $0.03M | 4 / 58 / 5 | 2 |
| Philadelphia Fury | Gabe Eckard | Temperamental / Normal / Personable | Lenient | Normal | Hands-off | Profit | $11.60M | $8.47M | $3.13M | 7 / 70 / 6 | 0 |
| Houston Mavericks | Xavier Ferguson | Controlling / Temperamental / Personable | Unmerciful | Controlling | Hands-off | Balanced | $9.60M | $10.10M | −$0.50M | 5 / 80 / 8 | 5 |
| Seattle Comets | Tony Manning | Normal / Temperamental / Controlling | Lenient | Normal | Hands-off | Profit | $12.30M | $6.81M | $5.49M | 7 / 84 / 7 | 2 |

What this supports: owners have distinct recorded display labels and stated priorities, budgets differ substantially, and Houston’s listed player payroll is above its listed budget while Seattle’s is well below. “Type / positive / negative” above transcribes the owner snapshot’s fields; it is not a narrative diagnosis. What it does not support: that any owner ordered a specific trade, that a given GM was pressured or free to spend, or that priorities caused the transaction pattern. No owner goals, mood/opinion-on-performance report, private instructions, or owner/GM conversation was located.

## 3. The 24 current general managers

The raw tuple is printed in this order: `trade_aggressiveness / trade_frequency / player_loyalty / favor_veterans_to_prospects / trade_preference / value_this_year / value_last_year / value_two_years`. It is deliberately not translated into labels. “Off / regular / late” counts group the companion packet’s verified 66 transaction rows into pre-opening offseason (through April 5), regular season before the July 14 All-Star date (April 6–July 13), and post-All-Star through the July 31 deadline. They describe club involvement. The “notable move” is a verified example, not necessarily the club’s only transaction.

| Club | Current GM (experience in OOTP export) | Raw GM tuple | Trades: off / regular / late = total | What the actual record shows at the cutoff |
|---|---|---|---:|---|
| Miami | Daniel Baker (20) | −6 / 1 / 3 / −5 / −1 / 9 / 2 / 2 | 5 / 0 / 3 = 8 | 55–49, tied with Chicago, one game behind Dallas in the NBC wild-card race. A busy offseason followed by three post-break transactions; the late flow was mixed, not a pure buying or selling pattern. |
| Phoenix | Bryan Somers (10) | 0 / −1 / 1 / −5 / −1 / 12 / 2 / 1 | 2 / 0 / 2 = 4 | 53–51, fourth in the provisional NBC wild-card queue. One July 28 transaction with Houston exchanged LF Alex Monge for RHP Kerry Darling and minor-league LHP Willie Saunders (#19865); Phoenix gave up an established bat and got a veteran arm plus a younger pitcher. |
| Atlanta | Hector Naranjo (11) | 6 / −1 / 4 / 0 / 0 / 8 / 4 / 1 | 3 / 1 / 2 = 6 | 46–58, 16 games behind Charlotte; tied with Tampa Bay on record. Six trades and seven contract/signing notices in the audit window, but no evidence identifies the GM’s private plan. |
| Dallas | Jim Griffith (10) | 4 / 0 / 0 / −1 / 0 / 9 / 4 / 1 | 5 / 0 / 3 = 8 | 56–48, provisional NBC wild-card leader by one game. Acquisitions included Brian Manchester, Doug Farmer, and Eric Strickland (trade IDs #19798, #19799, #19849). Three late trades as the club held the first non-division slot. |
| Detroit | Jesus Martinez (15) | 1 / −1 / 3 / −1 / 0 / 8 / 4 / 2 | 3 / 0 / 2 = 5 | 57–47, leading the NBC Central. Acquired 30-year-old starter Kennie Murdock from San Diego for minor-league RHP Bob Morales (#19867); Murdock made one Detroit start by July 31. |
| Charlotte | Jose Ruiz (6) | 0 / 1 / 3 / 0 / 0 / 11 / 2 / 1 | 2 / 0 / 1 = 3 | 62–42, league-best winning percentage (.596) and 7 games ahead in the NBC East. Three total trades, two in the offseason and one late; it was comparatively quiet beside Houston’s 12, but “quiet” is not proof of satisfaction or reluctance. |
| San Diego | Matt Johnson (20) | −5 / −1 / 2 / −6 / −1 / 11 / 2 / 0 | 2 / 0 / 5 = 7 | 39–65, 19 games out in the NBC West. Five late-window deals included 33-year-old Victor Hernandez (#504) to San Francisco and 30-year-old Kennie Murdock to Detroit; the return classes differed. Specific pitcher departures support present-for-future language for those deals only, not a proven rebuild. |
| Tampa Bay | Angel Gonzalez (12) | −2 / 1 / 1 / 5 / 0 / 14 / 1 / 1 | 2 / 0 / 3 = 5 | 46–58, tied with Atlanta and 16 games behind Charlotte. Three late-window transactions, including sending Jeff Mitchell to Houston for two minor-league position players (#19853). No specific motivation appears in the records. |
| Los Angeles | Gary Jones (13) | −2 / 1 / 2 / −5 / −1 / 8 / 3 / 2 | 1 / 0 / 0 = 1 | 51–53, third in the NBC West and seven games behind San Francisco. One trade in the full window and none after the All-Star date. The export establishes low activity, not why. |
| San Francisco | Luis Martinez (5) | 7 / −1 / 0 / 1 / 0 / 11 / 3 / 0 | 3 / 0 / 1 = 4 | 58–46, leading the NBC West. Acquired reliever Victor Hernandez (#504) from San Diego for catcher Omari Benjamin (#19806). Hernandez had 17 saves and a 2.78 ERA before the trade date; no San Francisco appearance is recorded by cutoff. |
| Minneapolis | Ken Sears (6) | 5 / 0 / 3 / 2 / 0 / 11 / 3 / 1 | 3 / 1 / 1 = 5 | 46–58, 11 behind Detroit. Its July 31 trade sent 27-year-old LHP Chris Jenkins (9–6, 3.48 ERA, 98⅓ IP through the cutoff) to Houston for two minor-league prospects (#19908). The message and summary disagree on the prospects’ positions; avoid naming positions. |
| Chicago | Mike Hatton (28) | 8 / 0 / 2 / 1 / 0 / 11 / 3 / 1 | 2 / 0 / 5 = 7 | 55–49, two games behind Detroit and tied with Miami one game behind Dallas for the wild card. Five late-window deals; Chicago traded with Boston in both directions on July 27 and July 31 (#19851, #19901), a two-way roster exchange rather than a simple buyer/seller story. |
| Denver | Kevin Bradley (5) | 2 / 0 / 0 / −4 / −1 / 10 / 3 / 2 | 2 / 1 / 1 = 4 | 56–48, third in the ABC West, two behind Seattle for the wild card and four behind Las Vegas for the division. Acquired RHP Tony Oubre from New York in June (#19446); Oubre’s Denver starts through July 31 totaled 53⅓ IP with a 3.21 ERA. No July 24–31 trade. |
| Cincinnati | Phil Trammel (6) | −2 / 1 / 1 / 0 / 0 / 9 / 3 / 3 | 3 / 2 / 0 = 5 | 53–51, tied on record with Nashville and 8.5 games behind Houston. Cincinnati’s 1980 team won 97 games and its division; that is franchise context, not proof of 1981 intent. No post-All-Star transaction is listed in the audited window. |
| New York | Jeremy Burk (10) | −2 / −1 / 3 / −1 / 0 / 9 / 4 / 0 | 2 / 1 / 3 = 6 | 47–57, third in the ABC East. July 16 it sent LHP Devon John to Nashville for two minor-league players (#19729); June it sent Tony Oubre to Denver for a minor-league pitcher and prospect. Those flows moved a veteran pitcher outward in two packages; no long-term plan is documented. |
| Nashville | Justin Gomez (24) | 7 / 1 / 2 / −5 / −1 / 10 / 2 / 1 | 5 / 0 / 1 = 6 | 53–51, tied with Cincinnati. Its July 16 acquisition of Devon John for minor-league players (#19729) is a specific established-arm addition; the standings still put Nashville behind Houston and outside the current projected eight places. |
| Las Vegas | Roberto Choto (17) | 0 / 0 / 0 / −4 / −1 / 9 / 4 / 1 | 3 / 0 / 2 = 5 | 60–44, leading the ABC West. It added RHP Nate Souza from Portland (#19854) and RHP Bill Hall from Pittsburgh (#19878) in late July. They are two veteran pitching additions for a division leader; roles and intent are not stated. |
| Portland | Ben Hayter (12) | 2 / 0 / 2 / 4 / 0 / 9 / 4 / 1 | 4 / 0 / 1 = 5 | 41–63, 19 behind in the ABC West. Sent 33-year-old RHP Nate Souza to Las Vegas for minor-league RF Manny Moran and RHP Luis Gortarez (#19854). This is one observable veteran-for-younger-assets deal, not proof of a system-wide rebuild. |
| St. Louis | David Ward (6) | 5 / 1 / 3 / 4 / 0 / 9 / 4 / 3 | 2 / 0 / 3 = 5 | 48–55 through 103 games, fourth in the ABC Central. July deals sent RHP Harry Benjamin to Seattle (#19866) and 3B Chris Sullivan to Houston (#19889). St. Louis reached the tournament in three consecutive seasons, 1972–74, but did not qualify in 1978–80; current standings are remote. |
| Boston | Bobby Ramsey (14) | 0 / 0 / 3 / 1 / 0 / 10 / 2 / 2 | 3 / 1 / 2 = 6 | 54–50, leading the ABC East. Boston and Chicago exchanged players in late July (#19851, #19901), including 26-year-old Victor Hernandez in and RHP Justin Bowen out. The Martinez position differs between source texts; do not resolve it by guess. |
| Pittsburgh | Nick Heller (22) | 4 / 1 / 2 / 1 / 0 / 9 / 3 / 2 | 1 / 0 / 4 = 5 | 44–60, fourth in the ABC East. Sent 32-year-old RHP Ricky Bautista to Houston on July 31 for minor-league CF/RF Tim Mathis (position conflict) and RHP Tyler Hayes (#19904). This is a specific veteran arm for younger assets. |
| Philadelphia | Jon Ramos (10) | 6 / 1 / 2 / −2 / 0 / 9 / 3 / 2 | 3 / 0 / 0 = 3 | 48–56, second in the ABC East and six behind Boston. All three trades were in the offseason; no audited in-season move is recorded. The record shows inactivity late, not an owner or GM decision rationale. |
| Houston | Larry Bell (10) | 2 / 0 / 1 / 1 / 0 / 8 / 4 / 1 | 5 / 1 / 6 = 12 | 61–42 through 103 games, leading the ABC Central at .592; Charlotte is the league percentage leader at .596. Houston had a league-high 12 trades (14 player rows in, 21 out). It obtained RHP Ricky Bautista and LHP Chris Jenkins in **two separate July 31 trades** (#19904, #19908), after July 28’s Darling/Saunders deal (#19865). This is high-volume two-way turnover plus specific pitching additions, not “three deadline-day additions.” |
| Seattle | David Suarez (1) | −3 / 0 / 4 / −1 / 0 / 10 / 3 / 2 | 6 / 0 / 1 = 7 | 58–46, the ABC wild-card leader and second in the West. Added 30-year-old RHP Harry Benjamin from St. Louis for minor-league RF Isaiah Gibson (#19866). Seattle’s four pitcher contract notices and this trade are observable roster facts; their relation to owner Tony Manning’s recorded “Profit” priority is unknown. |

The time-slice count is useful as an index, but cannot establish personality. Charlotte’s three deals versus Houston’s twelve are the clearest volume contrast. Detroit, San Francisco, Las Vegas, Seattle, and Houston all added pitching while positioned in or close to a tournament place; their specific pitcher types and timing differ. San Diego and Minneapolis sent established arms to contenders for younger returns. Dallas and Chicago entered an active wild-card race through different transaction patterns. These are evidence-based questions for a reporter, not verdicts on who “won” the deadline.

## 4. Owner–GM relationship: what can be responsibly said

The 24 owner/GM pairs are documented above. A relationship or conflict is not. The stored records do not show how long the current GM or owner has held the job, whether either party approved a specific trade, or what private instructions were given. The owner priority labels can sharpen questions, but are not a causal key. Examples:

- Las Vegas owner Terry Vinson is labeled “Winning”; Las Vegas is first in the ABC West and added Souza and Hall. That is consistent with a win-oriented outcome at the snapshot, but it does not prove the owner prompted the additions.
- Boston owner Alex Mendez is labeled “Winning”; Boston leads the ABC East and made a late two-way exchange with Chicago. The source does not establish that the owner requested either player.
- Charlotte owner Adam Butler is labeled “Profit”; the Colonels lead the league in winning percentage and made three trades in the audit window. That does not prove the label caused quiet activity or that ownership is unhappy.
- Houston owner Xavier Ferguson is labeled “Balanced”; Houston’s payroll is about $0.50M above the listed budget in this snapshot and the club leads its division while trading at league-high volume. That supports a factual financial/roster contrast, not a story that the owner is pressing the GM.
- New York, Cincinnati, Nashville, and St. Louis carry profit-oriented labels in the owner extract while their 1981 records differ. The export has no owner-specific season goal with which to judge those teams.

OOTP’s public ownership manual describes owner priorities, budget control, patience, and season expectations as distinct systems. Here, owner patience/spending/involvement/priority labels come from the owner snapshot; budgets and payroll come from `team_financials.csv`; numeric `owner_expectation` comes from the raw export and remains undecoded. No mismatch in categories can be called a GM–owner conflict without direct evidence. Do not infer private motives from a roster move.

## 5. The ABL’s first nine completed seasons, and its tenth underway

The ABL began in 1972. At the August 1 snapshot, 1981 is its tenth season but is not complete. Therefore historical championship totals below cover the nine completed seasons, 1972–80, not a final 1972–81 decade record.

| Completed season | ABC division winners / WC | NBC division winners / WC | Tournament champion |
|---|---|---|---|
| 1972 | St. Louis, New York, Denver / Las Vegas | Chicago, Miami, San Francisco / Atlanta | Atlanta Kings |
| 1973 | St. Louis, New York, Denver / Nashville | Chicago, Miami, San Francisco / Phoenix | Denver Rocketeers |
| 1974 | St. Louis, Pittsburgh, Denver / Portland | Detroit, Miami, Phoenix / San Diego | Miami Hurricanes |
| 1975 | Houston, Pittsburgh, Seattle / New York | Detroit, Miami, San Francisco / Phoenix | Phoenix Firebirds |
| 1976 | Houston, New York, Portland / Cincinnati | Detroit, Atlanta, Phoenix / Tampa Bay | New York Aces |
| 1977 | Cincinnati, New York, Portland / Houston | Dallas, Atlanta, Los Angeles / Chicago | Dallas Rustlers |
| 1978 | Houston, Boston, Portland / New York | Minneapolis, Miami, San Diego / Phoenix | Miami Hurricanes |
| 1979 | Houston, Pittsburgh, Seattle / Cincinnati | Chicago, Miami, Phoenix / Detroit | Houston Mavericks |
| 1980 | Cincinnati, Pittsburgh, Las Vegas / Houston | Chicago, Charlotte, Phoenix / Detroit | Detroit Dukes |

No single club dominated this championship ledger: nine completed seasons produced eight distinct champions, with Miami the only repeat winner (1974 and 1978). Phoenix won in 1975; the 1976–80 run then produced five different champions—New York, Dallas, Miami, Houston, Detroit. Houston is the 1979 champion and Detroit enters 1981 as defending champion. Cincinnati’s 97–65 division-winning 1980 regular season is notable, but Detroit won the postseason. These are franchise facts; no evidence assigns those seasons to the current GMs or owners.

| Club | 1972–80 playoff appearances | Titles (year) |
|---|---:|---|
| Miami Hurricanes | 6 | 2 (1974, 1978) |
| Phoenix Firebirds | 7 | 1 (1975) |
| Atlanta Kings | 3 | 1 (1972) |
| Dallas Rustlers | 1 | 1 (1977) |
| Detroit Dukes | 5 | 1 (1980) |
| Charlotte Colonels | 1 | 0 |
| San Diego Seraphs | 2 | 0 |
| Tampa Bay Storm | 1 | 0 |
| Los Angeles Cobras | 1 | 0 |
| San Francisco Warriors | 3 | 0 |
| Minneapolis Blizzard | 1 | 0 |
| Chicago Fire | 5 | 0 |
| Denver Rocketeers | 3 | 1 (1973) |
| Cincinnati Cougars | 4 | 0 |
| New York Aces | 6 | 1 (1976) |
| Nashville Blues | 1 | 0 |
| Las Vegas Gamblers | 2 | 0 |
| Portland Lumberjacks | 4 | 0 |
| St. Louis Stallions | 3 | 0 |
| Boston Patriots | 1 | 0 |
| Pittsburgh Express | 4 | 0 |
| Philadelphia Fury | 0 | 0 |
| Houston Mavericks | 6 | 1 (1979) |
| Seattle Comets | 2 | 0 |

The repeated postseason franchises through 1980 include Phoenix (seven appearances), Miami/New York/Houston (six each), and Detroit/Chicago (five each). Philadelphia has no recorded appearance in the first nine seasons; that is a “still seeking a first tournament berth” fact, not a judgment on the current executive. Chicago’s five appearances without a title, Houston’s 1979 title, and Detroit’s 1980 title are the cleanest context for the present story. The current standings and moves—not inherited executive intent—must do the work of explaining 1981.

The season-level sources do not establish a durable historical rivalry narrative or an executive’s tenure across those years. `trade_history.csv` contains older transactions, but attributing a franchise’s old roster decisions to a current GM would be unsound without a dated GM-tenure history; this repository does not provide one. Current divisional races below are factual competitive matchups; calling them long-running rivalries would require a separate head-to-head and continuity investigation.

## 6. Pregame race board — August 1, 1981

There are six division places and two wild cards: three division winners and one wild card in each conference, under the stored ABL playoff configuration. All placements below are provisional standings arithmetic; tied records remain ties, and this is not a final bracket or a tiebreak ruling.

| Conference / race | Club | Record | Position / gap |
|---|---|---:|---|
| NBC East | Charlotte Colonels | 62–42 | 1st; leads Miami by 7 games |
|  | Miami Hurricanes | 55–49 | 2nd; 7 GB |
|  | Atlanta Kings | 46–58 | Tied 3rd; 16 GB |
|  | Tampa Bay Storm | 46–58 | Tied 3rd; 16 GB |
| NBC Central | Detroit Dukes | 57–47 | 1st |
|  | Dallas Rustlers | 56–48 | 2nd; 1 GB; provisional NBC WC leader |
|  | Chicago Fire | 55–49 | 3rd; 2 GB; tied with Miami one game behind WC lead |
|  | Minneapolis Blizzard | 46–58 | 4th; 11 GB |
| NBC West | San Francisco Warriors | 58–46 | 1st |
|  | Phoenix Firebirds | 53–51 | 2nd; 5 GB |
|  | Los Angeles Cobras | 51–53 | 3rd; 7 GB |
|  | San Diego Seraphs | 39–65 | 4th; 19 GB |
| ABC East | Boston Patriots | 54–50 | 1st |
|  | Philadelphia Fury | 48–56 | 2nd; 6 GB |
|  | New York Aces | 47–57 | 3rd; 7 GB |
|  | Pittsburgh Express | 44–60 | 4th; 10 GB |
| ABC Central | Houston Mavericks | 61–42 | 1st |
|  | Cincinnati Cougars | 53–51 | Tied 2nd; 8½ GB |
|  | Nashville Blues | 53–51 | Tied 2nd; 8½ GB |
|  | St. Louis Stallions | 48–55 | 4th; 13 GB after 103 games |
| ABC West | Las Vegas Gamblers | 60–44 | 1st |
|  | Seattle Comets | 58–46 | 2nd; 2 GB; provisional ABC WC leader |
|  | Denver Rocketeers | 56–48 | 3rd; 4 GB in division and 2 GB behind Seattle for WC |
|  | Portland Lumberjacks | 41–63 | 4th; 19 GB |

At this cutoff the division leaders are Charlotte, Detroit, San Francisco, Boston, Houston, and Las Vegas. Dallas and Seattle occupy the provisional wild-card spots. Miami and Chicago are tied one game behind Dallas. Denver is two games behind Seattle for the ABC wild card, and four games behind Las Vegas in the ABC West. Cincinnati and Nashville are tied on record, 8½ behind Houston; do not convert that record tie into a tiebreak result. St. Louis played one fewer game than most clubs, so compare both wins and games played.

The most direct pressure points are Detroit’s one-game Central lead over Dallas; San Francisco’s five-game cushion over Phoenix; the tight ABC West gap among Las Vegas, Seattle, and Denver; the Dallas/Miami/Chicago wild-card cluster; and Houston’s strong division lead with Cincinnati/Nashville tied behind it. Boston leads an East where the chasing teams are 6, 7, and 10 games back. A club can still influence the eight-place race without being its current leader, but do not predict who will qualify.

## 7. Baseball consequences of the most consequential moves

These roles and stat lines come from player-level season/game records through July 31 and the verified transaction inventory. Current-season totals spanning both clubs are not automatically receiving-club production. A possible role is labeled as such; no role is asserted without a roster or game record.

- **Detroit / Kennie Murdock (#19867).** The 30-year-old right-hander had been 10–6 with a 2.43 ERA over 137 innings for San Diego before the July 28 trade. Detroit sent minor-league RHP Bob Morales. Detroit’s game-level record shows one start by Murdock through July 31 (8 innings, 1 earned run). For a division leader, that is an actual rotation addition already tested once, not merely an announced veteran arm.
- **San Francisco / Victor Hernandez (#19806).** The 33-year-old right-hander had 17 saves and a 2.78 ERA before the July 25 deal; his season line shows a relief role with San Diego. San Francisco sent minor-league catcher Omari Benjamin. No San Francisco pitching appearance is in the cutoff game log, so “added a closer” or “became the closer” would go too far. The safe baseball angle is another experienced leverage option joining the NBC West leader.
- **Houston / Ricky Bautista and Chris Jenkins (#19904, #19908).** These are two separate July 31 trades. Bautista, 32, arrived from Pittsburgh as a right-handed pitcher; the available pre-trade season line shows 26 games, 70 innings, and three saves, but does not establish Houston’s intended role. Jenkins, 27, arrived from Minneapolis as a left-handed pitcher, with a 9–6 record, 3.48 ERA, and 98⅓ innings. Because both deals occurred July 31 and the export has no transaction timestamps, do not credit either pitcher with a Houston appearance unless a dated game record confirms it; none is established here. Two established arms add pitching depth/options to the 61–42 division leader, not a proven rotation/closer redesign.
- **Houston / Kerry Darling (#19865).** The July 28 deal brought in 35-year-old RHP Darling and minor-league LHP Willie Saunders while sending LF Alex Monge to Phoenix. It adds another experienced arm, but no verified role assignment is available in this packet.
- **Denver / Tony Oubre (#19446).** Acquired June 18 from New York, Oubre made later-date starts for Denver through the cutoff: 53⅓ innings with a 3.21 ERA in the post-trade starts counted by the verified packet. This is the clearest earlier-season rotation addition among Denver’s wild-card chase; the club still entered August two games behind Seattle.
- **Seattle / Harry Benjamin (#19866).** The 30-year-old right-hander came from St. Louis on July 28 for minor-league RF Isaiah Gibson. The available season line through cutoff is 4–4 with two saves in 38 innings before the move; the schedule/game records show one Seattle appearance for one inning. Do not assign a starting or closing job from that sample.
- **Las Vegas / Nate Souza and Bill Hall (#19854, #19878).** Two right-handed veteran pitchers arrived in late July for minor-league returns. Las Vegas led the ABC West at 60–44, so the baseball question is how the new innings fit behind a league-leading 3.31 team ERA noted in the verified packet—not whether the trades “guarantee” a pennant.
- **The seller-side counterpart.** San Diego’s veteran arms went to two NBC leaders; Minneapolis exchanged Jenkins for two minor-league prospects while 11 back; Pittsburgh sent Bautista to Houston; St. Louis sent Benjamin to Seattle. These transactions made specific contenders’ pitching staffs deeper while returning younger assets to the clubs moving those pitchers. Do not generalize each club’s entire plan from one or two trades.
- **Boston–Chicago, #19851 and #19901.** Two late-July deals moved established players both ways. Chicago received 34-year-old RHP Justin Bowen, who had 18 saves before July 31, plus a minor-league player in the July 31 deal; Boston received Miguel Martinez. The position field conflicts (first base vs. second base) across the matched transaction descriptions. Preserve that limitation; do not claim a definitive defensive fit.

The likely on-field consequences to watch are who takes the next rotation turns, whether San Francisco gives Hernandez late-inning work, whether Houston uses Bautista in relief or starts, and whether Seattle/Las Vegas can convert additional arms into quality innings. Those are questions for August game usage, not facts already in hand.

## 8. Ernie’s beat-writer notebook: what a ten-season ABL veteran notices

“Ten-season veteran” here means a writer following a league whose tenth campaign is underway, not ten completed seasons. The historical frame is the nine completed years above.

- Charlotte has the best record and the best winning percentage, yet only three audited trades. Houston, a Central leader with a lower .592 percentage, leads the league with twelve trades. That contrast is striking without being a moral: quiet can reflect a roster already built, constraints, valuation, or unrecorded negotiations; volume does not prove urgency.
- Detroit and San Francisco did not add interchangeable “pitchers.” Detroit bought a starter who had made one start for the Dukes by cutoff; San Francisco bought a 17-save reliever who had not yet appeared for the Warriors. The two leaders give Ernie different role questions.
- Dallas is only one game behind Detroit in the NBC Central and leads the NBC wild-card arithmetic. Its three July acquisitions—Manchester, Farmer, and Strickland—sit in a crowded race where Miami and Chicago are one game behind Dallas. Chicago’s two Boston exchanges make the Fire a two-way participant, not a simple buyer.
- Seattle’s move for Benjamin came while the Comets led the ABC wild-card race; Denver had added Oubre in June but had no July 24–31 deal. Denver is two back of Seattle for the wild card and four back of Las Vegas in the division. “Two games back” applies to the wild card; “four” to its division deficit.
- Boston worked with Chicago while holding the ABC East lead. Its lead is real, but the two exchanges and Martinez position conflict call for roster-usage reporting rather than a simplistic “all-in” label.
- San Diego and Minneapolis moved established pitching to clubs in the tournament picture. The return assets and each club’s position make rebuilding a reasonable question to ask, not an established organizational identity.
- The championship ledger gives Cincinnati a 1980 division crown, Houston the 1979 title, and Detroit the 1980 title. It does not make the current GM the author of prior success. Miami’s two championships and Phoenix’s seven early postseason trips are institutional memory; the current races are a new test.
- Ownership labels can inform interview questions—why Las Vegas’s “Winning” priority sits alongside a very modest market/payroll profile, or how Houston’s “Balanced” label coexists with heavy trade volume and payroll above budget—but the records do not show what an owner told the GM.

Strongest supported discoveries: (1) multiple contenders addressed pitching through very different roles; (2) a league-high trade count belongs to a leader whose incoming/outgoing player-row balance is outward; (3) two separate Houston deadline-day acquisitions must not be collapsed into one or inflated into three; and (4) raw GM tendency codes invite reporting but cannot yet supply the “personalities” as labels. There are no attributable fan quotes or verified reactions to these trades in the data reviewed.

## 9. What comes next (schedule, not forecast)

The current `games.csv` schedules the following twelve August 1 games; each is marked unplayed in the stored schedule. This list supplies a real next-game hook without importing a result after July 31:

| Game ID | Scheduled matchup |
|---:|---|
| 1249 | Phoenix at Atlanta |
| 1250 | Los Angeles at Charlotte |
| 1251 | Minneapolis at Miami |
| 1252 | Detroit at Tampa Bay |
| 1253 | San Francisco at Chicago |
| 1254 | San Diego at Dallas |
| 1255 | Las Vegas at Boston |
| 1256 | Denver at New York |
| 1257 | St. Louis at Philadelphia |
| 1258 | Nashville at Pittsburgh |
| 1259 | Seattle at Cincinnati |
| 1260 | Portland at Houston |

Watch the August 1 matchups that put recently active clubs under immediate test: Detroit at Tampa Bay, San Francisco at Chicago, Las Vegas at Boston, Denver at New York, Seattle at Cincinnati, and Portland at Houston. On the next game logs, check Murdock’s turn, any Hernandez/Bautista/Jenkins appearances, the deployment of Seattle’s Benjamin and Las Vegas’s new arms, and whether Boston’s two-way exchange changes its lineup or bullpen. None of those usage decisions is established by the pregame file.

The bigger questions are whether Dallas holds its provisional NBC wild-card lead over Chicago and Miami; whether Seattle protects its two-game cushion over Denver while also chasing Las Vegas; whether Houston’s pitching additions widen its Central lead or merely add depth; and whether Boston, Detroit, Charlotte, San Francisco, and Las Vegas preserve division positions as the schedule advances. All six division leaders and both current wild-card leaders are still playing for the eight tournament places. The results must answer what the transaction ledger cannot.

## 10. THE PENNANT RACE IS THE STORY

The July 31 deadline matters because it arrived with every division and wild-card route still legible—and still contestable. This is the tenth season since the 1972 start, with 1981 still in progress. Nine prior seasons have left a wide-open championship history: Miami repeated, the last five completed seasons had five different champions, Houston won in 1979, and Detroit won in 1980. That gives the writer institutional context without pretending that today’s executives authored yesterday’s banners.

The real people story is partly documented and partly still a reporting assignment. We can name all 24 current GMs and owners, describe the owners’ captured patience/spending/involvement/priority labels, and show the actual movement each front office made. We cannot responsibly turn undocumented numeric GM fields into familiar personality types, decode `owner_expectation`, or claim that one owner ordered one deal. The strongest evidence-based executive distinctions are observable: Charlotte made three trades; Houston made twelve; San Diego and Minneapolis moved established pitching; Detroit, San Francisco, Houston, Seattle, Las Vegas, and Denver added arms at different points in the race; Dallas, Miami, and Chicago confronted an unusually close NBC wild-card chase; Boston and Chicago exchanged players in both directions.

The baseball stakes are not abstract. A 10–6 starter joined Detroit and already made one start. A 17-save reliever joined San Francisco but had not pitched for the new club by the cutoff. Houston took two separate July 31 pitchers—one left-handed starter with 98⅓ innings, one right-hander with a mixed 70-inning line—and their future roles remain open. Seattle and Las Vegas added arms while chasing tournament security; Denver’s earlier addition was already working in the rotation. Those are things an August scorecard can verify.

That is what distinguishes this deadline from a transaction ledger: it connected named front offices to real roster questions, and those questions now sit inside live races. Dallas is one behind Detroit and only one game clear of Miami/Chicago for the provisional wild card; Seattle is two ahead of Denver in the wild-card race and Las Vegas remains four ahead of Denver in the division; Boston protects the East while Houston and Las Vegas lead their divisions; Charlotte owns the league’s best record. The game IDs for August 1 put those clubs on the field against one another or against immediate pursuers.

Ernie can build anticipation without selling a forecast. The clubs have made their choices; the schedule now tests what those choices are worth. The race is the story, and the next game is where the story moves.

## Source register and reproducibility

Primary repository files (absolute paths under `C:\sbv_repo\abl_csv_repo`):

- `C:\sbv_repo\abl_csv_repo\csv\ootp_csv\leagues.csv` — league ID 200, current date, July 31 deadline, 162-game schedule, playoff settings.
- `C:\sbv_repo\abl_csv_repo\csv\ootp_csv\coaches.csv` — GM and owner records, current position, preferred occupation, experience, raw strategy/valuation fields.
- `C:\sbv_repo\abl_csv_repo\csv\ootp_csv\team_roster_staff.csv` — current GM and owner IDs by club; cross-check for staff roles.
- `C:\sbv_repo\abl_csv_repo\csv\ootp_csv\teams.csv`, `divisions.csv`, `sub_leagues.csv` — canonical 24-team names and race alignment.
- `C:\sbv_repo\abl_csv_repo\csv\ootp_csv\team_record.csv` — August 1 pregame standings; paired with team IDs and league/division dimensions.
- `C:\sbv_repo\abl_csv_repo\csv\ootp_csv\team_financials.csv` — budgets, payroll, market/fan/financial fields, raw owner-expectation code.
- `C:\sbv_repo\abl_csv_repo\csv\statsplus\current\01_abl_transactions_personnel_-_owner.csv` — owner-specific display labels; the header includes both `Pos` and `POS`, so case-sensitive parsing is required. Names are matched to owner IDs in the raw staff files; no August 1 capture timestamp is encoded in this CSV.
- `C:\sbv_repo\abl_csv_repo\csv\ootp_csv\games.csv`, `games_score.csv`, `players_game_pitching_stats.csv`, `players_career_pitching_stats.csv`, `players.csv` — scheduled games, game-level appearances, and player identities/stat lines through cutoff.
- `C:\sbv_repo\abl_csv_repo\csv\ootp_csv\team_history.csv`, `team_history_record.csv`, `league_playoffs.csv`, and `league_playoff_fixtures.csv` — historical appearances/champions and playoff structure.
- `C:\sbv_repo\abl_csv_repo\csv\out\almanac\1972\league_champions_1972_league200.csv` through the corresponding 1980 file — season qualifiers, cross-checked against raw `team_history.csv` playoff flags; the 1981 pregame standings are from raw `team_record.csv`, not an old season report.
- `C:\sbv_repo\abl_csv_repo\csv\ootp_csv\trade_history.csv` and `messages.csv` — primary transaction records; the companion packet matches its 66-trade inventory to messages and preserves transaction IDs, date/position discrepancies, and row counts.
- `C:\sbv_repo\abl_csv_repo\csv\out\control\abl_batch_manifest_current.md` and `abl_source_registry.csv` — provenance/freshness caveat for generated material. The current manifest describes an earlier July 19 capture for some raw rows and misclassifies a nested manager-tendency artifact; it is not used to certify that stale artifact as current.
- `C:\sbv_repo\abl_csv_repo\ernie_bewell_abl_trade_deadline_front_office_movement_1981_08_01.md` — companion verified transaction inventory, date/message matching, player-row counts, cutoff-specific deal descriptions, and source conflict notes. It remains unchanged.

Official OOTP documentation consulted:

- [Personnel Ratings and Tendencies](https://wiki.ootpdevelopments.com/index.php?title=OOTP_Baseball%3AImportant_Game_Concepts%2FThe_Personnel_Model%2FPersonnel_Ratings_and_Tendencies) — conceptual GM roster/valuation and trading tendencies; does not give this export’s field-by-field numeric codebook.
- [OOTP 20 Game Options](https://manuals.ootpdevelopments.com/index.php?man=ootp20&page=game_options) — describes global current/previous/two-years-ago statistical weights; this is a conceptual analogue, not a verified scale key for the per-GM CSV columns.
- [Personnel Roles](https://wiki.ootpdevelopments.com/index.php?title=OOTP_Baseball%3AImportant_Game_Concepts%2FThe_Personnel_Model%2FPersonnel_Roles) and [Personnel Editor](https://wiki.ootpdevelopments.com/index.php?title=OOTP_Baseball%3AImportant_Game_Concepts%2FTools%2C_Functions%2C_and_Editors%2FIn-Game_Editors%2FPersonnel_Editor) — job/function and general personnel rating-scale context. The general ratings-scale statement is not used to invent signed tendency scales.
- [Global and League Trading Options](https://wiki.ootpdevelopments.com/index.php?title=OOTP_Baseball%3AImportant_Game_Concepts%2FThe_Trading_Model%2FGlobal_and_League_Trading_Options%2FGlobal_and_League_Trading_Options) — global AI trading frequency and veterans/prospects concepts, not an individual CSV codebook.
- [OOTP 24 Ownership manual](https://manuals.ootpdevelopments.com/index.php?man=ootp24&page=ownership) and [OOTP 20 personnel ratings manual](https://manuals.ootpdevelopments.com/index.php?man=ootp20&page=personnel_ratings) — concept references from earlier published versions. No public OOTP 26 page or export schema located here that resolves the exact CSV numeric encodings.

**Limitations to preserve in any article:** GM numerical tendency scales and exact valid ranges remain unknown; `owner_expectation` integer labels remain unknown; owner-profile capture timing is not encoded; GM/owner tenure and private instructions are not established; transaction timestamps within a date are unavailable; two-way full-season player totals must not be mistaken for receiving-club production; and player-position conflicts noted in the verified transaction packet remain unresolved. The article-ready use is for sourced reporting leads and race context, not unsupported personality or ownership claims.
