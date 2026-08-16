"""Build the governed Chicago-Dallas July 20 investigation packet.

This script combines an existing raw-proof investigation with the separately
promoted StatsPlus enrichment note.  It does not read or change official story
candidates, rankings, menus, editorial boards, production packages, or scripts.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "csv/out/story/investigations"
STEM = "1981_07_20_asof_1981-07-19"
INV_CSV = OUT / f"chicago_dallas_series_investigation_{STEM}.csv"
INV_MD = OUT / f"chicago_dallas_series_investigation_{STEM}.md"
INV_JSON = OUT / f"chicago_dallas_series_investigation_{STEM}.json"
EVIDENCE_MD = OUT / f"chicago_dallas_series_evidence_packet_{STEM}.md"
EVIDENCE_JSON = OUT / f"chicago_dallas_series_evidence_packet_{STEM}.json"
FEED3_JSON = OUT / f"chicago_dallas_statsplus_enrichment_note_{STEM}.json"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    investigation = load_json(INV_JSON)
    feed3 = load_json(FEED3_JSON)

    # Keep the prior raw-proof evidence and make reruns idempotent.
    with INV_CSV.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if not row["evidence_id"].startswith("CHI-DAL-F3-")]
        fields = list(rows[0])

    feed3_source = "csv/out/story/investigations/chicago_dallas_statsplus_enrichment_note_1981_07_20_asof_1981-07-19.json"
    additions = [
        ("01", "Feed 3 playoff model", "Dallas is the model favorite", "Chicago division/playoff odds are 15.0%/35.9%; Dallas is 60.0%/79.7%.", "Odds are enrichment, not standings or destiny."),
        ("02", "Feed 3 ELO", "The clubs are close by ELO", "Chicago is 1513.4 and Dallas is 1516.8, a 3.4-point difference.", "ELO does not override Dallas's four-game standings lead."),
        ("03", "Feed 3 BaseRuns", "Neither record is far from modeled expectation", "Chicago has xW delta -1; Dallas has xW delta +1.", "Do not label either club lucky or due for regression."),
        ("04", "Feed 3 team WAR", "Dallas has the stronger aggregate WAR", "Chicago total WAR is 21.79; Dallas is 25.12.", "Team WAR enriches the matchup; it does not prove a result."),
        ("05", "Feed 3 injuries", "Both clubs carry injury context", "Chicago lists 3 injuries/90 DL days; Dallas 5/255.", "Do not infer series impact without validated player availability."),
        ("06", "Feed 3 fans/finance", "The series has supported attendance and financial context", "Chicago: interest 82, 18,249 average attendance, 98.6% full, $8.54M payroll. Dallas: 71, 22,539, 96.3%, $6.59M.", "Do not infer fan emotion or organizational intent."),
        ("07", "Feed 3 age/baserunning", "Chicago and Dallas have veteran-age rosters and negative UBR", "Chicago MLB age 29.3 and UBR -11.50; Dallas 30.1 and -6.00.", "Context only; no automated candidate or ranking effect."),
        ("08", "Feed 3 reference-only", "Owner traits and best-game discovery remain reference-only", "Owner/front-office traits and discovered best-game notes were not automated.", "Raw verification is required before publishing a discovered game claim."),
    ]
    for suffix, category, claim, finding, boundary in additions:
        rows.append({
            "evidence_id": f"CHI-DAL-F3-{suffix}",
            "category": category,
            "claim": claim,
            "status": "ENRICHMENT_ONLY" if suffix != "08" else "REFERENCE_ONLY",
            "finding": finding,
            "source_files": feed3_source,
            "source_record_ids": "CHI|DAL",
            "editorial_boundary": boundary,
        })
    with INV_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    investigation["feed3_enrichment"] = {
        "authority": "enrichment_only",
        "strongest_enhancement": "Dallas holds a large playoff-odds advantage (60.0% division and 79.7% playoff versus Chicago's 15.0% and 35.9%), while ELO is close at 1516.8 to 1513.4.",
        "chicago": feed3["chicago"],
        "dallas": feed3["dallas"],
        "interpretation": feed3["verdict"],
        "reference_only": ["owner/front-office traits", "best-game discovery", "historical fan interpretation"],
    }
    investigation["editorial_decision"] = {
        "classification": "PROMOTE_SERIES_SETUP",
        "add_to_july_20_slate": True,
        "replace_existing_story": False,
        "annotation_only": True,
        "save_for_ballpark_feed": True,
        "placement": "Add as a concise Chicago-Dallas series annotation beneath the Dallas-Detroit lead; do not replace the existing lead or secondary slate.",
    }
    if feed3_source not in investigation["source_files"]:
        investigation["source_files"].append(feed3_source)
    write_json(INV_JSON, investigation)

    md = f"""# Chicago–Dallas Series Investigation

**Newsroom date:** July 20, 1981  
**Evidence cutoff:** Games completed through July 19, 1981  
**Classification:** PROMOTE_SERIES_SETUP  
**Authority rule:** Raw OOTP proves games, scores, schedule, records, and standings. StatsPlus adds context only.

## Verdict

The central premise is proven, with one important limit. Chicago split four road games at Detroit, winning the July 19 finale 4–1, and returns home for four scheduled games against first-place Dallas from July 20–23. Chicago is 48–45, third in the NBC Central and four games behind 52–41 Dallas. A Chicago sweep would produce a tie at 52–45; it would not put Chicago ahead.

The historical claim is also accurate: Chicago reached the playoffs five times from 1972 through 1980 and won no championship. Its best finish was the 1973 Grand Series, lost to Denver in seven games. That history is context, not evidence of present pressure inside the club.

## 1. Detroit Road-Series Proof

| Date | Game ID | Site | Result |
|---|---:|---|---|
| July 16 | 1072 | Chicago at Detroit | Detroit 4, Chicago 2 |
| July 17 | 1084 | Chicago at Detroit | Detroit 10, Chicago 0 |
| July 18 | 1096 | Chicago at Detroit | Chicago 4, Detroit 2 |
| July 19 | 1108 | Chicago at Detroit | Chicago 4, Detroit 1 |

Chicago lost the first two and won the last two. The July 19 win completed a 2–2 road split.

## 2. NBC Central Through July 19

| Place | Team | Record | GB |
|---:|---|---:|---:|
| 1 | Dallas Rustlers | 52–41 | — |
| 2 | Detroit Dukes | 50–43 | 2 |
| 3 | Chicago Fire | 48–45 | 4 |
| 4 | Minneapolis Blizzard | 40–53 | 12 |

Series-only outcomes, holding all other games outside this head-to-head aside: a Chicago 4–0 result creates a tie; 3–1 leaves Chicago two back; 2–2 leaves the margin at four; 1–3 makes it six; 0–4 makes it eight. The series can change the direct margin by four games in either direction.

## 3. Upcoming Dallas Series

| Date | Game ID | Matchup | Ballpark | Status | Starters |
|---|---:|---|---|---|---|
| July 20 | 1119 | Dallas at Chicago | Chicago Grounds | Scheduled/unplayed | Unavailable |
| July 21 | 1131 | Dallas at Chicago | Chicago Grounds | Scheduled/unplayed | Unavailable |
| July 22 | 1143 | Dallas at Chicago | Chicago Grounds | Scheduled/unplayed | Unavailable |
| July 23 | 1155 | Dallas at Chicago | Chicago Grounds | Scheduled/unplayed | Unavailable |

No July 20 result is used or implied.

## 4. Chicago Franchise History, 1972–1980

- 1972: lost first postseason series to San Francisco, 4–3.
- 1973: defeated Miami and Phoenix; lost the Grand Series to Denver, 4–3.
- 1977: lost first postseason series to Atlanta, 4–1.
- 1979: defeated Miami; lost the conference series to Phoenix, 4–3.
- 1980: defeated Charlotte; lost the conference series to Detroit, 4–0.

Verified total: five playoff appearances, one Grand Series appearance, zero championships.

## 5. Chicago Team Texture

Alex O'Donell leads the listed Chicago hitters at 2.8 WAR and a .309/.372/.478 line. Sal Gamez has 2.4 WAR, 15 home runs, and 67 RBI; Miguel Morales has 2.3 WAR and 15 home runs. Gamez hit two home runs in the July 19 finale, while Morales and Victor Hernandez also homered.

Heriberto Jimenez leads the listed pitchers at 3.2 WAR, 12–8, and a 3.86 ERA. John Jury is 9–6 with a 2.44 ERA; Edgar Gonzalez is 10–4 with a 3.21 ERA and won July 18. Tony Suarez earned the July 19 win and Matt Hyman the save.

Feed 3 lists Chicago at 15.0% division odds, 35.9% playoff odds, 1513.4 ELO, 21.79 team WAR, and BaseRuns xW delta of -1. It lists three injuries and 90 DL days. These are enrichment measures, not current-state authority or a forecast of this series.

## 6. Dallas Team Texture

Dallas is front-running because the raw record places it first at 52–41. Devon Barlow leads the listed hitters at 2.9 WAR with a .301/.368/.519 line; Rich Vela has 2.1 WAR. Anton Garay leads the listed pitchers at 2.9 WAR, 11–2, and a 3.30 ERA; Mark Marmon has 1.9 WAR and a 2.80 ERA.

Feed 3 supplies the strongest enhancement: Dallas has 60.0% division odds and 79.7% playoff odds, compared with Chicago's 15.0% and 35.9%. Dallas also leads in team WAR, 25.12 to 21.79. ELO is much closer, 1516.8 to 1513.4, and BaseRuns places both near expected record. Dallas is the model favorite, but the scheduled head-to-head opportunity is real.

## 7. Ballpark and Fan Context

Chicago Grounds has a listed capacity of 18,500, an average park factor of 0.965, and a home-run factor of 1.131. Chicago's listed average attendance is 18,249—98.6% of capacity—with fan interest 82. Dallas lists 22,539 average attendance, 96.3% of capacity, and fan interest 71. These measures establish scale, not fan emotion.

## Editorial Decision

- **Classification:** PROMOTE_SERIES_SETUP
- **Add to July 20 slate:** Yes, as an annotation/concise series setup.
- **Replace anything:** No. Preserve Dallas–Detroit as the standings lead and the existing slate order.
- **Best use:** Ballpark Feed setup, with a brief Baseball Observer annotation.
- **Angle:** Chicago can directly change a four-game deficit against the division leader after earning a road split. The five-playoff/no-title history adds franchise context, not motive or destiny.

## Cautions

- A sweep ties Dallas; it does not move Chicago into sole first.
- Do not use or imply a July 20 result.
- Scheduled starters are unavailable.
- Feed 3 odds, ELO, WAR, injuries, and financial/fan measures enrich; they do not replace raw proof.
- Do not infer fan emotion, injury impact, owner intent, manager tendencies, clubhouse pressure, redemption, or a championship burden.
- Owner/front-office traits and best-game discovery remain reference-only.
"""
    INV_MD.write_text(md, encoding="utf-8")

    evidence_packet = {
        "newsroom_date": "1981-07-20",
        "as_of_date": "1981-07-19",
        "authority": {"raw_ootp": "current proof", "sortable_stats": "player/team context", "statsplus_feed3": "enrichment only", "historical": "governed context"},
        "confirmed_facts": investigation["core_findings"],
        "detroit_series": investigation["detroit_series"],
        "standings": investigation["standings"],
        "upcoming_series": investigation["dallas_series"],
        "conditional_series_math": investigation["conditional_series_math"],
        "franchise_history": investigation["franchise_history"],
        "team_texture": investigation["player_context"],
        "setting": investigation["setting"],
        "feed3_enrichment": investigation["feed3_enrichment"],
        "editorial_decision": investigation["editorial_decision"],
        "cautions": investigation["caution_flags"] + feed3["cautions"],
        "evidence_index": rows,
        "source_files": investigation["source_files"],
    }
    write_json(EVIDENCE_JSON, evidence_packet)

    evidence_md = f"""# Chicago–Dallas Series Evidence Packet

**Newsroom date:** July 20, 1981  
**Cutoff:** Games completed through July 19  
**Editorial status:** PROMOTE_SERIES_SETUP

## Proof Ledger

- **Detroit road split:** Proven. Chicago went 2–2 at Detroit, July 16–19 (game IDs 1072, 1084, 1096, 1108).
- **July 19 win:** Proven. Chicago won 4–1 as the road team in game 1108.
- **Dallas home series:** Proven. Four unplayed Dallas-at-Chicago games are scheduled July 20–23 (1119, 1131, 1143, 1155).
- **Standings:** Dallas 52–41, Detroit 50–43 (2 GB), Chicago 48–45 (4 GB).
- **Series leverage:** A Chicago sweep ties Dallas; other Chicago outcomes leave the Fire 2, 4, 6, or 8 games back.
- **Franchise claim:** Proven. Five playoff appearances from 1972–1980, one Grand Series appearance, no championships.

## Baseball Evidence

The strongest evidence is direct and schedule-bound: Chicago recovered from two losses in Detroit to win the final two, then immediately hosts the division leader for four games. Those four head-to-head games can erase or double the present four-game margin.

Chicago's current listed leaders include Alex O'Donell (2.8 hitter WAR), Heriberto Jimenez (3.2 pitcher WAR), Sal Gamez (15 HR, 67 RBI), and John Jury (2.44 ERA). Gamez homered twice in the July 19 finale. Dallas counters with Devon Barlow (2.9 WAR, .301/.368/.519) and Anton Garay (2.9 WAR, 11–2, 3.30 ERA).

## Feed 3 Enhancement

| Measure | Chicago | Dallas | Use |
|---|---:|---:|---|
| Division odds | 15.0% | 60.0% | Model context only |
| Playoff odds | 35.9% | 79.7% | Model context only |
| ELO | 1513.4 | 1516.8 | Shows a close rating, not equal standings position |
| Total team WAR | 21.79 | 25.12 | Dallas aggregate advantage |
| BaseRuns xW delta | -1 | +1 | Both near expected record |
| Injuries / DL days | 3 / 90 | 5 / 255 | Availability context; impact unproven |
| Fan interest | 82 | 71 | Descriptive only |
| Average attendance / capacity | 18,249 / 98.6% | 22,539 / 96.3% | Scale only; no emotion inference |
| MLB roster age | 29.3 | 30.1 | Team-age context |
| UBR | -11.50 | -6.00 | Team baserunning context |

The odds contrast is the strongest Feed 3 enhancement. It frames Chicago as a direct challenger with a meaningful schedule opportunity, not as the model favorite. Owner traits and best-game discovery remain reference-only.

## Historical Context

Chicago's five playoff years were 1972, 1973, 1977, 1979, and 1980. The 1973 club reached the Grand Series and lost to Denver, 4–3. The record supports a franchise-history sidebar, but not claims that current players feel a burden or that this series offers redemption.

## Setting

Chicago Grounds: capacity 18,500; average park factor 0.965; home-run factor 1.131. Scheduled starters are unavailable. Manager tendencies remain disabled.

## Editorial Use

Add the story to the July 20 slate as a concise annotation and Ballpark Feed setup. Do not replace Dallas–Detroit or change the existing candidate ranking. The supported line is: Chicago earned the split, comes home four games back, and now gets four direct games against first-place Dallas.

## Do Not Overstate

- No July 20 outcome is known here.
- A Chicago sweep creates a tie, not sole possession of first.
- Odds are probabilities, not proof of what will happen.
- Injury counts do not establish who will miss this series or how performance will change.
- Attendance and interest do not establish fan emotion.
- Five playoff appearances without a title do not prove pressure, curse, motive, or destiny.
"""
    EVIDENCE_MD.write_text(evidence_md, encoding="utf-8")

    print(f"investigation_rows={len(rows)}")
    print("classification=PROMOTE_SERIES_SETUP")
    print("official_story_artifacts_touched=false")


if __name__ == "__main__":
    main()
