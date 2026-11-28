from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STORY = ROOT / "csv" / "out" / "story"
SLUG = "1981_07_20_asof_1981-07-19"
BOARD = STORY / "editorial" / f"observer_editorial_board_{SLUG}.csv"
CANDIDATES = STORY / "candidates" / f"story_candidates_{SLUG}.csv"
EVIDENCE = STORY / "candidates" / f"story_evidence_{SLUG}.csv"
OUT_DIR = STORY / "production"
OUT_STEM = OUT_DIR / f"eb_production_package_{SLUG}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def item(row: dict[str, str], candidate: dict[str, str], **extra: object) -> dict[str, object]:
    return {
        "candidate_id": row["candidate_id"], "title": row["candidate_title"],
        "hierarchy_level": row["hierarchy_level"], "priority": row["recommended_priority"],
        "evidence_summary": row["evidence_summary"], "why_it_matters": row["why_it_matters"],
        "recommended_use": row["recommended_use"], "confidence": row["confidence"],
        "source_notes": row["source_notes"], "source_files": candidate["source_files"],
        "guardrail": row["disabled_signal_warning"], **extra,
    }


def main() -> int:
    board = read_csv(BOARD)
    candidates = {row["candidate_id"]: row for row in read_csv(CANDIDATES)}
    evidence = read_csv(EVIDENCE)
    if len(board) != 15 or len(evidence) != 15 or {row["candidate_id"] for row in board} != set(candidates):
        raise ValueError("Production package requires the verified 15-item editorial board")
    by_title = {row["candidate_title"]: row for row in board}
    lead_row = next(row for row in board if row["recommended_priority"] == "Lead")
    secondary_rows = [row for row in board if row["recommended_priority"] == "Secondary"]
    mention_rows = [row for row in board if row["recommended_priority"] == "Mention"]
    watch_rows = [row for row in board if row["recommended_priority"] == "Watch"]
    held_rows = [row for row in board if row["recommended_priority"] == "Hold"]

    lead = item(
        lead_row, candidates[lead_row["candidate_id"]],
        proven="Dallas is 52-41 and Detroit is 50-43 through games completed July 19; Dallas leads by two games.",
        do_not_overstate="The division is not settled. Do not infer Detroit fan sentiment from attendance or assign managerial blame.",
        talking_points=[
            "Here’s the skinny: two games separate Dallas and Detroit in the NBC Central.",
            "The records establish the pressure—52-41 against 50-43—with less schedule left to absorb each result.",
            "Detroit’s listed attendance leads the league, which gives the race public scale without telling us what those fans feel.",
            "Data points where to look; the next games prove whether the margin changes.",
        ],
        transition="Dallas–Detroit is not an isolated squeeze. The NBC West is also separated by two games, so move next to San Francisco–Phoenix.",
    )

    secondary_details = {
        "Las Vegas Gamblers leads Seattle Comets by 3 game(s) in American Baseball Conference Western Division": {
            "phrasing": "Las Vegas owns a three-game edge over Seattle, 54-39 to 51-42. Name the lead, then watch what the next series does to it.",
            "avoid": "Do not call the ABC West decided or use unavailable baserunning metrics to explain the gap.",
        },
        "San Francisco Warriors leads Phoenix Firebirds by 2 game(s) in National Baseball Conference Western Division": {
            "phrasing": "San Francisco and Phoenix give the league a second two-game division race: 50-43 against 48-45 through July 19.",
            "avoid": "Do not imply a clinch path or playoff certainty; qualification arithmetic is not part of this evidence.",
        },
        "Las Vegas Gamblers at Denver Rocketeers places current standings pressure on the field": {
            "phrasing": "The Ballpark Feed setup is clean: 54-39 Las Vegas visits 50-43 Denver on July 20, and both clubs carry current standings weight onto the field.",
            "avoid": "This is an unplayed schedule fact. Do not imply or search for a July 20 result.",
        },
        "Jose Coronado has built the strongest current starter value case": {
            "phrasing": "José Coronado’s line—4.3 WAR, a 2.86 ERA and 141.7 innings—points to the strongest current starter-value case, even with a 7-7 record.",
            "avoid": "Do not turn a value lead into an award result or claim his win-loss record measures his pitching quality.",
        },
    }
    secondary = [
        item(row, candidates[row["candidate_id"]], suggested_eb_phrasing=secondary_details[row["candidate_title"]]["phrasing"],
             avoid_overstating=secondary_details[row["candidate_title"]]["avoid"])
        for row in secondary_rows
    ]

    mention_text = {
        "Boston Patriots leads New York Aces by 4 game(s) in American Baseball Conference Eastern Division": (
            "Boston leads New York by four in the ABC East, 48-45 to 44-49. It belongs on the board, but behind the two- and three-game races.",
            "Current raw standings only; do not imply the race is closed."
        ),
        "Houston Mavericks leads Cincinnati Cougars by 5 game(s) in American Baseball Conference Central Division": (
            "Houston’s 53-39 record gives it a five-game ABC Central lead over 48-45 Cincinnati—the widest race still selected for mention.",
            "Mention the current margin; do not infer postseason qualification."
        ),
        "Manny Flores owns the strongest current position-player value line": (
            "Manny Flores has the current position-player value marker at 4.3 WAR with 18 homers in 384 plate appearances.",
            "Current leader, not a final award verdict."
        ),
        "1980 qualifier Charlotte Colonels remains central to the 1981 race": (
            "Charlotte went 91-71 as a 1980 qualifier and is 57-36 through July 19 this season—a useful historical echo for the current race.",
            "History supplies context, not a prediction."
        ),
    }
    mentions = [
        item(row, candidates[row["candidate_id"]], usable_mention=mention_text[row["candidate_title"]][0],
             source_caution=mention_text[row["candidate_title"]][1])
        for row in mention_rows
    ]

    watch_text = {
        "Atlanta Kings posted the week's sharpest rise": (
            "The 3-1 July 13-19 window is only four games; it shows movement, not a turnaround.",
            "Another one to two weeks of winning plus measurable movement in the division standings."
        ),
        "Atlanta Kings's record and run balance remain far apart": (
            "Atlanta is 40-53 with a 44.7-win Pythagorean expectation. The gap is descriptive and does not prove bad luck.",
            "A sustained change in run differential, record gap, and game-level outcomes over a larger window."
        ),
        "Seattle Comets's record and run balance remain far apart": (
            "Seattle is 51-42 despite a 405-405 run balance and 46.5 expected wins. That is a watch condition, not guaranteed regression.",
            "Several more series showing whether run balance and record move together or remain separated."
        ),
        "Detroit Dukes is drawing the league's largest listed home crowd per game": (
            "The listed 26,990 per game gives scale to Detroit’s race, but one attendance measure cannot establish fan emotion.",
            "A time series of attendance, race movement, and verified fan-interest measures—not inferred sentiment."
        ),
        "Tampa Bay Storm's payroll commitment and record create an organization question": (
            "A listed $10.8 million payroll and 42-51 record create a factual resource/results comparison, not proof of blame or intent.",
            "Verified roster decisions, organizational statements, or a sustained record change; manager tendencies remain unavailable."
        ),
    }
    watch = [
        item(row, candidates[row["candidate_id"]], why_watch_only=watch_text[row["candidate_title"]][0],
             evidence_needed=watch_text[row["candidate_title"]][1])
        for row in watch_rows
    ]

    held = [
        item(held_rows[0], candidates[held_rows[0]["candidate_id"]],
             why_held="Philadelphia went 1-3 from July 13-19. Four games do not establish collapse, organizational failure, or a durable trend.",
             evidence_needed="A sustained multi-week losing run, clear standings deterioration, worsening run differential, and any verified availability or roster context before using collapse language.")
    ]

    package = {
        "newsroom_date": "1981-07-20", "as_of_date": "1981-07-19",
        "editorial_rule": "Data points where to look; the game proves the story.",
        "voice": "Notice it. Name it. Connect it to the stakes. Then play ball.",
        "editorial_frame": {
            "angle": "Multiple division races have compressed at once; Dallas–Detroit is the lead example, and the other races show that the pressure is league-wide rather than isolated.",
            "why_lead": "Dallas–Detroit is a high-confidence, two-game current race and connects to a separate Detroit attendance fact without requiring invented emotion.",
            "supporting_picture": "San Francisco–Phoenix is also two games, Las Vegas–Seattle is three, and Las Vegas–Denver supplies an unplayed July 20 field-level setup.",
        },
        "lead_story": lead, "secondary_stories": secondary, "quick_mentions": mentions,
        "watch_list": watch, "held_items": held,
        "content_usage": {
            "baseball_observer_main_segment": "Lead Dallas–Detroit; widen to San Francisco–Phoenix and Las Vegas–Seattle; use Coronado as the player counterpoint.",
            "daily_411": "Division margins first, then Atlanta/Philadelphia weekly movement and the Coronado/Flores value markers.",
            "ballpark_feed_setup": "Las Vegas at Denver as an unplayed July 20 matchup framed by current records and ABC West pressure.",
            "forum_written_post": "A standings-board roundup with the two Pythagorean watch conditions and clearly labeled historical context.",
            "future_watch_list": "Atlanta and Seattle record/run gaps, Detroit attendance trend, Tampa Bay resource/results gap, and Philadelphia only if the fall persists.",
        },
        "voice_rules": [
            "Neutral observer.", "‘Here’s the skinny’ is allowed.", "‘Inside dope’ is allowed only where natural and sourced.",
            "No invented motives or fake clubhouse drama.", "No unsupported collapse language.",
            "Data points where to look; the game proves the story.",
            "Notice it, name it, connect it to the stakes, then play ball.",
        ],
        "safe_for_eb_use": True,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_STEM.with_suffix(".json").write_text(json.dumps(package, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# EB Production Package — July 20, 1981", "",
             "> Data points where to look; the game proves the story.", "",
             "*Notice it. Name it. Connect it to the stakes. Then play ball.*", "",
             "## Editorial frame", "",
             f"**Today's angle:** {package['editorial_frame']['angle']}", "",
             f"**Why Dallas–Detroit leads:** {package['editorial_frame']['why_lead']}", "",
             f"**Larger ABL picture:** {package['editorial_frame']['supporting_picture']}", "",
             "## Lead story prep", "", f"### {lead['title']}", "",
             f"- Evidence: {lead['evidence_summary']}", f"- Stakes: {lead['why_it_matters']}",
             f"- Proven: {lead['proven']}", f"- Do not overstate: {lead['do_not_overstate']}", "",
             "**Suggested EB talking points**", ""]
    lines += [f"- {point}" for point in lead["talking_points"]]
    lines += ["", f"**Transition:** {lead['transition']}", "", "## Secondary story prep", ""]
    for story in secondary:
        lines += [f"### {story['title']}", "", f"- Evidence: {story['evidence_summary']}",
                  f"- Why it matters: {story['why_it_matters']}", f"- Suggested use: {story['recommended_use']}",
                  f"- Suggested EB phrasing: {story['suggested_eb_phrasing']}", f"- Avoid: {story['avoid_overstating']}", ""]
    lines += ["## Quick mentions", ""]
    for story in mentions:
        lines += [f"- **{story['title']}** — {story['usable_mention']} *Caution: {story['source_caution']}*"]
    lines += ["", "## Watch list", ""]
    for story in watch:
        lines += [f"### {story['title']}", "", f"- Why watch only: {story['why_watch_only']}",
                  f"- Evidence needed: {story['evidence_needed']}", ""]
    lines += ["## Held item", ""]
    for story in held:
        lines += [f"### {story['title']}", "", f"- Why held: {story['why_held']}",
                  f"- Evidence needed: {story['evidence_needed']}", ""]
    lines += ["## Suggested content usage", ""]
    for key, value in package["content_usage"].items():
        lines += [f"### {key.replace('_', ' ').title()}", "", value, ""]
    lines += ["## Voice rules", ""] + [f"- {rule}" for rule in package["voice_rules"]]
    lines += ["", "Package status: **SAFE FOR EB USE**", ""]
    OUT_STEM.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "production_package_created": True, "lead_story": lead["title"],
        "secondary_stories": [row["title"] for row in secondary],
        "quick_mentions": [row["title"] for row in mentions],
        "held_items": [row["title"] for row in held], "safe_for_eb_use": True,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

