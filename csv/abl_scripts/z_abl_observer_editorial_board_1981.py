from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STORY = ROOT / "csv" / "out" / "story"
SLUG = "1981_07_20_asof_1981-07-19"
CANDIDATES = STORY / "candidates" / f"story_candidates_{SLUG}.csv"
EVIDENCE = STORY / "candidates" / f"story_evidence_{SLUG}.csv"
OUT_DIR = STORY / "editorial"
BOARD_STEM = OUT_DIR / f"observer_editorial_board_{SLUG}"
SLATE_PATH = OUT_DIR / f"observer_story_slate_{SLUG}.md"
FIELDS = [
    "editorial_rank", "candidate_id", "candidate_title", "candidate_type",
    "hierarchy_level", "evidence_summary", "why_it_matters", "recommended_use",
    "recommended_priority", "confidence", "evidence_strength_score",
    "current_relevance_score", "editorial_freshness_score", "stakes_score",
    "video_usefulness_score", "hierarchy_score", "editorial_score",
    "source_notes", "disabled_signal_warning",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def scores(row: dict[str, str]) -> dict[str, int]:
    hierarchy = row["hierarchy_level"]
    signal = row["signal_type"]
    evidence = 5 if row["confidence"] == "high" else 4
    relevance = {
        "standings": 5, "game": 5, "race": 4, "players": 4,
        "team": 4, "fans": 3, "management_organization": 3, "tournament": 3,
    }[hierarchy]
    freshness = 5 if signal in {"weekly_rise", "weekly_fall", "scheduled_matchup_stakes"} else (2 if row["candidate_class"] == "historical_echo" else (3 if signal == "pythag_record_gap" else 4))
    if signal == "division_race":
        lead = int(row["evidence_summary"].split("leads ")[0]) if row["evidence_summary"].startswith("leads ") else None
        stakes = 5 if "by 2 game" in row["headline_factual"] or "by 3 game" in row["headline_factual"] else 4
    else:
        stakes = {"scheduled_matchup_stakes": 5, "ace_value_leader": 4, "position_player_value_leader": 4,
                  "weekly_rise": 3, "weekly_fall": 3, "pythag_record_gap": 3,
                  "attendance_interest": 3, "payroll_record_pressure": 3,
                  "prior_qualifier_echo": 3}.get(signal, 3)
    video = {"standings": 5, "game": 5, "players": 4, "team": 4, "fans": 4,
             "management_organization": 4, "tournament": 4, "race": 3}[hierarchy]
    hierarchy_score = {"tournament": 5, "standings": 5, "race": 4, "fans": 3,
                       "management_organization": 3, "players": 3, "team": 3, "game": 3}[hierarchy]
    total = evidence*4 + relevance*4 + freshness*3 + stakes*4 + video*3 + hierarchy_score*2
    return {"evidence": evidence, "relevance": relevance, "freshness": freshness,
            "stakes": stakes, "video": video, "hierarchy": hierarchy_score, "total": total}


def priority(row: dict[str, str]) -> str:
    title = row["headline_factual"]
    signal = row["signal_type"]
    if "Dallas Rustlers leads Detroit Dukes" in title:
        return "Lead"
    if ("San Francisco Warriors leads Phoenix Firebirds" in title
            or "Las Vegas Gamblers leads Seattle Comets" in title
            or signal == "ace_value_leader" or signal == "scheduled_matchup_stakes"):
        return "Secondary"
    if ("Boston Patriots leads New York Aces" in title or "Houston Mavericks leads Cincinnati Cougars" in title
            or signal in {"position_player_value_leader", "prior_qualifier_echo"}):
        return "Mention"
    if signal in {"weekly_rise", "pythag_record_gap", "attendance_interest", "payroll_record_pressure"}:
        return "Watch"
    if signal == "weekly_fall":
        return "Hold"
    return "Watch"


def uses(row: dict[str, str], selected_priority: str) -> str:
    signal = row["signal_type"]
    hierarchy = row["hierarchy_level"]
    if selected_priority == "Lead":
        return "Baseball Observer main segment|Daily 411|forum written post"
    if signal == "scheduled_matchup_stakes":
        return "Ballpark Feed setup|Daily 411|Baseball Observer main segment"
    if signal == "ace_value_leader":
        return "Baseball Observer main segment|Ballpark Feed setup|Daily 411"
    if hierarchy == "standings":
        return "Daily 411|Baseball Observer main segment|forum written post"
    if signal == "prior_qualifier_echo":
        return "Baseball Observer main segment|forum written post"
    if hierarchy == "players":
        return "Daily 411|Baseball Observer main segment"
    if hierarchy in {"team", "management_organization", "fans"}:
        return "forum written post|Baseball Observer main segment|hold/watch list"
    return "Daily 411|hold/watch list"


def warning(row: dict[str, str]) -> str:
    signal = row["signal_type"]
    if signal == "payroll_record_pressure":
        return "Manager tendencies and complete staff-ID linkage are disabled; do not assign motive or blame."
    if signal == "attendance_interest":
        return "Attendance is not fan sentiment; do not infer emotion or approval."
    if signal == "pythag_record_gap":
        return "Do not call the gap luck or guaranteed regression; BatR/wSB/UBR/BsR remain unavailable."
    if signal in {"weekly_rise", "weekly_fall"}:
        return "Short window; do not declare a turnaround or collapse."
    if signal == "scheduled_matchup_stakes":
        return "No July 20 result is available or assumed."
    if signal == "prior_qualifier_echo":
        return "Historical echo is context, not a forecast."
    return ""


def source_notes(row: dict[str, str]) -> str:
    if row["candidate_class"] == "proven_current_fact":
        return f"Proven current-state fact through 1981-07-19: {row['source_files']}"
    if row["candidate_class"] == "enrichment_signal":
        return f"Promoted sortable enrichment joined to raw current record: {row['source_files']}"
    return f"Governed historical context paired with raw current evidence: {row['source_files']}"


def main() -> int:
    candidates = read_csv(CANDIDATES)
    evidence = read_csv(EVIDENCE)
    by_candidate: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in evidence:
        by_candidate[item["candidate_id"]].append(item)
    if len(candidates) != 15 or {row["candidate_id"] for row in candidates} != set(by_candidate):
        raise ValueError("Editorial board requires the verified 15-candidate/15-evidence run")
    working = []
    for row in candidates:
        scoring = scores(row)
        selected_priority = priority(row)
        working.append({
            "candidate_id": row["candidate_id"], "candidate_title": row["headline_factual"],
            "candidate_type": row["signal_type"], "hierarchy_level": row["hierarchy_level"],
            "evidence_summary": row["evidence_summary"], "why_it_matters": row["stakes"],
            "recommended_use": uses(row, selected_priority), "recommended_priority": selected_priority,
            "confidence": row["confidence"], "evidence_strength_score": scoring["evidence"],
            "current_relevance_score": scoring["relevance"], "editorial_freshness_score": scoring["freshness"],
            "stakes_score": scoring["stakes"], "video_usefulness_score": scoring["video"],
            "hierarchy_score": scoring["hierarchy"], "editorial_score": scoring["total"],
            "source_notes": source_notes(row), "disabled_signal_warning": warning(row),
        })
    priority_order = {"Lead": 0, "Secondary": 1, "Mention": 2, "Watch": 3, "Hold": 4}
    working.sort(key=lambda row: (priority_order[row["recommended_priority"]], -int(row["editorial_score"]), row["candidate_title"]))
    board = [{"editorial_rank": index, **row} for index, row in enumerate(working, 1)]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with BOARD_STEM.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader(); writer.writerows(board)
    payload = {
        "newsroom_date": "1981-07-20", "as_of_date": "1981-07-19",
        "editorial_rule": "Data points where to look; the game proves the story.",
        "voice": "Notice it. Name it. Connect it to the stakes. Then play ball.",
        "candidate_count": len(board), "board": board,
    }
    BOARD_STEM.with_suffix(".json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Baseball Observer Editorial Board — July 20, 1981", "",
             "> Data points where to look; the game proves the story.", "",
             "*Notice it. Name it. Connect it to the stakes. Then play ball.*", "",
             "| Rank | Priority | Hierarchy | Candidate | Score | Confidence | Recommended use |",
             "|---:|---|---|---|---:|---|---|"]
    for row in board:
        lines.append(f"| {row['editorial_rank']} | **{row['recommended_priority']}** | {row['hierarchy_level'].replace('_',' ').title()} | {row['candidate_title']} | {row['editorial_score']} | {row['confidence']} | {row['recommended_use']} |")
    lines += ["", "## Editorial notes", ""]
    for row in board:
        lines += [f"### {row['editorial_rank']}. {row['candidate_title']}", "",
                  f"- Evidence: {row['evidence_summary']}", f"- Why it matters: {row['why_it_matters']}",
                  f"- Sources: {row['source_notes']}"]
        if row["disabled_signal_warning"]:
            lines.append(f"- Guardrail: {row['disabled_signal_warning']}")
        lines.append("")
    BOARD_STEM.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")

    lead = next(row for row in board if row["recommended_priority"] == "Lead")
    secondary = [row for row in board if row["recommended_priority"] == "Secondary"]
    mentions = [row for row in board if row["recommended_priority"] == "Mention"]
    watch = [row for row in board if row["recommended_priority"] == "Watch"]
    held = [row for row in board if row["recommended_priority"] == "Hold"]
    slate = ["# Baseball Observer Story Slate — July 20, 1981", "",
             "> Data points where to look; the game proves the story.", "",
             "*Notice it. Name it. Connect it to the stakes. Then play ball.*", "",
             "## Recommended lead", "", f"**{lead['candidate_title']}**", "",
             f"{lead['evidence_summary']} {lead['why_it_matters']}", "",
             "## Secondary stories", ""]
    slate += [f"- **{row['candidate_title']}** — {row['evidence_summary']}" for row in secondary]
    slate += ["", "## Quick mentions", ""] + [f"- {row['candidate_title']} — {row['evidence_summary']}" for row in mentions]
    slate += ["", "## Watch list", ""] + [f"- {row['candidate_title']} — {row['disabled_signal_warning'] or row['why_it_matters']}" for row in watch]
    slate += ["", "## Hold", ""] + [f"- {row['candidate_title']} — {row['disabled_signal_warning']}" for row in held]
    slate += ["", "## What not to overstate", "",
              "- Do not turn a two-game lead into a settled division.",
              "- Do not call a one-week rise a turnaround or a one-week fall a collapse.",
              "- Do not convert Pythagorean gaps into luck or guaranteed regression.",
              "- Do not treat attendance as fan emotion or payroll as proof of blame.",
              "- Do not use manager tendencies, complete staff-ID linkage, CS/SB%, BatR, wSB, UBR, or BsR.",
              "- Do not imply a July 20 result; the Las Vegas–Denver game is unplayed setup.", "",
              "## Suggested Baseball Observer angle", "",
              "Start with the two-game Dallas–Detroit race: the standings establish the pressure, while Detroit's league-leading listed home attendance gives that pressure public scale. Keep the claim factual—the crowd number says people are present, not what they feel. Then widen to the other close division races.", "",
              "## Suggested Daily 411 angle", "",
              "Run the division board first, then the July 13–19 riser/faller, followed by José Coronado and Manny Flores as the current player-value markers. Keep every item tied to games completed through July 19.", "",
              "## Suggested Ballpark Feed angle", "",
              "Las Vegas at Denver is an unplayed July 20 setup: Las Vegas carries a three-game division lead into a matchup between 54–39 and 50–43 clubs. Name the standings pressure and let the game supply the result.", ""]
    SLATE_PATH.write_text("\n".join(slate), encoding="utf-8")
    print(json.dumps({
        "lead": lead["candidate_title"], "secondary": [row["candidate_title"] for row in secondary],
        "quick_mentions": [row["candidate_title"] for row in mentions],
        "watch_list": [row["candidate_title"] for row in watch],
        "held": [row["candidate_title"] for row in held],
        "safe_for_editorial_use": True,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

