from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ACTIVE_SEASON = 1981
ACTIVE_AS_OF_DATE = "1981-07-12"
ACTIVE_COVERAGE_LABEL = "1981_week_15"
EXPECTED_GAMES_PER_TEAM = 89
HIERARCHY = ["tournament", "standings", "race", "fans", "management", "players", "team", "game"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a weekly Baseball Observer packet from story candidates.")
    parser.add_argument("--week-label", default=ACTIVE_COVERAGE_LABEL)
    parser.add_argument("--run-id", default="sprint1_1981_week_15")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidate_path = ROOT / "csv" / "out" / "story" / "candidates" / f"story_candidates_{args.week_label}.csv"
    evidence_path = ROOT / "csv" / "out" / "story" / "candidates" / f"story_evidence_{args.week_label}.csv"
    manifest_path = ROOT / "csv" / "out" / "story" / "manifests" / f"story_engine_source_manifest_{args.week_label}.json"
    candidates = read_csv(candidate_path)
    evidence = read_csv(evidence_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_candidate: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in evidence:
        by_candidate[row["candidate_id"]].append(row)
    missing = [row["candidate_id"] for row in candidates if not row["source_files"] or not row["evidence_summary"] or not by_candidate[row["candidate_id"]]]
    if missing:
        raise ValueError(f"Candidates missing required source/evidence data: {missing}")

    order = {name: index for index, name in enumerate(HIERARCHY)}
    candidates.sort(key=lambda row: (order[row["hierarchy_level"]], -float(row["signal_score"])))
    packet = {
        "schema_version": "1.0-sprint1", "run_id": args.run_id,
        "week_label": args.week_label, "as_of_date": candidates[0]["as_of_date"],
        "editorial_principle": "Do not invent drama. Notice pressure. Name stakes. Let the game prove the story.",
        "candidate_count": len(candidates),
        "source_state": {
            "as_of_date": manifest["as_of_date"],
            "coverage_label": manifest["coverage_label"],
            "games_per_team": manifest["expected_games_per_team"],
            "current_source": "raw date-filterable OOTP exports",
            "excluded": "older Week 5 derivative snapshots",
            "disabled": "manager signals until compatible 89-game manager data exists",
        },
        "sections": [],
    }
    lines = [
        f"# Baseball Observer Weekly Packet — {args.week_label}", "",
        f"As of: `{candidates[0]['as_of_date']}`  ",
        f"Run: `{args.run_id}`", "",
        "> Do not invent drama. Notice pressure. Name stakes. Let the game prove the story.", "",
        "This packet names verified pressure and open questions. It does not predict what the next game will prove.", "",
        "## Source State", "",
        "- As of: July 12, 1981",
        "- Coverage: 89 games per team",
        "- Current source: raw date-filterable OOTP exports",
        "- Excluded: older Week 5 derivative snapshots",
        "- Disabled: manager signals until compatible 89-game manager data exists", "",
    ]
    for level in HIERARCHY:
        section_candidates = [row for row in candidates if row["hierarchy_level"] == level]
        if not section_candidates:
            continue
        section = {"hierarchy_level": level, "stories": []}
        lines.extend([f"## {level.title()}", ""])
        for row in section_candidates:
            ev = by_candidate[row["candidate_id"]]
            story = {**row, "evidence": ev, "proving_question": "What does the next game or week add to this evidence?"}
            section["stories"].append(story)
            lines.extend([
                f"### {row['headline_factual']}", "",
                f"**Pressure/stakes:** {row['stakes']}", "",
                f"**Evidence:** {row['evidence_summary']}", "",
                "**What lets the game prove it:** What does the next game or week add to this evidence?", "",
                f"**Sources:** {row['source_files']}", "",
            ])
        packet["sections"].append(section)
    lines.extend([
        "## Reporting guardrails", "",
        "- Do not convert Pythagorean gaps into claims of luck or guaranteed regression.",
        "- Do not call a scheduled starter, injury status, fan reaction, or playoff path verified unless a cited source establishes it.",
        "- Do not treat a prior champion or historical echo as a prediction.",
        "- Keep outcomes open. The next game supplies evidence; it does not fulfill a script.", "",
    ])
    out_dir = ROOT / "csv" / "out" / "story" / "packets"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"baseball_observer_packet_{args.week_label}.json"
    md_path = out_dir / f"baseball_observer_packet_{args.week_label}.md"
    json_path.write_text(json.dumps(packet, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"packet_json": str(json_path), "packet_markdown": str(md_path), "candidate_count": len(candidates)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
