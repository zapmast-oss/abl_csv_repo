from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path


PRIORITY_ORDER = {"A": 0, "B": 1, "C": 2}


def load_story_rows(csv_path: Path):
    if not csv_path.exists():
        print(f"Error: story dictionary not found at {csv_path}", file=sys.stderr)
        sys.exit(1)

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = [
            "story_id",
            "name",
            "level",
            "facet",
            "summary",
            "trigger_stats",
            "time_window",
            "story_type",
            "priority",
        ]
        missing = [c for c in required_cols if c not in (reader.fieldnames or [])]
        if missing:
            print(
                f"Error: story dictionary missing columns: {', '.join(missing)}",
                file=sys.stderr,
            )
            sys.exit(1)
        rows = list(reader)
    return rows


def priority_key(priority: str) -> int:
    return PRIORITY_ORDER.get((priority or "").strip().upper(), 3)


def sort_stories(stories):
    return sorted(
        stories,
        key=lambda r: (
            priority_key(r.get("priority", "")),
            (r.get("story_type") or ""),
            (r.get("name") or ""),
        ),
    )


def print_story_menu(rows):
    groups = defaultdict(list)
    for row in rows:
        level = (row.get("level") or "").strip()
        facet = (row.get("facet") or "").strip()
        groups[(level, facet)].append(row)

    for (level, facet) in sorted(groups.keys()):
        stories = sort_stories(groups[(level, facet)])
        print("=" * 50)
        print(f"LEVEL: {level.upper() or '?'}   FACET: {facet or '?'}")
        print("=" * 50)

        for row in stories:
            priority = (row.get("priority") or "").strip().upper()
            story_type = (row.get("story_type") or "").strip()
            story_id = (row.get("story_id") or "").strip()
            name = (row.get("name") or "").strip()
            summary = (row.get("summary") or "").strip()
            time_window = (row.get("time_window") or "").strip()
            trigger_stats = (row.get("trigger_stats") or "").strip()

            print(f"[{priority}][{story_type}] {story_id} - {name}")
            print(f"    summary: {summary}")
            print(f"    time window: {time_window}")
            print(f"    trigger stats: {trigger_stats}")
            print()

        print()


def main():
    script_path = Path(__file__).resolve()
    csv_path = script_path.parent.parent / "csv" / "story_dictionary.csv"
    rows = load_story_rows(csv_path)
    print_story_menu(rows)


if __name__ == "__main__":
    main()
