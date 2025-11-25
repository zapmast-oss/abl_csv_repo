from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

PRIORITY_ORDER: Dict[str, int] = {"A": 0, "B": 1, "C": 2}


def priority_rank(value: str) -> int:
    return PRIORITY_ORDER.get((value or "").strip().upper(), 3)


def load_story_rows(csv_path: Path, required_cols: List[str]) -> List[dict]:
    if not csv_path.exists():
        print(f"Error: story dictionary not found at {csv_path}", file=sys.stderr)
        sys.exit(1)

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing = [c for c in required_cols if c not in fieldnames]
        if missing:
            print(
                f"Error: story dictionary missing columns: {', '.join(missing)}",
                file=sys.stderr,
            )
            sys.exit(1)
        return list(reader)


def sort_stories(rows: List[dict]) -> List[dict]:
    return sorted(
        rows,
        key=lambda r: (
            priority_rank(r.get("priority", "")),
            (r.get("story_type") or ""),
            (r.get("name") or ""),
        ),
    )


def filter_by_priority(rows: List[dict], min_priority: str) -> List[dict]:
    min_rank = priority_rank(min_priority)
    return [
        row
        for row in rows
        if priority_rank(row.get("priority", "")) <= min_rank
    ]


def write_story_menu(rows: List[dict], out_path: Path) -> None:
    out_cols = [
        "story_id",
        "name",
        "level",
        "facet",
        "summary",
        "story_type",
        "priority",
        "time_window",
        "trigger_stats",
        "trigger_logic",
        "eb_prompt_seed",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in out_cols})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a weekly story menu CSV from story_dictionary.csv."
    )
    parser.add_argument(
        "week_label",
        help="Week label used in output filename, e.g. 1981_week_07",
    )
    parser.add_argument(
        "--min-priority",
        choices=["A", "B", "C"],
        default="A",
        help="Minimum priority to include (default: A)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve()
    csv_dir = script_path.parent.parent / "csv"
    story_csv = csv_dir / "story_dictionary.csv"

    required_cols = [
        "story_id",
        "name",
        "level",
        "facet",
        "summary",
        "trigger_stats",
        "trigger_logic",
        "time_window",
        "story_type",
        "eb_prompt_seed",
        "priority",
    ]

    rows = load_story_rows(story_csv, required_cols)
    filtered = filter_by_priority(rows, args.min_priority)
    sorted_rows = sort_stories(filtered)

    out_path = csv_dir / f"story_menu_{args.week_label}.csv"
    write_story_menu(sorted_rows, out_path)

    print(
        f"Wrote {len(sorted_rows)} stories to {out_path} (min priority: {args.min_priority})"
    )


if __name__ == "__main__":
    main()
