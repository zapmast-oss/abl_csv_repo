from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

# Priority handling
PRIORITY_ORDER = {"A": 0, "B": 1, "C": 2}

# Pythag trigger thresholds
GAMES_PLAYED_MIN = 20          # lowered from 40 to 20
PYTHAG_DIFF_OVER_MIN = 5.0     # overachiever: >= +5 wins vs pythag
PYTHAG_DIFF_UNDER_MAX = -5.0   # underachiever: <= -5 wins vs pythag

# Story IDs we handle in this script
STORY_OVER_ID = "TEAM_PYTHAG_OVERACHIEVER"
STORY_UNDER_ID = "TEAM_PYTHAG_UNDERACHIEVER"


def priority_rank(value: str) -> int:
    return PRIORITY_ORDER.get((value or "").strip().upper(), 3)


def load_story_menu(
    menu_path: Path,
    min_priority: str,
) -> Dict[str, Dict[str, Any]]:
    """
    Load the story menu for a given week and return a dict keyed by story_id
    for the Pythag stories we care about, filtered by min_priority.
    """
    if not menu_path.exists():
        print(f"Error: story menu not found at {menu_path}", file=sys.stderr)
        sys.exit(1)

    with menu_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = [
            "story_id",
            "name",
            "level",
            "facet",
            "story_type",
            "priority",
            "time_window",
        ]
        missing = [c for c in required_cols if c not in reader.fieldnames]
        if missing:
            print(
                f"Error: story menu {menu_path} missing columns: {', '.join(missing)}",
                file=sys.stderr,
            )
            sys.exit(1)

        rows = list(reader)

    min_rank = priority_rank(min_priority)
    stories_by_id: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        sid = (row.get("story_id") or "").strip()
        if sid not in {STORY_OVER_ID, STORY_UNDER_ID}:
            # Ignore patterns this script does not implement
            continue

        pr = priority_rank(row.get("priority", ""))
        if pr > min_rank:
            # Below minimum priority, skip
            continue

        stories_by_id[sid] = row

    return stories_by_id


def parse_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def load_pythag_report(pythag_path: Path) -> List[Dict[str, Any]]:
    """
    Load the season pythag report for 1981.
    """
    if not pythag_path.exists():
        print(f"Error: pythag report not found at {pythag_path}", file=sys.stderr)
        sys.exit(1)

    with pythag_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = [
            "team_id",
            "team_abbr",
            "team_name",
            "season",
            "wins",
            "losses",
            "pythag_expected_wins",
            "pythag_expected_losses",
            "pythag_diff",
        ]
        missing = [c for c in required_cols if c not in reader.fieldnames]
        if missing:
            print(
                f"Error: pythag report {pythag_path} missing columns: {', '.join(missing)}",
                file=sys.stderr,
            )
            sys.exit(1)

        rows = list(reader)

    return rows


def build_candidates_for_team(
    team_row: Dict[str, Any],
    stories_by_id: Dict[str, Dict[str, Any]],
    week_label: str,
) -> List[Dict[str, Any]]:
    """
    Given a team row and the active Pythag stories, return any candidate rows
    for that team (0, 1, or 2).
    """
    candidates: List[Dict[str, Any]] = []

    wins = parse_int(team_row.get("wins"))
    losses = parse_int(team_row.get("losses"))
    games_played = wins + losses

    if games_played < GAMES_PLAYED_MIN:
        return candidates

    pythag_diff = parse_float(team_row.get("pythag_diff"))

    # Overachiever
    if STORY_OVER_ID in stories_by_id and pythag_diff >= PYTHAG_DIFF_OVER_MIN:
        story = stories_by_id[STORY_OVER_ID]
        candidates.append(
            {
                "story_id": STORY_OVER_ID,
                "story_name": story.get("name", "").strip(),
                "level": story.get("level", "").strip(),
                "facet": story.get("facet", "").strip(),
                "story_type": story.get("story_type", "").strip(),
                "priority": story.get("priority", "").strip().upper(),
                "week_label": week_label,
                "team_id": (team_row.get("team_id") or "").strip(),
                "team_abbr": (team_row.get("team_abbr") or "").strip(),
                "team_name": (team_row.get("team_name") or "").strip(),
                "season": (team_row.get("season") or "").strip(),
                "wins": str(wins),
                "losses": str(losses),
                "pythag_expected_wins": (team_row.get("pythag_expected_wins") or "").strip(),
                "pythag_expected_losses": (team_row.get("pythag_expected_losses") or "").strip(),
                "pythag_diff": f"{pythag_diff:.3f}",
                "angle_label": "overachiever",
            }
        )

    # Underachiever
    if STORY_UNDER_ID in stories_by_id and pythag_diff <= PYTHAG_DIFF_UNDER_MAX:
        story = stories_by_id[STORY_UNDER_ID]
        candidates.append(
            {
                "story_id": STORY_UNDER_ID,
                "story_name": story.get("name", "").strip(),
                "level": story.get("level", "").strip(),
                "facet": story.get("facet", "").strip(),
                "story_type": story.get("story_type", "").strip(),
                "priority": story.get("priority", "").strip().upper(),
                "week_label": week_label,
                "team_id": (team_row.get("team_id") or "").strip(),
                "team_abbr": (team_row.get("team_abbr") or "").strip(),
                "team_name": (team_row.get("team_name") or "").strip(),
                "season": (team_row.get("season") or "").strip(),
                "wins": str(wins),
                "losses": str(losses),
                "pythag_expected_wins": (team_row.get("pythag_expected_wins") or "").strip(),
                "pythag_expected_losses": (team_row.get("pythag_expected_losses") or "").strip(),
                "pythag_diff": f"{pythag_diff:.3f}",
                "angle_label": "underachiever",
            }
        )

    return candidates


def sort_candidates(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def sort_key(r: Dict[str, Any]):
        story_id = r.get("story_id") or ""
        team_abbr = r.get("team_abbr") or ""
        try:
            diff = float(r.get("pythag_diff", "0") or 0.0)
        except ValueError:
            diff = 0.0
        # sort by story_id, then biggest |pythag_diff| first, then team_abbr
        return (story_id, -abs(diff), team_abbr)

    return sorted(rows, key=sort_key)


def write_candidates(
    out_path: Path,
    rows: List[Dict[str, Any]],
) -> None:
    fieldnames = [
        "story_id",
        "story_name",
        "level",
        "facet",
        "story_type",
        "priority",
        "week_label",
        "team_id",
        "team_abbr",
        "team_name",
        "season",
        "wins",
        "losses",
        "pythag_expected_wins",
        "pythag_expected_losses",
        "pythag_diff",
        "angle_label",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Pythagorean story triggers for a given week label and "
            "emit story_candidates_<week_label>.csv"
        )
    )
    parser.add_argument("week_label", help="Week label, e.g. 1981_week_05")
    parser.add_argument(
        "--min-priority",
        dest="min_priority",
        choices=["A", "B", "C"],
        default="A",
        help="Minimum story priority to consider (default: A)",
    )
    args = parser.parse_args()

    week_label = args.week_label
    min_priority = args.min_priority

    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent
    csv_root = repo_root / "csv"

    menu_path = csv_root / f"story_menu_{week_label}.csv"
    pythag_path = csv_root / "out" / "star_schema" / "fact_team_season_1981_pythag_report.csv"
    out_path = csv_root / f"story_candidates_{week_label}.csv"

    # Load menu and pythag data
    stories_by_id = load_story_menu(menu_path, min_priority)
    if not stories_by_id:
        print(
            f"No eligible Pythag stories in menu {menu_path} at min priority {min_priority}.",
            file=sys.stderr,
        )
        sys.exit(1)

    pythag_rows = load_pythag_report(pythag_path)

    # Evaluate candidates
    all_candidates: List[Dict[str, Any]] = []
    over_count = 0
    under_count = 0

    for team_row in pythag_rows:
        team_candidates = build_candidates_for_team(team_row, stories_by_id, week_label)
        for c in team_candidates:
            all_candidates.append(c)
            if c.get("angle_label") == "overachiever":
                over_count += 1
            elif c.get("angle_label") == "underachiever":
                under_count += 1

    if not all_candidates:
        print(
            f"No candidates found for week {week_label} at min priority {min_priority} "
            f"(games >= {GAMES_PLAYED_MIN}, pythag_diff +/-{int(PYTHAG_DIFF_OVER_MIN)}).",
            file=sys.stderr,
        )
        sys.exit(1)

    sorted_rows = sort_candidates(all_candidates)
    write_candidates(out_path, sorted_rows)

    print(
        f"Wrote {len(sorted_rows)} story candidates to {out_path} "
        f"(overachievers: {over_count}, underachievers: {under_count}, "
        f"min priority: {min_priority}, games >= {GAMES_PLAYED_MIN})."
    )


if __name__ == "__main__":
    main()
