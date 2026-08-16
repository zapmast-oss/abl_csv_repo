from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

# Paths
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
CSV_DIR = REPO_ROOT / "csv"
OUT_DIR = CSV_DIR / "out"

SOURCE_CSV = OUT_DIR / "story_candidates_1981_week5.csv"
MARKDOWN_OUT = OUT_DIR / "story_menu_1981_week5.md"

# Simple story dictionary scaffold; edit/extend as needed.
STORY_DICTIONARY: Dict[str, Dict[str, object]] = {
    "TEAM_STREAK": {
        "label": "Team on a Tear",
        "default_hook": "{team_name} have ripped off {streak_len} straight and just outscored opponents {runs_for}-{runs_against}.",
        "category": "team",
        "base_weight": 1.2,
    },
    "PLAYER_HOT_BAT": {
        "label": "Hot Bat",
        "default_hook": "{player_name} hit {batting_avg:.3f} with {home_runs} HR and {rbi} RBI this week, dragging {team_name} up the standings.",
        "category": "player",
        "base_weight": 1.0,
    },
    "TEAM_PYTHAG_OVERACHIEVER": {
        "label": "Pythag Overachiever",
        "default_hook": "{team_name} are beating the math by {pythag_diff:+.1f} wins over expectation.",
        "category": "team",
        "base_weight": 1.1,
    },
    "TEAM_PYTHAG_UNDERACHIEVER": {
        "label": "Pythag Underachiever",
        "default_hook": "{team_name} are under their Pythag by {pythag_diff:+.1f} wins.",
        "category": "team",
        "base_weight": 1.1,
    },
    # Add more patterns here as needed.
}


class SafeDict(dict):
    """Return blanks for missing keys when formatting hooks."""

    def __missing__(self, key):
        return ""


def load_candidates(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"ERROR: candidates CSV not found: {path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {len(df)} rows from {path}")
    print("[INFO] Columns:", list(df.columns))
    return df


def pick_story_type_col(df: pd.DataFrame) -> Optional[str]:
    for cand in ["story_type", "type", "storyType"]:
        if cand in df.columns:
            return cand
    for col in df.columns:
        if "story" in col.lower() and "type" in col.lower():
            return col
    return None


def pick_conference_col(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        lc = col.lower()
        if "conference" in lc:
            return col
    return None


def candidate_score(row: pd.Series, numeric_cols: Sequence[str], base_weight: float) -> float:
    for col in numeric_cols:
        val = row.get(col)
        try:
            num = float(val)
            return num * base_weight
        except Exception:
            continue
    return 0.0


def format_hook(template: str, row: pd.Series) -> str:
    try:
        return template.format_map(SafeDict(row.to_dict()))
    except Exception:
        return template


def build_title(label: str, row: pd.Series) -> str:
    for key in ["team_name", "team_abbr", "player_name"]:
        if key in row and pd.notna(row[key]) and str(row[key]).strip():
            return f"{label} – {row[key]}"
    return label


def pick_key_stats(row: pd.Series, max_fields: int = 4) -> List[str]:
    keys_to_skip = {
        "story_id",
        "story_name",
        "story_type",
        "priority",
        "level",
        "facet",
        "week_label",
        "angle_label",
        "team_id",
        "team_name",
        "team_abbr",
        "player_name",
        "season",
    }
    stats = []
    for col in row.index:
        if col in keys_to_skip:
            continue
        val = row[col]
        if pd.isna(val):
            continue
        if isinstance(val, float):
            stats.append(f"{col}={val:.3f}")
        else:
            stats.append(f"{col}={val}")
        if len(stats) >= max_fields:
            break
    return stats


def build_story_entries(df: pd.DataFrame) -> List[dict]:
    story_type_col = pick_story_type_col(df)
    if not story_type_col:
        print("ERROR: Could not find a story_type column in candidates.", file=sys.stderr)
        sys.exit(1)

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    conference_col = pick_conference_col(df)

    entries = []
    for idx, row in df.iterrows():
        stype = str(row.get(story_type_col, "")).strip()
        story_def = STORY_DICTIONARY.get(stype, {
            "label": "Generic Story",
            "default_hook": "{team_name} have a story worth telling.",
            "category": "generic",
            "base_weight": 1.0,
        })
        score = candidate_score(row, numeric_cols, float(story_def.get("base_weight", 1.0)))
        hook = format_hook(str(story_def.get("default_hook", "")), row)
        title = build_title(str(story_def.get("label", "Story")), row)

        entries.append(
            {
                "id": f"S-{idx+1:03d}",
                "story_type": stype,
                "category": story_def.get("category", "generic"),
                "title": title,
                "hook": hook,
                "score": score,
                "team_abbr": row.get("team_abbr", ""),
                "team_name": row.get("team_name", ""),
                "player_name": row.get("player_name", ""),
                "conference": row.get(conference_col, "") if conference_col else "",
                "key_stats": pick_key_stats(row),
            }
        )

    # sort by score descending
    entries.sort(key=lambda e: e["score"], reverse=True)
    return entries


def render_markdown(entries: List[dict], conference_col_present: bool) -> str:
    lines: List[str] = []
    lines.append("# ABL 1981 – Week 5 Story Menu")
    lines.append("")
    lines.append("## Top League-Wide Stories")
    for entry in entries[:12]:
        lines.append(f"1. [{entry['id']}] {entry['title']}")
        lines.append(f"   - Hook: {entry['hook']}")
        lines.append(f"   - Type: {entry['story_type']} / {entry['category']}")
        who = entry.get("player_name") or entry.get("team_name") or entry.get("team_abbr")
        if who:
            lines.append(f"   - Team/Player: {who}")
        if entry["key_stats"]:
            lines.append(f"   - Key stats: {', '.join(entry['key_stats'])}")
        lines.append("")

    if conference_col_present:
        for conf in ["NBC", "ABC"]:
            conf_entries = [e for e in entries if str(e.get("conference", "")).upper() == conf][:3]
            if not conf_entries:
                continue
            lines.append(f"## Conference Spotlight – {conf}")
            for entry in conf_entries:
                lines.append(f"1. [{entry['id']}] {entry['title']}")
                lines.append(f"   - Hook: {entry['hook']}")
                if entry["key_stats"]:
                    lines.append(f"   - Key stats: {', '.join(entry['key_stats'])}")
                lines.append("")

    return "\n".join(lines).strip() + "\n"


def print_summary(entries: List[dict]) -> None:
    print(f"[INFO] Total candidates: {len(entries)}")
    print("[INFO] Top 5 stories by score:")
    for entry in entries[:5]:
        print(f"  {entry['id']} ({entry['score']:.3f}) - {entry['title']}")


def main() -> None:
    df = load_candidates(SOURCE_CSV)
    entries = build_story_entries(df)
    conference_present = pick_conference_col(df) is not None

    print_summary(entries)

    md = render_markdown(entries, conference_present)
    MARKDOWN_OUT.parent.mkdir(parents=True, exist_ok=True)
    MARKDOWN_OUT.write_text(md, encoding="utf-8")
    print(f"[INFO] Wrote story menu to {MARKDOWN_OUT}")


if __name__ == "__main__":
    main()
