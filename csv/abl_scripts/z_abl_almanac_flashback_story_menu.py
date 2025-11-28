import argparse
from pathlib import Path
import pandas as pd

STORY_META = {
    "TOP_SEASON_TEAM_RUN_DIFF": {
        "section": "Season Giants – Run Differential",
        "order": 1,
        "max_items": 5,
    },
    "TOP_SEASON_TEAM_WIN_PCT": {
        "section": "Season Giants – Winning Percentage",
        "order": 2,
        "max_items": 5,
    },
    "BEST_SECOND_HALF_SURGE": {
        "section": "Second-Half Surges",
        "order": 3,
        "max_items": 5,
    },
    "WORST_SECOND_HALF_COLLAPSE": {
        "section": "Second-Half Collapses",
        "order": 4,
        "max_items": 5,
    },
    "BEST_MONTH_VS_SEASON": {
        "section": "Month of Glory – Overachievers",
        "order": 5,
        "max_items": 10,
    },
    "WORST_MONTH_VS_SEASON": {
        "section": "Month of Misery – Slumps",
        "order": 6,
        "max_items": 10,
    },
}


def build_story_menu(df: pd.DataFrame, season: int, league_id: int) -> str:
    required_cols = [
        "season",
        "league_id",
        "story_type",
        "rank",
        "team_id",
        "team_abbr",
        "team_name",
        "conf",
        "division",
        "focus_label",
        "metric_name",
        "metric_value",
        "comparison_note",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"story_candidates is missing required columns: {missing}. "
            f"Available: {sorted(df.columns.tolist())}"
        )

    header_lines = [
        f"# ABL Flashback {season} – Story Menu",
        "",
        f"_League ID {league_id}_",
        "",
        "This menu collects high-level story candidates for the full season. "
        "Use it as an outline – pick a few anchors from each section, then dive "
        "back into the CSVs for deeper detail.",
        "",
    ]
    sections = []

    # order story_types by STORY_META order, then any unknowns at the end
    def story_sort_key(story_type: str):
        meta = STORY_META.get(story_type)
        return meta["order"] if meta is not None else 999

    for story_type in sorted(df["story_type"].unique(), key=story_sort_key):
        meta = STORY_META.get(story_type, {})
        section_title = meta.get("section", story_type)
        max_items = meta.get("max_items", None)

        sub = df[df["story_type"] == story_type].copy()
        if sub.empty:
            continue

        sub = sub.sort_values(["rank", "metric_value"], ascending=[True, False])
        if max_items is not None:
            sub = sub.head(max_items)

        lines = [f"## {section_title}", ""]
        for _, row in sub.iterrows():
            team_name = str(row["team_name"])
            abbr = str(row["team_abbr"])
            conf = str(row["conf"])
            division = str(row["division"])
            focus = str(row["focus_label"])
            note = str(row["comparison_note"]).strip()
            metric_name = str(row["metric_name"])
            metric_value = row["metric_value"]

            context_bits = []
            if focus and focus.lower() not in ("", "season"):
                context_bits.append(focus)
            context_bits.append(f"{conf} {division}".strip())
            context = " | ".join(context_bits)

            metric_str = f"{metric_name}={metric_value}" if metric_name else ""
            extra = "  ".join(bit for bit in (context, metric_str) if bit)

            bullet = f"- **{team_name}** ({abbr}) — {note}"
            if extra:
                bullet += f"  \n  _{extra}_"

            lines.append(bullet)

        lines.append("")  # blank line after section
        sections.extend(lines)

    return "\n".join(header_lines + sections)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Flashback story menu markdown from story candidates."
    )
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--league-id", type=int, required=True)
    parser.add_argument(
        "--almanac-root",
        type=str,
        default="csv/out/almanac",
        help="Root folder for almanac outputs.",
    )
    args = parser.parse_args()

    season = args.season
    league_id = args.league_id
    root = Path(args.almanac_root)

    cand_path = (
        root
        / str(season)
        / f"flashback_story_candidates_{season}_league{league_id}.csv"
    )
    out_path = (
        root
        / str(season)
        / f"flashback_story_menu_{season}_league{league_id}.md"
    )

    print(f"[DEBUG] season={season}, league_id={league_id}")
    print(f"[DEBUG] almanac_root={root}")
    print(f"[DEBUG] candidates={cand_path}")

    if not cand_path.exists():
        raise FileNotFoundError(f"Story candidates file not found: {cand_path}")

    df = pd.read_csv(cand_path)
    print(f"[INFO] Loaded {len(df)} story candidates")

    md = build_story_menu(df, season=season, league_id=league_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")

    print(f"[OK] Wrote Flashback story menu to {out_path}")
    return 0


if __name__ == "__main__":import argparse
from pathlib import Path
import pandas as pd

STORY_META = {
    "TOP_SEASON_TEAM_RUN_DIFF": {
        "section": "Season Giants – Run Differential",
        "order": 1,
        "max_items": 5,
    },
    "TOP_SEASON_TEAM_WIN_PCT": {
        "section": "Season Giants – Winning Percentage",
        "order": 2,
        "max_items": 5,
    },
    "BEST_SECOND_HALF_SURGE": {
        "section": "Second-Half Surges",
        "order": 3,
        "max_items": 5,
    },
    "WORST_SECOND_HALF_COLLAPSE": {
        "section": "Second-Half Collapses",
        "order": 4,
        "max_items": 5,
    },
    "BEST_MONTH_VS_SEASON": {
        "section": "Month of Glory – Overachievers",
        "order": 5,
        "max_items": 10,
    },
    "WORST_MONTH_VS_SEASON": {
        "section": "Month of Misery – Slumps",
        "order": 6,
        "max_items": 10,
    },
}


def build_story_menu(df: pd.DataFrame, season: int, league_id: int) -> str:
    required_cols = [
        "season",
        "league_id",
        "story_type",
        "rank",
        "team_id",
        "team_abbr",
        "team_name",
        "conf",
        "division",
        "focus_label",
        "metric_name",
        "metric_value",
        "comparison_note",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"story_candidates is missing required columns: {missing}. "
            f"Available: {sorted(df.columns.tolist())}"
        )

    header_lines = [
        f"# ABL Flashback {season} – Story Menu",
        "",
        f"_League ID {league_id}_",
        "",
        "This menu collects high-level story candidates for the full season. "
        "Use it as an outline – pick a few anchors from each section, then dive "
        "back into the CSVs for deeper detail.",
        "",
    ]
    sections = []

    # order story_types by STORY_META order, then any unknowns at the end
    def story_sort_key(story_type: str):
        meta = STORY_META.get(story_type)
        return meta["order"] if meta is not None else 999

    for story_type in sorted(df["story_type"].unique(), key=story_sort_key):
        meta = STORY_META.get(story_type, {})
        section_title = meta.get("section", story_type)
        max_items = meta.get("max_items", None)

        sub = df[df["story_type"] == story_type].copy()
        if sub.empty:
            continue

        sub = sub.sort_values(["rank", "metric_value"], ascending=[True, False])
        if max_items is not None:
            sub = sub.head(max_items)

        lines = [f"## {section_title}", ""]
        for _, row in sub.iterrows():
            team_name = str(row["team_name"])
            abbr = str(row["team_abbr"])
            conf = str(row["conf"])
            division = str(row["division"])
            focus = str(row["focus_label"])
            note = str(row["comparison_note"]).strip()
            metric_name = str(row["metric_name"])
            metric_value = row["metric_value"]

            context_bits = []
            if focus and focus.lower() not in ("", "season"):
                context_bits.append(focus)
            context_bits.append(f"{conf} {division}".strip())
            context = " | ".join(context_bits)

            metric_str = f"{metric_name}={metric_value}" if metric_name else ""
            extra = "  ".join(bit for bit in (context, metric_str) if bit)

            bullet = f"- **{team_name}** ({abbr}) — {note}"
            if extra:
                bullet += f"  \n  _{extra}_"

            lines.append(bullet)

        lines.append("")  # blank line after section
        sections.extend(lines)

    return "\n".join(header_lines + sections)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Flashback story menu markdown from story candidates."
    )
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--league-id", type=int, required=True)
    parser.add_argument(
        "--almanac-root",
        type=str,
        default="csv/out/almanac",
        help="Root folder for almanac outputs.",
    )
    args = parser.parse_args()

    season = args.season
    league_id = args.league_id
    root = Path(args.almanac_root)

    cand_path = (
        root
        / str(season)
        / f"flashback_story_candidates_{season}_league{league_id}.csv"
    )
    out_path = (
        root
        / str(season)
        / f"flashback_story_menu_{season}_league{league_id}.md"
    )

    print(f"[DEBUG] season={season}, league_id={league_id}")
    print(f"[DEBUG] almanac_root={root}")
    print(f"[DEBUG] candidates={cand_path}")

    if not cand_path.exists():
        raise FileNotFoundError(f"Story candidates file not found: {cand_path}")

    df = pd.read_csv(cand_path)
    print(f"[INFO] Loaded {len(df)} story candidates")

    md = build_story_menu(df, season=season, league_id=league_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")

    print(f"[OK] Wrote Flashback story menu to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

    raise SystemExit(main())
