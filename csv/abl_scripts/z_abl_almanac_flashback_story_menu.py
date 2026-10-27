from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


TITLE_OVERRIDES: Dict[str, str] = {
    "season_run_diff_giants": "Season Giants – Run Differential",
    "season_win_pct_giants": "Season Giants – Winning Percentage",
    "second_half_surges": "Second-Half Surges",
    "second_half_collapses": "Second-Half Collapses",
    "month_of_glory_overachievers": "Month of Glory – Overachievers",
    "month_of_misery_slumps": "Month of Misery – Slumps",
}


def _safe_float(val):
    try:
        return float(val)
    except Exception:
        return None


def _format_pct(val) -> str:
    f = _safe_float(val)
    if f is None:
        return "N/A"
    return f"{f:.3f}"


def _format_run_diff(val) -> str:
    f = _safe_float(val)
    if f is None:
        return "N/A"
    return f"{int(round(f))}"


def _format_delta(val) -> str:
    f = _safe_float(val)
    if f is None:
        return "N/A"
    return f"{f:+.3f}"


def make_section_title(story_group: str) -> str:
    key = str(story_group)
    if key in TITLE_OVERRIDES:
        return TITLE_OVERRIDES[key]
    # Fallback: snake_case -> Title Case
    parts = [p for p in key.replace("-", "_").split("_") if p]
    if not parts:
        return "Stories"
    return " ".join(w.capitalize() for w in parts)


def format_story_line(group_key: str, row: pd.Series) -> str:
    g = str(group_key).lower()
    team_name = str(row.get("team_name", "")).strip() or "Unknown Team"
    team_abbr = str(row.get("team_abbr", "")).strip()
    conf = str(row.get("conf", "")).strip()
    division = str(row.get("division", "")).strip()
    metric_name = str(row.get("metric_name", "")).strip()
    metric_value = row.get("metric_value", None)

    metric_str = str(metric_value)
    context_metric = f"{metric_name}={metric_str}"

    # Season run-diff giants
    if "run_diff" in g and "season" in g:
        metric_str = _format_run_diff(row.get("run_diff", metric_value))
        context_metric = f"run_diff={metric_str}"
        main = (
            f"**{team_name}** ({team_abbr}) — "
            f"{team_name} led the league with a run differential of {metric_str}."
        )

    # Season win-pct giants
    elif "win_pct" in g and "season" in g:
        pct_str = _format_pct(row.get("pct", metric_value))
        context_metric = f"pct={pct_str}"
        main = (
            f"**{team_name}** ({team_abbr}) — "
            f"{team_name} posted a {pct_str} winning percentage, among the best in the league."
        )

    # Half surges / collapses
    elif "half" in g:
        half_label = str(row.get("half_label", "2nd half")).strip() or "2nd half"
        half_pct = _format_pct(row.get("half_win_pct", None))
        delta_str = _format_delta(metric_value)
        if any(word in g for word in ["collapse", "slump", "misery", "fade"]):
            verb = "faded after midseason"
        else:
            verb = "caught fire after midseason"
        main = (
            f"**{team_name}** ({team_abbr}) — "
            f"{team_name} {verb}: {half_pct} in the {half_label} (delta {delta_str} vs season)."
        )
        context_metric = f"{half_label} | half_win_pct_delta_vs_season={delta_str}"

    # Monthly over / under performance
    elif "month" in g:
        month_label = str(row.get("month_label", row.get("month", ""))).strip()
        if not month_label:
            month_label = "that month"
        month_pct = _format_pct(row.get("month_win_pct", None))
        delta_str = _format_delta(metric_value)
        is_slump = any(word in g for word in ["slump", "misery", "collapse", "worst"])
        if is_slump or (_safe_float(metric_value) is not None and _safe_float(metric_value) < 0):
            phrase = "slumped below their season pace"
        else:
            phrase = "played above their season pace"
        main = (
            f"**{team_name}** ({team_abbr}) — "
            f"In {month_label}, {team_name} {phrase}: {month_pct} (delta {delta_str})."
        )
        context_metric = f"{month_label} | month_win_pct_delta_vs_season={delta_str}"

    # Generic fallback
    else:
        val_str = metric_str
        if "pct" in metric_name:
            val_str = _format_pct(metric_value)
        elif "run_diff" in metric_name:
            val_str = _format_run_diff(metric_value)
        main = (
            f"**{team_name}** ({team_abbr}) — "
            f"{team_name} stood out on {metric_name}: {val_str}."
        )
        context_metric = f"{metric_name}={val_str}"

    context_bits = []
    if conf:
        context_bits.append(conf)
    if division:
        context_bits.append(division)
    context_bits.append(context_metric)
    context = "  ".join(context_bits)

    return f"- {main}  \n  _{context}_"


def build_story_menu(df: pd.DataFrame, season: int, league_id: int) -> str:
    required = ["story_group", "story_type", "metric_name", "metric_value", "team_name", "team_abbr"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"story_candidates is missing required columns: {missing}. "
            f"Available: {sorted(df.columns.tolist())}"
        )

    lines: List[str] = []
    lines.append(f"# ABL Flashback {season} – Story Menu")
    lines.append("")
    lines.append(f"_League ID {league_id}_")
    lines.append("")
    lines.append(
        "This menu collects high-level story candidates for the full season. "
        "Use it as an outline – pick a few anchors from each section, then dive back into the CSVs for deeper detail."
    )
    lines.append("")

    groups = sorted(df["story_group"].dropna().unique().tolist())
    if not groups:
        raise ValueError("No story_group values found in candidates CSV.")

    # Helpful debug of what we actually saw
    lines.append(f"<!-- Story groups detected: {', '.join(str(g) for g in groups)} -->")
    lines.append("")

    for group_key in groups:
        section = df[df["story_group"] == group_key].copy()
        if section.empty:
            continue

        title = make_section_title(group_key)
        lines.append(f"## {title}")
        lines.append("")

        g_lower = str(group_key).lower()
        is_negative = any(word in g_lower for word in ["collapse", "slump", "misery", "worst"])
        asc = is_negative

        if "rank_in_group" in section.columns:
            section = section.sort_values(
                by=["rank_in_group", "metric_value"],
                ascending=[True, asc],
                kind="mergesort",
            )
        else:
            section = section.sort_values(by=["metric_value"], ascending=asc, kind="mergesort")

        section = section.head(10)

        for _, row in section.iterrows():
            lines.append(format_story_line(group_key, row))

        lines.append("")

    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Build Flashback story menu Markdown from story candidates."
    )
    p.add_argument("--season", type=int, required=True, help="Season year (e.g., 1972)")
    p.add_argument("--league-id", type=int, required=True, help="League ID (ABL canon: 200)")
    p.add_argument(
        "--almanac-root",
        type=str,
        default="csv/out/almanac",
        help="Root folder for almanac outputs (default: csv/out/almanac)",
    )
    p.add_argument(
        "--candidates",
        type=str,
        default=None,
        help=(
            "Optional explicit path to flashback_story_candidates CSV. "
            "If not provided, built from almanac-root/season."
        ),
    )
    p.add_argument(
        "--out-path",
        type=str,
        default=None,
        help=(
            "Optional explicit output path for the Markdown menu. "
            "If not provided, uses "
            "almanac-root/<season>/flashback_story_menu_<season>_league<id>.md"
        ),
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    season = args.season
    league_id = args.league_id

    almanac_root = Path(args.almanac_root)
    print(f"[DEBUG] season={season}, league_id={league_id}")
    print(f"[DEBUG] almanac_root={almanac_root}")

    if args.candidates:
        candidates_path = Path(args.candidates)
    else:
        candidates_path = (
            almanac_root
            / str(season)
            / f"flashback_story_candidates_{season}_league{league_id}.csv"
        )

    print(f"[DEBUG] candidates={candidates_path}")

    if not candidates_path.is_file():
        raise SystemExit(f"[ERROR] story candidates CSV not found: {candidates_path}")

    df = pd.read_csv(candidates_path)
    print(f"[INFO] Loaded {len(df)} story candidates")

    md = build_story_menu(df, season=season, league_id=league_id)

    if args.out_path:
        out_path = Path(args.out_path)
    else:
        out_path = (
            almanac_root
            / str(season)
            / f"flashback_story_menu_{season}_league{league_id}.md"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"[OK] Wrote Flashback story menu to {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
