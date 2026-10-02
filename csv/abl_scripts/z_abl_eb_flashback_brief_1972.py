#!/usr/bin/env python
"""
Build an EB-friendly flashback data brief for a given season/league from almanac outputs.

Outputs:
  csv/out/almanac/{season}/eb_flashback_brief_{season}_league{league_id}.md

Assumptions:
  - Run from repo root.
  - Required source CSVs already exist under csv/out/almanac/{season}/.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from eb_text_utils import normalize_eb_text


def log(msg: str) -> None:
    print(msg)


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found at {path}")
    df = pd.read_csv(path)
    log(f"[INFO] Loaded {len(df)} rows from {label}")
    log(f"[DEBUG] {label} columns: {list(df.columns)}")
    return df


def find_first(df: pd.DataFrame, substrings: Iterable[str]) -> str:
    for col in df.columns:
        col_low = col.lower()
        if all(sub.lower() in col_low for sub in substrings):
            return col
    raise RuntimeError(f"Could not find column containing {list(substrings)} in {list(df.columns)}")


def detect_league_cols(df: pd.DataFrame) -> dict[str, str]:
    cols = {
        "team_name": find_first(df, ["team"]),
        "team_abbr": find_first(df, ["abbr"]),
        "wins": find_first(df, ["win"]),
        "losses": find_first(df, ["loss"]),
        "pct": find_first(df, ["pct"]),
        "runs_for": find_first(df, ["runs_for"]),
        "runs_against": find_first(df, ["runs_against"]),
        "run_diff": find_first(df, ["run_diff"]),
        "conf": find_first(df, ["conf"]),
        "division": find_first(df, ["div"]),
    }
    return cols


def detect_agg_cols(df: pd.DataFrame, label: str) -> dict[str, str]:
    def pick(fn):
        for col in df.columns:
            if fn(col.lower()):
                return col
        raise RuntimeError(f"Could not detect {label} aggregate columns in {list(df.columns)}")

    wins = pick(lambda c: "win" in c and ("total" in c or "sum" in c))
    losses = pick(lambda c: "loss" in c and ("total" in c or "sum" in c))
    run_diff = pick(lambda c: "run_diff" in c)
    pct = pick(lambda c: "pct" in c)
    return {"wins": wins, "losses": losses, "run_diff": run_diff, "pct": pct}


def build_league_section(df: pd.DataFrame, cols: dict[str, str]) -> list[str]:
    lines: list[str] = []
    top_rd = (
        df[[cols["team_name"], cols["team_abbr"], cols["run_diff"], cols["runs_for"], cols["runs_against"]]]
        .sort_values(cols["run_diff"], ascending=False)
        .head(5)
    )
    top_pct = df[[cols["team_name"], cols["team_abbr"], cols["pct"], cols["wins"], cols["losses"]]].sort_values(
        cols["pct"], ascending=False
    ).head(5)

    lines.append("## 5k View – League Snapshot")
    lines.append("")
    lines.append("Top run differential:")
    for _, r in top_rd.iterrows():
        lines.append(
            f"- {r[cols['team_name']]} ({r[cols['team_abbr']]}) — run_diff={r[cols['run_diff']]}, RS={r[cols['runs_for']]}, RA={r[cols['runs_against']]}"
        )
    lines.append("")
    lines.append("Top winning percentage:")
    for _, r in top_pct.iterrows():
        lines.append(
            f"- {r[cols['team_name']]} ({r[cols['team_abbr']]}) — pct={r[cols['pct']]:.3f}, record={int(r[cols['wins']])}-{int(r[cols['losses']])}"
        )
    lines.append("")
    return lines


def build_division_section(df_div: pd.DataFrame, div_cols: dict[str, str]) -> list[str]:
    lines: list[str] = []
    lines.append("## 4k View – Division Snapshot")
    lines.append("")
    for _, r in df_div.iterrows():
        lines.append(
            f"- {r['conf']} / {r['division']}: wins={r[div_cols['wins']]}, losses={r[div_cols['losses']]}, pct_avg={r[div_cols['pct']]:.3f}, run_diff_total={r[div_cols['run_diff']]}"
        )
    lines.append("")
    return lines


def build_story_section(df_story: pd.DataFrame) -> list[str]:
    required = {"story_group", "story_type", "team_name", "team_abbr", "metric_name", "metric_value"}
    missing = required - set(df_story.columns)
    if missing:
        raise RuntimeError(f"flashback story candidates missing columns: {missing}")
    lines: list[str] = []
    lines.append("## 3k View – Flashback Story Candidates")
    lines.append("")
    for group, sub in df_story.groupby("story_group"):
        lines.append(f"### {group}")
        for _, r in sub.iterrows():
            if "Month of Glory" in group or "Month of Misery" in group:
                try:
                    mw = int(r.get("month_wins", 0))
                    ml = int(r.get("month_losses", 0))
                    mp = float(r.get("month_win_pct", 0.0))
                    delta = float(r.get("metric_value", 0.0))
                    lines.append(
                        f"- {r['team_name']} ({r['team_abbr']}): went {mw}-{ml} that month ({mp:.3f}), delta vs season={delta:+.3f}"
                    )
                except Exception:
                    lines.append(
                        f"- {r['team_name']} ({r['team_abbr']}): metric={r['metric_name']} value={r['metric_value']}"
                    )
            elif "Second-Half Surges" in group or "Second-Half Collapses" in group:
                try:
                    hw = int(r.get("half_wins", 0))
                    hl = int(r.get("half_losses", 0))
                    hp = float(r.get("half_win_pct", 0.0))
                    delta = float(r.get("metric_value", 0.0))
                    lines.append(
                        f"- {r['team_name']} ({r['team_abbr']}): went {hw}-{hl} in the 2nd half ({hp:.3f}), delta vs season={delta:+.3f}"
                    )
                except Exception:
                    lines.append(
                        f"- {r['team_name']} ({r['team_abbr']}): metric={r['metric_name']} value={r['metric_value']}"
                    )
            else:
                lines.append(
                    f"- {r['team_name']} ({r['team_abbr']}): {r['metric_name']}={r['metric_value']}"
                )
        lines.append("")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Build EB flashback data brief for a season/league.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--league-id", type=int, required=True)
    args = parser.parse_args()

    season = args.season
    league_id = args.league_id

    base = Path("csv/out/almanac") / str(season)
    league_path = base / f"league_season_summary_{season}_league{league_id}.csv"
    conf_path = base / f"conference_summary_{season}_league{league_id}.csv"
    div_path = base / f"division_summary_{season}_league{league_id}.csv"
    story_path = base / f"flashback_story_candidates_{season}_league{league_id}.csv"
    out_path = base / f"eb_flashback_brief_{season}_league{league_id}.md"

    league_df = read_csv(league_path, "league_season_summary")
    conf_df = read_csv(conf_path, "conference_summary")
    div_df = read_csv(div_path, "division_summary")
    story_df = read_csv(story_path, "flashback_story_candidates")

    league_cols = detect_league_cols(league_df)
    conf_cols = detect_agg_cols(conf_df, "conference_summary")
    div_cols = detect_agg_cols(div_df, "division_summary")

    lines: list[str] = []
    lines.append(f"# EB Flashback {season} – Data Brief (DO NOT PUBLISH)")
    lines.append("")
    lines.append(f"_League ID {league_id}_")
    lines.append("")
    lines.append(
        "This brief is for Ernie Bewell’s internal use. It summarizes key 5k/4k/3k data for the season."
    )
    lines.append("")

    lines.extend(build_league_section(league_df, league_cols))
    lines.extend(build_division_section(div_df, div_cols))
    lines.extend(build_story_section(story_df))

    full_text = "\n".join(lines)
    full_text = normalize_eb_text(full_text)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(full_text, encoding="utf-8")
    log(f"[OK] Wrote EB flashback brief to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
