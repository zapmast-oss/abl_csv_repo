#!/usr/bin/env python
"""
Summarize Grand Series box scores into a simple markdown recap.

Inputs (defaults assume prior extractor run):
  - --season (int, required), --league-id (int, default 200)
  - --gs-dir (path, default csv/out/eb/gs_<season>)
    containing game_box_*.html (and optionally game_log_*.html)

Output:
  - csv/out/eb/gs_<season>_summary.md with a generated-on timestamp and per-game bullets.

We keep parsing lightweight: use pandas.read_html to grab the primary line-score
table, pull team names and the R column, and derive a simple series tally.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import logging
import re
from io import StringIO
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Grand Series box scores.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--league-id", type=int, default=200)
    parser.add_argument(
        "--gs-dir",
        type=Path,
        help="Directory with game_box_*.html; defaults to csv/out/eb/gs_<season>.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        help="Optional explicit output path for summary markdown.",
    )
    return parser.parse_args()


def load_box_table(html_text: str) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Load the primary line-score table (teams as rows, innings + R/H/E as columns).
    Returns the table and the date string if present in the title/subtitle.
    """
    tables = pd.read_html(StringIO(html_text))
    table = None
    for t in tables:
        cols = [str(c).strip().lower() for c in t.columns]
        if any(c == "r" for c in cols) and len(t) >= 2:
            table = t
            break
    if table is None:
        raise ValueError("No line-score table with an R column found in box HTML.")

    date_match = re.search(r"([A-Za-z]+\s+\d{1,2},\s+\d{4})", html_text)
    date_str = date_match.group(1) if date_match else None
    return table, date_str


def parse_game(box_path: Path) -> dict:
    html_text = box_path.read_text(encoding="utf-8", errors="ignore")
    table, date_str = load_box_table(html_text)

    # Normalize column names and find R column
    table.columns = [str(c).strip() for c in table.columns]
    r_col = next((c for c in table.columns if str(c).strip().lower() == "r"), None)
    if r_col is None:
        raise ValueError(f"No R column found in {box_path}")

    team_a = str(table.iloc[0, 0]).strip()
    team_b = str(table.iloc[1, 0]).strip()
    runs_a = float(table.iloc[0][r_col])
    runs_b = float(table.iloc[1][r_col])

    winner = team_a if runs_a > runs_b else team_b

    # Derive game number from filename if possible
    m = re.search(r"game_box_(\d+)", box_path.name)
    game_id = m.group(1) if m else box_path.name

    return {
        "game_id": game_id,
        "date": date_str,
        "team_a": team_a,
        "team_b": team_b,
        "runs_a": runs_a,
        "runs_b": runs_b,
        "winner": winner,
        "path": box_path,
    }


def summarize_series(box_files: List[Path]) -> List[dict]:
    games = [parse_game(p) for p in box_files]
    # Keep order as provided (assumed chronological if filenames sorted)
    tally: dict[str, int] = {}
    for g in games:
        tally[g["winner"]] = tally.get(g["winner"], 0) + 1
        g["tally"] = dict(tally)
    return games


def build_markdown(
    games: List[dict], season: int, league_id: int, generated_at: datetime
) -> str:
    lines: List[str] = []
    lines.append(f"# Grand Series {season} Recap (League {league_id})")
    lines.append(
        f"Generated on: {generated_at.strftime('%Y-%m-%d %H:%M:%S')} (local time)"
    )
    lines.append("")
    for idx, g in enumerate(games, 1):
        date_str = g["date"] or "Unknown date"
        tally_str = ", ".join(f"{k}:{v}" for k, v in g["tally"].items())
        lines.append(
            f"{idx}. {date_str} — {g['team_a']} {int(g['runs_a'])} vs {g['team_b']} {int(g['runs_b'])} "
            f"(winner: {g['winner']}; tally: {tally_str})"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    season = args.season
    gs_dir = args.gs_dir or Path("csv") / "out" / "eb" / f"gs_{season}"
    if not gs_dir.exists():
        raise FileNotFoundError(f"GS directory not found: {gs_dir}")

    box_files = sorted(gs_dir.glob("game_box_*.html"))
    if not box_files:
        raise FileNotFoundError(f"No game_box_*.html found in {gs_dir}")

    games = summarize_series(box_files)
    generated_at = datetime.now()
    md = build_markdown(games, season=season, league_id=args.league_id, generated_at=generated_at)

    out_path = args.out_md or (Path("csv") / "out" / "eb" / f"gs_{season}_summary.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")

    logging.info("Wrote Grand Series summary to %s", out_path)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
