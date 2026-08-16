#!/usr/bin/env python
"""
Assemble the EB regular-season pack markdown for any season/league.

Stitches together EB briefs, playoff fields, and preseason hype output.
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Sequence

from eb_text_utils import normalize_eb_text
from z_abl_almanac_champions_from_standings import get_league_champions_from_standings


def log(msg: str) -> None:
    print(msg, flush=True)


def read_md(path: Path, label: str) -> str:
    if not path.exists():
        log(f"[WARN] Missing {label} at {path}; skipping that section.")
        return ""
    text = path.read_text(encoding="utf-8").strip()
    log(f"[INFO] Loaded {label}: {len(text)} chars")
    return text


def _read_optional_fragment(path: Path) -> str | None:
    if not path.exists():
        logging.warning("EB fragment not found: %s", path)
        return None
    return path.read_text(encoding="utf-8")


def playoff_section(champions: dict, conf_order: Sequence[str] = ("ABC", "NBC")) -> str:
    lines: list[str] = []
    lines.append("## Champions & All-Star Context")
    lines.append("")
    slot_order = ["East", "Central", "West", "Wild Card"]
    for conf in conf_order:
        if conf not in champions:
            raise RuntimeError(f"Champions dict missing conference {conf}")
        lines.append(f"### Playoff Field – {conf}")
        for slot in slot_order:
            champ = champions[conf].get(slot)
            if not champ:
                raise RuntimeError(f"Missing {conf} {slot} champion")
            rd = champ.get("run_diff")
            rd_str = int(rd) if rd is not None else "n/a"
            lines.append(
                f"- {slot}: {champ['team_name']} ({champ['team_abbr']}) — {champ['wins']}-{champ['losses']}, run_diff={rd_str}"
            )
        lines.append("")
    return "\n".join(lines).strip()


def splice_month_glory_misery(flashback_md: str, month_md: str) -> str:
    """
    Replace the old Month-of-Glory / Month-of-Misery block in flashback_md
    with the new fragment from month_md, if both are available.
    """
    if not month_md:
        return flashback_md

    marker_start = "### Month of Glory"
    marker_end = "### Season Giants"

    start_idx = flashback_md.find(marker_start)
    end_idx = flashback_md.find(marker_end)

    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        print("[WARN] Could not locate Month-of-Glory block in flashback_md; leaving as-is.")
        return flashback_md

    month_block = month_md.strip() + "\n\n"
    new_md = flashback_md[:start_idx] + month_block + flashback_md[end_idx:]
    return new_md


def main() -> int:
    parser = argparse.ArgumentParser(description="Build EB regular-season pack markdown for any season/league.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--league-id", type=int, default=200)
    parser.add_argument("--almanac-zip", type=Path, default=None)
    parser.add_argument("--dim-team-park", type=Path, default=None)
    args = parser.parse_args()

    season = args.season
    league_id = args.league_id

    repo_root = Path(__file__).resolve().parents[2]
    almanac_dir = repo_root / "csv" / "out" / "almanac" / str(season)
    eb_out_dir = repo_root / "csv" / "out" / "eb"

    log(f"[INFO] Building EB pack for season={season}, league={league_id}")

    champions = get_league_champions_from_standings(
        season,
        league_id=league_id,
        almanac_zip=args.almanac_zip,
        dim_team_park_path=args.dim_team_park,
    )

    flashback = read_md(
        almanac_dir / f"eb_flashback_brief_{season}_league{league_id}.md",
        "eb_flashback_brief",
    )
    month_md_path = eb_out_dir / f"eb_month_glory_misery_{season}_league{league_id}.md"
    month_md = ""
    if month_md_path.exists():
        month_md = month_md_path.read_text(encoding="utf-8")
    else:
        log(f"[WARN] Month-of-Glory fragment not found: {month_md_path}")

    flashback = splice_month_glory_misery(flashback, month_md)

    components = [
        ("eb_monthly_timeline", almanac_dir / f"eb_monthly_timeline_{season}_league{league_id}.md"),
        ("eb_player_context", almanac_dir / f"eb_player_context_{season}_league{league_id}.md"),
        ("eb_preseason_hype", eb_out_dir / f"eb_preseason_hype_{season}_league{league_id}.md"),
        ("eb_player_leaders", almanac_dir / f"eb_player_leaders_{season}_league{league_id}.md"),
        ("eb_player_spotlights", almanac_dir / f"eb_player_spotlights_{season}_league{league_id}.md"),
        ("eb_series_spotlights", almanac_dir / f"eb_series_spotlights_{season}_league{league_id}.md"),
        ("eb_all_star", almanac_dir / f"eb_all_star_{season}_league{league_id}.md"),
    ]

    schedule_fragment_path = eb_out_dir / f"eb_schedule_context_{season}_league{league_id}.md"
    schedule_fragment = _read_optional_fragment(schedule_fragment_path)

    body_parts: list[str] = []
    if flashback:
        body_parts.append(flashback)

    body_parts.append(playoff_section(champions))

    body_parts.append("## EB Schedule Context")
    if schedule_fragment:
        body_parts.append(schedule_fragment.rstrip())
    else:
        body_parts.append(
            "[WARN] Schedule context fragment not found; run z_abl_eb_schedule_context_any.py first."
        )
    body_parts.append("")

    for label, path in components:
        content = read_md(path, label)
        if content:
            body_parts.append(content)

    header = f"# ABL {season} Regular Season – EB Pack (League {league_id})"
    generated_at = datetime.now().strftime("Generated on: %Y-%m-%d %H:%M:%S (local time)")
    sections = [header, generated_at, "", "---", ""]
    sections.append("\n\n---\n\n".join(body_parts).strip())

    full_text = "\n".join([s for s in sections if s != ""])
    full_text = normalize_eb_text(full_text).rstrip() + "\n"

    eb_out_dir.mkdir(parents=True, exist_ok=True)
    out_path = eb_out_dir / f"eb_regular_season_pack_{season}_league{league_id}.md"
    out_path.write_text(full_text, encoding="utf-8")
    legacy_path = almanac_dir / f"eb_regular_season_pack_{season}_league{league_id}.md"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text(full_text, encoding="utf-8")

    log(f"[OK] Wrote EB regular-season pack to {out_path} (legacy copy at {legacy_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
