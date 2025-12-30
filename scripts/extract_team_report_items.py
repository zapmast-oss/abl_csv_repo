#!/usr/bin/env python
"""Extract per-team lines from text_out reports."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


def find_team_name(base: Path, team_abbr: str) -> Optional[str]:
    """Try to resolve team name from star-schema dim tables."""
    candidates = [
        base / "csv" / "out" / "star_schema" / "dim_team_park.csv",
        base / "csv" / "out" / "star_schema" / "dim_team.csv",
        base / "csv" / "out" / "star_schema" / "dim_team_lookup.csv",
        base / "csv" / "out" / "star_schema" / "dim_teams.csv",
    ]
    team_abbr = team_abbr.strip().upper()
    for path in candidates:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    abbr = (row.get("team_abbr") or row.get("abbr") or "").strip().upper()
                    if abbr != team_abbr:
                        continue
                    name = (row.get("team_name") or row.get("name") or "").strip()
                    if name:
                        return name
                    city = (row.get("city") or "").strip()
                    nick = (row.get("nickname") or "").strip()
                    if city or nick:
                        return f"{city} {nick}".strip()
        except Exception:
            continue
    return None


def iter_report_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".txt", ".md"}:
            continue
        yield path


def build_matchers(team_abbr: str, team_name: Optional[str]) -> List[Tuple[str, re.Pattern]]:
    patterns: List[Tuple[str, re.Pattern]] = []
    if team_abbr:
        abbr = team_abbr.strip().upper()
        # avoid matching inside other words (e.g., CHICAGO)
        abbr_pat = re.compile(rf"(?<![A-Z0-9]){re.escape(abbr)}(?![A-Z0-9])")
        patterns.append((abbr, abbr_pat))
    if team_name:
        name = team_name.strip()
        if name:
            name_pat = re.compile(re.escape(name), re.IGNORECASE)
            patterns.append((name, name_pat))
    return patterns


def extract_matches(path: Path, matchers: List[Tuple[str, re.Pattern]]) -> List[str]:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return []
    hits: List[str] = []
    for idx, line in enumerate(lines, start=1):
        for _, pat in matchers:
            if pat.search(line):
                hits.append(f"L{idx}: {line}")
                break
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract per-team lines from text_out reports.")
    parser.add_argument("--team", required=True, help="Team abbr (e.g., CHI)")
    parser.add_argument("--team-name", default=None, help="Optional team name (e.g., Chicago Fire)")
    parser.add_argument("--base", default=None, help="Repo root (defaults to script location)")
    args = parser.parse_args()

    base = Path(args.base).resolve() if args.base else Path(__file__).resolve().parents[1]
    text_out = base / "csv" / "out" / "text_out"
    if not text_out.exists():
        print(f"ERROR: text_out not found at {text_out}")
        return 1

    team_abbr = args.team.strip().upper()
    team_name = args.team_name or find_team_name(base, team_abbr)

    matchers = build_matchers(team_abbr, team_name)
    if not matchers:
        print("ERROR: no team matchers built")
        return 1

    print(f"Team: {team_name or ''} ({team_abbr})")
    print(f"Reports root: {text_out}")
    print("")

    files = sorted(iter_report_files(text_out))
    for path in files:
        rel = path.relative_to(base)
        hits = extract_matches(path, matchers)
        print(f"== {rel} ==")
        if hits:
            for line in hits:
                print(line)
        else:
            print("No match")
        print("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
