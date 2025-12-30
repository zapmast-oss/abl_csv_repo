#!/usr/bin/env python
"""Extract per-team lines from text_out reports and write a markdown report."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


def _norm(txt: str) -> str:
    return re.sub(r"\s+", " ", str(txt or "").strip().lower())


def _safe_int(val: str) -> Optional[int]:
    try:
        num = int(str(val).strip())
        return num
    except Exception:
        return None


def find_team_info(base: Path, team_abbr: str, team_name_override: Optional[str]) -> dict:
    """Resolve team name/city/nickname/team_id from star-schema dim tables."""
    candidates = [
        base / "csv" / "out" / "star_schema" / "dim_team_park.csv",
        base / "csv" / "out" / "star_schema" / "dim_team.csv",
        base / "csv" / "out" / "star_schema" / "dim_team_lookup.csv",
        base / "csv" / "out" / "star_schema" / "dim_teams.csv",
    ]
    abbr = team_abbr.strip().upper()
    name_norm = _norm(team_name_override) if team_name_override else ""

    for path in candidates:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row_abbr = (row.get("team_abbr") or row.get("abbr") or "").strip().upper()
                    row_name = (row.get("team_name") or row.get("name") or row.get("team") or "").strip()
                    row_city = (row.get("city") or "").strip()
                    row_nick = (row.get("nickname") or "").strip()
                    row_full = f"{row_city} {row_nick}".strip()
                    matches = False
                    if abbr and row_abbr == abbr:
                        matches = True
                    elif name_norm and _norm(row_name) == name_norm:
                        matches = True
                    elif name_norm and row_full and _norm(row_full) == name_norm:
                        matches = True
                    if not matches:
                        continue

                    team_id = _safe_int(row.get("team_id") or row.get("id") or "")
                    return {
                        "team_abbr": row_abbr or abbr,
                        "team_name": team_name_override or row_name or row_full or "",
                        "city": row_city,
                        "nickname": row_nick,
                        "team_id": team_id,
                        "source": str(path),
                    }
        except Exception:
            continue

    return {
        "team_abbr": abbr,
        "team_name": team_name_override or "",
        "city": "",
        "nickname": "",
        "team_id": None,
        "source": "",
    }


def iter_report_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".txt", ".md", ".json"}:
            continue
        yield path


def _token_regex(token: str) -> Optional[re.Pattern]:
    tok = token.strip()
    if not tok:
        return None
    if tok.isdigit():
        return re.compile(rf"(?<!\d){re.escape(tok)}(?!\d)")
    if re.match(r"^[A-Za-z]+$", tok):
        return re.compile(rf"\b{re.escape(tok)}\b", re.IGNORECASE)
    return re.compile(re.escape(tok), re.IGNORECASE)


def build_matchers(tokens: List[str]) -> List[Tuple[str, re.Pattern]]:
    patterns: List[Tuple[str, re.Pattern]] = []
    seen = set()
    for tok in tokens:
        norm = tok.strip().lower()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        pat = _token_regex(tok)
        if pat is None:
            continue
        patterns.append((tok, pat))
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


def render_markdown(
    base: Path,
    text_out: Path,
    info: dict,
    tokens: List[str],
    matchers: List[Tuple[str, re.Pattern]],
) -> str:
    lines: List[str] = []
    team_abbr = info.get("team_abbr", "")
    team_name = info.get("team_name", "")
    team_id = info.get("team_id")

    lines.append(f"# Team Matches: {team_name or team_abbr} ({team_abbr})")
    lines.append("")
    lines.append("## Team Info")
    lines.append(f"- Team abbr: {team_abbr or 'N/A'}")
    lines.append(f"- Team name: {team_name or 'N/A'}")
    lines.append(f"- City: {info.get('city') or 'N/A'}")
    lines.append(f"- Nickname: {info.get('nickname') or 'N/A'}")
    lines.append(f"- Team id: {team_id if team_id is not None else 'N/A'}")
    if info.get("source"):
        lines.append(f"- Team info source: {info.get('source')}")
    lines.append("")

    lines.append("## Search Tokens")
    for tok, _ in matchers:
        lines.append(f"- {tok}")
    lines.append("")

    files = sorted(iter_report_files(text_out))
    for path in files:
        rel = path.relative_to(base)
        hits = extract_matches(path, matchers)
        lines.append(f"## {rel}")
        if hits:
            for hit in hits:
                lines.append(f"- {hit}")
        else:
            lines.append("- No match")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract per-team lines from text_out reports.")
    parser.add_argument("--team", required=True, help="Team abbr (e.g., CHI)")
    parser.add_argument("--team-name", default=None, help="Optional team name (e.g., Chicago Fire)")
    parser.add_argument("--team-id", default=None, help="Optional team id override")
    parser.add_argument("--out", default=None, help="Output markdown file path")
    parser.add_argument("--base", default=None, help="Repo root (defaults to script location)")
    args = parser.parse_args()

    base = Path(args.base).resolve() if args.base else Path(__file__).resolve().parents[1]
    text_out = base / "csv" / "out" / "text_out"
    if not text_out.exists():
        print(f"ERROR: text_out not found at {text_out}")
        return 1

    team_abbr = args.team.strip().upper()
    info = find_team_info(base, team_abbr, args.team_name)
    if args.team_id is not None:
        info["team_id"] = _safe_int(args.team_id)

    tokens: List[str] = []
    if team_abbr:
        tokens.append(team_abbr)
    if info.get("team_name"):
        tokens.append(info["team_name"])
    if info.get("city"):
        tokens.append(info["city"])
    if info.get("nickname"):
        tokens.append(info["nickname"])
    if info.get("team_id") is not None:
        tokens.append(str(info["team_id"]))

    # Ensure explicit requested tokens are included
    tokens.extend(["Chicago", "Fire", "CHI", "Chi"])

    matchers = build_matchers(tokens)
    if not matchers:
        print("ERROR: no team matchers built")
        return 1

    out_path = Path(args.out) if args.out else text_out / f"team_matches_{team_abbr}.md"
    md = render_markdown(base, text_out, info, tokens, matchers)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")

    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
