from __future__ import annotations

import argparse
import csv
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from abl_config import LEAGUE_ID, RAW_CSV_ROOT
from abl_team_helper import allowed_team_ids

CSV_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CSV_ROOT.parent
DEFAULT_DB = REPO_ROOT / "data_work" / "abl.db"
DEFAULT_CSV = CSV_ROOT / "out" / "csv_out" / "abl_managers_summary.csv"
DEFAULT_OUT = CSV_ROOT / "out" / "text_out" / "prep"
TEAMS_CSV = RAW_CSV_ROOT / "teams.csv"

UTC_FMT = "%Y-%m-%d %H:%M:%SZ"

def normalize_team_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in name.upper())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or None

def normalize_match_key(value: str) -> str:
    base = re.sub(r"\s*\([^)]*\)", "", value or "")
    return " ".join(base.replace("-", " ").lower().split())

def load_team_lookup() -> Dict[str, Dict[str, Optional[object]]]:
    lookup: Dict[str, Dict[str, Optional[object]]] = {}
    allowed_ids = set(allowed_team_ids())
    if not TEAMS_CSV.exists():
        return lookup
    with TEAMS_CSV.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                team_id = int(row.get("team_id", 0))
                league_id = int(row.get("league_id", 0))
            except ValueError:
                continue
            if team_id not in allowed_ids or league_id != LEAGUE_ID:
                continue
            city = (row.get("name") or "").strip()
            nickname = (row.get("nickname") or "").strip()
            abbr = (row.get("abbr") or "").strip()
            display = f"{city} {nickname}".strip() or city or nickname or (row.get("team_name") or "").strip()
            if not display:
                display = f"Team {team_id}"
            info = {"team_id": team_id, "display": display, "league_id": league_id}
            variants = {display, city, nickname, f"{abbr} {nickname}".strip()}
            for variant in variants:
                key = normalize_match_key(variant)
                if key:
                    lookup[key] = info
    return lookup

def load_csv_fallback(path: Path) -> Dict[str, str]:
    fallback: Dict[str, str] = {}
    if not path.exists():
        return fallback
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = (row.get("name") or "").strip()
            team = (row.get("team") or "").strip()
            if name and team:
                fallback[name.lower()] = team
    return fallback

def match_allowed_team(raw: Optional[str], lookup: Dict[str, Dict[str, Optional[object]]]) -> Optional[Dict[str, Optional[object]]]:
    if not raw:
        return None
    key = normalize_match_key(raw)
    if key in lookup:
        return lookup[key]
    for lookup_key, info in lookup.items():
        if lookup_key == key:
            return info
        if lookup_key.startswith(key) or key.startswith(lookup_key) or key in lookup_key:
            return info
    return None

def fetch_managers(db_path: Path) -> List[Dict[str, Optional[object]]]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT d.manager_id, d.name, d.current_team, d.last_refreshed_at,
                   v.total_titles, v.career_wins, v.career_losses, v.career_win_pct
            FROM dim_manager d
            LEFT JOIN v_manager_career v ON d.manager_id = v.manager_id
            ORDER BY v.total_titles DESC, v.career_win_pct DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]

def ten_second_take(titles: int, win_pct: float) -> str:
    if titles and win_pct >= 0.55:
        return "Championship-tested skipper with elite track record."
    if titles:
        return "Title winner who knows how to close tight races."
    if win_pct >= 0.55:
        return "High-win manager pushing toward that first crown."
    if win_pct >= 0.5:
        return "Steady hand keeping the club above water."
    return "Scrappy leader fighting to change the narrative."

def strength_cue(win_pct: float) -> str:
    if win_pct >= 0.55:
        return "Strength cue: high win% vs peers."
    return "Strength cue: scrappy, above water." if win_pct >= 0.5 else "Strength cue: resilience despite bumps."

def risk_cue(win_pct: float) -> str:
    if 0.48 <= win_pct <= 0.52:
        return "Risk cue: thin margin; sub-.500 vs elites."
    return "Risk cue: variance watch." if win_pct < 0.55 else "Risk cue: guard against complacency."

def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)

def parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None

def pick_better(existing: Dict[str, Optional[object]], candidate: Dict[str, Optional[object]]) -> Dict[str, Optional[object]]:
    existing_ts = parse_timestamp(existing.get("last_refreshed_at"))
    candidate_ts = parse_timestamp(candidate.get("last_refreshed_at"))
    if existing_ts and candidate_ts:
        if candidate_ts > existing_ts:
            return candidate
        if existing_ts > candidate_ts:
            return existing
    elif candidate_ts and not existing_ts:
        return candidate
    elif existing_ts and not candidate_ts:
        return existing
    existing_wins = existing.get("career_wins") or 0
    candidate_wins = candidate.get("career_wins") or 0
    if candidate_wins != existing_wins:
        return candidate if candidate_wins > existing_wins else existing
    existing_pct = existing.get("career_win_pct") or 0.0
    candidate_pct = candidate.get("career_win_pct") or 0.0
    if candidate_pct != existing_pct:
        return candidate if candidate_pct > existing_pct else existing
    return candidate

def build_top_report(rows: List[Dict[str, Optional[object]]], top_n: int) -> str:
    timestamp = datetime.now(timezone.utc).strftime(UTC_FMT)
    header = "League Top Managers"
    lines = [header, "=" * len(header), f"Generated: {timestamp}", ""]
    width_line = f"{'Rank':<4} {'Name':<28} {'Team':<20} {'W':>5} {'L':>5} {'Win%':>7} {'Titles':>7}"
    lines.append(width_line)
    lines.append("-" * len(width_line))
    for idx, row in enumerate(rows[:top_n], 1):
        name = row.get("name") or "n/a"
        team = row.get("current_team") or "n/a"
        wins = row.get("career_wins") or 0
        losses = row.get("career_losses") or 0
        win_pct = row.get("career_win_pct") or 0.0
        titles = row.get("total_titles") or 0
        lines.append(
            f"{idx:<4} {name:<28.28} {team:<20.20} {wins:>5} {losses:>5} {win_pct:>7.3f} {titles:>7}"
        )
    return "\n".join(lines)

def build_team_report(team: str, row: Dict[str, Optional[object]]) -> str:
    name = row.get("name") or "Unknown"
    titles = row.get("total_titles") or 0
    wins = row.get("career_wins") or 0
    losses = row.get("career_losses") or 0
    win_pct = row.get("career_win_pct") or 0.0
    refreshed = row.get("last_refreshed_at") or "n/a"
    timestamp = datetime.now(timezone.utc).strftime(UTC_FMT)
    take = ten_second_take(int(titles), float(win_pct))
    lines = [
        f"{team} - Manager Prep",
        f"Manager: {name} | Team: {team} | Titles: {titles}",
        f"Record: {wins}-{losses} ({win_pct:.3f})",
        f"Ten-second take: {take}",
        "Bullets:",
        f"  - Career snapshot: {wins}-{losses}, {win_pct:.3f} win%, Titles: {titles}, Last Refreshed: {refreshed}",
        f"  - {strength_cue(float(win_pct))}",
        f"  - {risk_cue(float(win_pct))}",
        f"Generated: {timestamp}",
    ]
    return "\n".join(lines)

def main() -> None:
    parser = argparse.ArgumentParser(description="Broadcast prep report for managers")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--topN", type=int, default=10)
    args = parser.parse_args()

    db_path = Path(args.db)
    csv_path = Path(args.csv)
    out_dir = Path(args.out)

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"Manager CSV not found: {csv_path}")

    fallback = load_csv_fallback(csv_path)
    team_lookup = load_team_lookup()
    if not team_lookup:
        raise SystemExit("No allowed teams available; check teams.csv")

    rows = fetch_managers(db_path)
    active: List[Dict[str, Optional[object]]] = []
    for row in rows:
        if not row.get("current_team"):
            fb = fallback.get((row.get("name") or "").lower())
            if fb:
                row["current_team"] = fb
        info = match_allowed_team(row.get("current_team"), team_lookup)
        if not info:
            continue
        row["current_team"] = info["display"]
        row["team_id"] = info["team_id"]
        row["league_id"] = info["league_id"]
        active.append(row)

    if not active:
        raise SystemExit("No active managers matched allowed teams.")

    league_ids = {row.get("league_id") for row in active if row.get("league_id" ) is not None}
    print(f"[check] Broadcast prep league_ids after filter: {league_ids}")

    top_report = build_top_report(active, max(1, args.topN))
    league_path = out_dir / "LEAGUE_top_managers.txt"
    atomic_write(league_path, top_report)

    team_best: Dict[str, Dict[str, Optional[object]]] = {}
    for row in active:
        code = normalize_team_name(row.get("current_team"))
        if not code:
            continue
        existing = team_best.get(code)
        if existing is None:
            team_best[code] = row
        else:
            team_best[code] = pick_better(existing, row)

    produced: set[Path] = set()
    team_files: List[Path] = []
    for code, row in sorted(team_best.items()):
        team_name = row.get("current_team") or "Unknown"
        content = build_team_report(team_name, row)
        dest = out_dir / f"{code}_manager_prep.txt"
        atomic_write(dest, content)
        produced.add(dest.resolve())
        team_files.append(dest)

    for existing in out_dir.glob("*_manager_prep.txt"):
        if existing.resolve() not in produced:
            try:
                existing.unlink()
            except OSError:
                pass

    print(f"Broadcast prep: wrote {league_path} and {len(team_files)} team sheets in {out_dir}")


if __name__ == "__main__":
    main()
