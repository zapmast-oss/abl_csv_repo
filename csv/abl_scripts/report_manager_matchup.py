from __future__ import annotations

import argparse
import csv
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from abl_team_helper import allowed_team_ids

CSV_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CSV_ROOT.parent
DEFAULT_DB = REPO_ROOT / "data_work" / "abl.db"
DEFAULT_CSV = CSV_ROOT / "out" / "csv_out" / "abl_managers_summary.csv"
DEFAULT_OUT_DIR = CSV_ROOT / "out" / "text_out" / "prep" / "matchups"
TEAMS_CSV = CSV_ROOT / "ootp_csv" / "teams.csv"
LEAGUE_ID = 200
UTC_FMT = "%Y-%m-%d %H:%M:%S"

def normalize_token(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in text.upper())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "UNK"

def normalize_match_key(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    base = re.sub(r"\s*\(.*\)$", "", value)
    base = base.replace("-", " ")
    clean = " ".join(base.lower().split())
    return clean or None

def load_csv_fallback(path: Path) -> Dict[str, str]:
    fallback: Dict[str, str] = {}
    if not path.exists():
        return fallback
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = (row.get("name") or "").strip().lower()
            team = (row.get("team") or "").strip()
            if name and team:
                fallback[name] = team
    return fallback

def build_allowed_lookup() -> Dict[str, Dict[str, Optional[object]]]:
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
            info = {"team_id": team_id, "league_id": league_id, "display": display}
            variants = {display, city, nickname, abbr, f"{abbr} {nickname}".strip()}
            for variant in variants:
                key = normalize_match_key(variant)
                if key:
                    lookup[key] = info
    return lookup

def match_allowed_team(raw: Optional[str], lookup: Dict[str, Dict[str, Optional[object]]]) -> Optional[Dict[str, Optional[object]]]:
    key = normalize_match_key(raw)
    if not key:
        return None
    if key in lookup:
        return lookup[key]
    for lookup_key, info in lookup.items():
        if key == lookup_key or key in lookup_key or lookup_key in key:
            return info
    return None

def fetch_managers(db_path: Path) -> List[Dict[str, Optional[object]]]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        query = (
            "SELECT v.name, COALESCE(v.current_team, d.current_team) AS current_team, "
            "v.career_wins, v.career_losses, v.career_win_pct, v.total_titles, d.last_refreshed_at "
            "FROM v_manager_career v LEFT JOIN dim_manager d ON d.manager_id = v.manager_id"
        )
        rows = conn.execute(query).fetchall()
    return [dict(row) for row in rows]

def score_team(team: str, target: str) -> int:
    team_lower = team.lower()
    if team_lower == target:
        return 0
    if team_lower.startswith(target):
        return 1
    if target in team_lower:
        return 2
    return 99

def filter_allowed_records(records: List[Dict[str, Optional[object]]], lookup: Dict[str, Dict[str, Optional[object]]]) -> List[Dict[str, Optional[object]]]:
    filtered: List[Dict[str, Optional[object]]] = []
    for row in records:
        info = match_allowed_team(row.get("current_team"), lookup)
        if not info:
            continue
        row["team_id"] = info["team_id"]
        row["league_id"] = info["league_id"]
        row["current_team"] = info["display"]
        filtered.append(row)
    return filtered

def resolve_team(records: List[Dict[str, Optional[object]]], query: str, lookup: Dict[str, Dict[str, Optional[object]]]) -> Dict[str, Optional[object]]:
    target_info = match_allowed_team(query, lookup)
    if target_info:
        for rec in records:
            if rec.get("team_id") == target_info["team_id"]:
                return rec
    target = query.strip().lower()
    candidates = [rec for rec in records if rec.get("current_team")]
    if not candidates:
        raise RuntimeError("No active teams available")
    scored = [(score_team(rec["current_team"], target), rec) for rec in candidates]
    scored = [item for item in scored if item[0] != 99]
    if not scored:
        raise RuntimeError(f"No manager found matching '{query}'")
    scored.sort(key=lambda item: (item[0], item[1]["current_team"]))
    best_score = scored[0][0]
    matches = [rec for score, rec in scored if score == best_score]
    names = {rec["current_team"] for rec in matches}
    if len(names) > 1:
        raise RuntimeError(f"Ambiguous team match for '{query}'. Candidates: {', '.join(sorted(names))}")
    return matches[0]

def ten_second_take(titles: int, win_pct: float) -> str:
    snippets: List[str] = []
    if titles >= 2:
        snippets.append("proven big-game skipper; ring equity")
    elif titles == 1:
        snippets.append("championship-tested; steady profile")
    if win_pct >= 0.560:
        snippets.append("high-efficiency operator")
    elif win_pct >= 0.520:
        snippets.append("above-water, consistent")
    elif win_pct >= 0.480:
        snippets.append("coin-flip profile; leverage matters")
    else:
        snippets.append("volatile outcomes; needs margins")
    return "; ".join(snippets)[:120]

def edge_win(home_pct: Optional[float], away_pct: Optional[float]) -> Tuple[str, str]:
    if home_pct is None or away_pct is None:
        return "EVEN", "+0.000"
    delta = home_pct - away_pct
    if abs(delta) < 1e-4:
        return "EVEN", "+0.000"
    return ("HOME" if delta > 0 else "AWAY"), f"{delta:+.3f}"

def edge_titles(home_titles: Optional[int], away_titles: Optional[int]) -> Tuple[str, str]:
    if home_titles is None or away_titles is None:
        return "EVEN", "+0"
    delta = (home_titles or 0) - (away_titles or 0)
    if delta == 0:
        return "EVEN", "+0"
    return ("HOME" if delta > 0 else "AWAY"), f"{delta:+d}"

def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)

def format_side(label: str, row: Optional[Dict[str, Optional[object]]]) -> List[str]:
    if row is None:
        return [
            f"{label}: [unknown]",
            "Manager: [unknown]   W-L: --   Win%: n/a  Titles: n/a",
            "",
        ]
    name = row.get("name") or "[unknown]"
    team = row.get("current_team") or "[unknown]"
    wins = row.get("career_wins")
    losses = row.get("career_losses")
    win_pct = row.get("career_win_pct")
    titles = row.get("total_titles")
    wins_disp = "--" if wins is None else str(wins)
    losses_disp = "--" if losses is None else str(losses)
    pct_disp = "n/a" if win_pct is None else f"{win_pct:.3f}"
    titles_disp = "--" if titles is None else str(titles)
    return [
        f"{label}: {team}",
        f"Manager: {name:<18} W-L: {wins_disp}-{losses_disp}  Win%: {pct_disp}  Titles: {titles_disp}",
        "",
    ]

def build_card(home_team: str, away_team: str, home_row: Optional[Dict[str, Optional[object]]], away_row: Optional[Dict[str, Optional[object]]]) -> str:
    timestamp = datetime.now(timezone.utc)
    header = f"ABL Matchup Card - {home_team} vs {away_team} (UTC: {timestamp.strftime('%Y-%m-%d %H:%M')})"
    lines: List[str] = [header, "=" * len(header)]
    lines.extend(format_side("HOME", home_row))
    lines.extend(format_side("AWAY", away_row))
    lines.append("-" * len(header))
    win_edge, win_delta = edge_win(
        home_row.get("career_win_pct") if home_row else None,
        away_row.get("career_win_pct") if away_row else None,
    )
    title_edge, title_delta = edge_titles(
        home_row.get("total_titles") if home_row else None,
        away_row.get("total_titles") if away_row else None,
    )
    lines.append(f"Edge (Win%):   {win_edge:<5}   (delta = {win_delta})")
    lines.append(f"Edge (Titles): {title_edge:<5}   (delta = {title_delta})")
    home_take = ten_second_take(int(home_row.get("total_titles") or 0), float(home_row.get("career_win_pct") or 0.0)) if home_row else "Unknown profile"
    away_take = ten_second_take(int(away_row.get("total_titles") or 0), float(away_row.get("career_win_pct") or 0.0)) if away_row else "Unknown profile"
    lines.append(f"Ten-second take (HOME): {home_take}")
    lines.append(f"Ten-second take (AWAY): {away_take}")
    return "\n".join(lines)

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate manager matchup card")
    parser.add_argument("--home", help="Home team name/code")
    parser.add_argument("--away", help="Away team name/code")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    db_path = Path(args.db)
    csv_path = Path(args.csv)
    out_dir = Path(args.out)

    if not db_path.exists():
        raise SystemExit(f"Database missing: {db_path}")
    if not csv_path.exists():
        raise SystemExit(f"Manager CSV missing: {csv_path}")

    fallback = load_csv_fallback(csv_path)
    records = fetch_managers(db_path)
    for row in records:
        if not row.get("current_team"):
            fb = fallback.get((row.get("name") or "").lower())
            if fb:
                row["current_team"] = fb

    lookup = build_allowed_lookup()
    if not lookup:
        raise SystemExit(f"Unable to load allowed teams from {TEAMS_CSV}")
    records = filter_allowed_records(records, lookup)
    if not records:
        raise SystemExit("No managers found for the 24 ABL clubs.")
    league_ids = {row.get("league_id") for row in records if row.get("league_id") is not None}
    team_ids = sorted({row.get("team_id") for row in records if row.get("team_id") is not None})
    print(f"[check] Manager matchup league_ids after filter: {league_ids}")
    print(f"[check] Manager matchup team_ids after filter: {team_ids}")

    if args.verify:
        home_query = "Miami Hurricanes"
        away_query = "Chicago Fire"
    else:
        if not args.home or not args.away:
            raise SystemExit("Both --home and --away are required")
        home_query = args.home
        away_query = args.away

    home_row = resolve_team(records, home_query, lookup)
    away_row = resolve_team(records, away_query, lookup)

    home_team = home_row.get("current_team") or home_query
    away_team = away_row.get("current_team") or away_query
    card = build_card(home_team, away_team, home_row, away_row)
    date_stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    filename = f"{date_stamp}_{normalize_token(home_team)}_vs_{normalize_token(away_team)}.txt"
    output_path = out_dir / filename
    atomic_write(output_path, card)

    if args.verify:
        print(f"Matchup card written: {output_path}")
        preview = "\n".join(card.splitlines()[:25])
        print(preview)
    else:
        print(f"Matchup card written: {output_path}")

if __name__ == "__main__":
    main()
