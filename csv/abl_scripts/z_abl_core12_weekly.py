"""Generate weekly ABL Core 12 summaries (JSON + Markdown)."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
CSV_DIR = SCRIPT_DIR.parent
OUT_DIR = CSV_DIR / "out"
TEXT_OUT_DIR = OUT_DIR / "text_out"

FORBIDDEN_TOKENS = [
    "generated on",
    "team (w-l)",
    "rating",
    "idx",
    "meaning:",
    "legend",
    "neutral (->",
    "gauntlet (->",
    "soft (->",
    "last-14",
    "stress index",
    "saved",
    "spotlights",
    "great for",
    "abl rookie",
    "watch)",
]

STRESS_FORBIDDEN = ["last-14", "last-14/7-day", "meaning", "stress index"]
POW_FORBIDDEN = ["abl week miner", "generated", "report"]

TEAM_DISPLAY_ABBR = {
    "Los Angeles": "LA",
    "Los": "LA",
    "LA": "LA",
    "Las Vegas": "LV",
    "Las": "LV",
    "LV": "LV",
    "St. Louis": "STL",
    "St.": "STL",
    "St": "STL",
    "STL": "STL",
    "San Francisco": "SF",
    "SF": "SF",
    "Tampa Bay": "TB",
    "TB": "TB",
    "New York": "NY",
    "NY": "NY",
}


def warn(msg: str) -> None:
    print(f"WARNING: {msg}")


def find_csv_root(start: Path) -> Path:
    current = start
    while True:
        if any(current.rglob("teams*.csv")):
            return current
        if current.parent == current:
            break
        current = current.parent
    raise RuntimeError("Missing teams CSV in csv/")


def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return s
    replacements = {
        "\u00e2\u0080\u0094": "\u2014",
        "\u00e2\u0080\u0093": "\u2013",
        "\u00e2\u20ac\u2014": "\u2014",
        "\u00e2\u20ac\u201c": "\u2013",
        "\u0394": "\u0394",
    }
    for bad, good in replacements.items():
        s = s.replace(bad, good)
    return s.strip()


def normalize_team_name(team: str) -> str:
    if not team:
        return team
    t = team.strip()
    if t in {"St.", "St"}:
        return "St. Louis"
    if t.startswith("St.") and "Louis" not in t:
        return "St. Louis"
    return team


def display_team(team: str) -> str:
    if not team:
        return team
    t = team.strip()
    return TEAM_DISPLAY_ABBR.get(t, t)


def contains_team_name(s: str, teams: set[str]) -> bool:
    return any(team and team.lower() in s.lower() for team in teams)


def has_stat_token(s: str) -> bool:
    return bool(
        re.search(r"\d+-\d+", s)
        or re.search(r"\d\.\d{2,3}", s)
        or re.search(r"\bOPS\b", s, flags=re.I)
        or re.search(r"\bWAR\b", s, flags=re.I)
    )


def looks_like_player(s: str) -> bool:
    return bool(re.search(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", s))


def looks_like_team_row(row: dict) -> bool:
    return (
        isinstance(row, dict)
        and "team" in row
        and isinstance(row["team"], str)
        and row["team"].strip() != ""
        and not any(tok.lower() in row["team"].lower() for tok in ["team", "generated", "abl", "neutral", "soft", "gauntlet", "meaning"])
    )


def looks_like_player_row(row: dict) -> bool:
    return (
        isinstance(row, dict)
        and "player" in row
        and isinstance(row["player"], str)
        and len(row["player"].split()) >= 2
        and "generated" not in row["player"].lower()
        and not any(tok in row["player"].lower() for tok in POW_FORBIDDEN)
    )


def is_real_player_name(name: str) -> bool:
    if not name or not isinstance(name, str):
        return False
    name_clean = name.strip()
    if "rookie" in name_clean.lower() or "player" in name_clean.lower():
        return False
    return bool(re.match(r"^[A-Za-z][A-Za-z.'-]* [A-Za-z][A-Za-z.'-]*$", name_clean))


def is_valid_rookie_row(row: dict) -> bool:
    if not looks_like_player_row(row):
        return False
    name = row.get("player", "")
    if not is_real_player_name(name):
        return False
    rating = (row.get("rating") or "").strip()
    stat = (row.get("stat") or "").strip()
    has_stat_num = bool(re.search(r"\d", stat))
    return bool(rating) or has_stat_num


def looks_like_manager_row(row: dict) -> bool:
    return (
        isinstance(row, dict)
        and "manager" in row
        and "(" in row["manager"]
        and ")" in row["manager"]
        and "rating" not in row["manager"].lower()
    )


def allowed_line(line: str, teams: set[str]) -> bool:
    low = line.lower()
    if any(tok in low for tok in FORBIDDEN_TOKENS):
        return False
    if contains_team_name(line, teams) or looks_like_player(line):
        return True
    return False


def read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        warn(f"Missing file: {path}")
    except Exception as exc:  # pragma: no cover - defensive
        warn(f"Could not read {path}: {exc}")
    return None


def parse_table_whitespace(lines: List[str], min_cols: int = 3) -> List[List[str]]:
    rows: List[List[str]] = []
    for line in lines:
        clean = line.rstrip()
        if not clean:
            if rows:
                break
            continue
        parts = clean.split()
        if len(parts) >= min_cols:
            rows.append(parts)
    return rows


def parse_pipe_table(text: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for line in text.splitlines():
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) < 2:
            continue
        rows.append({f"col{i+1}": part for i, part in enumerate(parts)})
    return rows


def load_show_notes(text: str) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    standings: List[Dict[str, str]] = []
    hot: List[Dict[str, str]] = []
    cold: List[Dict[str, str]] = []

    sections = re.split(r"^===+\s*", text, flags=re.MULTILINE)
    for block in sections:
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        header = lines[0].lower()
        table = parse_table_whitespace(lines[1:], min_cols=4)
        if not table:
            continue
        if "standings" in header:
            for row in table:
                entry = {"team": row[0]}
                if len(row) >= 3:
                    entry["w"] = row[1]
                    entry["l"] = row[2]
                if len(row) >= 4:
                    entry["pct"] = row[3]
                standings.append(entry)
        elif "last 10" in header:
            parsed: List[Dict[str, str]] = []
            for row in table:
                team = row[0]
                diff = None
                last10 = None
                for part in row[1:]:
                    if re.match(r"[+-]?\d+", part):
                        diff = int(part)
                        break
                for part in row[1:]:
                    if re.match(r"\d+-\d+", part):
                        last10 = part
                        break
                parsed.append({"team": team, "diff": diff, "last10": last10})
            parsed.sort(key=lambda r: (r["diff"] is None, -(r["diff"] or 0)))
            hot = parsed[:5]
            cold = list(reversed(parsed))[:5]
    return standings, hot, cold


def load_simple_list(text: str, min_cols: int = 3) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for line in text.splitlines():
        clean = line.strip()
        if not clean or clean.startswith("==="):
            continue
        parts = clean.split()
        if len(parts) < min_cols:
            continue
        rows.append({"entry": clean})
    return rows


def load_one_run(text: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for line in text.splitlines():
        clean = line.strip()
        if not clean:
            continue
        if re.search(r"team", clean, flags=re.I) and re.search(r"1R|1-run|one-run", clean, flags=re.I):
            continue
        m = re.match(r"([A-Za-z].*?)(\d+-\d+)", clean)
        if not m:
            continue
        team_part = m.group(1).strip()
        record = m.group(2).strip()
        rows.append({"team": team_part, "record": record})
    return rows


def load_strength_of_schedule(text: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for line in text.splitlines():
        clean = line.strip()
        if not clean or clean.startswith("="):
            continue
        parts = clean.split()
        if not parts:
            continue
        team = parts[0]
        sos_match = re.search(r"([0-1]\.\d{3})", clean)
        record_match = re.search(r"(\d+-\d+)", clean)
        sos_val = float(sos_match.group(1)) if sos_match else None
        record_val = record_match.group(1) if record_match else ""
        note = clean[len(team) :].strip()
        rows.append({"team": team, "note": note, "sos": sos_val, "record": record_val})
    return rows


def load_rookie_watch(hitters: str, pitchers: str) -> List[Dict[str, str]]:
    def parse_block(text: str, role: str) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        for line in text.splitlines():
            clean = line.strip()
            if not clean or clean.startswith("==="):
                continue
            lower = clean.lower()
            skip_prefixes = [
                "abl rookie",
                "generated on",
                "spotlights rookie",
                "great for",
                "player ",
                "----",
                "threshold",
                "key:",
                "definitions",
                "war pace",
                "ace track",
                "helps flag",
                "era uses",
            ]
            if any(lower.startswith(pref) for pref in skip_prefixes):
                continue
            tokens = clean.split()
            name = " ".join(tokens[:2]) if len(tokens) >= 2 else (tokens[0] if tokens else "")
            team = tokens[2] if len(tokens) >= 3 else ""
            rating = next((t for t in tokens if re.match(r"(Meteoric|Impact|Steady|Rising)", t, flags=re.I)), "")
            stat = next((t for t in tokens if re.match(r"\d+\.\d{3}", t)), "")
            rows.append(
                {
                    "player": name,
                    "team": team,
                    "rating": rating,
                    "stat": stat,
                    "raw": clean,
                    "role": role,
                }
            )
        return rows

    return parse_block(hitters or "", "hitter") + parse_block(pitchers or "", "pitcher")


def load_player_of_week(text: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for ln in text.splitlines():
        clean = ln.strip()
        if not clean or clean.startswith("==="):
            continue
        tokens = clean.split()
        if len(tokens) >= 2:
            rows.append({"player": " ".join(tokens[:2]), "team": tokens[2] if len(tokens) >= 3 else "", "line": clean})
    return rows


def load_manager_tendencies(text: str) -> List[Dict[str, str]]:
    rows = parse_pipe_table(text)
    result: List[Dict[str, str]] = []
    if rows:
        for row in rows:
            entry = {
                "manager": row.get("col1", ""),
                "team": row.get("col2", ""),
                "rating": row.get("col3", ""),
                "hook": row.get("col4", ""),
                "platoon": row.get("col5", ""),
            }
            result.append(entry)
        return result

    for line in text.splitlines():
        clean = line.strip()
        if not clean or "|" in clean or clean.startswith("="):
            continue
        parts = clean.split()
        if len(parts) < 2:
            continue
        manager = " ".join(parts[:2])
        team = parts[2] if len(parts) >= 3 else ""
        result.append({"manager": manager, "team": team})
    return result


def load_matchups(text: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for line in text.splitlines():
        clean = line.strip()
        if not clean or clean.startswith("==="):
            continue
        m = re.match(r"(.+?)\s*@\s*(.+)", clean)
        if not m:
            m = re.match(r"(.+?)\s+vs\.?\s+(.+)", clean, flags=re.I)
        if m:
            away = m.group(1).strip()
            home = m.group(2).strip()
            if away and home and away != "@" and home != "@":
                rows.append({"away": away, "home": home})
    return rows


def load_player_lookup(csv_dir: Path) -> Dict[str, Dict[str, Optional[float]]]:
    candidates = [csv_dir / "ootp_csv" / "players.csv", csv_dir / "players.csv"]
    path = next((p for p in candidates if p.exists()), None)
    lookup: Dict[str, Dict[str, Optional[float]]] = {}
    if path:
        try:
            with path.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    first = (row.get("first_name") or "").strip()
                    last = (row.get("last_name") or "").strip()
                    if not first or not last:
                        continue
                    name_key = f"{first} {last}".lower()
                    age = row.get("age")
                    exp = row.get("experience")
                    try:
                        age_val = float(age) if age not in (None, "") else None
                    except ValueError:
                        age_val = None
                    try:
                        exp_val = float(exp) if exp not in (None, "") else None
                    except ValueError:
                        exp_val = None
                    current = lookup.get(name_key, {"age": None, "experience": None, "rook": False})
                    new_age = age_val if current["age"] is None else (max(current["age"], age_val) if age_val is not None else current["age"])
                    new_exp = exp_val if current["experience"] is None else (max(current["experience"], exp_val) if exp_val is not None else current["experience"])
                    lookup[name_key] = {"age": new_age, "experience": new_exp, "rook": current.get("rook", False)}
        except Exception:
            pass
    misc_path = csv_dir / "abl_statistics" / "abl_statistics_player_statistics_-_sortable_stats_player_misc_info.csv"
    if misc_path.exists():
        try:
            rows = [ln for ln in misc_path.read_text(encoding="utf-8").splitlines() if ln and not ln.startswith("#")]
            if rows:
                reader = csv.DictReader(rows)
                for row in reader:
                    name_field = (row.get("Name") or "").strip()
                    if not name_field:
                        continue
                    rook_flag = (row.get("ROOK") or "").strip().lower()
                    is_rook = rook_flag in {"yes", "true", "1", "rookie"}
                    if not is_rook:
                        continue
                    name_key = name_field.lower()
                    current = lookup.get(name_key, {"age": None, "experience": None, "rook": False})
                    current["rook"] = True
                    lookup[name_key] = current
        except Exception:
            pass
    return lookup


def is_rookie(name: str, lookup: Dict[str, Dict[str, Optional[float]]]) -> bool:
    if not name:
        return False
    info = lookup.get(name.lower())
    if not info:
        return False
    if info.get("rook"):
        return True
    age = info.get("age")
    exp = info.get("experience")
    if exp is not None:
        return exp <= 1
    if age is not None:
        return age <= 27
    return False


def ensure_weekly_report(year: int, week: int) -> Path:
    path = OUT_DIR / f"league_report_abl_{year}_w{week:02d}.md"
    if path.exists():
        return path
    script = CSV_DIR / "abl_scripts" / "z_abl_weekly_league_report.py"
    cmd = [sys.executable, str(script), "--year", str(year), "--week", str(week)]
    subprocess.run(cmd, check=True, cwd=CSV_DIR)
    return path


def parse_weekly_report(report_path: Path) -> Dict[str, object]:
    data: Dict[str, object] = {
        "divisions": [],
        "teams": {},
        "team_war": {},
        "top_war_players": [],
        "hr_leaders": [],
        "rbi_leaders": [],
    }
    lines = report_path.read_text(encoding="utf-8").splitlines()
    team_stats: Dict[str, dict] = {}
    divisions = []
    pattern = re.compile(r"^(\S+)\s+(.*?)\s+(\d+)-(\d+)\s+([0-9.]+)\s+([-0-9.]+)\s+([+\-]?\d+|NA)$")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if "Division" in line and " - " in line:
            div_name = line
            i += 2
            teams = []
            while i < len(lines) and lines[i].strip():
                m = pattern.match(lines[i].strip())
                if m:
                    abbr, name, w, l, pct, gb, rd_txt = m.groups()
                    rd_val = None if rd_txt == "NA" else int(rd_txt)
                    entry = {"abbr": abbr, "name": name, "w": int(w), "l": int(l), "pct": float(pct), "gb": float(gb), "rd": rd_val}
                    teams.append(entry)
                    team_stats[abbr] = entry
                i += 1
            if teams:
                divisions.append({"name": div_name, "teams": teams})
        else:
            i += 1
    data["divisions"] = divisions
    data["teams"] = team_stats

    def parse_list_after(anchor: str) -> List[str]:
        entries: List[str] = []
        for idx, ln in enumerate(lines):
            if ln.strip().startswith(anchor):
                j = idx + 1
                while j < len(lines) and lines[j].strip().startswith("- "):
                    entries.append(lines[j].strip()[2:])
                    j += 1
                break
        return entries

    top_players = parse_list_after("Top 10 players by WAR:")
    for item in top_players:
        m = re.match(r"(.+?) \((.+?)\) - WAR ([0-9.]+)", item)
        if m:
            data["top_war_players"].append({"name": m.group(1), "abbr": m.group(2), "war": m.group(3)})

    def parse_leader(anchor: str, suffix: str) -> List[dict]:
        results = []
        lst = parse_list_after(anchor)
        for ent in lst:
            m = re.match(r"(.+?) \((.+?)\) - ([0-9.]+)", ent)
            if m:
                results.append({"name": m.group(1), "abbr": m.group(2), "val": m.group(3)})
                continue
            m = re.match(r"(.+?) \((.+?)\) - (\d+) " + suffix, ent)
            if m:
                results.append({"name": m.group(1), "abbr": m.group(2), "val": m.group(3)})
        return results

    data["hr_leaders"] = parse_leader("Home Runs (Top 5):", "HR")
    data["rbi_leaders"] = parse_leader("RBI (Top 5):", "RBI")

    for anchor in ["Team WAR (Top 5):", "Team WAR (Bottom 5):"]:
        entries = parse_list_after(anchor)
        for ent in entries:
            m = re.match(r"(\S+) (.+?) - Team WAR ([0-9.]+)", ent)
            if m:
                data["team_war"][m.group(1)] = m.group(3)
    return data


def build_core_recap(parsed: Dict[str, object]) -> List[str]:
    lines: List[str] = []
    lines.append("## Core Recap (3/2/3)")
    lines.append("")
    lines.append("### Division Races (3)")
    div_entries: List[Tuple[float, str]] = []
    for div in parsed.get("divisions", []):
        teams = div.get("teams", [])
        if len(teams) < 2:
            continue
        leader, chaser = teams[0], teams[1]
        gb = chaser.get("gb", 0.0)
        conf_abbr = "NBC" if "National Baseball Conference" in div["name"] else ("ABC" if "American Baseball Conference" in div["name"] else "")
        div_abbr = "E" if "Eastern" in div["name"] else ("C" if "Central" in div["name"] else ("W" if "Western" in div["name"] else ""))
        rd1 = leader.get("rd")
        rd2 = chaser.get("rd")
        rd1_txt = f"{rd1:+d}" if isinstance(rd1, int) else "NA"
        rd2_txt = f"{rd2:+d}" if isinstance(rd2, int) else "NA"
        div_entries.append((gb, f"- {leader['abbr']} leads {chaser['abbr']} in {conf_abbr} {div_abbr} by {gb:.1f} GB (Leader RD {rd1_txt}, Chaser RD {rd2_txt})"))
    div_entries.sort(key=lambda x: x[0])
    for _, txt in div_entries[:3]:
        lines.append(txt)
    lines.append("")
    lines.append("### Team Trends (2)")
    teams: Dict[str, dict] = parsed.get("teams", {})
    team_war: Dict[str, str] = parsed.get("team_war", {})

    def sort_key(item):
        abbr, info = item
        pct = info.get("pct", 0.0)
        rd = info.get("rd")
        rd_val = rd if rd is not None else -9999
        war_val = float(team_war.get(abbr)) if abbr in team_war else -9999.0
        return (pct, rd_val, war_val)

    if teams:
        best_abbr, best_info = max(teams.items(), key=sort_key)
        worst_abbr, worst_info = min(teams.items(), key=sort_key)

        def fmt_team(abbr: str, info: dict) -> str:
            rd_txt = f"{info['rd']:+d}" if info.get("rd") is not None else "NA"
            war_txt = team_war.get(abbr, "NA")
            return f"- {abbr} {info['name']}: {info['w']}-{info['l']} ({info['pct']:.3f}) | RD {rd_txt} | Team WAR {war_txt}"

        lines.append(fmt_team(best_abbr, best_info))
        lines.append(fmt_team(worst_abbr, worst_info))
    lines.append("")
    lines.append("### Player Spotlights (3)")
    bullets: List[str] = []
    used: set[str] = set()
    top_war = parsed.get("top_war_players", [])
    if top_war:
        p = top_war[0]
        bullets.append(f"- WAR: {p['name']} ({p['abbr']}) - {p['war']}")
        used.add(p["name"])

    def add_unique(entries: List[dict], label: str) -> None:
        for ent in entries:
            if ent["name"] not in used:
                bullets.append(f"- {label}: {ent['name']} ({ent['abbr']}) - {ent['val']}")
                used.add(ent["name"])
                return

    add_unique(parsed.get("rbi_leaders", []), "RBI")
    add_unique(parsed.get("hr_leaders", []), "HR")
    while len(bullets) < 3:
        bullets.append("- HR: N/A")
    lines.extend(bullets[:3])
    lines.append("")
    return lines


def render_forum_post(core12: dict) -> str:
    year = core12.get("year")
    week = core12.get("week")
    lines: List[str] = [f"# Action Baseball League - Week {week} {year} Core 12 Report", ""]
    player_lookup = load_player_lookup(CSV_DIR)

    def format_team_record(entry: Dict[str, str]) -> Tuple[str, float]:
        team = normalize_team_name(entry.get("team", ""))
        team_disp = display_team(team)
        w = entry.get("w")
        l = entry.get("l")
        pct = 0.0
        if w and l and w.isdigit() and l.isdigit():
            w_i, l_i = int(w), int(l)
            pct = w_i / max(w_i + l_i, 1)
            rec_str = f"{team_disp} ({w}-{l})"
        else:
            rec_str = team_disp
        return rec_str, pct

    standings_raw = core12.get("standings") or []
    standings = [s for s in standings_raw if looks_like_team_row(s)]
    team_set: set[str] = {normalize_team_name(e.get("team", "")) for e in standings if e.get("team")}
    ranked_standings = []
    for e in standings:
        rec, pct = format_team_record(e)
        ranked_standings.append((rec, pct))
    ranked_standings.sort(key=lambda x: -x[1])
    top_table = [rec for rec, _ in ranked_standings[:3]]
    bottom_table = [rec for rec, _ in list(reversed(ranked_standings))[:3]]

    lines.append("## Around the League")
    if top_table:
        lines.append("- Top of the table: " + ", ".join(top_table))
    if bottom_table:
        lines.append("- Bottom of the table: " + ", ".join(bottom_table))

    one_run_raw = core12.get("one_run_records") or []
    one_run = [r for r in one_run_raw if looks_like_team_row(r) and re.match(r"\d+-\d+", r.get("record", "")) and "generated" not in r.get("record", "").lower()]
    one_run_stats: List[Tuple[str, float, str]] = []
    for r in one_run:
        team = normalize_team_name(r.get("team", ""))
        team_disp = display_team(team)
        record = r.get("record", "")
        m = re.match(r"(\d+)-(\d+)", record)
        if not m:
            continue
        w_i, l_i = int(m.group(1)), int(m.group(2))
        total = w_i + l_i
        pct = w_i / total if total else 0.0
        one_run_stats.append((team, pct, f"{team_disp} ({record})"))
    one_run_stats.sort(key=lambda x: -x[1])
    best_one_run = [entry[2] for entry in one_run_stats[:3]]
    worst_one_run = [entry[2] for entry in list(reversed(one_run_stats))[:3]]

    bullpen_raw = core12.get("bullpen_stress") or []
    bullpen = []
    for b in bullpen_raw:
        entry = b.get("entry", "") or b.get("team", "")
        token = entry.split()[0] if entry else ""
        if token and not any(tok in entry.lower() for tok in STRESS_FORBIDDEN) and looks_like_team_row({"team": token}):
            bullpen.append(display_team(normalize_team_name(token)))
    bullpen_top = bullpen[:5]

    close_lines: List[str] = ["", "## Close Games & Bullpens"]
    if best_one_run:
        close_lines.append("### One-Run Games")
        close_lines.append("- Best in one-run games: " + ", ".join(best_one_run))
    if worst_one_run:
        if "### One-Run Games" not in close_lines:
            close_lines.append("### One-Run Games")
        close_lines.append("- Struggling late: " + ", ".join(worst_one_run))
    if bullpen_top:
        close_lines.append("### Bullpen Stress")
        close_lines.append("- Heaviest workloads: " + ", ".join(bullpen_top))
    lines.extend(close_lines)

    sos_raw = core12.get("strength_of_schedule_last14") or []
    sos = [
        s
        for s in sos_raw
        if looks_like_team_row(s)
        and s.get("sos") is not None
        and isinstance(s.get("sos"), (int, float))
        and re.match(r"\d+-\d+", s.get("record", "") or "")
        and not any(t in s.get("team", "").lower() for t in ["neutral", "soft", "gauntlet"])
    ]
    sos.sort(key=lambda x: -x["sos"])
    gauntlet = sos[:3]
    soft = list(reversed(sos))[:3]
    lines += ["", "## Strength of Schedule"]
    if gauntlet:
        lines.append(
            "- Toughest recent slate: "
            + ", ".join([f"{display_team(normalize_team_name(g['team']))} ({g.get('sos'):.3f} SOS, {g.get('record','')})" for g in gauntlet])
        )
    if soft:
        lines.append(
            "- Easiest recent slate: "
            + ", ".join([f"{display_team(normalize_team_name(s['team']))} ({s.get('sos'):.3f} SOS, {s.get('record','')})" for s in soft])
        )

    rookies_raw = core12.get("rookie_watch") or []
    rookies = [r for r in rookies_raw if is_valid_rookie_row(r)]
    if player_lookup:
        rookies = [r for r in rookies if is_rookie(r.get("player", ""), player_lookup)]

    def rookie_stat_val(r: Dict[str, str]) -> float:
        try:
            return float(r.get("stat") or 0)
        except Exception:
            return 0.0

    rookies_sorted = sorted(rookies, key=rookie_stat_val, reverse=True)[:4]
    if rookies_sorted:
        lines += ["", "## Rookie Watch"]
        for r in rookies_sorted:
            name = r.get("player", "")
            team = display_team(normalize_team_name(r.get("team", "")))
            rating = r.get("rating", "")
            stat = r.get("stat", "")
            parts = [name]
            if team:
                parts.append(f"({team})")
            desc = []
            if rating:
                desc.append(rating)
            if stat:
                desc.append(stat)
            line = " ".join(parts)
            if desc:
                line += " - " + ", ".join(desc)
            lines.append(f"- {line}")

    pow_rows = load_player_of_week(read_text(TEXT_OUT_DIR / "z_ABL_Week_Miner.txt") or "")
    pow_valid = [p for p in pow_rows if looks_like_player_row(p)]
    if pow_valid:
        top_pow = pow_valid[0]
        player = top_pow.get("player", "")
        team_raw = normalize_team_name(top_pow.get("team", ""))
        team = display_team(team_raw)
        stat_line = top_pow.get("line", "").replace(player, "").replace(team_raw, "").replace(team, "").strip()
        pieces = [player]
        if team:
            pieces.append(f"({team})")
        blurb = " ".join(pieces)
        if stat_line:
            blurb += f" - {stat_line}"
        lines += ["", "## Player of the Week", f"- {blurb}"]

    managers_raw = core12.get("manager_tendencies") or []
    managers = [m for m in managers_raw if looks_like_manager_row({"manager": m.get("manager") or m.get("col1", "")})]
    mgr_lines = []
    for m in managers:
        manager_field = m.get("manager") or m.get("col1") or ""
        rating = m.get("rating") or m.get("col3") or ""
        idx = m.get("hook") or m.get("col4") or ""
        name = manager_field.split("(")[0].strip()
        team_code_match = re.search(r"\(([^)]+)\)", manager_field)
        team_code = team_code_match.group(1) if team_code_match else ""
        team = display_team(normalize_team_name(team_code))
        parts = [f"{team}: {name}" if team else name]
        rating_bits = []
        if rating:
            rating_bits.append(rating)
        if idx:
            rating_bits.append(str(idx))
        if rating_bits:
            parts.append(" - " + " ".join(rating_bits))
        mgr_lines.append("".join(parts))
    if mgr_lines:
        lines += ["", "## Manager's Corner"]
        for mgr in mgr_lines:
            lines.append(f"- {mgr}")

    featured_raw = core12.get("featured_matchups") or []
    featured = [m for m in featured_raw if looks_like_team_row({"team": m.get("home", "")}) and looks_like_team_row({"team": m.get("away", "")})]
    sunday_raw = core12.get("sunday_matchups") or []
    sunday = [m for m in sunday_raw if looks_like_team_row({"team": m.get("home", "")}) and looks_like_team_row({"team": m.get("away", "")})]
    feat_lines = [
        f"{display_team(normalize_team_name(m.get('away','')))} at {display_team(normalize_team_name(m.get('home','')))}"
        for m in featured
        if m.get("away") and m.get("home")
    ][:3]
    sun_lines = [
        f"{display_team(normalize_team_name(m.get('away','')))} at {display_team(normalize_team_name(m.get('home','')))}"
        for m in sunday
        if m.get("away") and m.get("home")
    ][:3]
    if feat_lines or sun_lines:
        lines += ["", "## On Deck - Featured & Sunday Matchups"]
        if feat_lines:
            lines.append("- Featured: " + "; ".join(feat_lines))
        if sun_lines:
            lines.append("- Sunday set: " + "; ".join(sun_lines))

    md = "\n".join(normalize_text(line) for line in lines if line is not None).strip()
    return md + "\n"


def render_video_outline(core12: dict, recap_lines: List[str]) -> str:
    year = core12.get("year")
    week = core12.get("week")
    lines: List[str] = [f"# It's Monday - ABL Week {week}, {year}", ""]
    player_lookup = load_player_lookup(CSV_DIR)
    def short_team_label(team: str) -> str:
        t = normalize_team_name(team)
        if not t:
            return t
        return TEAM_DISPLAY_ABBR.get(t.strip(), t.strip()[:3].upper())

    standings_raw = core12.get("standings") or []
    standings = [s for s in standings_raw if looks_like_team_row(s)]
    team_set: set[str] = {normalize_team_name(e.get("team", "")) for e in standings if e.get("team")}
    ranked_standings = []
    for e in standings:
        team = short_team_label(e.get("team", ""))
        w = e.get("w")
        l = e.get("l")
        pct = 0.0
        if w and l and w.isdigit() and l.isdigit():
            w_i, l_i = int(w), int(l)
            pct = w_i / max(w_i + l_i, 1)
        ranked_standings.append((team, pct))
    ranked_standings.sort(key=lambda x: -x[1])
    top_table = [r[0] for r in ranked_standings[:3]]
    bottom_table = [r[0] for r in list(reversed(ranked_standings))[:3]]

    lines += ["## Open", "- Quick vibe; standings and headlines."]
    lines.extend(recap_lines)
    lines += ["", "## Standings & Momentum"]
    if top_table:
        lines.append("- Top: " + ", ".join(top_table))
    if bottom_table:
        lines.append("- Bottom: " + ", ".join(bottom_table))

    one_run_raw = core12.get("one_run_records") or []
    one_run = [r for r in one_run_raw if looks_like_team_row(r) and re.match(r"\d+-\d+", r.get("record", "")) and "generated" not in r.get("record", "").lower()]
    one_run_stats: List[Tuple[str, float, str]] = []
    for r in one_run:
        team = normalize_team_name(r.get("team", ""))
        record = r.get("record", "")
        m = re.match(r"(\d+)-(\d+)", record)
        if not m:
            continue
        w_i, l_i = int(m.group(1)), int(m.group(2))
        total = w_i + l_i
        pct = w_i / total if total else 0.0
        one_run_stats.append((team, pct, f"{team} {record}"))
    one_run_stats.sort(key=lambda x: -x[1])
    best_one_run = [entry[2] for entry in one_run_stats[:3]]
    worst_one_run = [entry[2] for entry in list(reversed(one_run_stats))[:3]]

    bullpen_raw = core12.get("bullpen_stress") or []
    bullpen = []
    for b in bullpen_raw:
        entry = b.get("entry", "") or b.get("team", "")
        token = entry.split()[0] if entry else ""
        if token and not any(tok in entry.lower() for tok in STRESS_FORBIDDEN) and looks_like_team_row({"team": token}):
            bullpen.append(normalize_team_name(token))

    lines += ["", "## Close Games & Bullpens"]
    if best_one_run:
        lines.append("- Best 1-run: " + ", ".join(best_one_run))
    if worst_one_run:
        lines.append("- Cold 1-run: " + ", ".join(worst_one_run))
    if bullpen:
        lines.append("- Bullpen stress: " + ", ".join(bullpen[:5]))

    sos_raw = core12.get("strength_of_schedule_last14") or []
    sos = [
        s
        for s in sos_raw
        if looks_like_team_row(s)
        and s.get("sos") is not None
        and isinstance(s.get("sos"), (int, float))
        and re.match(r"\d+-\d+", s.get("record", "") or "")
        and not any(t in s.get("team", "").lower() for t in ["neutral", "soft", "gauntlet"])
    ]
    sos.sort(key=lambda x: -x["sos"])
    gauntlet = sos[:3]
    soft = list(reversed(sos))[:3]
    lines += ["", "## Strength of Schedule"]
    if gauntlet:
        lines.append("- Tough: " + ", ".join([normalize_team_name(g["team"]) for g in gauntlet]))
    if soft:
        lines.append("- Soft: " + ", ".join([normalize_team_name(s["team"]) for s in soft]))

    rookies_raw = core12.get("rookie_watch") or []
    rookies = [r for r in rookies_raw if is_valid_rookie_row(r)]
    if player_lookup:
        rookies = [r for r in rookies if is_rookie(r.get("player", ""), player_lookup)]

    def rookie_val(r: Dict[str, str]) -> float:
        try:
            return float(r.get("stat") or 0)
        except Exception:
            return 0.0

    rookies_sorted = sorted(rookies, key=rookie_val, reverse=True)[:4]
    lines += ["", "## Rookie Watch & Player of the Week"]
    if rookies_sorted:
        lines.append("- Rookies: " + ", ".join([r.get("player", "") for r in rookies_sorted]))
    pow_rows = load_player_of_week(read_text(TEXT_OUT_DIR / "z_ABL_Week_Miner.txt") or "")
    pow_valid = [p for p in pow_rows if looks_like_player_row(p)]
    if pow_valid:
        top_pow = pow_valid[0]
        player = top_pow.get("player", "")
        team = normalize_team_name(top_pow.get("team", ""))
        stat_line = top_pow.get("line", "").replace(player, "").replace(team, "").strip()
        pieces = [player]
        if team:
            pieces.append(f"({team})")
        blurb = " ".join(pieces)
        if stat_line:
            blurb += f" - {stat_line}"
        lines.append("- POW: " + blurb)

    managers_raw = core12.get("manager_tendencies") or []
    managers = [m for m in managers_raw if looks_like_manager_row({"manager": m.get("manager") or m.get("col1", "")})]
    mgrs = []
    for m in managers:
        manager_field = m.get("manager") or m.get("col1") or ""
        rating = m.get("rating") or m.get("col3") or ""
        idx = m.get("hook") or m.get("col4") or ""
        name = manager_field.split("(")[0].strip()
        team_code_match = re.search(r"\(([^)]+)\)", manager_field)
        team_code = team_code_match.group(1) if team_code_match else ""
        team = normalize_team_name(team_code)
        parts = [f"{team} {name}".strip()]
        rating_bits = []
        if rating:
            rating_bits.append(rating)
        if idx:
            rating_bits.append(str(idx))
        if rating_bits:
            parts.append(" - " + " ".join(rating_bits))
        mgrs.append("".join(parts))
    if mgrs:
        lines += ["", "## Manager's Corner"]
        for mgr in mgrs:
            lines.append(f"- {mgr}")

    featured_raw = core12.get("featured_matchups") or []
    featured = [m for m in featured_raw if looks_like_team_row({"team": m.get("home", "")}) and looks_like_team_row({"team": m.get("away", "")})]
    sunday_raw = core12.get("sunday_matchups") or []
    sunday = [m for m in sunday_raw if looks_like_team_row({"team": m.get("home", "")}) and looks_like_team_row({"team": m.get("away", "")})]
    feat_lines = [
        f"{normalize_team_name(m.get('away',''))} at {normalize_team_name(m.get('home',''))}"
        for m in featured
        if m.get("away") and m.get("home")
    ][:3]
    sun_lines = [
        f"{normalize_team_name(m.get('away',''))} at {normalize_team_name(m.get('home',''))}"
        for m in sunday
        if m.get("away") and m.get("home")
    ][:3]
    if feat_lines or sun_lines:
        lines += ["", "## On Deck"]
        if feat_lines:
            lines.append("- Featured: " + "; ".join(feat_lines))
        if sun_lines:
            lines.append("- Sunday: " + "; ".join(sun_lines))

    md = "\n".join(normalize_text(line) for line in lines if line is not None).strip()
    return md + "\n"


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate ABL Core 12 weekly summary.")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    args = parser.parse_args(argv)

    year = args.year
    week = args.week

    global CSV_DIR, OUT_DIR, TEXT_OUT_DIR
    csv_root = find_csv_root(SCRIPT_DIR)
    CSV_DIR = csv_root
    OUT_DIR = CSV_DIR / "out"
    TEXT_OUT_DIR = OUT_DIR / "text_out"

    show_notes_path = TEXT_OUT_DIR / "ABL_Show_Notes.txt"
    eb_pack_path = TEXT_OUT_DIR / "eb_data_pack_1981_monday.txt"
    momentum_path = TEXT_OUT_DIR / "z_ABL_Momentum_Windows.txt"
    blowout_path = TEXT_OUT_DIR / "z_ABL_Blowout_Resilience.txt"
    bullpen_path = TEXT_OUT_DIR / "z_ABL_Bullpen_Stress_Index.txt"
    one_run_path = TEXT_OUT_DIR / "z_ABL_One_Run_Record.txt"
    sos_path = TEXT_OUT_DIR / "z_ABL_SOS_Last14.txt"
    rookie_hit_path = TEXT_OUT_DIR / "z_ABL_Rookie_Watch_Hitters.txt"
    rookie_pitch_path = TEXT_OUT_DIR / "z_ABL_Rookie_Watch_Pitchers.txt"
    week_miner_path = TEXT_OUT_DIR / "z_ABL_Week_Miner.txt"
    mgr_tend_path = TEXT_OUT_DIR / "z_ABL_Manager_Tendencies.txt"
    featured_path = TEXT_OUT_DIR / "z_ABL_Featured_Matchups_1981.txt"
    sunday_path = TEXT_OUT_DIR / "z_ABL_Sunday_Matchups.txt"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_OUT_DIR.mkdir(parents=True, exist_ok=True)

    weekly_report_path = ensure_weekly_report(year, week)
    weekly_parsed = parse_weekly_report(weekly_report_path)

    core12 = {
        "year": year,
        "week": week,
        "standings": [],
        "hot_teams": [],
        "cold_teams": [],
        "run_differential": [],
        "momentum_windows": [],
        "blowout_resilience": [],
        "bullpen_stress": [],
        "one_run_records": [],
        "strength_of_schedule_last14": [],
        "rookie_watch": [],
        "player_of_the_week": {},
        "manager_tendencies": [],
        "featured_matchups": [],
        "sunday_matchups": [],
    }

    show_text = read_text(show_notes_path)
    if show_text:
        standings, hot, cold = load_show_notes(show_text)
        core12["standings"] = standings
        core12["hot_teams"] = hot
        core12["cold_teams"] = cold

    momentum_text = read_text(momentum_path)
    if momentum_text:
        core12["momentum_windows"] = load_simple_list(momentum_text)

    blowout_text = read_text(blowout_path)
    if blowout_text:
        core12["blowout_resilience"] = load_simple_list(blowout_text)

    bullpen_text = read_text(bullpen_path)
    if bullpen_text:
        core12["bullpen_stress"] = load_simple_list(bullpen_text)

    one_run_text = read_text(one_run_path)
    if one_run_text:
        core12["one_run_records"] = load_one_run(one_run_text)

    sos_text = read_text(sos_path)
    if sos_text:
        core12["strength_of_schedule_last14"] = load_strength_of_schedule(sos_text)

    hit_text = read_text(rookie_hit_path)
    pitch_text = read_text(rookie_pitch_path)
    if hit_text or pitch_text:
        core12["rookie_watch"] = load_rookie_watch(hit_text or "", pitch_text or "")

    week_miner_text = read_text(week_miner_path)
    if week_miner_text:
        core12["player_of_the_week"] = load_player_of_week(week_miner_text)

    mgr_text = read_text(mgr_tend_path)
    if mgr_text:
        core12["manager_tendencies"] = load_manager_tendencies(mgr_text)

    feat_text = read_text(featured_path)
    if feat_text:
        core12["featured_matchups"] = load_matchups(feat_text)

    sunday_text = read_text(sunday_path)
    if sunday_text:
        core12["sunday_matchups"] = load_matchups(sunday_text)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / f"core12_{year}_w{week:02d}.json"
    forum_path = OUT_DIR / f"forum_abl_{year}_w{week:02d}.md"
    video_path = OUT_DIR / f"video_outline_abl_{year}_w{week:02d}.md"

    json_path.write_text(json.dumps(core12, indent=2), encoding="utf-8")
    forum_path.write_text(render_forum_post(core12), encoding="utf-8")
    recap_lines = build_core_recap(weekly_parsed)
    video_path.write_text(render_video_outline(core12, recap_lines), encoding="utf-8")

    print(f"Wrote: {video_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
