"""Generate weekly ABL Core 12 summaries (JSON + Markdown)."""

from __future__ import annotations

import argparse
import json
import re
import sys
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

SECTION_FALLBACKS = {
    "around": "- No major movement at the top of the table this week.",
    "one_run": "- No teams clearly separating in one-run results this week.",
    "bullpen": "- No bullpens currently flagged at Critical or High levels.",
    "sos": "- Schedule strength balanced across the league this week.",
    "rookies": "- No rookies met the reporting threshold this week.",
    "player_week": "- No consensus Player of the Week emerged.",
    "managers": "- No notable managerial pattern shifts this week.",
    "matchups": "- No featured matchups flagged this week.",
}


def warn(msg: str) -> None:
    print(f"WARNING: {msg}")


def normalize_text(s: str) -> str:
    """
    Normalize common encoding artifacts and trim whitespace.
    """
    if not isinstance(s, str):
        return s
    replacements = {
        "\u00e2\u0080\u0094": "\u2014",  # utf-8 em dash bytes mis-decoded
        "\u00e2\u0080\u0093": "\u2013",  # utf-8 en dash bytes mis-decoded
        "\u00e2\u20ac\u2014": "\u2014",  # common mojibake for em dash
        "\u00e2\u20ac\u201c": "\u2013",  # common mojibake for en dash
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


def allowed_line(line: str, teams: set[str]) -> bool:
    low = line.lower()
    if any(tok in low for tok in FORBIDDEN_TOKENS):
        return False
    if has_stat_token(line) and (contains_team_name(line, teams) or looks_like_player(line)):
        return True
    # Allow manager-style lines that include both a team and a name
    if contains_team_name(line, teams) and looks_like_player(line):
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
            hot = [r for r in parsed if r.get("diff") is not None and r["diff"] > 0][:5]
            cold = [r for r in reversed(parsed) if r.get("diff") is not None and r["diff"] < 0][:5]
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
        tag = ""
        if re.search(r"clutch|hot|elite", clean, flags=re.I):
            tag = "Clutch"
        elif re.search(r"cold|ice|struggle|meltdown", clean, flags=re.I):
            tag = "Cold"
        rows.append({"team": team_part, "record": record, "tag": tag})
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


def load_player_of_week(text: str) -> Dict[str, str]:
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("===")]
    for ln in lines:
        if re.search(r"player of the week", ln, flags=re.I):
            continue
        return {"blurb": ln.strip()}
    return {}


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


def render_forum_post(core12: dict) -> str:
    year = core12.get("year")
    week = core12.get("week")
    lines: List[str] = [f"# Action Baseball League - Week {week} {year} Core 12 Report", ""]

    def format_team_record(entry: Dict[str, str]) -> str:
        team = entry.get("team", "")
        w, l = entry.get("w"), entry.get("l")
        if w and l:
            return f"{team} ({w}-{l})"
        return team

    standings = core12.get("standings") or []
    hot = core12.get("hot_teams") or []
    cold = core12.get("cold_teams") or []
    team_set: set[str] = set()
    for e in standings:
        team = e.get("team", "")
        if team:
            team_set.add(team)
            team_set.add(normalize_team_name(team))

    top_table = [
        format_team_record(e).replace(e.get("team", ""), normalize_team_name(e.get("team", "")))
        for e in standings
        if e.get("team")
        and re.search(r"\d+-\d+", format_team_record(e))
        and contains_team_name(format_team_record(e), team_set)
    ][:3]
    top_hot = []
    for e in hot[:3]:
        team = normalize_team_name(e.get("team", ""))
        diff = e.get("diff")
        last10 = e.get("last10")
        if team:
            label = f"{team}"
            if last10:
                label += f" ({last10})"
            elif diff is not None:
                label += f" ({diff} in last 10)"
            if contains_team_name(label, team_set) and has_stat_token(label):
                top_hot.append(label)
    top_cold = []
    for e in cold[:3]:
        team = normalize_team_name(e.get("team", ""))
        diff = e.get("diff")
        last10 = e.get("last10")
        if team:
            label = f"{team}"
            if last10:
                label += f" ({last10})"
            elif diff is not None:
                label += f" ({diff} in last 10)"
            if contains_team_name(label, team_set) and has_stat_token(label):
                top_cold.append(label)

    around_bullets: List[str] = []
    if top_table:
        bullet = "- Top of the table: " + ", ".join(top_table)
        if allowed_line(bullet, team_set):
            around_bullets.append(bullet)
    if top_hot:
        bullet = "- Hottest clubs (last 10): " + ", ".join(top_hot)
        if allowed_line(bullet, team_set):
            around_bullets.append(bullet)
    if top_cold:
        bullet = "- Cold spell: " + ", ".join(top_cold)
        if allowed_line(bullet, team_set):
            around_bullets.append(bullet)
    lines.append("## Around the League")
    if not around_bullets:
        around_bullets = [SECTION_FALLBACKS["around"]]
    lines.extend(around_bullets)

    one_run = core12.get("one_run_records") or []
    clutch, cold_run = [], []
    for r in one_run:
        team = normalize_team_name(r.get("team", ""))
        record = r.get("record", "")
        if "generated" in record.lower():
            continue
        tag = (r.get("tag") or "").lower()
        if tag == "clutch":
            clutch.append({"team": team, "record": record})
        elif tag == "cold":
            cold_run.append({"team": team, "record": record})
    clutch_sorted = clutch[:3]
    cold_sorted = cold_run[:3]

    bullpen = core12.get("bullpen_stress") or []
    critical, high = [], []
    for b in bullpen:
        entry = b.get("entry", "") or b.get("team", "")
        if not entry:
            continue
        lower = entry.lower()
        team_token = normalize_team_name(entry.split()[0])
        if any(tok in lower for tok in ["meaning", "stress", "last-14"]):
            continue
        if "critical" in lower and contains_team_name(team_token, team_set):
            critical.append(team_token)
        elif "high" in lower and contains_team_name(team_token, team_set):
            high.append(team_token)

    close_lines: List[str] = []
    if clutch_sorted or cold_sorted:
        one_run_lines: List[str] = []
        if clutch_sorted:
            bullet = "- Clutch: " + ", ".join(
                [f"{c['team']} ({c['record']})" for c in clutch_sorted if contains_team_name(c["team"], team_set)]
            )
            if allowed_line(bullet, team_set):
                one_run_lines.append(bullet)
        if cold_sorted:
            bullet = "- Cold: " + ", ".join(
                [f"{c['team']} ({c['record']})" for c in cold_sorted if contains_team_name(c["team"], team_set)]
            )
            if allowed_line(bullet, team_set):
                one_run_lines.append(bullet)
        if not one_run_lines:
            one_run_lines = [SECTION_FALLBACKS["one_run"]]
        close_lines.append("### One-Run Games")
        close_lines.extend(one_run_lines)
    else:
        close_lines.append("### One-Run Games")
        close_lines.append(SECTION_FALLBACKS["one_run"])
    if critical or high:
        bullpen_lines: List[str] = []
        if critical:
            bullet = "- Critical workload: " + ", ".join(critical)
            if allowed_line(bullet, team_set):
                bullpen_lines.append(bullet)
        if high:
            bullet = "- High alert: " + ", ".join(high)
            if allowed_line(bullet, team_set):
                bullpen_lines.append(bullet)
        if not bullpen_lines:
            bullpen_lines = [SECTION_FALLBACKS["bullpen"]]
        close_lines.append("### Bullpen Stress")
        close_lines.extend(bullpen_lines)
    else:
        close_lines.append("### Bullpen Stress")
        close_lines.append(SECTION_FALLBACKS["bullpen"])
    lines += ["", "## Close Games & Bullpens"]
    lines.extend(close_lines)

    sos = core12.get("strength_of_schedule_last14") or []
    filtered_sos = [
        s
        for s in sos
        if s.get("team")
        and contains_team_name(normalize_team_name(s.get("team", "")), team_set)
        and re.search(r"\d+-\d+", s.get("record", "") or "")
        and not any(tok in (s.get("note", "").lower()) for tok in ["neutral", "gauntlet (->", "soft (->"])
    ]
    gauntlet = sorted([s for s in filtered_sos if s.get("sos") is not None], key=lambda x: -x["sos"])[:3]
    soft = sorted([s for s in filtered_sos if s.get("sos") is not None], key=lambda x: x["sos"])[:3]
    sos_section: List[str] = []
    if gauntlet:
        bullet = "- Gauntlet: " + ", ".join(
            [f"{normalize_team_name(g['team'])} ({g.get('record','').strip() or g.get('note','').strip()})" for g in gauntlet]
        )
        if allowed_line(bullet, team_set):
            sos_section.append(bullet)
    if soft:
        bullet = "- Soft: " + ", ".join(
            [f"{normalize_team_name(s['team'])} ({s.get('record','').strip() or s.get('note','').strip()})" for s in soft]
        )
        if allowed_line(bullet, team_set):
            sos_section.append(bullet)
    lines += ["", "## Strength of Schedule"]
    if not sos_section:
        sos_section = [SECTION_FALLBACKS["sos"]]
    lines.extend(sos_section)

    rookies = core12.get("rookie_watch") or []

    def format_rookie(r: Dict[str, str]) -> str:
        name = r.get("player") or ""
        team = normalize_team_name(r.get("team") or "")
        rating = r.get("rating") or ""
        stat = r.get("stat") or ""
        pieces = [name]
        if team:
            pieces.append(f"({team})")
        desc_parts = []
        if rating:
            desc_parts.append(rating)
        if stat:
            desc_parts.append(stat)
        if desc_parts:
            pieces.append(" - " + ", ".join(desc_parts))
        return "".join(pieces).strip()

    rookie_lines = []
    for r in rookies:
        line = format_rookie(r)
        if looks_like_player(line) and has_stat_token(line) and not any(tok in line.lower() for tok in FORBIDDEN_TOKENS):
            rookie_lines.append(line)
        if len(rookie_lines) >= 4:
            break
    lines += ["", "## Rookie Watch"]
    if not rookie_lines:
        rookie_lines = [SECTION_FALLBACKS["rookies"]]
    for rl in rookie_lines:
        bullet = rl if rl.startswith("-") else f"- {rl}"
        if bullet in SECTION_FALLBACKS.values() or allowed_line(bullet, team_set):
            lines.append(bullet)

    pow_entry = core12.get("player_of_the_week") or {}
    pow_blurb = (pow_entry.get("blurb") or "").strip()
    lines += ["", "## Player of the Week"]
    pow_lines: List[str] = []
    if pow_blurb and "week miner" not in pow_blurb.lower() and allowed_line(f"- {pow_blurb}", team_set):
        pow_lines.append(f"- {pow_blurb}")
    if not pow_lines:
        pow_lines = [SECTION_FALLBACKS["player_week"]]
    lines.extend(pow_lines)

    managers = core12.get("manager_tendencies") or []
    mgr_lines = []
    for m in managers[:7]:
        name = m.get("manager") or m.get("col1") or ""
        team = normalize_team_name(m.get("team") or m.get("col2") or "")
        rating = ""
        if name:
            team_clean = team
            if "(" in name and ")" in name and not team_clean:
                # try to split "Name (TEAM)"
                m_match = re.search(r"\(([^)]+)\)", name)
                if m_match:
                    team_clean = m_match.group(1)
                    name = name.split("(")[0].strip()
            summary = f"{team_clean}: {name}" if team_clean else name
            mgr_lines.append(summary)
    valid_mgrs = []
    for ml in mgr_lines:
        if contains_team_name(ml, team_set) and not any(tok in ml.lower() for tok in FORBIDDEN_TOKENS):
            valid_mgrs.append(f"- {ml}")
    lines += ["", "## Manager's Corner"]
    if not valid_mgrs:
        valid_mgrs = [SECTION_FALLBACKS["managers"]]
    lines.extend(valid_mgrs)

    featured = core12.get("featured_matchups") or []
    sunday = core12.get("sunday_matchups") or []
    feat_lines = [
        f"{normalize_team_name(m.get('away',''))} at {normalize_team_name(m.get('home',''))}".strip()
        for m in featured
        if m.get("away") and m.get("home")
    ]
    sun_lines = [
        f"{normalize_team_name(m.get('away',''))} at {normalize_team_name(m.get('home',''))}".strip()
        for m in sunday
        if m.get("away") and m.get("home")
    ]
    matchup_lines: List[str] = []
    if feat_lines:
        bullet = "- Featured: " + "; ".join(feat_lines)
        if allowed_line(bullet, team_set):
            matchup_lines.append(bullet)
    if sun_lines:
        bullet = "- Sunday set: " + "; ".join(sun_lines)
        if allowed_line(bullet, team_set):
            matchup_lines.append(bullet)
    lines += ["", "## On Deck - Featured & Sunday Matchups"]
    if not matchup_lines:
        matchup_lines = [SECTION_FALLBACKS["matchups"]]
    lines.extend(matchup_lines)

    filtered_lines: List[str] = []
    for line in lines:
        if line.startswith("-") and line not in SECTION_FALLBACKS.values():
            if allowed_line(line, team_set):
                filtered_lines.append(line)
        else:
            filtered_lines.append(line)

    md = "\n".join(normalize_text(line) for line in filtered_lines if line is not None).strip()
    return md + "\n"


def render_video_outline(core12: dict) -> str:
    year = core12.get("year")
    week = core12.get("week")
    lines: List[str] = [f"# It's Monday - ABL Week {week}, {year}", ""]

    standings = core12.get("standings") or []
    hot = core12.get("hot_teams") or []
    cold = core12.get("cold_teams") or []
    team_set: set[str] = set()
    for e in standings:
        team = e.get("team", "")
        if team:
            team_set.add(team)
            team_set.add(normalize_team_name(team))

    top_table = [
        normalize_team_name(e.get("team", ""))
        for e in standings
        if e.get("team")
        and contains_team_name(normalize_team_name(e.get("team", "")), team_set)
        and re.search(r"\d+-\d+", str(e.values()))
    ][:3]
    top_hot = [
        f"{normalize_team_name(e.get('team',''))} ({e.get('last10') or e.get('diff','')})"
        for e in hot[:3]
        if e.get("team") and contains_team_name(normalize_team_name(e.get("team", "")), team_set)
    ]
    top_cold = [
        f"{normalize_team_name(e.get('team',''))} ({e.get('last10') or e.get('diff','')})"
        for e in cold[:3]
        if e.get("team") and contains_team_name(normalize_team_name(e.get("team", "")), team_set)
    ]

    lines += ["## Open", "- Quick vibe; standings and headlines."]
    lines += ["", "## Standings & Momentum"]
    stand_bullets: List[str] = []
    if top_table:
        stand_bullets.append("- Top of table: " + ", ".join(top_table))
    if top_hot:
        stand_bullets.append("- Hot: " + ", ".join(top_hot))
    if top_cold:
        stand_bullets.append("- Cold: " + ", ".join(top_cold))
    if not stand_bullets:
        stand_bullets = [SECTION_FALLBACKS["around"]]
    lines.extend(stand_bullets)

    one_run = core12.get("one_run_records") or []
    clutch = []
    cold_run = []
    for r in one_run:
        team = normalize_team_name(r.get("team", ""))
        record = r.get("record", "")
        tag = (r.get("tag") or "").lower()
        if not contains_team_name(team, team_set):
            continue
        if tag == "clutch":
            clutch.append(f"{team} {record}")
        elif tag == "cold":
            cold_run.append(f"{team} {record}")
    bullpen = core12.get("bullpen_stress") or []
    stress_notes = []
    for b in bullpen:
        entry = b.get("entry", "") or b.get("team", "")
        if entry and re.search(r"critical|high", entry, flags=re.I) and contains_team_name(entry, team_set):
            stress_notes.append(entry.split()[0])

    close_lines: List[str] = []
    if clutch or cold_run:
        if clutch:
            bullet = "- Clutch one-run teams: " + ", ".join(clutch[:3])
            if allowed_line(bullet, team_set):
                close_lines.append(bullet)
        if cold_run:
            bullet = "- Cold in one-run: " + ", ".join(cold_run[:3])
            if allowed_line(bullet, team_set):
                close_lines.append(bullet)
    if not close_lines:
        close_lines = [SECTION_FALLBACKS["one_run"]]
    stress_added = False
    if stress_notes:
        bullet = "- Bullpen stress: " + ", ".join(stress_notes[:6])
        if allowed_line(bullet, team_set):
            close_lines.append(bullet)
            stress_added = True
    if not stress_added and SECTION_FALLBACKS["bullpen"] not in close_lines:
        close_lines.append(SECTION_FALLBACKS["bullpen"])
    lines += ["", "## Close Games & Bullpens"]
    lines.extend(close_lines)

    sos = core12.get("strength_of_schedule_last14") or []
    filtered_sos = [
        s
        for s in sos
        if s.get("team")
        and contains_team_name(normalize_team_name(s.get("team", "")), team_set)
        and re.search(r"\d+-\d+", s.get("record", "") or "")
        and not any(tok in (s.get("note", "").lower()) for tok in ["neutral", "gauntlet (->", "soft (->"])
    ]
    gauntlet = sorted([s for s in filtered_sos if s.get("sos") is not None], key=lambda x: -x["sos"])[:3]
    soft = sorted([s for s in filtered_sos if s.get("sos") is not None], key=lambda x: x["sos"])[:3]
    lines += ["", "## Strength of Schedule"]
    sos_lines = []
    if gauntlet:
        sos_lines.append("- Gauntlet: " + ", ".join([normalize_team_name(g["team"]) for g in gauntlet]))
    if soft:
        sos_lines.append("- Soft: " + ", ".join([normalize_team_name(s["team"]) for s in soft]))
    if not sos_lines:
        sos_lines = [SECTION_FALLBACKS["sos"]]
    lines.extend(sos_lines)

    rookies = core12.get("rookie_watch") or []
    section_lines: List[str] = []
    rookie_bullets = []
    for r in rookies:
        player = r.get("player", "")
        team = normalize_team_name(r.get("team", ""))
        stat = r.get("stat", "")
        if looks_like_player(player) and has_stat_token(stat or "") and team:
            text = f"{player} ({team}) {stat}".strip()
            rookie_bullets.append(text)
        if len(rookie_bullets) >= 4:
            break
    if rookie_bullets:
        bullet = "- Rookies: " + ", ".join(rookie_bullets)
        if allowed_line(bullet, team_set):
            section_lines.append(bullet)
    pow_entry = core12.get("player_of_the_week") or {}
    pow_blurb = (pow_entry.get("blurb") or "").strip()
    if pow_blurb and "week miner" not in pow_blurb.lower() and allowed_line(f"- {pow_blurb}", team_set):
        section_lines.append("- POW: " + pow_blurb)
    if not section_lines:
        section_lines = [SECTION_FALLBACKS["rookies"], SECTION_FALLBACKS["player_week"]]
    lines += ["", "## Rookie Watch & Player of the Week"]
    lines.extend(section_lines)

    featured = core12.get("featured_matchups") or []
    sunday = core12.get("sunday_matchups") or []
    feat_lines = [
        f"{normalize_team_name(m.get('away',''))} at {normalize_team_name(m.get('home',''))}".strip()
        for m in featured
        if m.get("away") and m.get("home")
    ]
    sun_lines = [
        f"{normalize_team_name(m.get('away',''))} at {normalize_team_name(m.get('home',''))}".strip()
        for m in sunday
        if m.get("away") and m.get("home")
    ]
    matchup_lines = []
    if feat_lines:
        bullet = "- Featured: " + "; ".join(feat_lines)
        if allowed_line(bullet, team_set):
            matchup_lines.append(bullet)
    if sun_lines:
        bullet = "- Sunday set: " + "; ".join(sun_lines)
        if allowed_line(bullet, team_set):
            matchup_lines.append(bullet)
    if not matchup_lines:
        matchup_lines = [SECTION_FALLBACKS["matchups"]]
    lines += ["", "## On Deck"]
    lines.extend(matchup_lines)

    filtered_lines: List[str] = []
    for line in lines:
        if line.startswith("-") and line not in SECTION_FALLBACKS.values():
            if allowed_line(line, team_set):
                filtered_lines.append(line)
        else:
            filtered_lines.append(line)

    md = "\n".join(normalize_text(line) for line in filtered_lines if line is not None).strip()
    return md + "\n"


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate ABL Core 12 weekly summary.")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    args = parser.parse_args(argv)

    year = args.year
    week = args.week

    # Paths
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

    # Show notes
    show_text = read_text(show_notes_path)
    if show_text:
        standings, hot, cold = load_show_notes(show_text)
        core12["standings"] = standings
        core12["hot_teams"] = hot
        core12["cold_teams"] = cold

    # Other reports
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

    # Output paths
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / f"core12_{year}_w{week:02d}.json"
    forum_path = OUT_DIR / f"forum_abl_{year}_w{week:02d}.md"
    video_path = OUT_DIR / f"video_outline_abl_{year}_w{week:02d}.md"

    json_path.write_text(json.dumps(core12, indent=2), encoding="utf-8")
    forum_path.write_text(render_forum_post(core12), encoding="utf-8")
    video_path.write_text(render_video_outline(core12), encoding="utf-8")

    print(f"Core12 JSON written to {json_path}")
    print(f"Forum draft written to {forum_path}")
    print(f"Video outline written to {video_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
