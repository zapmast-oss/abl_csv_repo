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


def warn(msg: str) -> None:
    print(f"WARNING: {msg}")


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
    )


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

    def format_team_record(entry: Dict[str, str]) -> Tuple[str, float]:
        team = normalize_team_name(entry.get("team", ""))
        w = entry.get("w")
        l = entry.get("l")
        pct = 0.0
        if w and l and w.isdigit() and l.isdigit():
            w_i, l_i = int(w), int(l)
            pct = w_i / max(w_i + l_i, 1)
            rec_str = f"{team} ({w}-{l})"
        else:
            rec_str = team
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
        record = r.get("record", "")
        m = re.match(r"(\d+)-(\d+)", record)
        if not m:
            continue
        w_i, l_i = int(m.group(1)), int(m.group(2))
        total = w_i + l_i
        pct = w_i / total if total else 0.0
        one_run_stats.append((team, pct, f"{team} ({record})"))
    one_run_stats.sort(key=lambda x: -x[1])
    best_one_run = [entry[2] for entry in one_run_stats[:3]]
    worst_one_run = [entry[2] for entry in list(reversed(one_run_stats))[:3]]

    bullpen_raw = core12.get("bullpen_stress") or []
    bullpen = [b for b in bullpen_raw if looks_like_team_row({"team": (b.get("team") or b.get("entry") or "").split()[0]})]
    bullpen_list = []
    for b in bullpen:
        entry = b.get("entry", "") or b.get("team", "")
        if entry:
            bullpen_list.append(normalize_team_name(entry.split()[0]))
    bullpen_top = bullpen_list[:5]

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
            + ", ".join([f"{normalize_team_name(g['team'])} ({g.get('sos'):.3f} SOS, {g.get('record','')})" for g in gauntlet])
        )
    if soft:
        lines.append(
            "- Easiest recent slate: "
            + ", ".join([f"{normalize_team_name(s['team'])} ({s.get('sos'):.3f} SOS, {s.get('record','')})" for s in soft])
        )

    rookies_raw = core12.get("rookie_watch") or []
    rookies = [r for r in rookies_raw if looks_like_player_row(r)]

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
            team = normalize_team_name(r.get("team", ""))
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

    pow_entry = core12.get("player_of_the_week") or {}
    pow_blurb = (pow_entry.get("blurb") or "").strip()
    if pow_blurb:
        lines += ["", "## Player of the Week", f"- {pow_blurb}"]

    managers_raw = core12.get("manager_tendencies") or []
    managers = [m for m in managers_raw if looks_like_manager_row({"manager": m.get("manager") or m.get("col1", "")})]
    mgr_lines = []
    for m in managers:
        name = m.get("manager") or m.get("col1") or ""
        team = normalize_team_name(m.get("team") or m.get("col2") or "")
        if name and team:
            mgr_lines.append(f"{team}: {name}")
    if mgr_lines:
        lines += ["", "## Manager's Corner"]
        lines.append("- Top tendencies: " + ", ".join(mgr_lines[:3]))

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
        lines += ["", "## On Deck - Featured & Sunday Matchups"]
        if feat_lines:
            lines.append("- Featured: " + "; ".join(feat_lines))
        if sun_lines:
            lines.append("- Sunday set: " + "; ".join(sun_lines))

    md = "\n".join(normalize_text(line) for line in lines if line is not None).strip()
    return md + "\n"


def render_video_outline(core12: dict) -> str:
    year = core12.get("year")
    week = core12.get("week")
    lines: List[str] = [f"# It's Monday - ABL Week {week}, {year}", ""]

    standings_raw = core12.get("standings") or []
    standings = [s for s in standings_raw if looks_like_team_row(s)]
    team_set: set[str] = {normalize_team_name(e.get("team", "")) for e in standings if e.get("team")}
    ranked_standings = []
    for e in standings:
        team = normalize_team_name(e.get("team", ""))
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
    bullpen = [b for b in bullpen_raw if looks_like_team_row({"team": (b.get("team") or b.get("entry") or "").split()[0]})]
    bullpen_list = []
    for b in bullpen:
        entry = b.get("entry", "") or b.get("team", "")
        if entry:
            bullpen_list.append(normalize_team_name(entry.split()[0]))

    lines += ["", "## Close Games & Bullpens"]
    if best_one_run:
        lines.append("- Best 1-run: " + ", ".join(best_one_run))
    if worst_one_run:
        lines.append("- Cold 1-run: " + ", ".join(worst_one_run))
    if bullpen_list:
        lines.append("- Bullpen stress: " + ", ".join(bullpen_list[:5]))

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
    rookies = [r for r in rookies_raw if looks_like_player_row(r)]

    def rookie_val(r: Dict[str, str]) -> float:
        try:
            return float(r.get("stat") or 0)
        except Exception:
            return 0.0

    rookies_sorted = sorted(rookies, key=rookie_val, reverse=True)[:4]
    lines += ["", "## Rookie Watch & Player of the Week"]
    if rookies_sorted:
        lines.append("- Rookies: " + ", ".join([r.get("player", "") for r in rookies_sorted]))
    pow_entry = core12.get("player_of_the_week") or {}
    pow_blurb = (pow_entry.get("blurb") or "").strip()
    if pow_blurb:
        lines.append("- POW: " + pow_blurb)

    managers_raw = core12.get("manager_tendencies") or []
    managers = [m for m in managers_raw if looks_like_manager_row({"manager": m.get("manager") or m.get("col1", "")})]
    mgrs = []
    for m in managers:
        name = m.get("manager") or m.get("col1") or ""
        team = normalize_team_name(m.get("team") or m.get("col2") or "")
        if name and team:
            mgrs.append(f"{team} {name}")
    if mgrs:
        lines += ["", "## Manager's Corner"]
        lines.append("- Tendencies: " + ", ".join(mgrs[:3]))

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
    video_path.write_text(render_video_outline(core12), encoding="utf-8")

    print(f"Core12 JSON written to {json_path}")
    print(f"Forum draft written to {forum_path}")
    print(f"Video outline written to {video_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
