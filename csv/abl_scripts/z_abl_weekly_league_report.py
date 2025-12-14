"""Generate the ABL Weekly League Report (fundamentals + WAR) in Markdown."""

import argparse
import csv
import datetime
import math
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

LEAGUE_ID = 200
TEAM_IDS = set(range(1, 25))
SKIP_STRINGS = {"", "team", "generated", "meaning", "neutral", "gauntlet", "soft"}
WAR_KEYS = ["war", "war_total", "war_tot", "total_war"]


def should_skip_row(row: Dict[str, str]) -> bool:
    for val in row.values():
        if isinstance(val, str):
            text = val.strip()
            if text == "":
                continue
            if text.lower() in SKIP_STRINGS:
                return True
    return False


def to_lower_map(row: Dict[str, str]) -> Dict[str, str]:
    return {str(k).lower(): v for k, v in row.items()}


def get_str(row: Dict[str, str], keys: Iterable[str]) -> Optional[str]:
    lower = to_lower_map(row)
    for key in keys:
        val = lower.get(key.lower())
        if val is None:
            continue
        text = str(val).strip()
        if text == "":
            continue
        return text
    return None


def get_int(row: Dict[str, str], keys: Iterable[str]) -> Optional[int]:
    value = get_str(row, keys)
    if value is None:
        return None
    try:
        if "." in value:
            return int(float(value))
        return int(value)
    except ValueError:
        return None


def get_float(row: Dict[str, str], keys: Iterable[str]) -> Optional[float]:
    value = get_str(row, keys)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def safe_div(num: Optional[float], denom: Optional[float]) -> Optional[float]:
    if num is None or denom is None or denom == 0:
        return None
    return num / denom


def ip_to_outs(ip: Optional[float]) -> float:
    if ip is None:
        return 0.0
    whole = int(ip)
    frac = round((ip - whole) * 10)
    frac_outs = 0
    if frac == 1:
        frac_outs = 1
    elif frac == 2:
        frac_outs = 2
    return whole * 3 + frac_outs


def normalize_headers(path: Path) -> List[str]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [h.lower() for h in (reader.fieldnames or [])]


def best_match(patterns: List[str], search_dirs: List[Path]) -> Optional[Path]:
    for pattern in patterns:
        for base in search_dirs:
            matches = sorted(base.rglob(pattern))
            if matches:
                return matches[0]
    return None


def find_csv_root(start: Path) -> Path:
    current = start
    while True:
        if any(current.rglob("teams*.csv")):
            return current
        if current.parent == current:
            break
        current = current.parent
    raise RuntimeError("Missing teams CSV in csv/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ABL Weekly League Report")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def filter_row_common(row: Dict[str, str], year: int) -> bool:
    if should_skip_row(row):
        return False
    if get_int(row, ["league_id", "leagueid", "league"]) not in (None, LEAGUE_ID):
        return False
    if get_int(row, ["year"]) not in (None, year):
        return False
    level_id = get_int(row, ["level_id", "level"])
    if level_id is not None and level_id != 1:
        return False
    split_id = get_int(row, ["split_id", "split"])
    if split_id is not None and split_id not in (0, 1):
        return False
    return True


def load_sub_leagues(path: Optional[Path]) -> Dict[int, str]:
    if not path:
        return {}
    names: Dict[int, str] = {}
    for row in read_rows(path):
        if should_skip_row(row):
            continue
        if get_int(row, ["league_id"]) not in (None, LEAGUE_ID):
            continue
        sid = get_int(row, ["sub_league_id", "subleague_id", "sub_lg_id"])
        if sid is None:
            continue
        names[sid] = (get_str(row, ["name", "abbr"]) or f"Conference {sid}").strip()
    return names


def load_divisions(path: Optional[Path]) -> Dict[Tuple[int, int], str]:
    if not path:
        return {}
    names: Dict[Tuple[int, int], str] = {}
    for row in read_rows(path):
        if should_skip_row(row):
            continue
        if get_int(row, ["league_id"]) not in (None, LEAGUE_ID):
            continue
        sid = get_int(row, ["sub_league_id", "subleague_id", "sub_lg_id"]) or 0
        did = get_int(row, ["division_id", "div_id"])
        if did is None:
            continue
        names[(sid, did)] = (get_str(row, ["name", "abbr"]) or f"Division {did}").strip()
    return names


def load_players(path: Optional[Path]) -> Dict[int, str]:
    if not path:
        return {}
    players: Dict[int, str] = {}
    for row in read_rows(path):
        if should_skip_row(row):
            continue
        pid = get_int(row, ["player_id", "id"])
        if pid is None:
            continue
        name = get_str(row, ["player_name", "name"])
        if not name:
            first = get_str(row, ["first_name", "firstname"]) or ""
            last = get_str(row, ["last_name", "lastname"]) or ""
            name = f"{first} {last}".strip()
        players[pid] = name or f"Player {pid}"
    return players


def load_teams(path: Path) -> Dict[int, dict]:
    teams: Dict[int, dict] = {}
    headers = normalize_headers(path)
    has_w = any(h in {"w", "wins"} for h in headers)
    has_l = any(h in {"l", "losses"} for h in headers)
    has_rf = any(h in {"r", "rf", "runs"} for h in headers)
    has_ra = any(h in {"ra", "runs_against", "runs_allowed", "rallowed"} for h in headers)
    for row in read_rows(path):
        if should_skip_row(row):
            continue
        if get_int(row, ["league_id", "leagueid", "league"]) != LEAGUE_ID:
            continue
        tid = get_int(row, ["team_id", "id"])
        if tid is None or tid not in TEAM_IDS:
            continue
        abbr = get_str(row, ["abbr", "team_abbr"])
        name = get_str(row, ["name"]) or ""
        nickname = get_str(row, ["nickname"]) or ""
        display = name
        if name and nickname:
            display = f"{name} {nickname}"
        elif nickname and not name:
            display = nickname
        teams[tid] = {
            "abbr": abbr or f"T{tid}",
            "name": display.strip() or (abbr or f"T{tid}"),
            "sub_league_id": get_int(row, ["sub_league_id", "subleague_id", "sub_lg_id"]) or 0,
            "division_id": get_int(row, ["division_id", "div_id"]) or 0,
            "w": get_int(row, ["w", "wins"]) if has_w else None,
            "l": get_int(row, ["l", "losses"]) if has_l else None,
            "rf": get_float(row, ["r", "rf", "runs"]) if has_rf else None,
            "ra": get_float(row, ["ra", "runs_against", "runs_allowed", "rallowed"]) if has_ra else None,
        }
    return teams


def load_team_records(path: Optional[Path], year: int, team_ids: set) -> Dict[int, dict]:
    records: Dict[int, dict] = {}
    if not path:
        return records
    for row in read_rows(path):
        if should_skip_row(row):
            continue
        ry = get_int(row, ["year"])
        if ry is not None and ry != year:
            continue
        tid = get_int(row, ["team_id"])
        if tid is None or tid not in team_ids:
            continue
        records[tid] = {"w": get_int(row, ["w", "wins"]) or 0, "l": get_int(row, ["l", "losses"]) or 0}
    return records


def load_team_stats(path: Optional[Path], year: int, team_ids: set) -> Dict[int, dict]:
    if not path:
        return {}
    data: Dict[int, dict] = {}
    for row in read_rows(path):
        if not filter_row_common(row, year):
            continue
        tid = get_int(row, ["team_id"])
        if tid is None or tid not in team_ids:
            continue
        row_split = get_int(row, ["split_id", "split"])
        if tid in data:
            existing_split = get_int(data[tid], ["split_id", "split"])
            if existing_split == 1 and row_split != 1:
                continue
        data[tid] = to_lower_map(row)
    return data


def detect_war_col(headers: List[str]) -> Optional[str]:
    lower = [h.lower() for h in headers]
    for key in WAR_KEYS:
        if key in lower:
            return key
    return None


def load_player_stats(path: Optional[Path], year: int, team_ids: set) -> Tuple[Dict[int, List[Dict[str, str]]], Optional[str]]:
    stats: Dict[int, List[Dict[str, str]]] = {}
    if not path:
        return stats, None
    war_col = detect_war_col(normalize_headers(path))
    for row in read_rows(path):
        if not filter_row_common(row, year):
            continue
        tid = get_int(row, ["team_id"])
        if tid is None or tid not in team_ids:
            continue
        pid = get_int(row, ["player_id", "id"])
        if pid is None:
            continue
        stats.setdefault(pid, []).append(row)
    return stats, war_col


def pct(w: int, l: int) -> float:
    return w / (w + l) if (w + l) > 0 else 0.0


def compute_rd(tid: int, team_bat: Dict[int, dict], team_pit: Dict[int, dict], teams: Dict[int, dict]) -> Optional[float]:
    bat = team_bat.get(tid)
    pit = team_pit.get(tid)
    if bat and pit:
        rf = get_float(bat, ["r", "runs"])
        ra = get_float(pit, ["ra", "runs_against", "runs_allowed"])
        fallback_r = get_float(pit, ["r"])
        if ra is None or (fallback_r is not None and ra is not None and ra < fallback_r * 0.5):
            # Some exports only provide 'r' for pitching; treat it as runs allowed when dedicated keys are absent.
            ra = fallback_r
        if rf is not None and ra is not None:
            return rf - ra
    rf = teams[tid].get("rf")
    ra = teams[tid].get("ra")
    if rf is not None and ra is not None:
        return rf - ra
    return None


def standings_section(teams: Dict[int, dict], records: Dict[int, dict], team_bat: Dict[int, dict], team_pit: Dict[int, dict], sub_leagues: Dict[int, str], divisions: Dict[Tuple[int, int], str]) -> Tuple[List[str], List[Tuple[str, float]], List[Tuple[str, float]], List[Tuple[float, int]]]:
    lines: List[str] = []
    tight: List[Tuple[str, float]] = []
    leads: List[Tuple[str, float]] = []
    chasers: List[Tuple[float, int]] = []
    grouped: Dict[int, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
    for tid, info in teams.items():
        grouped[info["sub_league_id"]][info["division_id"]].append(tid)
    for sl_id in sorted(grouped.keys()):
        sl_name = sub_leagues.get(sl_id, f"Conference {sl_id}")
        for div_id in sorted(grouped[sl_id].keys()):
            div_name = divisions.get((sl_id, div_id), f"Division {div_id}")
            lines.append(f"{sl_name} - {div_name}")
            lines.append("ABBR Team W-L PCT GB RD")
            entries = []
            for tid in grouped[sl_id][div_id]:
                rec = records.get(tid)
                if not rec:
                    continue
                w = rec["w"]
                l = rec["l"]
                rd_val = compute_rd(tid, team_bat, team_pit, teams)
                entries.append((pct(w, l), w, tid, l, rd_val))
            entries.sort(key=lambda x: (-x[0], -x[1]))
            if not entries:
                lines.append("")
                continue
            leader_w = entries[0][1]
            leader_l = entries[0][3]
            second_gb = None
            for idx, (pval, w, tid, l, rd_val) in enumerate(entries):
                gb = ((leader_w - w) + (l - leader_l)) / 2.0
                if idx == 1:
                    second_gb = gb
                if idx > 0:
                    chasers.append((gb, tid))
                rd_text = "NA" if rd_val is None else f"{rd_val:+.0f}"
                lines.append(f"{teams[tid]['abbr']} {teams[tid]['name']} {w}-{l} {pval:.3f} {gb:.1f} {rd_text}")
            if second_gb is not None and second_gb <= 3.0:
                tight.append((f"{sl_name} - {div_name}", second_gb))
            if second_gb is not None:
                leader_name = f"{teams[entries[0][2]]['abbr']} {teams[entries[0][2]]['name']}"
                leads.append((leader_name, second_gb))
            lines.append("")
    tight.sort(key=lambda x: x[1])
    leads.sort(key=lambda x: x[1], reverse=True)
    chasers.sort(key=lambda x: x[0])
    return lines, tight, leads, chasers


def rd_snapshot(team_bat: Dict[int, dict], team_pit: Dict[int, dict], teams: Dict[int, dict]) -> Tuple[List[str], List[str]]:
    diffs: List[Tuple[int, float]] = []
    for tid in teams:
        rd_val = compute_rd(tid, team_bat, team_pit, teams)
        if rd_val is None:
            continue
        diffs.append((tid, rd_val))
    if not diffs:
        return [], []
    diffs.sort(key=lambda x: x[1], reverse=True)
    top = [f"- {teams[tid]['abbr']} {teams[tid]['name']} - RD {rd:+.0f}" for tid, rd in diffs[:5]]
    bottom = [f"- {teams[tid]['abbr']} {teams[tid]['name']} - RD {rd:+.0f}" for tid, rd in sorted(diffs, key=lambda x: x[1])[:5]]
    return top, bottom


def offense_snapshot(team_bat: Dict[int, dict], teams: Dict[int, dict]) -> List[str]:
    entries: List[Tuple[int, float, Optional[float], Optional[float]]] = []
    for tid, row in team_bat.items():
        r = get_float(row, ["r", "runs"])
        g = get_float(row, ["g", "games"])
        rg = safe_div(r, g)
        ops = get_float(row, ["ops"])
        if ops is None:
            obp = get_float(row, ["obp"])
            slg = get_float(row, ["slg"])
            if obp is not None and slg is not None:
                ops = obp + slg
        if rg is None and ops is None:
            continue
        primary = rg if rg is not None else (ops if ops is not None else -math.inf)
        entries.append((tid, primary, rg, ops))
    entries.sort(key=lambda x: -x[1])
    lines: List[str] = []
    for tid, _, rg, ops in entries[:5]:
        parts = []
        if rg is not None:
            parts.append(f"R/G {rg:.2f}")
        if ops is not None:
            parts.append(f"OPS {ops:.3f}")
        lines.append(f"- {teams[tid]['abbr']} {teams[tid]['name']} - {', '.join(parts)}")
    return lines


def pitching_snapshot(team_pit: Dict[int, dict], teams: Dict[int, dict]) -> List[str]:
    entries: List[Tuple[int, float, Optional[float]]] = []
    for tid, row in team_pit.items():
        era = get_float(row, ["era"])
        ip_val = get_float(row, ["ip"])
        if era is None:
            er = get_float(row, ["er"])
            outs = ip_to_outs(ip_val)
            innings = outs / 3.0 if outs else 0.0
            if er is not None and innings > 0:
                era = (er / innings) * 9.0
        whip = get_float(row, ["whip"])
        if whip is None:
            outs = ip_to_outs(ip_val)
            innings = outs / 3.0 if outs else 0.0
            if innings > 0:
                bb = get_float(row, ["bb"]) or 0.0
                h = get_float(row, ["ha", "h"]) or 0.0
                whip = (bb + h) / innings
        if era is None and whip is None:
            continue
        era_val = era if era is not None else math.inf
        entries.append((tid, era_val, whip))
    entries.sort(key=lambda x: (x[1], x[2] if x[2] is not None else math.inf))
    lines: List[str] = []
    for tid, era, whip in entries[:5]:
        parts = []
        if era != math.inf:
            parts.append(f"ERA {era:.2f}")
        if whip is not None:
            parts.append(f"WHIP {whip:.2f}")
        lines.append(f"- {teams[tid]['abbr']} {teams[tid]['name']} - {', '.join(parts)}")
    return lines


def aggregate_player_totals(stats: Dict[int, List[Dict[str, str]]], war_col: Optional[str]) -> Dict[int, dict]:
    totals: Dict[int, dict] = defaultdict(
        lambda: {
            "war": 0.0,
            "pa": 0.0,
            "ab": 0.0,
            "h": 0.0,
            "hr": 0,
            "rbi": 0,
            "team_war": defaultdict(float),
            "team_pa": defaultdict(float),
            "team_ip_outs": defaultdict(float),
            "er": 0.0,
            "k": 0,
            "w": 0,
            "ip_outs": 0.0,
            "name": "",
        }
    )
    for pid, rows in stats.items():
        for row in rows:
            war_val = get_float(row, [war_col]) if war_col else None
            war_val = war_val if war_val is not None else 0.0
            tid = get_int(row, ["team_id"])
            pa = get_float(row, ["pa"]) or 0.0
            ab = get_float(row, ["ab"]) or 0.0
            h = get_float(row, ["h"]) or 0.0
            hr = get_int(row, ["hr"]) or 0
            rbi = get_int(row, ["rbi"]) or 0
            ip_val = get_float(row, ["ip"])
            outs = ip_to_outs(ip_val)
            er = get_float(row, ["er"]) or 0.0
            k = get_int(row, ["k", "so"]) or 0
            w = get_int(row, ["w", "wins"]) or 0
            totals[pid]["war"] += war_val
            totals[pid]["pa"] += pa
            totals[pid]["ab"] += ab
            totals[pid]["h"] += h
            totals[pid]["hr"] += hr
            totals[pid]["rbi"] += rbi
            totals[pid]["ip_outs"] += outs
            totals[pid]["er"] += er
            totals[pid]["k"] += k
            totals[pid]["w"] += w
            row_name = get_str(row, ["player_name", "name"])
            if row_name:
                totals[pid]["name"] = row_name
            if tid is not None:
                totals[pid]["team_war"][tid] += war_val
                totals[pid]["team_pa"][tid] += pa
                totals[pid]["team_ip_outs"][tid] += outs
    return totals


def war_section(bat_totals: Dict[int, dict], pitch_totals: Dict[int, dict], players: Dict[int, str], teams: Dict[int, dict]) -> Tuple[List[str], List[str], List[str]]:
    combined: Dict[int, dict] = defaultdict(lambda: {"war": 0.0, "team_war": defaultdict(float)})
    for pid, data in bat_totals.items():
        combined[pid]["war"] += data["war"]
        for tid, war in data["team_war"].items():
            combined[pid]["team_war"][tid] += war
    for pid, data in pitch_totals.items():
        combined[pid]["war"] += data["war"]
        for tid, war in data["team_war"].items():
            combined[pid]["team_war"][tid] += war

    player_lines: List[Tuple[float, str]] = []
    team_war: Dict[int, float] = defaultdict(float)

    for pid, data in combined.items():
        total_war = data["war"]
        team_id = None
        if data["team_war"]:
            team_id = max(data["team_war"].items(), key=lambda x: x[1])[0]
        name = players.get(pid) or bat_totals.get(pid, {}).get("name") or pitch_totals.get(pid, {}).get("name") or f"Player {pid}"
        abbr = teams.get(team_id, {}).get("abbr", "N/A") if team_id else "N/A"
        player_lines.append((total_war, f"- {name} ({abbr}) - WAR {total_war:.1f}"))
        for tid, war in data["team_war"].items():
            team_war[tid] += war

    player_lines.sort(key=lambda x: x[0], reverse=True)
    top_players = [line for _, line in player_lines[:10]]

    team_sorted = sorted(team_war.items(), key=lambda x: x[1], reverse=True)
    top_team = [f"- {teams[tid]['abbr']} {teams[tid]['name']} - Team WAR {war:.1f}" for tid, war in team_sorted[:5]]
    bottom_team = [f"- {teams[tid]['abbr']} {teams[tid]['name']} - Team WAR {war:.1f}" for tid, war in sorted(team_war.items(), key=lambda x: x[1])[:5]]

    return top_players, top_team, bottom_team


def batting_leaders_section(totals: Dict[int, dict], players: Dict[int, str], teams: Dict[int, dict], team_batting: Dict[int, dict]) -> Dict[str, List[str]]:
    leaders = {"AVG": [], "HR": [], "RBI": []}
    pa_available = any(data.get("pa", 0.0) > 0 for data in totals.values())
    ab_available = any(data.get("ab", 0.0) > 0 for data in totals.values())
    for pid, data in totals.items():
        pa = data.get("pa", 0.0)
        ab = data.get("ab", 0.0)
        primary_team = None
        if data["team_pa"]:
            primary_team = max(data["team_pa"].items(), key=lambda x: x[1])[0]
        games = None
        if primary_team is not None and primary_team in team_batting:
            games = get_float(team_batting[primary_team], ["g", "games"])
        if games is not None:
            required_pa = games * 3.1
            if pa < required_pa:
                continue
        elif (pa_available and pa < 25) or (not pa_available and ab_available and ab < 25):
            continue
        h = data.get("h", 0.0)
        hr = data.get("hr", 0)
        rbi = data.get("rbi", 0)
        avg = None
        if ab > 0:
            avg = h / ab
        name = players.get(pid) or totals[pid].get("name") or f"Player {pid}"
        abbr = teams.get(primary_team, {}).get("abbr", "N/A") if primary_team is not None else "N/A"
        if avg is not None:
            leaders["AVG"].append((avg, f"- {name} ({abbr}) - {avg:.3f}"))
        leaders["HR"].append((hr, f"- {name} ({abbr}) - {hr} HR"))
        leaders["RBI"].append((rbi, f"- {name} ({abbr}) - {rbi} RBI"))
    result: Dict[str, List[str]] = {}
    for key in ["AVG", "HR", "RBI"]:
        rows = sorted(leaders[key], key=lambda x: x[0], reverse=True)[:5]
        result[key] = [r[1] for r in rows]
    return result


def pitching_leaders_section(totals: Dict[int, dict], players: Dict[int, str], teams: Dict[int, dict]) -> Dict[str, List[str]]:
    leaders = {"ERA": [], "SO": [], "W": []}
    for pid, data in totals.items():
        outs = data.get("ip_outs", 0.0)
        if outs and outs < 30.0:  # 10 innings = 30 outs
            continue
        er = data.get("er", 0.0)
        era = (er / (outs / 3.0)) * 9.0 if outs > 0 else None
        k = data.get("k", 0)
        w = data.get("w", 0)
        name = players.get(pid) or totals[pid].get("name") or f"Player {pid}"
        abbr = "N/A"
        if data["team_ip_outs"]:
            abbr = teams.get(max(data["team_ip_outs"].items(), key=lambda x: x[1])[0], {}).get("abbr", "N/A")
        if era is not None:
            leaders["ERA"].append((era, f"- {name} ({abbr}) - {era:.2f} ERA"))
        leaders["SO"].append((k, f"- {name} ({abbr}) - {k} SO"))
        leaders["W"].append((w, f"- {name} ({abbr}) - {w} W"))
    result: Dict[str, List[str]] = {}
    result["ERA"] = [r[1] for r in sorted(leaders["ERA"], key=lambda x: x[0])[:5]] if leaders["ERA"] else []
    for key in ["SO", "W"]:
        result[key] = [r[1] for r in sorted(leaders[key], key=lambda x: x[0], reverse=True)[:5]]
    return result


def build_report(year: int, week: int, context: dict) -> str:
    teams = context["teams"]
    records = context["records"]
    sub_leagues = context["sub_leagues"]
    divisions = context["divisions"]
    team_bat = context["team_batting"]
    team_pit = context["team_pitching"]
    bat_totals = context["bat_totals"]
    pitch_totals = context["pitch_totals"]
    players = context["players"]

    lines: List[str] = []
    lines.append("# Action Baseball League - Weekly League Report")
    lines.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Season: {year} Week: {week:02d}")
    lines.append("")

    lines.append("A) Standings (by Conference then Division)")
    standings_lines, tight, leads, chasers = standings_section(teams, records, team_bat, team_pit, sub_leagues, divisions)
    lines.extend(standings_lines)

    lines.append("B) Pennant Races")
    if tight:
        lines.append("Tight divisions:")
        for name, gb in tight:
            lines.append(f"- {name}: {gb:.1f} GB")
    if chasers:
        lines.append("Closest chasers:")
        for gb, tid in chasers[:5]:
            lines.append(f"- {teams[tid]['abbr']} {teams[tid]['name']} (GB {gb:.1f})")
    if leads:
        lines.append("Biggest leads:")
        for leader, lead in leads[:3]:
            lines.append(f"- {leader} (Lead {lead:.1f})")
    lines.append("")

    if team_bat and team_pit:
        lines.append("C) Team Fundamentals")
        top_rd, bottom_rd = rd_snapshot(team_bat, team_pit, teams)
        if top_rd:
            lines.append("Run Differential (Top 5):")
            lines.extend(top_rd)
        if bottom_rd:
            lines.append("Run Differential (Bottom 5):")
            lines.extend(bottom_rd)
        offense = offense_snapshot(team_bat, teams)
        if offense:
            lines.append("Offense snapshot (Top 5):")
            lines.extend(offense)
        pitching = pitching_snapshot(team_pit, teams)
        if pitching:
            lines.append("Pitching snapshot (Top 5):")
            lines.extend(pitching)
        lines.append("")

    lines.append("D) WAR Report")
    top_players, top_team_war, bottom_team_war = war_section(bat_totals, pitch_totals, players, teams)
    lines.append("Top 10 players by WAR:")
    lines.extend(top_players)
    lines.append("Team WAR (Top 5):")
    lines.extend(top_team_war)
    lines.append("Team WAR (Bottom 5):")
    lines.extend(bottom_team_war)
    lines.append("")

    lines.append("E) League Leaders")
    bat_leaders = batting_leaders_section(bat_totals, players, teams, team_bat)
    if bat_leaders["AVG"]:
        lines.append("Batting AVG (Top 5):")
        lines.extend(bat_leaders["AVG"])
    if bat_leaders["HR"]:
        lines.append("Home Runs (Top 5):")
        lines.extend(bat_leaders["HR"])
    if bat_leaders["RBI"]:
        lines.append("RBI (Top 5):")
        lines.extend(bat_leaders["RBI"])

    pitch_leaders = pitching_leaders_section(pitch_totals, players, teams)
    if pitch_leaders["ERA"]:
        lines.append("ERA (Top 5):")
        lines.extend(pitch_leaders["ERA"])
    if pitch_leaders["SO"]:
        lines.append("Strikeouts (Top 5):")
        lines.extend(pitch_leaders["SO"])
    if pitch_leaders["W"]:
        lines.append("Wins (Top 5):")
        lines.extend(pitch_leaders["W"])

    return "\n".join(lines)


def preflight(year: int, search_dirs: List[Path]) -> dict:
    errors: List[str] = []
    teams_path = best_match(["teams.csv", "teams*.csv"], search_dirs)
    if not teams_path:
        errors.append("Missing teams CSV in csv/")
    team_record_path = best_match(["team_record.csv", "team_record*.csv"], search_dirs)
    sub_league_path = best_match(["sub_leagues.csv", "sub_leagues*.csv"], search_dirs)
    division_path = best_match(["divisions.csv", "divisions*.csv"], search_dirs)
    team_bat_path = best_match(["team_batting_stats.csv", "team_batting_stats*.csv"], search_dirs)
    team_pit_path = best_match(["team_pitching_stats.csv", "team_pitching_stats*.csv"], search_dirs)
    batting_path = best_match(
        ["batting_stats.csv", "batting_stats*.csv", "players_batting_stats.csv", "players_batting_stats*.csv", "players_career_batting_stats.csv", "players_career_batting_stats*.csv"],
        search_dirs,
    )
    pitching_path = best_match(
        ["pitching_stats.csv", "pitching_stats*.csv", "players_pitching_stats.csv", "players_pitching_stats*.csv", "players_career_pitching_stats.csv", "players_career_pitching_stats*.csv"],
        search_dirs,
    )
    players_path = best_match(["players.csv", "players*.csv"], search_dirs)

    if teams_path:
        headers = normalize_headers(teams_path)
        for col in ["team_id", "league_id", "abbr"]:
            if col not in headers:
                errors.append(f"teams.csv missing column '{col}'")
        if "name" not in headers and "nickname" not in headers:
            errors.append("teams.csv missing name or nickname column")

    if errors:
        raise RuntimeError("; ".join(errors))

    teams = load_teams(teams_path)
    if len(teams) != 24:
        errors.append(f"Expected 24 ABL teams (league_id=200, team_id 1..24) but found {len(teams)}")
    team_ids = set(teams.keys())

    team_records = load_team_records(team_record_path, year, team_ids) if team_record_path else {}
    have_wl = all(teams[tid].get("w") is not None and teams[tid].get("l") is not None for tid in team_ids)
    if not have_wl and team_records:
        for tid in team_ids:
            if tid in team_records:
                teams[tid]["w"] = team_records[tid]["w"]
                teams[tid]["l"] = team_records[tid]["l"]
    have_wl = all(teams[tid].get("w") is not None and teams[tid].get("l") is not None for tid in team_ids)
    if not have_wl:
        errors.append("Missing W/L data for canonical teams (teams.csv or team_record.csv)")

    sub_leagues = load_sub_leagues(sub_league_path)
    divisions = load_divisions(division_path)
    team_bat = load_team_stats(team_bat_path, year, team_ids)
    team_pit = load_team_stats(team_pit_path, year, team_ids)
    players = load_players(players_path)

    bat_stats, bat_war_col = load_player_stats(batting_path, year, team_ids)
    pitch_stats, pitch_war_col = load_player_stats(pitching_path, year, team_ids)
    if not ((bat_war_col and bat_stats) or (pitch_war_col and pitch_stats)):
        errors.append("No WAR data found for canonical teams")

    if errors:
        raise RuntimeError("; ".join(errors))

    bat_totals = aggregate_player_totals(bat_stats, bat_war_col)
    pitch_totals = aggregate_player_totals(pitch_stats, pitch_war_col)

    return {
        "teams": teams,
        "records": {tid: {"w": teams[tid]["w"], "l": teams[tid]["l"]} for tid in team_ids},
        "sub_leagues": sub_leagues,
        "divisions": divisions,
        "team_batting": team_bat,
        "team_pitching": team_pit,
        "bat_totals": bat_totals,
        "pitch_totals": pitch_totals,
        "players": players,
    }


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve()
    try:
        csv_root = find_csv_root(script_path.parent)
    except RuntimeError as exc:
        print(f"FATAL: {exc}")
        sys.exit(2)
    search_dirs = [csv_root]
    try:
        context = preflight(args.year, search_dirs)
        report = build_report(args.year, args.week, context)
        out_dir = csv_root / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        final_path = out_dir / f"league_report_abl_{args.year}_w{args.week:02d}.md"
        tmp_path = out_dir / f".tmp_league_report_abl_{args.year}_w{args.week:02d}.md"
        tmp_path.write_text(report, encoding="utf-8")
        tmp_path.replace(final_path)
        print(f"Wrote: {final_path.resolve()}")
    except RuntimeError as exc:
        print(f"FATAL: {exc}")
        sys.exit(2)
    except Exception:
        print("FATAL: Unexpected error")
        traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()
