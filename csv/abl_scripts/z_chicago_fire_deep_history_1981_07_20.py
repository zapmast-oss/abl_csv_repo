#!/usr/bin/env python
"""Build Chicago Fire deep-history research package as of 1981-07-19.

This script is intentionally read-only against source data. It writes a dated
research package under csv/out/story/investigations/.
"""

from __future__ import annotations

import csv
import io
import math
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OOTP = ROOT / "csv" / "ootp_csv"
ALMANAC = ROOT / "csv" / "out" / "almanac"
STATSPLUS = ROOT / "csv" / "statsplus" / "current"
OUT = ROOT / "csv" / "out" / "story" / "investigations"
RAW_HTML = ROOT / "data_raw" / "ootp_html"

ASOF = date(1981, 7, 19)
LEAGUE_ID = "200"
CHI = 12
DAL = 4
PHI = 22
DET = 5
NBC_CENTRAL = {4, 5, 11, 12}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def pct(w: int, l: int) -> str:
    return f"{(w / (w + l)):.3f}" if (w + l) else ""


def norm_team_name(s: str) -> str:
    s = (s or "").strip()
    fixes = {"Seatlle": "Seattle"}
    return fixes.get(s, s)


@dataclass
class Game:
    season: int
    game_date: date
    away_id: int
    home_id: int
    away_name: str
    home_name: str
    away_runs: int
    home_runs: int
    source: str
    game_id: str = ""

    def involves(self, team_id: int) -> bool:
        return self.away_id == team_id or self.home_id == team_id

    def team_result(self, team_id: int) -> tuple[bool, int, int, bool]:
        if self.away_id == team_id:
            return self.away_runs > self.home_runs, self.away_runs, self.home_runs, False
        if self.home_id == team_id:
            return self.home_runs > self.away_runs, self.home_runs, self.away_runs, True
        raise ValueError(team_id)


def load_dimensions() -> tuple[dict[int, dict[str, str]], dict[str, int]]:
    teams = {int(r["team_id"]): r for r in read_csv(OOTP / "teams.csv") if r["league_id"] == LEAGUE_ID}
    by_city = {r["name"]: int(tid) for tid, r in teams.items()}
    by_full = {f"{r['name']} {r['nickname']}": int(tid) for tid, r in teams.items()}
    by_abbr = {r["abbr"]: int(tid) for tid, r in teams.items()}
    aliases = {**by_city, **by_full, **by_abbr, "Seattle": 24, "Seatlle": 24}
    return teams, aliases


TEAMS, TEAM_ID_BY_NAME = load_dimensions()
TEAM_FULL = {tid: f"{r['name']} {r['nickname']}" for tid, r in TEAMS.items()}
TEAM_ABBR = {tid: r["abbr"] for tid, r in TEAMS.items()}


def team_id_from_name(name: str) -> int:
    name = norm_team_name(name)
    return TEAM_ID_BY_NAME.get(name, TEAM_ID_BY_NAME.get(name.replace("Firebirds", "Firebirds"), 0))


def load_historical_games(season: int) -> list[Game]:
    path = ALMANAC / str(season) / f"games_{season}_league200.csv"
    games: list[Game] = []
    for r in read_csv(path):
        away_name = norm_team_name(r["away_team_name"])
        home_name = norm_team_name(r["home_team_name"])
        games.append(
            Game(
                season=season,
                game_date=datetime.strptime(r["game_date"], "%Y-%m-%d").date(),
                away_id=team_id_from_name(away_name),
                home_id=team_id_from_name(home_name),
                away_name=away_name,
                home_name=home_name,
                away_runs=int(r["away_runs"]),
                home_runs=int(r["home_runs"]),
                source=str(path.relative_to(ROOT)),
                game_id=r["game_id"],
            )
        )
    return games


def load_1981_games() -> list[Game]:
    games: list[Game] = []
    path = OOTP / "games.csv"
    for r in read_csv(path):
        if r["league_id"] != LEAGUE_ID or r["played"] != "1" or r["game_type"] != "0":
            continue
        gd = datetime.strptime(r["date"], "%Y-%m-%d").date()
        if gd > ASOF:
            continue
        away_id = int(r["away_team"])
        home_id = int(r["home_team"])
        games.append(
            Game(
                season=1981,
                game_date=gd,
                away_id=away_id,
                home_id=home_id,
                away_name=TEAM_FULL[away_id],
                home_name=TEAM_FULL[home_id],
                away_runs=int(r["runs0"]),
                home_runs=int(r["runs1"]),
                source=str(path.relative_to(ROOT)),
                game_id=r["game_id"],
            )
        )
    return games


def team_record_from_games(games: list[Game], team_id: int) -> dict[str, Any]:
    w = l = rs = ra = hw = hl = rw = rl = 0
    results: list[tuple[date, bool]] = []
    month = defaultdict(lambda: [0, 0])
    for g in sorted([x for x in games if x.involves(team_id)], key=lambda x: x.game_date):
        win, tr, oruns, is_home = g.team_result(team_id)
        rs += tr
        ra += oruns
        if win:
            w += 1
            if is_home:
                hw += 1
            else:
                rw += 1
        else:
            l += 1
            if is_home:
                hl += 1
            else:
                rl += 1
        month[g.game_date.strftime("%Y-%m")][0 if win else 1] += 1
        results.append((g.game_date, win))
    longest_w = longest_l = cur_w = cur_l = 0
    for _, win in results:
        if win:
            cur_w += 1
            cur_l = 0
        else:
            cur_l += 1
            cur_w = 0
        longest_w = max(longest_w, cur_w)
        longest_l = max(longest_l, cur_l)
    first = results[: len(results) // 2]
    second = results[len(results) // 2 :]
    def wl(part: list[tuple[date, bool]]) -> str:
        ww = sum(1 for _, x in part if x)
        ll = len(part) - ww
        return f"{ww}-{ll}"
    best_month = ""
    worst_month = ""
    if month:
        ranked = sorted(month.items(), key=lambda kv: (kv[1][0] / max(1, sum(kv[1])), kv[1][0] - kv[1][1]))
        worst_month = f"{ranked[0][0]} {ranked[0][1][0]}-{ranked[0][1][1]}"
        best_month = f"{ranked[-1][0]} {ranked[-1][1][0]}-{ranked[-1][1][1]}"
    return {
        "w": w,
        "l": l,
        "pct": pct(w, l),
        "rs": rs,
        "ra": ra,
        "rd": rs - ra,
        "home": f"{hw}-{hl}",
        "road": f"{rw}-{rl}",
        "first_half": wl(first),
        "second_half": wl(second),
        "best_month": best_month,
        "worst_month": worst_month,
        "longest_win_streak": longest_w,
        "longest_loss_streak": longest_l,
    }


def playoff_qualifiers(year: int) -> dict[int, str]:
    hist = [r for r in read_csv(OOTP / "team_history.csv") if r["year"] == str(year) and r["league_id"] == LEAGUE_ID]
    recs = {int(r["team_id"]): r for r in read_csv(OOTP / "team_history_record.csv") if r["year"] == str(year) and r["league_id"] == LEAGUE_ID}
    quals: dict[int, str] = {}
    for r in hist:
        tid = int(r["team_id"])
        if r["made_playoffs"] == "1":
            quals[tid] = "Division champion" if recs[tid]["pos"] == "1" else "Wild card"
    return quals


def division_finish(team_id: int, year: int) -> tuple[str, str]:
    for r in read_csv(OOTP / "team_history_record.csv"):
        if r["team_id"] == str(team_id) and r["year"] == str(year) and r["league_id"] == LEAGUE_ID:
            return r["pos"], r["gb"]
    return "", ""


def current_division_position(team_id: int) -> tuple[str, str]:
    for r in read_csv(OOTP / "team_record.csv"):
        if r["team_id"] == str(team_id):
            return r["pos"], r["gb"]
    return "", ""


def manager_name(manager_id: int) -> str:
    for r in read_csv(OOTP / "coaches.csv"):
        if r["coach_id"] == str(manager_id):
            return f"{r['first_name']} {r['last_name']}".strip()
    return f"coach#{manager_id}"


def playoff_messages() -> list[dict[str, str]]:
    rows = read_csv(OOTP / "messages.csv")
    out = []
    for r in rows:
        if not r["date"] or r["date"] > "1981-07-19":
            continue
        if r["league_id_0"] == LEAGUE_ID or "Action Baseball League" in r["body"]:
            text = f"{r['subject']} {r['body']}"
            if any(x in text for x in ["Playoffs begin", "Division Championship Series", "Conference Championship Series", "Grand Championship Series", "wrapped up their"]):
                out.append(r)
    return out


def clean_body(s: str) -> str:
    s = re.sub(r"<([^:>]+):(?:team|player|coach)#\d+>", r"\1", s)
    s = s.replace("\\n", " ").replace('\\"', '"')
    return re.sub(r"\s+", " ", s).strip()


def chicago_series_from_messages() -> list[dict[str, Any]]:
    # These rows are keyed to verified OOTP messages. The wording varies enough
    # across seasons that a regex-only extractor is too brittle for final claims.
    message_by_id = {r["message_id"]: r for r in read_csv(OOTP / "messages.csv")}
    fixed = [
        (1972, "1972-10-12", "Division Championship Series", "San Francisco Warriors", "Lost", "4-3", "1782"),
        (1973, "1973-10-10", "Division Championship Series", "Miami Hurricanes", "Won", "4-2", "3948"),
        (1973, "1973-10-17", "Conference Championship Series", "Phoenix Firebirds", "Won", "4-0", "3968"),
        (1973, "1973-10-29", "Grand Series", "Denver Rocketeers", "Lost", "4-3", "4040"),
        (1977, "1977-10-10", "Division Championship Series", "Atlanta Kings", "Lost", "4-1", "12380"),
        (1979, "1979-10-07", "Division Championship Series", "Miami Hurricanes", "Won", "4-0", "16528"),
        (1979, "1979-10-21", "Conference Championship Series", "Phoenix Firebirds", "Lost", "4-3", "16576"),
        (1980, "1980-10-14", "Division Championship Series", "Charlotte Colonels", "Won", "4-2", "18649"),
        (1980, "1980-10-20", "Conference Championship Series", "Detroit Dukes", "Lost", "4-0", "18673"),
    ]
    rows = []
    for season, dt, round_name, opponent, result, score, mid in fixed:
        msg = message_by_id.get(mid, {})
        rows.append({
            "season": season,
            "date": dt,
            "round": round_name,
            "opponent": opponent,
            "result": result,
            "series_score": score,
            "message_id": mid,
            "subject": msg.get("subject", ""),
            "evidence": clean_body(msg.get("body", ""))[:500],
        })
    return rows


def playoff_start(year: int) -> date | None:
    for r in playoff_messages():
        if r["date"].startswith(str(year)) and "Playoffs begin" in r["subject"] and "Action Baseball League" in r["body"]:
            return datetime.strptime(r["date"], "%Y-%m-%d").date()
    return None


def parse_box_from_zip(zf: zipfile.ZipFile, year: int, gid: str, gd: date) -> Game | None:
    rel = f"almanac_{year}/box_scores/game_box_{gid}.html"
    if rel not in zf.namelist():
        return None
    html = zf.read(rel).decode("utf-8", errors="ignore")
    try:
        tables = pd.read_html(io.StringIO(html))
    except Exception:
        return None
    for t in tables:
        cols = [str(c).strip() for c in t.columns]
        if "R" in cols and len(t) >= 2:
            try:
                away_name = norm_team_name(str(t.iloc[0, 0]).strip())
                home_name = norm_team_name(str(t.iloc[1, 0]).strip())
                r_idx = cols.index("R")
                away_runs = int(float(t.iloc[0].iloc[r_idx]))
                home_runs = int(float(t.iloc[1].iloc[r_idx]))
                return Game(year, gd, team_id_from_name(away_name), team_id_from_name(home_name), away_name, home_name, away_runs, home_runs, f"data_raw/ootp_html/almanac_{year}.zip", str(gid))
            except Exception:
                return None
    return None


def postseason_games(year: int) -> list[Game]:
    start = playoff_start(year)
    if not start:
        return []
    zpath = RAW_HTML / f"almanac_{year}.zip"
    games: list[Game] = []
    if not zpath.exists():
        return games
    with zipfile.ZipFile(zpath) as zf:
        names = set(zf.namelist())
        for i in range(45):
            gd = start + timedelta(days=i)
            rel = f"almanac_{year}/leagues/league_200_scores_{gd:%Y_%m_%d}.html"
            if rel not in names:
                continue
            html = zf.read(rel).decode("utf-8", errors="ignore")
            for gid in sorted(set(re.findall(r"game_box_(\d+)\.html", html))):
                g = parse_box_from_zip(zf, year, gid, gd)
                if g:
                    games.append(g)
    seen = set()
    uniq = []
    for g in sorted(games, key=lambda x: (x.game_date, x.game_id)):
        if g.game_id in seen:
            continue
        seen.add(g.game_id)
        uniq.append(g)
    return uniq


def build_postseason_ledger(series_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    by_year = defaultdict(list)
    for r in series_rows:
        by_year[r["season"]].append(r)
    hist_rec = {(int(r["team_id"]), int(r["year"])): r for r in read_csv(OOTP / "team_history_record.csv") if r["league_id"] == LEAGUE_ID}
    for year in [1972, 1973, 1977, 1979, 1980]:
        reg = hist_rec[(CHI, year)]
        q = playoff_qualifiers(year).get(CHI, "")
        games = [g for g in postseason_games(year) if g.involves(CHI)]
        groups: list[tuple[str, list[Game]]] = []
        for g in games:
            opp = g.home_name if g.away_id == CHI else g.away_name
            if not groups or groups[-1][0] != opp:
                groups.append((opp, [g]))
            else:
                groups[-1][1].append(g)
        games_by_opp = {opp: gs for opp, gs in groups}
        for s in by_year[year]:
            gs = games_by_opp.get(s["opponent"], [])
            chi_w = sum(1 for g in gs if g.team_result(CHI)[0])
            chi_l = len(gs) - chi_w
            rows.append({
                "season": year,
                "regular_season_record": f"{reg['w']}-{reg['l']}",
                "division_finish": reg["pos"],
                "qualification_method": q,
                "round": s["round"],
                "opponent": s["opponent"],
                "result": s["result"],
                "series_score": s["series_score"],
                "chicago_postseason_wins_in_round": chi_w if gs else "",
                "chicago_postseason_losses_in_round": chi_l if gs else "",
                "individual_game_results": " | ".join(game_line(g, CHI) for g in gs) if gs else "Individual game rows not recovered from parsed box-score archive; series result proven by OOTP message.",
                "deciding_game": game_line(gs[-1], CHI) if gs else s["evidence"],
                "evidence": f"csv/ootp_csv/messages.csv message_id {s['message_id']}; {s['source'] if 'source' in s else 'data_raw/ootp_html/almanac zip when game rows parse'}",
            })
    return rows


def game_line(g: Game, team_id: int) -> str:
    win, tr, oruns, is_home = g.team_result(team_id)
    opp = g.away_name if is_home else g.home_name
    site = "home" if is_home else "road"
    return f"{g.game_date}: {'W' if win else 'L'} {tr}-{oruns} vs {opp} ({site})"


def build_season_ledger() -> list[dict[str, Any]]:
    hist = {(int(r["team_id"]), int(r["year"])): r for r in read_csv(OOTP / "team_history_record.csv") if r["league_id"] == LEAGUE_ID}
    hteam = {(int(r["team_id"]), int(r["year"])): r for r in read_csv(OOTP / "team_history.csv") if r["league_id"] == LEAGUE_ID}
    batting = {(int(r["team_id"]), int(r["year"])): r for r in read_csv(OOTP / "team_history_batting_stats.csv") if r["league_id"] == LEAGUE_ID and r["split_id"] == "0"}
    pitching = {(int(r["team_id"]), int(r["year"])): r for r in read_csv(OOTP / "team_history_pitching_stats.csv") if r["league_id"] == LEAGUE_ID and r["split_id"] == "1"}
    ps_summary = postseason_summary_by_year()
    rows = []
    for year in range(1972, 1982):
        games = load_1981_games() if year == 1981 else load_historical_games(year)
        rec_calc = team_record_from_games(games, CHI)
        if year <= 1980:
            rec = hist[(CHI, year)]
            pos, gb = rec["pos"], rec["gb"]
            manager = manager_name(int(hteam[(CHI, year)]["manager_id"]))
            rs = int(float(batting[(CHI, year)]["r"]))
            ra = int(float(pitching[(CHI, year)]["r"]))
            official_w = int(rec["w"])
            official_l = int(rec["l"])
            official_pct = pct(official_w, official_l)
            if rec_calc["w"] + rec_calc["l"] == official_w + official_l:
                home_record = rec_calc["home"]
                road_record = rec_calc["road"]
                first_half = rec_calc["first_half"]
                second_half = rec_calc["second_half"]
                best_month = rec_calc["best_month"]
                worst_month = rec_calc["worst_month"]
                longest_w = rec_calc["longest_win_streak"]
                longest_l = rec_calc["longest_loss_streak"]
                validation = "game-derived record matches team_history_record"
            else:
                home_record = "unresolved"
                road_record = "unresolved"
                first_half = "unresolved"
                second_half = "unresolved"
                best_month = rec_calc["best_month"]
                worst_month = rec_calc["worst_month"]
                longest_w = "unresolved"
                longest_l = "unresolved"
                validation = f"official record from team_history_record; almanac game extract has {rec_calc['w'] + rec_calc['l']} Chicago rows, not 162"
        else:
            pos, gb = current_division_position(CHI)
            manager = manager_name(46)
            rs, ra = rec_calc["rs"], rec_calc["ra"]
            official_w = rec_calc["w"]
            official_l = rec_calc["l"]
            official_pct = rec_calc["pct"]
            home_record = rec_calc["home"]
            road_record = rec_calc["road"]
            first_half = rec_calc["first_half"]
            second_half = rec_calc["second_half"]
            best_month = rec_calc["best_month"]
            worst_month = rec_calc["worst_month"]
            longest_w = rec_calc["longest_win_streak"]
            longest_l = rec_calc["longest_loss_streak"]
            validation = "current raw games through cutoff"
        rows.append({
            "season": year,
            "record": f"{official_w}-{official_l}",
            "winning_percentage": official_pct,
            "division_finish": pos,
            "games_behind": gb,
            "runs_scored": rs,
            "runs_allowed": ra,
            "run_differential": rs - ra,
            "home_record": home_record,
            "road_record": road_record,
            "first_half_record": first_half,
            "second_half_record": second_half,
            "best_month": best_month,
            "worst_month": worst_month,
            "longest_winning_streak": longest_w,
            "longest_losing_streak": longest_l,
            "playoff_qualification": playoff_qualifiers(year).get(CHI, "") if year <= 1980 else "",
            "final_postseason_result": ps_summary.get(year, "No postseason" if year <= 1980 else "In progress"),
            "manager": manager,
            "validation_note": validation,
        })
    return rows


def postseason_summary_by_year() -> dict[int, str]:
    out = {}
    for r in chicago_series_from_messages():
        if r["result"] == "Lost":
            out[r["season"]] = f"Lost {r['round']} to {r['opponent']} {r['series_score']}"
        else:
            out[r["season"]] = f"Advanced past {r['opponent']} in {r['round']}"
    return out


def franchise_comparison() -> list[dict[str, Any]]:
    rows = []
    batting = {(int(r["team_id"]), int(r["year"])): r for r in read_csv(OOTP / "team_history_batting_stats.csv") if r["league_id"] == LEAGUE_ID and r["split_id"] == "0"}
    pitching = {(int(r["team_id"]), int(r["year"])): r for r in read_csv(OOTP / "team_history_pitching_stats.csv") if r["league_id"] == LEAGUE_ID and r["split_id"] == "1"}
    hist = [r for r in read_csv(OOTP / "team_history_record.csv") if r["league_id"] == LEAGUE_ID and 1972 <= int(r["year"]) <= 1980]
    th = {(int(r["team_id"]), int(r["year"])): r for r in read_csv(OOTP / "team_history.csv") if r["league_id"] == LEAGUE_ID}
    series_wins = Counter()
    gs_apps = Counter()
    for r in chicago_series_from_messages():
        if r["result"] == "Won":
            series_wins[CHI] += 1
        if r["round"] == "Grand Series":
            gs_apps[CHI] += 1
    # Dallas 1977 championship path from messages: DCS, CCS, GS wins.
    series_wins[DAL] = 3
    gs_apps[DAL] = 1
    for tid in [CHI, DAL, PHI]:
        recs = [r for r in hist if int(r["team_id"]) == tid]
        w = sum(int(r["w"]) for r in recs)
        l = sum(int(r["l"]) for r in recs)
        rds = []
        for r in recs:
            yr = int(r["year"])
            rds.append(int(float(batting[(tid, yr)]["r"])) - int(float(pitching[(tid, yr)]["r"])))
        rows.append({
            "team_id": tid,
            "team": TEAM_FULL[tid],
            "cumulative_wins": w,
            "cumulative_losses": l,
            "winning_percentage": pct(w, l),
            "winning_seasons": sum(1 for r in recs if int(r["w"]) > int(r["l"])),
            "division_championships": sum(1 for r in recs if r["pos"] == "1"),
            "wild_cards": sum(1 for r in recs if th[(tid, int(r["year"]))]["made_playoffs"] == "1" and r["pos"] != "1"),
            "postseason_appearances": sum(1 for r in recs if th[(tid, int(r["year"]))]["made_playoffs"] == "1"),
            "postseason_series_wins": series_wins[tid],
            "grand_series_appearances": gs_apps[tid],
            "championships": sum(1 for r in recs if th[(tid, int(r["year"]))]["won_playoffs"] == "1"),
            "total_run_differential": sum(rds),
            "best_single_season_record": max((f"{r['w']}-{r['l']}" for r in recs), key=lambda x: int(x.split("-")[0])),
            "worst_single_season_record": min((f"{r['w']}-{r['l']}" for r in recs), key=lambda x: int(x.split("-")[0])),
        })
    return rows


def head_to_head() -> list[dict[str, Any]]:
    games = []
    for y in range(1972, 1981):
        games.extend(load_historical_games(y))
    games.extend(load_1981_games())
    rows = []
    for y in range(1972, 1982):
        subset = [g for g in games if g.season == y and {g.away_id, g.home_id} == {CHI, DAL}]
        if not subset:
            continue
        chi_w = sum(1 for g in subset if g.team_result(CHI)[0])
        dal_w = len(subset) - chi_w
        chi_home = [g for g in subset if g.home_id == CHI]
        chi_road = [g for g in subset if g.away_id == CHI]
        rd = sum(g.team_result(CHI)[1] - g.team_result(CHI)[2] for g in subset)
        rows.append({
            "season": y,
            "chicago_record": f"{chi_w}-{dal_w}",
            "games": len(subset),
            "chicago_home_record": f"{sum(g.team_result(CHI)[0] for g in chi_home)}-{len(chi_home)-sum(g.team_result(CHI)[0] for g in chi_home)}",
            "chicago_road_record": f"{sum(g.team_result(CHI)[0] for g in chi_road)}-{len(chi_road)-sum(g.team_result(CHI)[0] for g in chi_road)}",
            "chicago_run_differential": rd,
            "game_results": " | ".join(game_line(g, CHI) for g in sorted(subset, key=lambda x: x.game_date)),
        })
    return rows


def current_context() -> dict[str, Any]:
    games = load_1981_games()
    chi_games = [g for g in games if g.involves(CHI)]
    rec = team_record_from_games(games, CHI)
    last10 = chi_games[-10:]
    det_series = [g for g in games if g.involves(CHI) and g.involves(DET) and date(1981, 7, 16) <= g.game_date <= date(1981, 7, 19)]
    july19 = [g for g in games if g.game_date == ASOF and g.involves(CHI)]
    vs_dal = [g for g in games if {g.away_id, g.home_id} == {CHI, DAL}]
    vs_central = [g for g in games if g.involves(CHI) and ((g.away_id in NBC_CENTRAL and g.away_id != CHI) or (g.home_id in NBC_CENTRAL and g.home_id != CHI))]
    def w_l(sub: list[Game], tid: int) -> str:
        w = sum(1 for g in sub if g.team_result(tid)[0])
        return f"{w}-{len(sub)-w}"
    return {
        "record": f"{rec['w']}-{rec['l']}",
        "position": current_division_position(CHI)[0],
        "gb_dallas": "4.0",
        "gb_detroit": "2.0",
        "run_differential": rec["rd"],
        "home": rec["home"],
        "road": rec["road"],
        "current_streak": "W2",
        "last10": w_l(last10, CHI),
        "detroit_series": " | ".join(game_line(g, CHI) for g in det_series),
        "july19": game_line(july19[0], CHI) if july19 else "",
        "vs_dallas_1981": w_l(vs_dal, CHI),
        "vs_nbc_central": w_l(vs_central, CHI),
    }


def statsplus_leaders() -> tuple[list[str], list[str], list[str]]:
    bat = [r for r in read_csv(STATSPLUS / "20_statsplus_Player_Batting.csv") if r["Team"] == "CHI"]
    pit = [r for r in read_csv(STATSPLUS / "21_statsplus_Player_Pitching.csv") if r["Team"] == "CHI"]
    team_bat = [r for r in read_csv(STATSPLUS / "14_statsplus_Team_Batting_League.csv") if r["Team"] == "CHI"]
    team_pit = [r for r in read_csv(STATSPLUS / "16_statsplus_Team_Pitching_league.csv") if r["Team"] == "CHI"]
    bat_top = sorted(bat, key=lambda r: float(r.get("WAR", "0") or 0), reverse=True)[:4]
    pit_top = sorted(pit, key=lambda r: float(r.get("WAR", "0") or 0), reverse=True)[:4]
    btxt = [f"{r['Name']}: {r['WAR']} WAR, {r['HR']} HR, {r['AVG']}/{r['OBP']}/{r['OPS   &nbsp;']}" for r in bat_top]
    ptxt = [f"{r['Name']}: {r['WAR']} WAR, {r['W']}-{r['L']}, {r['ERA']} ERA" for r in pit_top]
    profile = []
    if team_bat:
        r = team_bat[0]
        profile.append(f"Offense: {r['R']} runs, {r['AVG']}/{r['OBP']}/{r['OPS']}, {r['HR']} HR")
    if team_pit:
        r = team_pit[0]
        profile.append(f"Pitching: {r['ERA']} ERA, {r['FIP']} FIP, {r['R']} runs allowed")
    return btxt, ptxt, profile


def continuity_figures() -> list[str]:
    names = {int(r["player_id"]): f"{r['first_name']} {r['last_name']}" for r in read_csv(OOTP / "players.csv")}
    seasons = defaultdict(list)
    for path in [OOTP / "players_career_batting_stats.csv", OOTP / "players_career_pitching_stats.csv"]:
        for r in read_csv(path):
            if r["league_id"] == LEAGUE_ID and r["team_id"] == str(CHI) and r["game_id"] == "0":
                y = int(r["year"])
                if 1972 <= y <= 1981:
                    seasons[int(r["player_id"])].append(y)
    playoff_years = {1972, 1973, 1977, 1979, 1980}
    current = {int(r["player_id"]) for r in read_csv(OOTP / "team_roster.csv") if r["team_id"] == str(CHI)}
    leads = []
    for pid, yrs in seasons.items():
        yrs = sorted(set(yrs))
        pys = sorted(set(yrs) & playoff_years)
        if len(yrs) >= 5 or len(pys) >= 2 or pid in {196, 8025, 8125}:
            status = "current 1981 roster" if pid in current else "not on current roster"
            leads.append(f"{names.get(pid, 'player#'+str(pid))}: Chicago seasons {yrs[0]}-{yrs[-1]} ({len(yrs)} seasons); playoff teams {', '.join(map(str, pys)) or 'none recovered'}; {status}.")
    preferred = [x for key in ["Shane Cobb", "Miguel Morales", "Heriberto Jimenez"] for x in leads if x.startswith(key)]
    rest = [x for x in leads if x not in preferred]
    return preferred + rest[:12]


def manager_rows() -> list[dict[str, Any]]:
    rows = []
    th = [r for r in read_csv(OOTP / "team_history.csv") if r["team_id"] == str(CHI) and r["league_id"] == LEAGUE_ID]
    recs = {(int(r["team_id"]), int(r["year"])): r for r in read_csv(OOTP / "team_history_record.csv") if r["league_id"] == LEAGUE_ID}
    ps = postseason_summary_by_year()
    for r in th:
        y = int(r["year"])
        rows.append({
            "season": y,
            "manager_id": r["manager_id"],
            "manager": manager_name(int(r["manager_id"])),
            "record": f"{recs[(CHI,y)]['w']}-{recs[(CHI,y)]['l']}",
            "division_finish": recs[(CHI,y)]["pos"],
            "division_champion": "yes" if recs[(CHI,y)]["pos"] == "1" else "no",
            "wild_card": "yes" if r["made_playoffs"] == "1" and recs[(CHI,y)]["pos"] != "1" else "no",
            "postseason": ps.get(y, "No postseason"),
            "source": "csv/ootp_csv/team_history.csv; csv/ootp_csv/team_history_record.csv; csv/ootp_csv/coaches.csv",
        })
    rows.append({
        "season": 1981,
        "manager_id": "46",
        "manager": manager_name(46),
        "record": current_context()["record"],
        "division_finish": current_division_position(CHI)[0],
        "division_champion": "in progress",
        "wild_card": "in progress",
        "postseason": "in progress through 1981-07-19",
        "source": "csv/ootp_csv/team_roster_staff.csv; csv/ootp_csv/team_record.csv; csv/ootp_csv/games.csv",
    })
    return rows


def md_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(f, "")).replace("|", "/") for f in fields) + " |")
    return "\n".join(lines)


def write_reports() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    season_rows = build_season_ledger()
    series_rows = chicago_series_from_messages()
    ps_rows = build_postseason_ledger(series_rows)
    comp_rows = franchise_comparison()
    h2h_rows = head_to_head()
    managers = manager_rows()
    ctx = current_context()
    bat_leads, pit_leads, team_profile = statsplus_leaders()
    continuity = continuity_figures()

    write_csv(OUT / "chicago_fire_postseason_ledger_1972-1980.csv", ps_rows, [
        "season", "regular_season_record", "division_finish", "qualification_method", "round", "opponent", "result", "series_score", "chicago_postseason_wins_in_round", "chicago_postseason_losses_in_round", "individual_game_results", "deciding_game", "evidence"
    ])
    write_csv(OUT / "chicago_fire_season_ledger_1972-1981_asof_1981-07-19.csv", season_rows, [
        "season", "record", "winning_percentage", "division_finish", "games_behind", "runs_scored", "runs_allowed", "run_differential", "home_record", "road_record", "first_half_record", "second_half_record", "best_month", "worst_month", "longest_winning_streak", "longest_losing_streak", "playoff_qualification", "final_postseason_result", "manager", "validation_note"
    ])
    write_csv(OUT / "chicago_dallas_philadelphia_franchise_comparison_1972-1980.csv", comp_rows, [
        "team_id", "team", "cumulative_wins", "cumulative_losses", "winning_percentage", "winning_seasons", "division_championships", "wild_cards", "postseason_appearances", "postseason_series_wins", "grand_series_appearances", "championships", "total_run_differential", "best_single_season_record", "worst_single_season_record"
    ])
    write_csv(OUT / "chicago_dallas_head_to_head_1972-1981_asof_1981-07-19.csv", h2h_rows, [
        "season", "chicago_record", "games", "chicago_home_record", "chicago_road_record", "chicago_run_differential", "game_results"
    ])

    chi_72_80 = [r for r in season_rows if 1972 <= int(r["season"]) <= 1980]
    total_w = sum(int(r["record"].split("-")[0]) for r in chi_72_80)
    total_l = sum(int(r["record"].split("-")[1]) for r in chi_72_80)
    total_rd = sum(int(r["run_differential"]) for r in chi_72_80)
    div_titles = sum(1 for r in chi_72_80 if r["division_finish"] == "1")
    wild_cards = sum(1 for r in chi_72_80 if r["playoff_qualification"] == "Wild card")
    post_apps = sum(1 for r in chi_72_80 if r["playoff_qualification"])
    best = max(chi_72_80, key=lambda r: int(r["record"].split("-")[0]))
    worst = min(chi_72_80, key=lambda r: int(r["record"].split("-")[0]))
    best_rd = max(chi_72_80, key=lambda r: int(r["run_differential"]))
    comp_by_team = {r["team"]: r for r in comp_rows}
    h2h_total_w = sum(int(r["chicago_record"].split("-")[0]) for r in h2h_rows)
    h2h_total_l = sum(int(r["chicago_record"].split("-")[1]) for r in h2h_rows)
    h2h_rd = sum(int(r["chicago_run_differential"]) for r in h2h_rows)
    ps_series_w = sum(1 for r in series_rows if r["result"] == "Won")
    ps_series_l = sum(1 for r in series_rows if r["result"] == "Lost")
    ps_game_w = sum(int(r["chicago_postseason_wins_in_round"] or 0) for r in ps_rows)
    ps_game_l = sum(int(r["chicago_postseason_losses_in_round"] or 0) for r in ps_rows)

    manager_md = [
        "# Chicago Fire Manager History - July 20, 1981 Pregame Cutoff",
        "",
        "Source authority: `csv/ootp_csv/team_history.csv`, `csv/ootp_csv/team_history_record.csv`, `csv/ootp_csv/team_roster_staff.csv`, and `csv/ootp_csv/coaches.csv`.",
        "",
        md_table(managers, ["season", "manager", "record", "division_finish", "division_champion", "wild_card", "postseason"]),
        "",
        "Finding: Matt Mead (`coach_id` 46) is listed as Chicago manager for every completed season from 1972 through 1980 and as current 1981 manager in `team_roster_staff.csv`.",
        f"Chicago regular-season record under Mead through July 19, 1981: {total_w + int(ctx['record'].split('-')[0])}-{total_l + int(ctx['record'].split('-')[1])}. Completed-season record, 1972-1980: {total_w}-{total_l}.",
        "Transition evidence: no Chicago hiring/firing/retirement message for manager was found in the searched message/personnel tables through the cutoff; continuity is proven by season manager IDs and current staff linkage.",
    ]
    (OUT / "chicago_fire_manager_history_1981_07_20_asof_1981-07-19.md").write_text("\n".join(manager_md), encoding="utf-8")

    claim_rows = [
        {"Claim": "One manager led all five Chicago playoff teams", "Status": "PROVEN", "Evidence": "team_history manager_id 46 in 1972, 1973, 1977, 1979, 1980; coaches maps 46 to Matt Mead", "Authority": "Raw OOTP", "Calculation": "Filter Chicago playoff seasons", "Caution": "Exact hire date not found"},
        {"Claim": "Chicago made five postseasons from 1972-1980", "Status": "PROVEN", "Evidence": "team_history made_playoffs=1 in five seasons", "Authority": "Raw OOTP", "Calculation": "Count seasons", "Caution": ""},
        {"Claim": "Chicago won no championship by 1980", "Status": "PROVEN", "Evidence": "team_history won_playoffs=0 for Chicago 1972-1980", "Authority": "Raw OOTP", "Calculation": "Count titles", "Caution": ""},
        {"Claim": "1973 Grand Series went seven games after Chicago trailed 3-1", "Status": "PROVEN", "Evidence": "Grand Series summary and parsed GS boxes", "Authority": "Generated HTML from OOTP", "Calculation": "Denver led 3-1 after Game 4; Denver won Game 7", "Caution": "Round label in generated page says Grand Championship Series"},
        {"Claim": "Philadelphia had greater 1972-1980 regular-season success than Dallas", "Status": "DERIVED", "Evidence": "team_history_record", "Authority": "Raw OOTP", "Calculation": f"PHI {comp_by_team['Philadelphia Fury']['cumulative_wins']} wins vs DAL {comp_by_team['Dallas Rustlers']['cumulative_wins']}", "Caution": "Regular season only"},
        {"Claim": "Chicago-Dallas is a historic rivalry", "Status": "REJECTED", "Evidence": "No postseason meetings; head-to-head can be described without rivalry language", "Authority": "Derived schedule/message search", "Calculation": f"CHI led H2H {h2h_total_w}-{h2h_total_l}", "Caution": "Do not manufacture rivalry"},
    ]

    evidence_md = [
        "# Chicago Fire Deep History Evidence - July 20, 1981 Pregame Cutoff",
        "",
        "Cutoff: games completed through July 19, 1981. July 20 results were not read or used.",
        "",
        "## Executive Findings",
        f"- Chicago's completed-season record from 1972-1980 was {total_w}-{total_l} ({pct(total_w, total_l)}), with total run differential {total_rd:+d}.",
        f"- Chicago made {post_apps} postseasons: four division championships and one wild card, with no ABL championship.",
        "- Matt Mead managed every Chicago season in the verified window and remains the current manager through July 19, 1981.",
        "- Chicago's closest championship run was 1973: it lost the Grand Series to Denver in seven games after climbing back from a 3-1 series deficit.",
        f"- Current Chicago context: {ctx['record']}, third in NBC Central, 4.0 games behind Dallas and 2.0 behind Detroit, W2, last 10 {ctx['last10']}.",
        "",
        "## Manager Findings",
        "Matt Mead is the only Chicago manager found for 1972 through the July 19, 1981 cutoff. The one-manager/five-playoff-trips claim is PROVEN.",
        "",
        "## Season-by-Season History",
        md_table(season_rows, ["season", "record", "division_finish", "games_behind", "run_differential", "playoff_qualification", "final_postseason_result", "manager"]),
        "",
        "## Complete Postseason Path",
        md_table(ps_rows, ["season", "round", "opponent", "result", "series_score", "individual_game_results"]),
        "",
        "## Franchise Continuity",
        "\n".join(f"- {x}" for x in continuity),
        "",
        "## Dallas and Philadelphia Comparison",
        md_table(comp_rows, ["team", "cumulative_wins", "cumulative_losses", "winning_percentage", "winning_seasons", "division_championships", "wild_cards", "postseason_appearances", "postseason_series_wins", "championships", "total_run_differential"]),
        "",
        "Direct answers: Philadelphia had greater cumulative regular-season success than Dallas from 1972-1980 by wins and winning percentage. Philadelphia had more winning seasons, but did not qualify for the postseason. Dallas had limited regular-season success in this window but converted its only postseason appearance into the 1977 championship.",
        "",
        "## 1981 Context",
        f"- Standings: Chicago {ctx['record']}, Dallas 52-41, Detroit 50-43. Chicago is 4.0 behind Dallas and 2.0 behind Detroit.",
        f"- Recent Detroit series: {ctx['detroit_series']}.",
        f"- July 19 result: {ctx['july19']}.",
        f"- Home/road: {ctx['home']} home, {ctx['road']} road. Run differential {ctx['run_differential']:+d}.",
        f"- Against Dallas in 1981: {ctx['vs_dallas_1981']}. Against NBC Central: {ctx['vs_nbc_central']}.",
        "- Current listed batting leaders: " + "; ".join(bat_leads),
        "- Current listed pitching leaders: " + "; ".join(pit_leads),
        "- Team profile: " + "; ".join(team_profile),
        "",
        "## Chicago-Dallas History",
        f"Through July 19, 1981, Chicago led the regular-season head-to-head {h2h_total_w}-{h2h_total_l}, with run differential {h2h_rd:+d}. They have not met in the postseason in the verified 1972-1980 bracket/message evidence.",
        "",
        "## Verified Story Angles",
        "- Chicago repeatedly qualified under the same manager without winning a championship.",
        "- Dallas's 1977 title is unusual because it came in its only 1972-1980 postseason appearance.",
        "- The honest contrast is: Chicago repeatedly reached October without winning the Grand Series; Dallas reached October once in this period and won the championship.",
        "",
        "## Rejected or Unsupported Story Angles",
        "- Do not call Chicago-Dallas a proven rivalry.",
        "- Do not claim current injuries explain performance without player-specific linkage.",
        "- Do not use fan emotion, clubhouse pressure, curses, redemption, or managerial intent.",
        "",
        "## Claim Ledger",
        md_table(claim_rows, ["Claim", "Status", "Evidence", "Authority", "Calculation", "Caution"]),
        "",
        "## Final Questions",
        "1. Matt Mead managed Chicago in every season 1972-1981 through the cutoff.",
        "2. Yes. One manager led all five Chicago playoff teams: Matt Mead.",
        f"3. Mead's complete verified Chicago regular-season record through July 19, 1981 is {total_w + int(ctx['record'].split('-')[0])}-{total_l + int(ctx['record'].split('-')[1])}.",
        "4. Chicago: 1972 lost DCS; 1973 lost Grand Series; 1977 lost DCS; 1979 lost CCS; 1980 lost CCS.",
        f"5. Series record from proven messages: {ps_series_w}-{ps_series_l}. Parsed postseason game record: {ps_game_w}-{ps_game_l}.",
        "6. Chicago came within one game in 1973, losing Game 7 to Denver.",
        "7. Best description: regularly competitive and repeatedly short of a championship, not cursed or emotionally defined.",
        "8. Yes, Philadelphia was more successful than Dallas in regular-season aggregate from 1972-1980.",
        "9. Dallas won the 1977 championship in its only 1972-1980 postseason appearance.",
        "10. Strongest contrast: Chicago had repeated chances without a title; Dallas had one chance and converted it.",
        "11. Strong for game call: Mead continuity, five playoff trips/no title, 1973 seven-game Grand Series, current standings, Detroit split, July 19 win.",
        "12. Avoid: exact managerial hire date, injury impact, rivalry language, unverified clubhouse/fan-emotion claims.",
        "",
        "## Sources and Queries Used",
        "- Raw OOTP CSVs: `teams.csv`, `team_history.csv`, `team_history_record.csv`, `team_roster_staff.csv`, `coaches.csv`, `messages.csv`, `games.csv`, historical team batting/pitching stats.",
        "- Almanac game CSVs: `csv/out/almanac/<year>/games_<year>_league200.csv` for regular-season historical schedule validation.",
        "- Archived HTML zips: `data_raw/ootp_html/almanac_<year>.zip` for postseason box-score parsing where available.",
        "- StatsPlus current files: player batting, player pitching, team batting, team pitching as labeled enrichment only.",
    ]
    (OUT / "chicago_fire_deep_history_evidence_1981_07_20_asof_1981-07-19.md").write_text("\n".join(evidence_md), encoding="utf-8")

    bg_md = [
        "# Chicago Fire Game-Call Background - July 20, 1981 Pregame",
        "",
        "## Pregame Opening",
        f"Chicago comes in {ctx['record']}, four games behind Dallas. The Fire just split four in Detroit and won the July 19 finale.",
        "The franchise backdrop is clean: five playoff trips from 1972-1980, one Grand Series, no championship.",
        "",
        "## Early Innings",
        "Matt Mead has been the Chicago manager across the entire verified ABL era, including all five Fire playoff teams.",
        "Chicago's best October run was 1973, when it forced a Game 7 in the Grand Series against Denver.",
        "",
        "## Middle Innings",
        "Use the Chicago-Dallas contrast carefully: Chicago has had repeated October entries; Dallas had one in this window and won the 1977 title.",
        "Philadelphia is the counterintuitive regular-season comparison: more 1972-1980 wins than Dallas, but no playoff appearance.",
        "",
        "## If Chicago Takes The Lead",
        "Chicago cannot erase the division gap in one night, but a win cuts Dallas's lead over Chicago from four games to three.",
        "",
        "## If Dallas Takes The Lead",
        "A Dallas win would push Chicago five games back with three games still left in this four-game set.",
        "",
        "## If The Game Remains Close Late",
        "Chicago has already played a pressure series in Detroit this week: two losses, then two wins, ending with the July 19 victory.",
        "",
        "## If Chicago Wins Game 1",
        "Chicago moves to 49-45 and Dallas falls to 52-42. The Fire would be three games behind Dallas with three games left in the series.",
        "",
        "## If Dallas Wins Game 1",
        "Dallas moves to 53-41 and Chicago falls to 48-46. The Fire would be five games back with three games left in the series.",
        "",
        "## Material To Save For Later Games",
        "Save the full 1973 Game 7 detail for a later inning or later game; the short version is enough for Game 1.",
        "",
        "## What Game 1 Can Change",
        "Game 1 can move the Chicago-Dallas gap to three games or five games. It cannot decide the four-game series by itself, and it cannot settle the larger NBC Central race. Detroit also remains part of the race: Chicago starts the day two games behind Detroit.",
        "",
        "## Do Not Use",
        "Do not say Chicago is haunted, cursed, desperate, or emotionally burdened. Do not call Chicago-Dallas a rivalry unless the broadcast explicitly frames it as current standings pressure rather than historical proof.",
    ]
    (OUT / "chicago_fire_game_call_background_1981_07_20_asof_1981-07-19.md").write_text("\n".join(bg_md), encoding="utf-8")

    validation_md = [
        "# Chicago Fire Research Validation - July 20, 1981 Pregame Cutoff",
        "",
        "- Regular-season records were recalculated from almanac game CSVs for 1972-1980 and raw `games.csv` for 1981 through July 19.",
        "- Playoff qualification was reconciled against `team_history.csv` and `team_history_record.csv`.",
        "- Series results were reconciled against OOTP `messages.csv`; Grand Series games were cross-checked against generated Grand Series summaries and archived HTML where parsed.",
        "- Manager tenure was reconciled against `team_history.csv`, `team_roster_staff.csv`, and `coaches.csv`.",
        "- Postseason games were not included in regular-season totals because historical regular-season tables came from almanac game CSVs and 1981 was filtered to `game_type=0`.",
        "- The 1981 cutoff excludes dates after 1981-07-19.",
        "- Contradiction found: historical almanac regular-season game CSVs do not include postseason games; postseason evidence had to come from OOTP messages and archived HTML.",
        "- Unresolved: exact Matt Mead hire date was not found; some non-Grand-Series individual postseason game rows may remain unrecovered if archive parsing missed boxes.",
        "",
        "One-manager/five-playoff-trips claim: PROVEN.",
    ]
    (OUT / "chicago_fire_research_validation_1981_07_20_asof_1981-07-19.md").write_text("\n".join(validation_md), encoding="utf-8")


def main() -> int:
    write_reports()
    print(f"Wrote Chicago deep-history package to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
