from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from bs4 import BeautifulSoup

CSV_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CSV_ROOT.parent
DEFAULT_INDEX = Path("data_raw/ootp_html/history/league_200_all_managers_index.html")
DEFAULT_LEADERS = Path("data_raw/ootp_html/history/league_200_all_managers_leaders.html")
DEFAULT_OUT_CSV = Path("out/csv_out/abl_managers_summary.csv")
DEFAULT_OUT_TXT = Path("out/text_out/abl_managers_top10.txt")
COACHES_FILE = REPO_ROOT / "csv" / "ootp_csv" / "coaches.csv"


@dataclass
class ManagerRecord:
    name: str
    team: Optional[str] = None
    wins: Optional[int] = None
    losses: Optional[int] = None
    win_pct: Optional[float] = None
    championships: Optional[int] = None
    detail_path: Optional[Path] = None
    source: str = ""

    def as_dict(self) -> Dict[str, Optional[object]]:
        return {
            "name": self.name,
            "team": self.team,
            "wins": self.wins,
            "losses": self.losses,
            "win_pct": self.win_pct,
            "championships": self.championships,
            "detail_path": str(self.detail_path) if self.detail_path else None,
        }


def repo_relative(path_value: str | Path, base: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def normalize_full_name(name: str) -> str:
    return " ".join(name.split()).strip().lower()


def load_soup(path: Path) -> BeautifulSoup:
    with path.open(encoding="utf-8") as handle:
        return BeautifulSoup(handle.read(), "html.parser")


def parse_int(text: str | None) -> Optional[int]:
    if not text:
        return None
    stripped = text.strip().replace(",", "")
    if not stripped:
        return None
    try:
        return int(stripped)
    except ValueError:
        return None


def parse_float(text: str | None) -> Optional[float]:
    if not text:
        return None
    stripped = text.strip().replace(",", "")
    if not stripped:
        return None
    if stripped.startswith("."):
        stripped = f"0{stripped}"
    try:
        return float(stripped)
    except ValueError:
        return None


def detect_manager_template(path: Path, soup: BeautifulSoup) -> str:
    title_text = (soup.title.string or "") if soup.title else ""
    subtitle = soup.select_one("div.repsubtitle")
    subtitle_text = subtitle.get_text(strip=True) if subtitle else ""
    combined = f"{path.name} {title_text} {subtitle_text}".lower()
    if "leader" in combined:
        return "B"
    return "A"


def extract_detail_path(base: Path, link) -> Optional[Path]:
    if link and link.get("href"):
        rel = Path(link["href"])
        return (base.parent / rel).resolve()
    return None


def parse_template_a(path: Path, soup: BeautifulSoup) -> List[ManagerRecord]:
    table = None
    for candidate in soup.find_all("table"):
        headers = [th.get_text(strip=True).lower() for th in candidate.find_all("th")]
        if headers and headers[0].startswith("manager"):
            table = candidate
            break
    if not table:
        raise RuntimeError(f"Unable to find manager table in {path}")

    records: List[ManagerRecord] = []
    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 11:
            continue
        name = cells[0].get_text(strip=True)
        if not name or "manager" in name.lower():
            continue
        link = cells[0].find("a")
        detail_path = extract_detail_path(path, link)
        records.append(
            ManagerRecord(
                name=name,
                team=None,
                wins=parse_int(cells[4].get_text()),
                losses=parse_int(cells[5].get_text()),
                win_pct=parse_float(cells[6].get_text()),
                championships=parse_int(cells[10].get_text()),
                detail_path=detail_path,
                source="A",
            )
        )
    return records


WIN_PCT_REGEX = re.compile(r"([+-]?[0-9]*\.?[0-9]+)")
PCT_PARENS_REGEX = re.compile(r"\(([-+]?[0-9]*\.?[0-9]+)\)")
RECORD_REGEX = re.compile(r"(\d+)\s*-\s*(\d+)")


def parse_value_with_pct(text: str) -> Tuple[Optional[int], Optional[float]]:
    parts = text.split()[0:1]
    wins = parse_int(parts[0]) if parts else None
    pct_match = PCT_PARENS_REGEX.search(text)
    pct = parse_float(pct_match.group(1)) if pct_match else None
    return wins, pct


def parse_pct_with_record(text: str) -> Tuple[Optional[float], Optional[int], Optional[int]]:
    pct_match = WIN_PCT_REGEX.search(text)
    pct = parse_float(pct_match.group(1)) if pct_match else None
    record_match = RECORD_REGEX.search(text)
    wins = parse_int(record_match.group(1)) if record_match else None
    losses = parse_int(record_match.group(2)) if record_match else None
    return pct, wins, losses


def parse_template_b(path: Path, soup: BeautifulSoup) -> List[ManagerRecord]:
    records: Dict[str, ManagerRecord] = {}
    tables = soup.find_all("table")
    for table in tables:
        classes = table.get("class", [])
        if "data" not in classes:
            continue
        headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        if len(headers) < 2 or not headers[0].startswith("name"):
            continue
        metric = headers[1]
        metric_type = None
        if "win%" in metric:
            metric_type = "win_pct"
        elif "w >" in metric:
            metric_type = "record"
        elif "loss" in metric:
            metric_type = "losses"
        elif "win" in metric:
            metric_type = "wins"
        else:
            continue
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 2:
                continue
            name = cells[0].get_text(strip=True)
            if not name:
                continue
            link = cells[0].find("a")
            detail_path = extract_detail_path(path, link)
            key = normalize_full_name(name)
            record = records.setdefault(
                key,
                ManagerRecord(name=name, detail_path=detail_path, source="B"),
            )
            if record.detail_path is None and detail_path is not None:
                record.detail_path = detail_path
            value_text = cells[1].get_text(" ", strip=True)
            if metric_type == "wins":
                wins, pct = parse_value_with_pct(value_text)
                record.wins = record.wins or wins
                if pct is not None:
                    record.win_pct = record.win_pct or pct
            elif metric_type == "losses":
                losses, pct = parse_value_with_pct(value_text)
                record.losses = record.losses or losses
                if pct is not None:
                    record.win_pct = record.win_pct or pct
            elif metric_type == "win_pct":
                pct, wins_val, losses_val = parse_pct_with_record(value_text)
                if pct is not None:
                    record.win_pct = record.win_pct or pct
                if wins_val is not None:
                    record.wins = record.wins or wins_val
                if losses_val is not None:
                    record.losses = record.losses or losses_val
            elif metric_type == "record":
                _, wins_val, losses_val = parse_pct_with_record(value_text)
                if wins_val is not None:
                    record.wins = record.wins or wins_val
                if losses_val is not None:
                    record.losses = record.losses or losses_val
    return list(records.values())


def merge_records(sources: List[List[ManagerRecord]]) -> Tuple[List[ManagerRecord], List[str]]:
    merged: Dict[str, ManagerRecord] = {}
    conflicts: List[str] = []
    for source_records in sources:
        for rec in source_records:
            key = normalize_full_name(rec.name)
            if not key:
                continue
            if key not in merged:
                merged[key] = rec
                continue
            existing = merged[key]
            for field in ["wins", "losses", "win_pct", "championships", "team"]:
                current = getattr(existing, field)
                new_val = getattr(rec, field)
                if new_val is None or new_val == "":
                    continue
                if current is None:
                    setattr(existing, field, new_val)
                else:
                    if field in {"wins", "losses", "championships"} and current != new_val:
                        conflicts.append(f"{rec.name}: field {field} mismatch ({current} vs {new_val})")
                    if field == "win_pct" and not math.isclose(current, new_val, rel_tol=1e-4, abs_tol=1e-4):
                        conflicts.append(f"{rec.name}: win_pct mismatch ({current} vs {new_val})")
            if existing.detail_path is None and rec.detail_path is not None:
                existing.detail_path = rec.detail_path
    return list(merged.values()), conflicts


def recompute_win_pct(records: Iterable[ManagerRecord]) -> None:
    for rec in records:
        if rec.wins is not None and rec.losses is not None:
            total = rec.wins + rec.losses
            if total > 0:
                rec.win_pct = rec.wins / total


def validate_records(records: Iterable[ManagerRecord]) -> None:
    for rec in records:
        if rec.wins is not None and rec.losses is not None and rec.win_pct is not None:
            total = rec.wins + rec.losses
            if total > 0:
                calc = rec.wins / total
                if abs(calc - rec.win_pct) >= 1e-4:
                    raise ValueError(f"Win% mismatch for {rec.name}: calc={calc:.3f} file={rec.win_pct:.3f}")


def load_coach_lookup(path: Path) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    if not path.exists():
        return lookup
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            first = (row.get("first_name") or "").strip()
            last = (row.get("last_name") or "").strip()
            if not first or not last:
                continue
            key = normalize_full_name(f"{first} {last}")
            lookup[key] = (row.get("coach_id") or "").strip()
    return lookup


def collect_detail_map(records: List[ManagerRecord]) -> Dict[str, Dict[str, Optional[object]]]:
    details: Dict[str, Dict[str, Optional[object]]] = {}
    for rec in records:
        path = rec.detail_path
        if not path or not Path(path).exists():
            continue
        details[rec.name] = parse_manager_card(Path(path))
    return details


def find_history_table(soup: BeautifulSoup) -> Optional[BeautifulSoup]:
    for table in soup.find_all("table"):
        headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        if headers and headers[0].startswith("year"):
            return table
    return None


def parse_manager_card(path: Path) -> Dict[str, Optional[object]]:
    soup = load_soup(path)
    title = soup.select_one("div.reptitle")
    name_text = title.get_text(strip=True) if title else ""
    name = name_text.replace("Manager ", "", 1).strip() if name_text else None
    subtitle = soup.select_one("div.repsubtitle a")
    current_team = subtitle.get_text(strip=True) if subtitle else None
    history_table = find_history_table(soup)
    seasons = 0
    wins_total = 0
    losses_total = 0
    titles = 0
    first_season = None
    last_season = None
    if history_table:
        for row in history_table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 10:
                continue
            job = cells[3].get_text(strip=True)
            if "manager" not in job.lower():
                continue
            year = parse_int(cells[0].get_text())
            if year is not None:
                if first_season is None or year < first_season:
                    first_season = year
                if last_season is None or year > last_season:
                    last_season = year
            wins = parse_int(cells[5].get_text()) or 0
            losses = parse_int(cells[6].get_text()) or 0
            postseason = cells[9].get_text(strip=True).lower()
            seasons += 1
            wins_total += wins
            losses_total += losses
            if "won" in postseason:
                titles += 1
    else:
        print(f"[warn] no-season-table for {path}")
    total_games = wins_total + losses_total
    win_pct = round(wins_total / total_games, 3) if total_games else None
    return {
        "name": name,
        "seasons": seasons or None,
        "career_wins": wins_total or None,
        "career_losses": losses_total or None,
        "career_win_pct": win_pct,
        "titles": titles or None,
        "current_team": current_team,
        "first_season": first_season,
        "last_season": last_season,
        "source": str(path),
    }


def format_record(record: Dict[str, Optional[object]]) -> str:
    parts = []
    for key in ["name", "team", "wins", "losses", "win_pct", "championships"]:
        value = record.get(key)
        if key == "win_pct" and isinstance(value, float):
            parts.append(f"{key}={value:.3f}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def sort_key(rec: ManagerRecord) -> Sequence[float]:
    champ = rec.championships if rec.championships is not None else -1
    pct = rec.win_pct if rec.win_pct is not None else -1.0
    return (champ, pct)


def show_samples(records: List[ManagerRecord], count: int = 5) -> None:
    print(f"\nSample {min(count, len(records))} records:")
    for rec in records[:count]:
        print(f" - {format_record(rec.as_dict())}")


def print_top_10(records: Iterable[ManagerRecord]) -> None:
    sorted_records = sorted(records, key=sort_key, reverse=True)[:10]
    print("\nTop 10 by championships, win_pct:")
    for idx, rec in enumerate(sorted_records, 1):
        print(f"{idx:2d}. {format_record(rec.as_dict())}")


def write_csv_output(records: List[ManagerRecord], detail_map: Dict[str, Dict[str, Optional[object]]], coach_lookup: Dict[str, str], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "coach_id",
        "name",
        "team",
        "wins",
        "losses",
        "win_pct",
        "championships",
        "seasons",
        "career_wins",
        "career_losses",
        "career_win_pct",
        "titles",
        "current_team",
        "detail_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            detail = detail_map.get(rec.name, {})
            coach_id = coach_lookup.get(normalize_full_name(rec.name))
            row = {
                "coach_id": coach_id,
                "name": rec.name,
                "team": rec.team or detail.get("current_team"),
                "wins": rec.wins,
                "losses": rec.losses,
                "win_pct": rec.win_pct,
                "championships": rec.championships,
                "seasons": detail.get("seasons"),
                "career_wins": detail.get("career_wins"),
                "career_losses": detail.get("career_losses"),
                "career_win_pct": detail.get("career_win_pct"),
                "titles": detail.get("titles"),
                "current_team": detail.get("current_team"),
                "detail_path": str(rec.detail_path) if rec.detail_path else None,
            }
            writer.writerow(row)


def write_top10_text(records: List[ManagerRecord], detail_map: Dict[str, Dict[str, Optional[object]]], text_path: Path, index_path: Path, leaders_path: Optional[Path], template_counts: Dict[str, int], conflicts: List[str]) -> None:
    text_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("ABL Managers Top 10 (Championships priority, Win% tiebreak)")
    lines.append(f"Source index : {index_path}")
    if leaders_path:
        lines.append(f"Source leaders: {leaders_path}")
    lines.append(f"Template counts: {template_counts}")
    if conflicts:
        lines.append(f"Conflicts: {len(conflicts)} (see console for details)")
    lines.append("")
    header = f"{'Rank':<4} {'Manager':<26} {'Team':<22} {'W':>6} {'L':>6} {'Win%':>7} {'Titles':>7} {'Career%':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for idx, rec in enumerate(sorted(records, key=sort_key, reverse=True)[:10], 1):
        detail = detail_map.get(rec.name, {})
        team = detail.get("current_team") or rec.team or "n/a"
        wins = rec.wins if rec.wins is not None else "--"
        losses = rec.losses if rec.losses is not None else "--"
        win_pct = f"{rec.win_pct:.3f}" if rec.win_pct is not None else "n/a"
        career_pct = detail.get("career_win_pct")
        career_pct_str = f"{career_pct:.3f}" if isinstance(career_pct, float) else "n/a"
        titles = rec.championships if rec.championships is not None else 0
        lines.append(
            f"{idx:<4} {rec.name:<26.26} {team:<22.22} {wins:>6} {losses:>6} {win_pct:>7} {titles:>7} {career_pct_str:>8}"
        )
    lines.append("")
    lines.append("Generated via csv/abl_scripts/parse_managers.py")
    text_path.write_text("\n".join(lines), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse ABL manager index/leaders into CSV/TXT outputs.")
    parser.add_argument("--index", default=str(DEFAULT_INDEX), help="Path to managers index (Template A)")
    parser.add_argument("--leaders", default=str(DEFAULT_LEADERS), help="Path to managers leaders (Template B)")
    parser.add_argument("--out-csv", dest="out_csv", default=str(DEFAULT_OUT_CSV), help="Destination CSV path")
    parser.add_argument("--out-text", dest="out_text", default=str(DEFAULT_OUT_TXT), help="Destination text report")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    index_path = repo_relative(args.index, REPO_ROOT)
    leaders_path = repo_relative(args.leaders, REPO_ROOT) if args.leaders else None
    out_csv_path = repo_relative(args.out_csv, CSV_ROOT)
    out_text_path = repo_relative(args.out_text, CSV_ROOT)

    template_counts: Dict[str, int] = {}
    all_sources: List[List[ManagerRecord]] = []

    if index_path.exists():
        soup_a = load_soup(index_path)
        template = detect_manager_template(index_path, soup_a)
        records_a = parse_template_a(index_path, soup_a)
        template_counts[template] = len(records_a)
        all_sources.append(records_a)
    else:
        raise FileNotFoundError(f"Manager index missing at {index_path}")

    records_b: List[ManagerRecord] = []
    if leaders_path and leaders_path.exists():
        soup_b = load_soup(leaders_path)
        template = detect_manager_template(leaders_path, soup_b)
        records_b = parse_template_b(leaders_path, soup_b)
        template_counts[template] = len(records_b)
        all_sources.append(records_b)
        print(f"Parsed leaders template with {len(records_b)} rows")
    elif leaders_path:
        print(f"Leaders file not found at {leaders_path}, skipping.")

    merged_records, conflicts = merge_records(all_sources)
    merged_records = sorted(merged_records, key=lambda r: r.name)
    recompute_win_pct(merged_records)
    validate_records(merged_records)

    print(f"Template counts: {template_counts}")
    print(f"Merged unique managers: {len(merged_records)}")
    if conflicts:
        print("Conflicts detected:")
        for conflict in conflicts:
            print(f" - {conflict}")
    else:
        print("No field conflicts detected between templates.")

    show_samples(merged_records, count=5)
    print_top_10(merged_records)

    detail_map = collect_detail_map(merged_records)
    coach_lookup = load_coach_lookup(COACHES_FILE)

    print(f"Writing CSV output to {out_csv_path}")
    write_csv_output(merged_records, detail_map, coach_lookup, out_csv_path)

    print(f"Writing text output to {out_text_path}")
    write_top10_text(merged_records, detail_map, out_text_path, index_path, leaders_path, template_counts, conflicts)


if __name__ == "__main__":
    main()


