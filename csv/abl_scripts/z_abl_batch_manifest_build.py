from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

from abl_path_policy import (
    is_authoritative_ootp_csv,
    is_immediate_sortable_csv,
    validate_output_paths,
)


ROOT = Path(__file__).resolve().parents[2]
CONTROL = ROOT / "csv" / "out" / "control"
DEFAULT_REGISTRY = CONTROL / "abl_source_registry.csv"
FIELDS = [
    "source_file_path", "source_family", "authority_level", "file_size_bytes",
    "modified_timestamp", "row_count", "column_count", "sha256",
    "can_drive_current_state", "can_support_current_state",
    "latest_game_date_detected", "games_per_team_detected",
    "sortable_promotion_status", "schema_drift_status",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_shape(path: Path) -> tuple[int, int]:
    csv.field_size_limit(100_000_000)
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        lines = (line for line in handle if line.strip() and not line.lstrip().startswith("#"))
        reader = csv.reader(lines)
        header = next(reader, [])
        count = sum(1 for row in reader if row and any(cell.strip() for cell in row))
    return count, len(header)


def parse_iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected YYYY-MM-DD date, got {value!r}") from exc


def parse_game_date(value: str) -> date:
    """Parse OOTP dates, which may omit leading zeroes."""

    return datetime.strptime(value, "%Y-%m-%d").date()


def resolve_repo_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def select_governed_sources(
    registry: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Select only immediate promoted CSV inputs, independent of stale labels."""

    governed: list[dict[str, str]] = []
    for entry in registry:
        rel = entry["file_path"]
        if is_authoritative_ootp_csv(rel):
            normalized = dict(entry)
            normalized["source_family"] = "ootp_csv"
            normalized["authority_level"] = "system_of_record_extract"
            governed.append(normalized)
        elif is_immediate_sortable_csv(rel):
            normalized = dict(entry)
            normalized["source_family"] = "sortable_stats"
            normalized["authority_level"] = "supplemental_report_extract"
            governed.append(normalized)
    return governed


def select_completed_games(
    rows: list[dict[str, str]], season: int, as_of_date: date, league_id: str
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for row in rows:
        if row.get("league_id") != league_id or row.get("game_type") != "0" or row.get("played") != "1":
            continue
        game_date = parse_game_date(row["date"])
        if game_date.year == season and game_date <= as_of_date:
            selected.append(row)
    return selected


def game_context(
    season: int, as_of_date: date, league_id: str = "200"
) -> tuple[dict[str, dict[str, str]], set[str]]:
    games_path = ROOT / "csv" / "ootp_csv" / "games.csv"
    with games_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    completed = select_completed_games(rows, season, as_of_date, league_id)
    counts: Counter[str] = Counter()
    for row in completed:
        counts[row["home_team"]] += 1
        counts[row["away_team"]] += 1
    dates = [parse_game_date(row["date"]) for row in completed]
    latest = max(dates).isoformat() if dates else ""
    games_range = f"{min(counts.values())}-{max(counts.values())}" if counts else ""
    context = {
        name: {"latest": latest, "games": games_range}
        for name in ("games.csv", "games_score.csv", "game_logs.csv")
    }
    return context, {row["game_id"] for row in completed}


def output_stem(output_dir: Path, season: int, as_of_date: date) -> Path:
    return output_dir / f"abl_batch_manifest_{season}_asof_{as_of_date.isoformat()}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a dated ABL input batch manifest with an explicit cutoff."
    )
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--as-of-date", type=parse_iso_date, required=True)
    parser.add_argument("--newsroom-date", type=parse_iso_date)
    parser.add_argument("--league-id", default="200")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--promotion-verification", type=Path)
    parser.add_argument("--output-dir", type=Path, default=CONTROL)
    args = parser.parse_args()
    if args.as_of_date.year != args.season:
        parser.error("--as-of-date year must match --season")
    if args.newsroom_date and args.newsroom_date < args.as_of_date:
        parser.error("--newsroom-date cannot precede --as-of-date")
    return args


def main() -> int:
    args = parse_args()
    newsroom_date = args.newsroom_date or args.as_of_date
    registry_path = resolve_repo_path(args.registry)
    output_dir = resolve_repo_path(args.output_dir)
    promotion_path = resolve_repo_path(
        args.promotion_verification
        or Path(
            "csv/out/control/"
            f"sortable_promotion_verification_{args.season}_asof_{args.as_of_date.isoformat()}.csv"
        )
    )
    stem = output_stem(output_dir, args.season, args.as_of_date)
    outputs = tuple(stem.with_suffix(ext) for ext in (".csv", ".json", ".md"))
    validate_output_paths(outputs, ROOT)

    with registry_path.open("r", encoding="utf-8-sig", newline="") as handle:
        registry = list(csv.DictReader(handle))
    governed = select_governed_sources(registry)
    if promotion_path.exists():
        with promotion_path.open("r", encoding="utf-8-sig", newline="") as handle:
            promoted = {row["official_file"]: row for row in csv.DictReader(handle)}
    else:
        promoted = {}

    context, completed_ids = game_context(args.season, args.as_of_date, args.league_id)
    driver_ids: dict[str, set[str]] = {}
    for name in ("games_score.csv", "game_logs.csv"):
        path = ROOT / "csv" / "ootp_csv" / name
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            driver_ids[name] = {row["game_id"] for row in csv.DictReader(handle)}

    rows: list[dict[str, object]] = []
    for entry in sorted(governed, key=lambda row: row["file_path"]):
        rel = entry["file_path"]
        path = ROOT / rel
        if not path.exists():
            raise FileNotFoundError(f"Registered capture source missing: {path}")
        row_count, column_count = csv_shape(path)
        name = path.name
        promotion = promoted.get(rel)
        latest = context.get(name, {}).get("latest", "")
        games_range = context.get(name, {}).get("games", "")
        if name in driver_ids and not completed_ids.issubset(driver_ids[name]):
            latest = "INCOMPLETE_GAME_ID_COVERAGE"
            games_range = "INCOMPLETE"
        rows.append({
            "source_file_path": rel,
            "source_family": entry["source_family"],
            "authority_level": entry["authority_level"],
            "file_size_bytes": path.stat().st_size,
            "modified_timestamp": datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
            "row_count": row_count,
            "column_count": column_count,
            "sha256": sha256(path),
            "can_drive_current_state": entry["can_drive_current_state"],
            "can_support_current_state": entry["can_support_current_state"],
            "latest_game_date_detected": latest,
            "games_per_team_detected": games_range,
            "sortable_promotion_status": (
                "PROMOTED_VERIFIED"
                if promotion and promotion["verification_status"] == "PASS"
                else ("not_applicable" if entry["source_family"] != "sortable_stats" else "UNVERIFIED")
            ),
            "schema_drift_status": promotion["schema_status"] if promotion else "not_applicable",
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    with outputs[0].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    family_counts = Counter(row["source_family"] for row in rows)
    batch_id = f"capture_{args.season}_asof_{args.as_of_date.isoformat()}"
    payload = {
        "batch_id": batch_id,
        "season": args.season,
        "newsroom_date": newsroom_date.isoformat(),
        "target_as_of_date": args.as_of_date.isoformat(),
        "entry_count": len(rows),
        "counts_by_source_family": dict(family_counts),
        "files": rows,
    }
    outputs[1].write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        f"# ABL Batch Manifest — {args.season} as of {args.as_of_date.isoformat()}",
        "",
        f"- Batch: `{batch_id}`",
        f"- Newsroom date: `{newsroom_date.isoformat()}`",
        f"- Target as-of date: `{args.as_of_date.isoformat()}`",
        f"- Entries: **{len(rows)}**",
        f"- Raw OOTP files: **{family_counts['ootp_csv']}**",
        f"- Sortable-stat files: **{family_counts['sortable_stats']}**",
        "",
        "Only immediate promoted CSV inputs are capture entries. Nested content beneath protected input roots is excluded.",
        "",
        "| Source | Family | Authority | Rows | Columns | Bytes | Current driver | Current support | Latest date | Games/team | Promotion | Drift |",
        "|---|---|---|---:|---:|---:|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['source_file_path']}` | `{row['source_family']}` | "
            f"`{row['authority_level']}` | {row['row_count']} | {row['column_count']} | "
            f"{row['file_size_bytes']} | {row['can_drive_current_state']} | "
            f"{row['can_support_current_state']} | {row['latest_game_date_detected'] or 'n/a'} | "
            f"{row['games_per_team_detected'] or 'n/a'} | "
            f"`{row['sortable_promotion_status']}` | `{row['schema_drift_status']}` |"
        )
    outputs[2].write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "batch_id": batch_id,
        "entry_count": len(rows),
        "counts_by_source_family": dict(family_counts),
        "outputs": [str(path) for path in outputs],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
