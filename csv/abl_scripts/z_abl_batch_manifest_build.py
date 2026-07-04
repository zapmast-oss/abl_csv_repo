from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONTROL = ROOT / "csv" / "out" / "control"
REGISTRY = CONTROL / "abl_source_registry.csv"
PROMOTION_VERIFICATION = CONTROL / "sortable_promotion_verification_1981_asof_1981-07-19.csv"
OUT_STEM = CONTROL / "abl_batch_manifest_current"
TARGET_DATE = "1981-07-19"
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


def parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def game_context() -> tuple[dict[str, dict[str, str]], set[str]]:
    games_path = ROOT / "csv" / "ootp_csv" / "games.csv"
    with games_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    completed = [
        row for row in rows
        if row["league_id"] == "200" and row["game_type"] == "0" and row["played"] == "1"
    ]
    counts: Counter[str] = Counter()
    for row in completed:
        counts[row["home_team"]] += 1; counts[row["away_team"]] += 1
    dates = [parse_date(row["date"]) for row in completed]
    latest = max(dates).date().isoformat() if dates else ""
    games_range = f"{min(counts.values())}-{max(counts.values())}" if counts else ""
    context = {
        name: {"latest": latest, "games": games_range}
        for name in ("games.csv", "games_score.csv", "game_logs.csv")
    }
    return context, {row["game_id"] for row in completed}


def main() -> int:
    with REGISTRY.open("r", encoding="utf-8-sig", newline="") as handle:
        registry = list(csv.DictReader(handle))
    governed = [row for row in registry if row["source_family"] in {"ootp_csv", "sortable_stats"}]
    if PROMOTION_VERIFICATION.exists():
        with PROMOTION_VERIFICATION.open("r", encoding="utf-8-sig", newline="") as handle:
            promoted = {row["official_file"]: row for row in csv.DictReader(handle)}
    else:
        promoted = {}
    context, completed_ids = game_context()
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
            raise FileNotFoundError(f"Registered current-capture source missing: {path}")
        row_count, column_count = csv_shape(path)
        name = path.name
        promotion = promoted.get(rel)
        latest = context.get(name, {}).get("latest", "")
        games_range = context.get(name, {}).get("games", "")
        if name in driver_ids and not completed_ids.issubset(driver_ids[name]):
            latest = "INCOMPLETE_GAME_ID_COVERAGE"
            games_range = "INCOMPLETE"
        rows.append({
            "source_file_path": rel, "source_family": entry["source_family"],
            "authority_level": entry["authority_level"], "file_size_bytes": path.stat().st_size,
            "modified_timestamp": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
            "row_count": row_count, "column_count": column_count, "sha256": sha256(path),
            "can_drive_current_state": entry["can_drive_current_state"],
            "can_support_current_state": entry["can_support_current_state"],
            "latest_game_date_detected": latest, "games_per_team_detected": games_range,
            "sortable_promotion_status": "PROMOTED_VERIFIED" if promotion and promotion["verification_status"] == "PASS" else ("not_applicable" if entry["source_family"] != "sortable_stats" else "UNVERIFIED"),
            "schema_drift_status": promotion["schema_status"] if promotion else "not_applicable",
        })
    OUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    with OUT_STEM.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    family_counts = Counter(row["source_family"] for row in rows)
    payload = {
        "batch_id": "current_capture_1981_asof_1981-07-19",
        "newsroom_date": "1981-07-20", "target_as_of_date": TARGET_DATE,
        "entry_count": len(rows), "counts_by_source_family": dict(family_counts),
        "files": rows,
    }
    OUT_STEM.with_suffix(".json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# ABL Current Batch Manifest", "",
             "- Batch: `current_capture_1981_asof_1981-07-19`",
             "- Newsroom date: `1981-07-20`", "- Target as-of date: `1981-07-19`",
             f"- Entries: **{len(rows)}**", f"- Raw OOTP files: **{family_counts['ootp_csv']}**",
             f"- Sortable-stat files: **{family_counts['sortable_stats']}**", "",
             "This manifest inventories the current raw and promoted sortable capture. Historical, generated, editorial, and documentation files are not capture entries.", "",
             "| Source | Family | Authority | Rows | Columns | Bytes | Current driver | Current support | Latest date | Games/team | Promotion | Drift |",
             "|---|---|---|---:|---:|---:|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| `{row['source_file_path']}` | `{row['source_family']}` | `{row['authority_level']}` | {row['row_count']} | {row['column_count']} | {row['file_size_bytes']} | {row['can_drive_current_state']} | {row['can_support_current_state']} | {row['latest_game_date_detected'] or 'n/a'} | {row['games_per_team_detected'] or 'n/a'} | `{row['sortable_promotion_status']}` | `{row['schema_drift_status']}` |")
    OUT_STEM.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"entry_count": len(rows), "counts_by_source_family": dict(family_counts), "outputs": [str(OUT_STEM.with_suffix(ext)) for ext in ('.csv','.md','.json')]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

