#!/usr/bin/env python3
"""Promote the approved 1981 StatsPlus Feed 3 capture with archive and verification."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SLUG = "1981_asof_1981-07-19"
CONTROL = REPO / "csv" / "out" / "control"
DRY_RUN = CONTROL / f"statsplus_promotion_dry_run_{SLUG}.csv"
SEMANTIC = CONTROL / f"statsplus_table_1_semantic_reconciliation_{SLUG}.json"
CURRENT = REPO / "csv" / "statsplus" / "current"
ARCHIVE = REPO / "csv" / "out" / "archive" / f"statsplus_pre_promotion_{SLUG}"
APPROVED = {
    "PROMOTE_AS_STATSPLUS_CURRENT",
    "PROMOTE_ACCEPT_SCHEMA_DRIFT",
    "PROMOTE_ACCEPT_RESHAPED_SCHEMA",
    "PROMOTE_ACCEPT_ADDED_DETAIL",
    "PROMOTE_REORDERED_OVERLAP",
}
ARCHIVE_FIELDS = [
    "relative_path", "original_current_path", "archive_path", "file_size", "modified_timestamp",
    "checksum_before", "checksum_archive", "archive_status",
]


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_shape(path: Path) -> tuple[int, int]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    headers = rows[0] if rows else []
    data = [row for row in rows[1:] if any(cell.strip() for cell in row)]
    return len(data), len(headers)


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else ["status"])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def md_table(headers: list[str], rows: list[list]) -> str:
    clean = lambda value: str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(clean(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def contained(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def main() -> None:
    dry = read_csv(DRY_RUN)
    semantic = json.loads(SEMANTIC.read_text(encoding="utf-8"))
    approved = [row for row in dry if row["current_action"] in APPROVED]
    excluded = {int(row["table_number"]): row for row in dry if row["current_action"] == "INTENTIONALLY_EXCLUDED"}
    holds = [row for row in dry if row["current_action"].startswith("HOLD_")]
    if len(approved) != 25 or holds or set(excluded) != {6, 27}:
        raise SystemExit(f"Dry-run gate failed: approved={len(approved)}, holds={len(holds)}, exclusions={sorted(excluded)}")

    # Validate every source and intended target before changing current state.
    target_names = set()
    for row in approved:
        source = Path(row["staging_source_path"])
        target = Path(row["proposed_official_target_path"])
        if not source.is_file():
            raise SystemExit(f"Missing staging source: {source}")
        if sha256(source) != row["sha256"]:
            raise SystemExit(f"Staging checksum changed since dry run: {source}")
        if int(row["table_number"]) in {6, 27}:
            raise SystemExit(f"Excluded table selected for promotion: {row['table_number']}")
        if not contained(target, CURRENT) or target.parent.resolve() != CURRENT.resolve():
            raise SystemExit(f"Unsafe or nested target path: {target}")
        if target.name in target_names:
            raise SystemExit(f"Duplicate target filename: {target.name}")
        target_names.add(target.name)

    # Refuse to reuse a populated archive, preventing accidental archive overwrite on rerun.
    if ARCHIVE.exists() and any(ARCHIVE.rglob("*")):
        raise SystemExit(f"Archive folder is already populated; refusing to overwrite: {ARCHIVE}")
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    CURRENT.mkdir(parents=True, exist_ok=True)

    archive_rows = []
    existing = sorted(path for path in CURRENT.rglob("*") if path.is_file())
    archive_sources = ARCHIVE / "current_files"
    for source in existing:
        relative = source.relative_to(CURRENT)
        destination = archive_sources / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        before = sha256(source)
        shutil.copy2(source, destination)
        after = sha256(destination)
        if before != after:
            raise SystemExit(f"Archive checksum mismatch: {source}")
        archive_rows.append({
            "relative_path": str(relative), "original_current_path": str(source), "archive_path": str(destination),
            "file_size": source.stat().st_size,
            "modified_timestamp": datetime.fromtimestamp(source.stat().st_mtime).astimezone().isoformat(timespec="seconds"),
            "checksum_before": before, "checksum_archive": after, "archive_status": "ARCHIVED_VERIFIED",
        })

    # Only remove current files after every existing file is safely archived and verified.
    for source in existing:
        source.unlink()

    archive_csv = ARCHIVE / "archive_manifest.csv"
    archive_json = ARCHIVE / "archive_manifest.json"
    archive_md = ARCHIVE / "archive_manifest.md"
    write_csv(archive_csv, archive_rows, ARCHIVE_FIELDS)
    write_json(archive_json, {
        "season": 1981, "as_of_date": "1981-07-19", "archive_location": str(ARCHIVE),
        "files_archived": len(archive_rows), "archive_verified": all(row["checksum_before"] == row["checksum_archive"] for row in archive_rows),
        "files": archive_rows,
    })
    archive_md.write_text(
        "# StatsPlus Pre-Promotion Archive Manifest\n\n"
        f"**Archive:** `{ARCHIVE}`  \n**Files archived:** {len(archive_rows)}\n\n"
        + (md_table(["Relative path", "Bytes", "Checksum", "Status"],
                    [[row["relative_path"], row["file_size"], row["checksum_archive"], row["archive_status"]] for row in archive_rows])
           if archive_rows else "No prior current Feed 3 files existed; archive count is zero.") + "\n",
        encoding="utf-8")

    promotion_rows = []
    for row in approved:
        source = Path(row["staging_source_path"])
        target = Path(row["proposed_official_target_path"])
        shutil.copy2(source, target)
        checksum_after = sha256(target)
        promotion_status = "PROMOTED_VERIFIED" if checksum_after == row["sha256"] else "FAILED_CHECKSUM"
        promotion_rows.append({
            "table_number": row["table_number"], "table_family": row["table_family"],
            "source_staging_path": str(source), "official_current_path": str(target), "action": row["current_action"],
            "row_count": row["row_count"], "column_count": row["column_count"],
            "checksum_before": row["sha256"], "checksum_after": checksum_after,
            "schema_status": row["schema_status"], "authority_classification": row["source_authority_classification"],
            "story_value": row["story_value"], "risk_note": row["risk_note"], "promotion_status": promotion_status,
        })
    if any(row["promotion_status"] != "PROMOTED_VERIFIED" for row in promotion_rows):
        raise SystemExit("One or more promoted files failed checksum verification.")

    report_stem = CONTROL / f"statsplus_promotion_report_{SLUG}"
    write_csv(report_stem.with_suffix(".csv"), promotion_rows)
    write_json(report_stem.with_suffix(".json"), {
        "season": 1981, "as_of_date": "1981-07-19", "promotion_completed": True,
        "files_promoted": len(promotion_rows), "files_archived": len(archive_rows), "archive_location": str(ARCHIVE),
        "intentional_exclusions": [6, 27], "story_signals_enabled": False, "files": promotion_rows,
    })
    report_stem.with_suffix(".md").write_text(
        "# StatsPlus Feed 3 Promotion Report\n\n> **StatsPlus enhances. It does not replace.**\n\n"
        f"**Files promoted:** {len(promotion_rows)}  \n**Files archived:** {len(archive_rows)}  \n"
        "**Intentional exclusions:** Tables 6 and 27  \n**Story signals enabled:** No\n\n"
        + md_table(["#", "Family", "Action", "Rows", "Cols", "Status"],
                   [[row["table_number"], row["table_family"], row["action"], row["row_count"], row["column_count"], row["promotion_status"]] for row in promotion_rows]) + "\n",
        encoding="utf-8")

    # Independent post-copy verification.
    current_files = sorted(path for path in CURRENT.rglob("*") if path.is_file())
    promoted_by_name = {Path(row["official_current_path"]).name: row for row in promotion_rows}
    verification_rows = []
    for path in current_files:
        promoted = promoted_by_name.get(path.name)
        actual_rows, actual_columns = csv_shape(path)
        verification_rows.append({
            "table_number": promoted["table_number"] if promoted else "", "file_name": path.name,
            "current_path": str(path), "exists": "yes", "checksum_match": "yes" if promoted and sha256(path) == promoted["checksum_before"] else "no",
            "expected_row_count": promoted["row_count"] if promoted else "", "actual_row_count": actual_rows,
            "row_count_match": "yes" if promoted and actual_rows == int(promoted["row_count"]) else "no",
            "expected_column_count": promoted["column_count"] if promoted else "", "actual_column_count": actual_columns,
            "column_count_match": "yes" if promoted and actual_columns == int(promoted["column_count"]) else "no",
            "schema_status": promoted["schema_status"] if promoted else "unexpected_file",
            "verification_status": "PASS" if promoted and sha256(path) == promoted["checksum_before"] and actual_rows == int(promoted["row_count"]) and actual_columns == int(promoted["column_count"]) else "FAIL",
        })

    table_numbers = {int(row["table_number"]) for row in verification_rows if row["table_number"]}
    special_checks = {
        "current_folder_exists": CURRENT.is_dir(), "current_file_count": len(current_files),
        "exactly_25_current_files": len(current_files) == 25, "table_6_excluded": 6 not in table_numbers,
        "table_27_excluded": 27 not in table_numbers, "all_checksums_match": all(row["checksum_match"] == "yes" for row in verification_rows),
        "all_row_counts_match": all(row["row_count_match"] == "yes" for row in verification_rows),
        "all_column_counts_match": all(row["column_count_match"] == "yes" for row in verification_rows),
        "table_1_accepted_reshaped_schema": promoted_by_name.get("01_abl_transactions_personnel_-_owner.csv", {}).get("action") == "PROMOTE_ACCEPT_RESHAPED_SCHEMA",
        "table_1_partial_meaning_loss": semantic.get("classification") == "PARTIAL_MEANING_LOSS",
        "owner_mood_unavailable": "Owner Mood" in semantic.get("removed_elements", []),
        "season_objective_unavailable": "Season Objective" in semantic.get("removed_elements", []),
        "specific_goals_unavailable": "Specific Goals" in semantic.get("removed_elements", []),
        "table_19_ubr_available": 19 in table_numbers and "UBR" in next((csv_shape_headers(Path(row["official_current_path"])) for row in promotion_rows if int(row["table_number"]) == 19), []),
        "table_21_rwar_available": 21 in table_numbers and "rWAR" in next((csv_shape_headers(Path(row["official_current_path"])) for row in promotion_rows if int(row["table_number"]) == 21), []),
        "hold_review_blockers": 0,
        "story_signals_enabled": False,
    }
    required_boolean_checks = [key for key, value in special_checks.items() if isinstance(value, bool) and key != "story_signals_enabled"]
    verification_passed = (all(special_checks[key] for key in required_boolean_checks)
                           and special_checks["hold_review_blockers"] == 0
                           and all(row["verification_status"] == "PASS" for row in verification_rows))
    special_checks["verification_passed"] = verification_passed

    verification_stem = CONTROL / f"statsplus_promotion_verification_{SLUG}"
    write_csv(verification_stem.with_suffix(".csv"), verification_rows)
    write_json(verification_stem.with_suffix(".json"), {"season": 1981, "as_of_date": "1981-07-19", "checks": special_checks, "files": verification_rows})
    verification_stem.with_suffix(".md").write_text(
        "# StatsPlus Feed 3 Promotion Verification\n\n"
        f"**Verification:** {'PASS' if verification_passed else 'FAIL'}\n\n"
        + md_table(["Check", "Result"], [[key, value] for key, value in special_checks.items()])
        + "\n\n" + md_table(["#", "File", "Checksum", "Rows", "Columns", "Status"],
                                  [[row["table_number"], row["file_name"], row["checksum_match"], row["row_count_match"], row["column_count_match"], row["verification_status"]] for row in verification_rows]) + "\n",
        encoding="utf-8")
    if not verification_passed:
        raise SystemExit("Post-promotion verification failed; inspect verification report.")

    action_counts = Counter(row["action"] for row in promotion_rows)
    current_status = {
        "season": 1981, "as_of_date": "1981-07-19", "feed3_promoted": True,
        "current_folder": str(CURRENT), "current_file_count": len(current_files), "intentional_exclusions": [6, 27],
        "accepted_schema_drift": [4], "accepted_reshaped_schema": [1], "accepted_added_detail": [15, 16, 19, 21, 24],
        "accepted_reordered_overlap": [7, 8],
        "unavailable_fields": ["Owner Mood", "Season Objective", "Specific Goals"],
        "available_new_enrichments": ["team UBR (table 19)", "player rWAR (table 21)", "expanded team-age detail (table 24)", "IP/SV/BS pitching detail (tables 15/16)"],
        "authority_boundaries": {
            "raw_ootp": "completed games, scores, logs, current-state date, and official standings proof",
            "sortable_stats": "governed current player/team enrichment",
            "statsplus": "supplemental model/context values with source labels",
        },
        "doctrine": "StatsPlus enhances. It does not replace.",
        "expandable_feed_note": "Future advanced batting, pitching, standardized metrics, or matchup matrices require staged intake and separate promotion.",
        "story_signals_enabled": False,
        "ready_for_later_story_engine_enrichment_task": True,
        "promotion_action_counts": dict(action_counts),
    }
    status_stem = CONTROL / f"statsplus_current_source_status_{SLUG}"
    write_json(status_stem.with_suffix(".json"), current_status)
    status_stem.with_suffix(".md").write_text(
        "# StatsPlus Feed 3 Current Source Status\n\n> **StatsPlus enhances. It does not replace.**\n\n"
        "- Feed 3 promoted: **Yes**.\n"
        f"- Current file count: **{len(current_files)}**.\n"
        "- Intentional exclusions: **tables 6 and 27**.\n"
        "- Accepted reshaped schema: **table 1**, with partial meaning loss.\n"
        "- Accepted schema/additive changes: **tables 4, 15, 16, 19, 21, and 24**.\n"
        "- Accepted reordered overlap: **tables 7 and 8**; do not double count.\n"
        "- Unavailable fields: **Owner Mood, Season Objective, Specific Goals**.\n"
        "- New enrichments: **team UBR, player rWAR, expanded age detail, and pitching IP/SV/BS**.\n"
        "- Story signals enabled: **No**.\n"
        "- Ready for a separate story-engine enrichment integration task: **Yes**.\n\n"
        "Feed 3 is expandable, but every future table family requires staged intake, profiling, classification, and separate promotion. Raw OOTP remains authoritative for completed-game and current-state proof.\n",
        encoding="utf-8")

    print(json.dumps({
        "promotion_completed": True, "files_promoted": len(promotion_rows), "files_archived": len(archive_rows),
        "archive_location": str(ARCHIVE), "verification_passed": verification_passed,
        "table_6_excluded": special_checks["table_6_excluded"], "table_27_excluded": special_checks["table_27_excluded"],
        "table_1_limitation_documented": special_checks["table_1_partial_meaning_loss"] and special_checks["owner_mood_unavailable"],
        "ubr_available": special_checks["table_19_ubr_available"], "rwar_available": special_checks["table_21_rwar_available"],
        "ready_for_story_engine_enrichment_integration": True,
    }, indent=2))


def csv_shape_headers(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return next(csv.reader(handle), [])


if __name__ == "__main__":
    main()
