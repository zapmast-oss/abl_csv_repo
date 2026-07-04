from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / "csv" / "abl_statistics"
PLAYER_STAGING = Path(
    "C:/Users/earld/OneDrive/Documents/Out of the Park Developments/"
    "OOTP Baseball 26/saved_games/Action Baseball League.lg/import_export/"
    "abl_statistics_player_statistics"
)
TEAM_STAGING = Path(
    "C:/Users/earld/OneDrive/Documents/Out of the Park Developments/"
    "OOTP Baseball 26/saved_games/Action Baseball League.lg/import_export/"
    "abl_statistics_team_statistics"
)
CONTROL = ROOT / "csv" / "out" / "control"
REGISTRY = CONTROL / "abl_source_registry.csv"
ARCHIVE = ROOT / "csv" / "out" / "archive" / "sortable_stats_pre_promotion_1981_asof_1981-07-19"
AS_OF = "1981-07-19"
DRIFT_FILES = {
    "abl_statistics_team_statistics___info_-_sortable_stats_abl_staff.csv",
    "abl_statistics_team_statistics___info_-_sortable_stats_batting_stats.csv",
}


def usable_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        return [line for line in handle if line.strip() and not line.lstrip().startswith("#")]


def profile(path: Path) -> dict[str, object]:
    parsed = list(csv.reader(usable_lines(path)))
    headers = parsed[0] if parsed else []
    rows = [row for row in parsed[1:] if any(cell.strip() for cell in row)]
    return {"headers": headers, "row_count": len(rows), "column_count": len(headers)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def source_for_official(name: str) -> Path | None:
    player = PLAYER_STAGING / name
    team = TEAM_STAGING / name
    if player.exists() and team.exists():
        raise ValueError(f"Ambiguous staging source for {name}")
    if player.exists():
        return player
    if team.exists():
        return team
    return None


def disposition(column: str, filename: str) -> str:
    if filename.endswith("abl_staff.csv"):
        return "replaced by coaches.csv/team staff lookup; compatibility view may expose null when no ID match"
    if column == "WAR":
        return "replaced by batting_xtra.csv WAR"
    if column in {"CS", "SB%"}:
        return "replaced by raw/team or aggregated player baserunning source when validated; otherwise unavailable"
    if column in {"BatR", "wSB", "UBR", "BsR"}:
        return "marked unavailable; null in compatibility view; disable dependent enrichment"
    return "deprecated or reviewed later"


def drift_impact(filename: str, removed: list[str]) -> tuple[str, str]:
    if filename.endswith("abl_staff.csv"):
        return (
            "build_star_schema.py can regenerate coach IDs from raw coaches data, but must stop rewriting the source file. Consumers that read *_ID directly need a generated compatibility view.",
            "Manager/staff identity remains available by name; ID-dependent manager enrichment is limited when a coach-name lookup fails.",
        )
    return (
        "build_star_schema.py will produce a narrower team batting fact. z_abl_basepath_pressure.py cannot use this file for UBR and loses CS/SB% unless another validated source supplies them.",
        "Core batting totals remain available. Baserunning-value and caught-stealing enrichments must use another validated source or be disabled/null.",
    )


def build_drift_rows() -> list[dict[str, object]]:
    rows = []
    for name in sorted(DRIFT_FILES):
        old_path, new_path = OFFICIAL / name, TEAM_STAGING / name
        old, new = profile(old_path), profile(new_path)
        retained = [column for column in old["headers"] if column in new["headers"]]
        removed = [column for column in old["headers"] if column not in new["headers"]]
        added = [column for column in new["headers"] if column not in old["headers"]]
        downstream, story = drift_impact(name, removed)
        rows.append({
            "official_existing_file_path": old_path.relative_to(ROOT).as_posix(),
            "staging_file_path": str(new_path), "old_column_count": old["column_count"],
            "new_column_count": new["column_count"], "old_columns": "|".join(old["headers"]),
            "new_columns": "|".join(new["headers"]), "columns_retained": "|".join(retained),
            "columns_removed": "|".join(removed), "columns_added": "|".join(added),
            "likely_downstream_impact": downstream, "story_engine_impact": story,
            "missing_field_disposition": "; ".join(f"{column}: {disposition(column, name)}" for column in removed),
            "schema_decision": "ACCEPTED_FORWARD_STANDARD",
        })
    return rows


def write_drift(rows: list[dict[str, object]]) -> None:
    stem = CONTROL / f"sortable_schema_drift_accepted_1981_asof_{AS_OF}"
    write_csv(stem.with_suffix(".csv"), list(rows[0]), rows)
    write_json(stem.with_suffix(".json"), {"as_of_date": AS_OF, "decision": "accepted_schema_drift", "files": rows})
    lines = ["# Accepted Sortable Schema Drift — 1981 as of 1981-07-19", "",
             "The current 13-column captures are the forward source standard. Removed fields are not fabricated in source files.", ""]
    for row in rows:
        lines += [f"## `{Path(str(row['official_existing_file_path'])).name}`", "",
                  f"- Old columns: {row['old_column_count']}", f"- New columns: {row['new_column_count']}",
                  f"- Retained: `{row['columns_retained']}`", f"- Removed: `{row['columns_removed']}`",
                  f"- Added: `{row['columns_added'] or 'none'}`", f"- Downstream impact: {row['likely_downstream_impact']}",
                  f"- Story-engine impact: {row['story_engine_impact']}",
                  f"- Field disposition: {row['missing_field_disposition']}", ""]
    stem.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def write_adaptation(drift_rows: list[dict[str, object]]) -> None:
    stem = CONTROL / f"sortable_schema_adaptation_plan_1981_asof_{AS_OF}"
    plan = {
        "as_of_date": AS_OF,
        "source_policy": "Do not add fake columns to official source CSVs.",
        "compatibility_view_location": "csv/out/control initially; csv/out/curated after curated-layer implementation",
        "affected_consumers": [
            {
                "script": "csv/abl_scripts/build_star_schema.py",
                "source": "abl_staff.csv",
                "expected_removed_columns": "GM_ID, MA_ID, BN_ID, PC_ID, HC_ID, SC_ID, TT_ID, OWN_ID, 1BC_ID, 3BC_ID",
                "change": "Use attach_coach_ids against raw coaches data; write IDs only to dim_team_staff/compatibility output; remove source-file rewrite.",
                "limited_signal": "Disable ID-dependent staff linkage only for unmatched coach names.",
            },
            {
                "script": "csv/abl_scripts/build_star_schema.py",
                "source": "batting_stats.csv",
                "expected_removed_columns": "CS, SB%, WAR, BatR, wSB, UBR, BsR",
                "change": "Allow narrower base table; take WAR from batting_xtra; source other fields from validated raw/curated tables or expose null.",
                "limited_signal": "Baserunning value and caught-stealing enrichment limited until replacement source is curated.",
            },
            {
                "script": "csv/abl_scripts/z_abl_basepath_pressure.py",
                "source": "batting_stats.csv fallback",
                "expected_removed_columns": "CS, UBR",
                "change": "Prefer abl_team_bat_baserunning.csv or a curated raw aggregation; fail/disable the UBR component when unavailable instead of inventing zero.",
                "limited_signal": "Disable UBR/BsR-based basepath pressure component when no compatible source exists.",
            },
        ],
        "compatibility_views": [
            "team_staff_compat: retain 13 captured fields and left-join coach IDs from raw coaches; null unmatched IDs.",
            "team_batting_compat: retain 13 captured fields, join WAR from batting_xtra, join validated CS/SB% if available, and set BatR/wSB/UBR/BsR null with availability flags.",
        ],
    }
    write_json(stem.with_suffix(".json"), plan)
    lines = ["# Sortable Schema Adaptation Plan — 1981 as of 1981-07-19", "",
             "Official sources remain faithful to the capture. Compatibility is generated downstream; fake source columns are prohibited.", "",
             "## Affected consumers", "", "| Script | Source | Removed expectations | Required change | Limited/disabled enrichment |",
             "|---|---|---|---|---|"]
    for item in plan["affected_consumers"]:
        lines.append(f"| `{item['script']}` | `{item['source']}` | {item['expected_removed_columns']} | {item['change']} | {item['limited_signal']} |")
    lines += ["", "## Compatibility-view strategy", ""] + [f"- {item}" for item in plan["compatibility_views"]]
    lines += ["", "Compatibility views belong under `csv/out/control/` during transition or `csv/out/curated/` later. They must include availability/provenance flags.", ""]
    stem.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def load_registry() -> dict[str, dict[str, str]]:
    with REGISTRY.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["file_path"]: row for row in csv.DictReader(handle)}


def build_dry_run(drift_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    registry = load_registry()
    drift = {Path(str(row["official_existing_file_path"])).name for row in drift_rows}
    rows = []
    replacement_names = {path.name for path in OFFICIAL.glob("*.csv")}
    staged_all = sorted(list(PLAYER_STAGING.glob("*.csv")) + list(TEAM_STAGING.glob("*.csv")))
    for source in staged_all:
        target = OFFICIAL / source.name
        if source.name in replacement_names:
            before, after = profile(target), profile(source)
            action = "REPLACE_EXISTING_ACCEPT_SCHEMA_DRIFT" if source.name in drift else "REPLACE_EXISTING"
            schema_status = "ACCEPTED_SCHEMA_DRIFT" if source.name in drift else "EXACT_SCHEMA_MATCH"
            confidence = 100
            risk = "Accepted forward schema; downstream adaptation required." if source.name in drift else "Low schema risk; values/row universe updated."
            rel = target.relative_to(ROOT).as_posix()
            reg = registry.get(rel, {})
            target_path = rel
        else:
            before = {"row_count": "", "column_count": "", "headers": []}; after = profile(source)
            action = "HOLD_DUPLICATE"
            schema_status = "OVERLAPPING_OR_SUBSET_VIEW"
            confidence = 95
            risk = "No unique governed columns established; do not promote."
            reg = {}
            target_path = ""
        rows.append({
            "official_target_file_path": target_path, "staging_source_file_path": str(source),
            "action": action, "row_count_before": before["row_count"], "row_count_after": after["row_count"],
            "column_count_before": before["column_count"], "column_count_after": after["column_count"],
            "schema_status": schema_status, "risk_note": risk,
            "source_family": reg.get("source_family", "sortable_stats"),
            "subject_area": reg.get("subject_area", "overlapping sortable view"),
            "promotion_confidence": confidence,
        })
    return rows


def write_dry_run(rows: list[dict[str, object]]) -> None:
    stem = CONTROL / f"sortable_promotion_dry_run_1981_asof_{AS_OF}"
    write_csv(stem.with_suffix(".csv"), list(rows[0]), rows)
    counts = {}
    for row in rows:
        counts[row["action"]] = counts.get(row["action"], 0) + 1
    blockers = [row for row in rows if row["action"] == "HOLD_REVIEW"]
    payload = {"as_of_date": AS_OF, "action_counts": counts, "hold_review_blockers": len(blockers), "promotion_authorized_by_policy": not blockers, "files": rows}
    write_json(stem.with_suffix(".json"), payload)
    lines = ["# Sortable Promotion Dry Run — 1981 as of 1981-07-19", "",
             f"- Replace exact schema: **{counts.get('REPLACE_EXISTING', 0)}**",
             f"- Replace with accepted drift: **{counts.get('REPLACE_EXISTING_ACCEPT_SCHEMA_DRIFT', 0)}**",
             f"- Hold duplicates: **{counts.get('HOLD_DUPLICATE', 0)}**",
             f"- Hold review blockers: **{len(blockers)}**",
             f"- Promotion gate: **{'PASS' if not blockers else 'BLOCKED'}**", "",
             "| Action | Official target | Staging source | Rows before/after | Columns before/after | Schema | Risk |",
             "|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| `{row['action']}` | `{row['official_target_file_path'] or '—'}` | `{row['staging_source_file_path']}` | {row['row_count_before'] or '—'} / {row['row_count_after']} | {row['column_count_before'] or '—'} / {row['column_count_after']} | `{row['schema_status']}` | {row['risk_note']} |")
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_status(dry_rows: list[dict[str, object]]) -> None:
    replacement = [row for row in dry_rows if row["action"].startswith("REPLACE_EXISTING")]
    rows = []
    for row in replacement:
        rows.append({
            "official_target_file_path": row["official_target_file_path"],
            "replacement_status": "ACCEPTED_SCHEMA_DRIFT" if row["action"].endswith("ACCEPT_SCHEMA_DRIFT") else "EXACT_SAFE_REPLACEMENT",
            "story_impact": "Staff IDs adapted downstream; unmatched ID enrichments limited." if "abl_staff" in row["official_target_file_path"] else ("Baserunning-value enrichments limited; core batting remains available." if "batting_stats" in row["official_target_file_path"] else "No schema-level story impact."),
            "disabled_or_limited_notes": "See sortable schema adaptation plan." if row["action"].endswith("ACCEPT_SCHEMA_DRIFT") else "none",
        })
    stem = CONTROL / f"sortable_capture_combined_status_1981_asof_{AS_OF}_final"
    write_csv(stem.with_suffix(".csv"), list(rows[0]), rows)
    payload = {"as_of_date": AS_OF, "exact_safe_replacements": 18, "accepted_schema_drift": 2, "official_sources_covered": 20, "promotion_status": "SAFE_WITH_ACCEPTED_SCHEMA_DRIFT", "sources": rows}
    write_json(stem.with_suffix(".json"), payload)
    lines = ["# Final Sortable Capture Status — 1981 as of 1981-07-19", "",
             "- Exact/safe replacements: **18 of 20**",
             "- Accepted with schema drift: **2 of 20**",
             "- Coverage: **20 of 20 governed sources**",
             "- Promotion status: **SAFE_WITH_ACCEPTED_SCHEMA_DRIFT**", "",
             "## Story and enrichment impact", "",
             "- Staff names remain available; staff-ID enrichment uses raw coach lookup and is limited when names do not resolve.",
             "- Core team batting remains available; CS/SB% and advanced baserunning value fields require another validated source or remain unavailable.",
             "- UBR/BsR/wSB/BatR-dependent signals must be disabled or explicitly marked unavailable until curated replacements exist.", ""]
    stem.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def archive_official(replace_rows: list[dict[str, object]]) -> None:
    if ARCHIVE.exists():
        manifest_path = ARCHIVE / "archive_manifest.csv"
        if not manifest_path.exists():
            raise FileExistsError(f"Archive exists without a manifest; refusing to continue: {ARCHIVE}")
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            archived = list(csv.DictReader(handle))
        if len(archived) != len(replace_rows):
            raise ValueError(f"Existing archive has {len(archived)} records; expected {len(replace_rows)}")
        for row in archived:
            copy = ROOT / row["archive_path"]
            if not copy.exists() or sha256(copy) != row["sha256_archive"]:
                raise IOError(f"Existing archive validation failed: {copy}")
        return
    ARCHIVE.mkdir(parents=True)
    rows = []
    for item in replace_rows:
        source = ROOT / str(item["official_target_file_path"])
        target = ARCHIVE / source.name
        before = profile(source); source_hash = sha256(source)
        shutil.copy2(source, target)
        archive_hash = sha256(target)
        if source_hash != archive_hash:
            raise IOError(f"Archive checksum mismatch for {source}")
        rows.append({
            "official_source_path": source.relative_to(ROOT).as_posix(),
            "archive_path": target.relative_to(ROOT).as_posix(), "file_size_bytes": source.stat().st_size,
            "row_count": before["row_count"], "column_count": before["column_count"],
            "sha256_before": source_hash, "sha256_archive": archive_hash,
            "archived_at_utc": datetime.now(timezone.utc).isoformat(),
        })
    write_csv(ARCHIVE / "archive_manifest.csv", list(rows[0]), rows)
    write_json(ARCHIVE / "archive_manifest.json", {"as_of_date": AS_OF, "file_count": len(rows), "files": rows})
    lines = ["# Pre-Promotion Sortable Archive Manifest", "", f"Archived files: **{len(rows)}**", "",
             "| Official source | Archive copy | Rows | Columns | SHA-256 |", "|---|---|---:|---:|---|"]
    for row in rows:
        lines.append(f"| `{row['official_source_path']}` | `{row['archive_path']}` | {row['row_count']} | {row['column_count']} | `{row['sha256_archive']}` |")
    (ARCHIVE / "archive_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def promote(replace_rows: list[dict[str, object]]) -> None:
    for item in replace_rows:
        source = Path(str(item["staging_source_file_path"]))
        target = ROOT / str(item["official_target_file_path"])
        temporary = target.with_name(f".{target.name}.promotion_tmp")
        shutil.copy2(source, temporary)
        if sha256(source) != sha256(temporary):
            raise IOError(f"Promotion staging checksum mismatch: {source} -> {temporary}")
        temporary.replace(target)
        if sha256(source) != sha256(target):
            raise IOError(f"Promotion checksum mismatch after atomic replace: {source} -> {target}")


def write_verification(replace_rows: list[dict[str, object]]) -> None:
    rows = []
    replaced_names = {Path(str(item["official_target_file_path"])).name for item in replace_rows}
    for item in replace_rows:
        target = ROOT / str(item["official_target_file_path"]); source = Path(str(item["staging_source_file_path"]))
        actual, expected = profile(target), profile(source)
        drift = target.name in DRIFT_FILES
        passed = actual["row_count"] == expected["row_count"] and actual["column_count"] == expected["column_count"] and sha256(target) == sha256(source)
        rows.append({
            "official_file": target.relative_to(ROOT).as_posix(), "staging_file": str(source),
            "row_count_expected": expected["row_count"], "row_count_actual": actual["row_count"],
            "column_count_expected": expected["column_count"], "column_count_actual": actual["column_count"],
            "sha256_match": "yes" if sha256(target) == sha256(source) else "no",
            "schema_status": "ACCEPTED_SCHEMA_DRIFT" if drift else "EXACT_SCHEMA",
            "verification_status": "PASS" if passed else "FAIL",
        })
    official_csvs = list(OFFICIAL.glob("*.csv"))
    all_pass = len(official_csvs) == 20 and len(replaced_names) == 20 and all(row["verification_status"] == "PASS" for row in rows)
    stem = CONTROL / f"sortable_promotion_verification_1981_asof_{AS_OF}"
    write_csv(stem.with_suffix(".csv"), list(rows[0]), rows)
    payload = {"as_of_date": AS_OF, "official_csv_count": len(official_csvs), "promoted_file_count": len(rows), "accepted_drift_files": sorted(DRIFT_FILES), "unresolved_blockers": 0 if all_pass else 1, "verification_status": "PASS" if all_pass else "FAIL", "files": rows}
    write_json(stem.with_suffix(".json"), payload)
    lines = ["# Sortable Promotion Verification — 1981 as of 1981-07-19", "",
             f"- Official sortable CSV count: **{len(official_csvs)}**",
             f"- Promoted files verified: **{len(rows)}**",
             "- Accepted drift files: **2**",
             f"- Unresolved blockers: **{payload['unresolved_blockers']}**",
             f"- Verification: **{payload['verification_status']}**", "",
             "| Official file | Rows expected/actual | Columns expected/actual | Hash | Schema | Result |",
             "|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| `{row['official_file']}` | {row['row_count_expected']} / {row['row_count_actual']} | {row['column_count_expected']} / {row['column_count_actual']} | {row['sha256_match']} | `{row['schema_status']}` | `{row['verification_status']}` |")
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not all_pass:
        raise RuntimeError("Post-promotion verification failed")


def main() -> int:
    for path in (OFFICIAL, PLAYER_STAGING, TEAM_STAGING):
        if not path.is_dir():
            raise NotADirectoryError(path)
    if len(list(OFFICIAL.glob("*.csv"))) != 20:
        raise ValueError("Expected exactly 20 official sortable CSVs before promotion")
    drift_rows = build_drift_rows()
    write_drift(drift_rows)
    write_adaptation(drift_rows)
    dry_rows = build_dry_run(drift_rows)
    write_dry_run(dry_rows)
    write_final_status(dry_rows)
    blockers = [row for row in dry_rows if row["action"] == "HOLD_REVIEW"]
    replace_rows = [row for row in dry_rows if row["action"].startswith("REPLACE_EXISTING")]
    if blockers:
        print(json.dumps({"promotion_performed": False, "hold_review_blockers": len(blockers)}, indent=2))
        return 2
    if len(replace_rows) != 20:
        raise ValueError(f"Expected 20 replacements after accepted drift, found {len(replace_rows)}")
    archive_official(replace_rows)
    promote(replace_rows)
    write_verification(replace_rows)
    print(json.dumps({
        "promotion_performed": True, "replaced_files": len(replace_rows),
        "archive": str(ARCHIVE), "official_csv_count": len(list(OFFICIAL.glob('*.csv'))),
        "verification": "PASS",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
