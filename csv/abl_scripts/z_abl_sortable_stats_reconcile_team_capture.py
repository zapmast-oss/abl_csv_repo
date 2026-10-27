from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXISTING_DIR = ROOT / "csv" / "abl_statistics"
TEAM_STAGING = Path(
    "C:/Users/earld/OneDrive/Documents/Out of the Park Developments/"
    "OOTP Baseball 26/saved_games/Action Baseball League.lg/import_export/"
    "abl_statistics_team_statistics"
)
CONTROL = ROOT / "csv" / "out" / "control"
PLAYER_RECONCILIATION = CONTROL / "sortable_stats_reconciliation_1981_asof_1981-07-19.csv"
MISSING_REPLACEMENTS = CONTROL / "missing_sortable_replacements_1981_asof_1981-07-19.csv"
NEW_PLAYER_REVIEW = CONTROL / "new_sortable_source_review_1981_asof_1981-07-19.csv"
REGISTRY = CONTROL / "abl_source_registry.csv"
AS_OF = "1981-07-19"
RECOMMENDATIONS = {
    "PROMOTE_REPLACE_EXISTING", "PROMOTE_AS_NEW_SOURCE", "HOLD_DUPLICATE_OR_OVERLAP",
    "HOLD_UNKNOWN_REVIEW", "IGNORE_NOT_ABL_SORTABLE_STATS",
}
RECON_FIELDS = [
    "record_type", "staging_file", "existing_file", "previously_missing",
    "staging_row_count", "existing_row_count", "staging_column_count",
    "existing_column_count", "header_jaccard", "staging_header_containment",
    "filename_similarity", "subject_area", "likely_grain", "match_type",
    "confidence_score", "recommendation", "promotion_reason", "warnings",
]


def normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.strip().lower())


def normalize_name(value: str) -> str:
    stem = Path(value).stem.lower()
    stem = re.sub(r"^abl_statistics_team_statistics_+", "", stem)
    stem = re.sub(r"(?:info_)?-?_?sortable_stats_", "", stem)
    return "_".join(re.findall(r"[a-z0-9]+", stem))


def profile(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        usable = [line for line in handle if line.strip() and not line.lstrip().startswith("#")]
    parsed = list(csv.reader(usable))
    headers = parsed[0] if parsed else []
    rows = [row for row in parsed[1:] if any(cell.strip() for cell in row)]
    normalized = [normalize_header(header) for header in headers]
    return {
        "path": path, "name": path.name, "normalized_name": normalize_name(path.name),
        "headers": headers, "header_list": normalized, "header_set": set(normalized),
        "row_count": len(rows), "column_count": len(headers),
    }


def subject(filename: str) -> str:
    lower = filename.lower()
    if "staff" in lower:
        return "team_staff"
    if "finan" in lower:
        return "team_financial_context"
    if "pers_park" in lower or "park_info" in lower:
        return "team_park_personality_context"
    if "history" in lower or "cur_rec" in lower or "default" in lower:
        return "team_record_history_context"
    if "batting" in lower:
        return "team_batting"
    if "pitching" in lower:
        return "team_pitching"
    if "fielding" in lower:
        return "team_fielding"
    return "unknown"


def metrics(new: dict[str, object], old: dict[str, object]) -> tuple[float, float, float]:
    a, b = new["header_set"], old["header_set"]
    shared = len(a & b)
    jaccard = shared / len(a | b) if a | b else 1.0
    containment = shared / len(a) if a else 0.0
    name_score = SequenceMatcher(None, str(new["normalized_name"]), str(old["normalized_name"])).ratio()
    return jaccard, containment, name_score


def score(new: dict[str, object], old: dict[str, object]) -> tuple[float, float, float, float]:
    jaccard, containment, name_score = metrics(new, old)
    subject_bonus = 0.10 if subject(str(new["name"])) == subject(str(old["name"])) else 0.0
    column_ratio = min(int(new["column_count"]), int(old["column_count"])) / max(int(new["column_count"]), int(old["column_count"]), 1)
    total = 0.35 * jaccard + 0.25 * containment + 0.20 * name_score + 0.10 * column_ratio + subject_bonus
    return min(total, 1.0), jaccard, containment, name_score


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def reconcile(existing: list[dict[str, object]], staging: list[dict[str, object]], missing: set[str]) -> list[dict[str, object]]:
    by_name = {str(item["name"]).lower(): item for item in existing}
    rows: list[dict[str, object]] = []
    safely_matched: set[str] = set()
    physically_matched: set[str] = set()
    for new in staging:
        exact = by_name.get(str(new["name"]).lower())
        if exact:
            old = exact
            total, jaccard, containment, name_score = score(new, old)
            physically_matched.add(str(old["name"]))
            if new["header_list"] == old["header_list"]:
                match_type = "EXACT_FILENAME_AND_SCHEMA"
                recommendation = "PROMOTE_REPLACE_EXISTING"
                confidence = 100.0
                reason = "Exact filename and normalized column sequence match."
                warnings = "row values changed with the new capture" if new["row_count"] == old["row_count"] else f"row count changed {old['row_count']} -> {new['row_count']}"
                safely_matched.add(str(old["name"]))
            else:
                match_type = "EXACT_FILENAME_SCHEMA_CHANGED"
                recommendation = "HOLD_UNKNOWN_REVIEW"
                confidence = round(max(total, 0.75) * 100, 1)
                reason = "Filename matches, but the staging export omits or changes existing columns."
                old_only = [header for header in old["headers"] if normalize_header(header) not in new["header_set"]]
                warnings = f"schema contraction/change; existing-only columns: {'|'.join(old_only)}"
        else:
            candidates = sorted(((score(new, old), old) for old in existing), key=lambda item: item[0][0], reverse=True)
            (total, jaccard, containment, name_score), old = candidates[0]
            if jaccard >= 0.88 and min(int(new["column_count"]), int(old["column_count"])) / max(int(new["column_count"]), int(old["column_count"]), 1) >= 0.90:
                match_type = "RENAMED_STRUCTURAL_MATCH"
                recommendation = "PROMOTE_REPLACE_EXISTING"
                reason = "Different filename with near-identical schema and team grain."
                confidence = round(total * 100, 1)
                warnings = "renamed replacement requires explicit review"
                safely_matched.add(str(old["name"]))
            elif containment >= 0.60 or (name_score >= 0.55 and subject(str(new["name"])) == subject(str(old["name"]))):
                match_type = "OVERLAPPING_VIEW"
                recommendation = "HOLD_DUPLICATE_OR_OVERLAP"
                reason = "Focused team report substantially overlaps an existing governed broader source."
                confidence = round(total * 100, 1)
                warnings = "do not register a subset view without a distinct curated consumer"
            elif "sortable_stats" in str(new["name"]).lower() and subject(str(new["name"])) != "unknown":
                match_type = "NEW_SOURCE_CANDIDATE"
                recommendation = "PROMOTE_AS_NEW_SOURCE"
                reason = "Recognized team sortable report with a distinct schema and subject role."
                confidence = round(total * 100, 1)
                warnings = "new-source status requires column-level review"
            else:
                match_type = "UNKNOWN_SORTABLE"
                recommendation = "HOLD_UNKNOWN_REVIEW"
                reason = "No safe structural correspondence could be established."
                confidence = round(total * 100, 1)
                warnings = "manual review required"
        rows.append({
            "record_type": "TEAM_STAGING_FILE", "staging_file": new["name"], "existing_file": old["name"],
            "previously_missing": "yes" if str(old["name"]) in missing else "no",
            "staging_row_count": new["row_count"], "existing_row_count": old["row_count"],
            "staging_column_count": new["column_count"], "existing_column_count": old["column_count"],
            "header_jaccard": round(jaccard, 4), "staging_header_containment": round(containment, 4),
            "filename_similarity": round(name_score, 4), "subject_area": subject(str(new["name"])),
            "likely_grain": "team capture" if int(new["row_count"]) in range(20, 31) else "unknown",
            "match_type": match_type, "confidence_score": confidence, "recommendation": recommendation,
            "promotion_reason": reason, "warnings": warnings,
        })
    for old_name in sorted(missing):
        if old_name not in safely_matched:
            physical = old_name in physically_matched
            rows.append({
                "record_type": "PREVIOUSLY_MISSING_STILL_UNRESOLVED", "staging_file": old_name if physical else "",
                "existing_file": old_name, "previously_missing": "yes", "staging_row_count": "",
                "existing_row_count": next(item["row_count"] for item in existing if item["name"] == old_name),
                "staging_column_count": "", "existing_column_count": next(item["column_count"] for item in existing if item["name"] == old_name),
                "header_jaccard": "", "staging_header_containment": "", "filename_similarity": "",
                "subject_area": subject(old_name), "likely_grain": "team capture",
                "match_type": "SCHEMA_INCOMPATIBLE_REPLACEMENT" if physical else "NO_STAGING_REPLACEMENT",
                "confidence_score": "", "recommendation": "HOLD_UNKNOWN_REVIEW",
                "promotion_reason": "A file is present but does not preserve the governed schema." if physical else "No team-staging file safely replaces this governed source.",
                "warnings": "promotion remains blocked for this source",
            })
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summary(rows: list[dict[str, object]], missing: set[str]) -> dict[str, object]:
    staging = [row for row in rows if row["record_type"] == "TEAM_STAGING_FILE"]
    safe_missing = sorted({str(row["existing_file"]) for row in staging if row["previously_missing"] == "yes" and row["recommendation"] == "PROMOTE_REPLACE_EXISTING"})
    unresolved = sorted(missing - set(safe_missing))
    recs = Counter(str(row["recommendation"]) for row in staging)
    types = Counter(str(row["match_type"]) for row in staging)
    staff_unresolved = any("staff" in name.lower() for name in unresolved)
    return {
        "existing_sortable_files_inspected": 20,
        "team_staging_files_inspected": len(staging),
        "exact_filename_schema_matches": types["EXACT_FILENAME_AND_SCHEMA"],
        "exact_filename_schema_changed": types["EXACT_FILENAME_SCHEMA_CHANGED"],
        "likely_renamed_matches": types["RENAMED_STRUCTURAL_MATCH"],
        "previously_missing_sources_with_safe_replacements": safe_missing,
        "previously_missing_replacements_found": len(safe_missing),
        "existing_sources_still_missing_safe_replacements": unresolved,
        "existing_sources_still_missing_count": len(unresolved),
        "new_source_candidates": recs["PROMOTE_AS_NEW_SOURCE"],
        "duplicates_or_overlaps": recs["HOLD_DUPLICATE_OR_OVERLAP"],
        "unknown_manual_review_staging_files": recs["HOLD_UNKNOWN_REVIEW"],
        "staff_manager_reports_still_missing": staff_unresolved,
        "team_staging_covers_missing_family": len(unresolved) == 0,
        "promotion_safe": len(unresolved) == 0 and recs["HOLD_UNKNOWN_REVIEW"] == 0 and recs["HOLD_DUPLICATE_OR_OVERLAP"] == 0,
    }


def write_reconciliation(rows: list[dict[str, object]], info: dict[str, object]) -> None:
    stem = CONTROL / f"sortable_team_stats_reconciliation_1981_asof_{AS_OF}"
    write_csv(stem.with_suffix(".csv"), RECON_FIELDS, rows)
    write_json(stem.with_suffix(".json"), {"as_of_date": AS_OF, "team_staging_folder": str(TEAM_STAGING), "summary": info, "reconciliation": rows})
    lines = ["# Team Sortable-Stats Reconciliation — 1981 as of 1981-07-19", "",
             f"- Existing sortable files inspected: **{info['existing_sortable_files_inspected']}**",
             f"- Team-staging files inspected: **{info['team_staging_files_inspected']}**",
             f"- Exact filename/schema matches: **{info['exact_filename_schema_matches']}**",
             f"- Exact filename/schema changes: **{info['exact_filename_schema_changed']}**",
             f"- Likely renamed matches: **{info['likely_renamed_matches']}**",
             f"- Previously missing sources now safely replaced: **{info['previously_missing_replacements_found']}**",
             f"- Existing sources still lacking safe replacements: **{info['existing_sources_still_missing_count']}**",
             f"- New-source candidates: **{info['new_source_candidates']}**",
             f"- Duplicates/overlaps: **{info['duplicates_or_overlaps']}**",
             f"- Unknown/manual-review staging files: **{info['unknown_manual_review_staging_files']}**",
             f"- Staff/manager reports still unresolved: **{'YES' if info['staff_manager_reports_still_missing'] else 'NO'}**",
             f"- Promotion safe: **{'YES' if info['promotion_safe'] else 'NO'}**", "",
             "## Previously missing sources now safely replaced", ""]
    lines += [f"- `{name}`" for name in info["previously_missing_sources_with_safe_replacements"]] or ["None."]
    lines += ["", "## Existing sources still lacking safe replacements", ""]
    lines += [f"- `{name}`" for name in info["existing_sources_still_missing_safe_replacements"]] or ["None."]
    sections = [
        ("Staging files with no exact existing match", lambda row: row["record_type"] == "TEAM_STAGING_FILE" and row["match_type"] not in {"EXACT_FILENAME_AND_SCHEMA", "EXACT_FILENAME_SCHEMA_CHANGED"}),
        ("Likely new source candidates", lambda row: row["recommendation"] == "PROMOTE_AS_NEW_SOURCE"),
        ("Duplicates or overlaps", lambda row: row["recommendation"] == "HOLD_DUPLICATE_OR_OVERLAP"),
        ("Unknown or manual review", lambda row: row["record_type"] == "TEAM_STAGING_FILE" and row["recommendation"] == "HOLD_UNKNOWN_REVIEW"),
    ]
    for title, predicate in sections:
        selected = [row for row in rows if predicate(row)]
        lines += ["", f"## {title}", ""]
        if not selected:
            lines.append("None.")
        else:
            lines += ["| Staging | Existing match | Type | Confidence | Recommendation | Warning |", "|---|---|---|---:|---|---|"]
            for row in selected:
                lines.append(f"| `{row['staging_file']}` | `{row['existing_file']}` | `{row['match_type']}` | {row['confidence_score']} | `{row['recommendation']}` | {row['warnings']} |")
    lines += ["", "## Recommended promotion plan", "",
              "1. Do not promote while `abl_staff.csv` and `batting_stats.csv` fail schema compatibility.",
              "2. Treat the eight exact schema matches as replacement candidates only after combined dry-run validation.",
              "3. Hold the seven focused team views as overlaps unless column review proves unique governed value.",
              "4. Re-export the full ABL staff report and full team batting-stats report with the governed columns.",
              "5. Reconcile those corrected files, then prepare a combined dry-run promotion manifest.", "",
              "## Warnings", "",
              "- Physical filename presence is not a safe replacement when columns were dropped.",
              "- The staff/manager family is still unresolved because the staged staff files are schema contractions.",
              "- No files were promoted, moved, renamed, replaced, or deleted.", ""]
    stem.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def write_manifest(staging_rows: list[dict[str, object]]) -> None:
    by_name = {str(row["staging_file"]): row for row in staging_rows}
    rows = []
    for path in sorted(TEAM_STAGING.glob("*.csv")):
        item = profile(path); rec = by_name[path.name]
        rows.append({
            "file_name": path.name, "full_path": str(path), "file_size_bytes": path.stat().st_size,
            "modified_timestamp": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
            "row_count": item["row_count"], "column_count": item["column_count"], "sha256": sha256(path),
            "reconciliation_status": rec["match_type"], "promotion_recommendation": rec["recommendation"],
        })
    stem = CONTROL / f"sortable_capture_manifest_1981_asof_{AS_OF}_team_statistics"
    fields = list(rows[0]); write_csv(stem.with_suffix(".csv"), fields, rows)
    write_json(stem.with_suffix(".json"), {"as_of_date": AS_OF, "capture_scope": "team_statistics", "staging_folder": str(TEAM_STAGING), "file_count": len(rows), "files": rows})
    lines = ["# Sortable Capture Manifest — Team Statistics — 1981 as of 1981-07-19", "",
             f"- Staging folder: `{TEAM_STAGING}`", f"- Files: **{len(rows)}**", "- Hash: SHA-256", "- Status: staged; not promoted", "",
             "| File | Bytes | Modified UTC | Rows | Columns | SHA-256 | Reconciliation | Recommendation |",
             "|---|---:|---|---:|---:|---|---|---|"]
    for row in rows:
        lines.append(f"| `{row['file_name']}` | {row['file_size_bytes']} | {row['modified_timestamp']} | {row['row_count']} | {row['column_count']} | `{row['sha256']}` | `{row['reconciliation_status']}` | `{row['promotion_recommendation']}` |")
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_combined(existing_names: list[str], player_rows: list[dict[str, str]], team_rows: list[dict[str, object]], team_info: dict[str, object]) -> None:
    player_by_existing = {row["existing_file"]: row for row in player_rows if row["record_type"] == "STAGING_FILE" and row["recommendation"] == "PROMOTE_REPLACE_EXISTING"}
    team_by_existing = {str(row["existing_file"]): row for row in team_rows if row["record_type"] == "TEAM_STAGING_FILE" and row["recommendation"] == "PROMOTE_REPLACE_EXISTING"}
    combined = []
    for name in sorted(existing_names):
        p = player_by_existing.get(name); t = team_by_existing.get(name)
        chosen = p or t
        combined.append({
            "existing_file": name, "player_staging_file": p["staging_file"] if p else "",
            "team_staging_file": t["staging_file"] if t else "",
            "safe_replacement_found": "yes" if chosen else "no",
            "replacement_source": "player_statistics" if p else ("team_statistics" if t else ""),
            "replacement_match_type": chosen["match_type"] if chosen else "UNRESOLVED",
            "promotion_recommendation": chosen["recommendation"] if chosen else "HOLD_UNKNOWN_REVIEW",
        })
    player_all = [row for row in player_rows if row["record_type"] == "STAGING_FILE"]
    team_all = [row for row in team_rows if row["record_type"] == "TEAM_STAGING_FILE"]
    new_review = read_csv_rows(NEW_PLAYER_REVIEW) if NEW_PLAYER_REVIEW.exists() else []
    approved_player_new = sum(row.get("recommendation") == "register_as_new_source" for row in new_review)
    reviewed_player_duplicates = sum(row.get("recommendation") == "hold_as_duplicate" for row in new_review)
    reviewed_player_unknown = sum(row.get("recommendation") == "hold_for_review" for row in new_review)
    safe = sum(row["safe_replacement_found"] == "yes" for row in combined)
    info = {
        "official_existing_sources": len(combined), "safe_replacements_found": safe,
        "existing_sources_still_missing": len(combined) - safe,
        "proposed_player_new_sources_before_column_review": sum(row["recommendation"] == "PROMOTE_AS_NEW_SOURCE" for row in player_all),
        "approved_new_source_candidates_after_review": approved_player_new + int(team_info["new_source_candidates"]),
        "duplicates_or_overlaps": sum(row["recommendation"] == "HOLD_DUPLICATE_OR_OVERLAP" for row in player_all + team_all) + reviewed_player_duplicates,
        "unknown_manual_review_files": sum(row["recommendation"] == "HOLD_UNKNOWN_REVIEW" for row in player_all + team_all) + reviewed_player_unknown,
        "staff_manager_reports_still_missing": team_info["staff_manager_reports_still_missing"],
        "promotion_safe": safe == len(combined) and not team_info["staff_manager_reports_still_missing"],
        "next_required_capture": "Re-export full 23-column abl_staff.csv and full 20-column batting_stats.csv into team staging." if safe < len(combined) else "none",
    }
    stem = CONTROL / f"sortable_capture_combined_status_1981_asof_{AS_OF}"
    write_csv(stem.with_suffix(".csv"), list(combined[0]), combined)
    write_json(stem.with_suffix(".json"), {"as_of_date": AS_OF, "summary": info, "sources": combined})
    lines = ["# Combined Sortable Capture Status — 1981 as of 1981-07-19", "",
             f"- Official governed sources: **{info['official_existing_sources']}**",
             f"- Safe replacements found: **{info['safe_replacements_found']}**",
             f"- Still missing safe replacements: **{info['existing_sources_still_missing']}**",
             f"- Approved new-source candidates after column review: **{info['approved_new_source_candidates_after_review']}**",
             f"- Duplicates/overlaps across staging captures: **{info['duplicates_or_overlaps']}**",
             f"- Unknown/manual-review staging files: **{info['unknown_manual_review_files']}**",
             f"- Staff/manager reports still unresolved: **{'YES' if info['staff_manager_reports_still_missing'] else 'NO'}**",
             f"- Promotion safe: **{'YES' if info['promotion_safe'] else 'NO'}**",
             f"- Next required capture: **{info['next_required_capture']}**", "",
             "| Existing governed source | Player staging | Team staging | Safe replacement | Recommendation |",
             "|---|---|---|---|---|"]
    for row in combined:
        lines.append(f"| `{row['existing_file']}` | `{row['player_staging_file'] or '—'}` | `{row['team_staging_file'] or '—'}` | {row['safe_replacement_found']} | `{row['promotion_recommendation']}` |")
    lines += ["", "Promotion remains blocked until all 20 governed sources have safe replacements and overlap/new-source decisions are reflected in a reviewed dry-run promotion manifest.", ""]
    stem.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if not TEAM_STAGING.is_dir():
        raise NotADirectoryError(TEAM_STAGING)
    for required in (PLAYER_RECONCILIATION, MISSING_REPLACEMENTS, REGISTRY):
        if not required.exists():
            raise FileNotFoundError(required)
    existing = [profile(path) for path in sorted(EXISTING_DIR.glob("*.csv"))]
    staging = [profile(path) for path in sorted(TEAM_STAGING.glob("*.csv"))]
    missing_rows = read_csv_rows(MISSING_REPLACEMENTS)
    missing = {row["existing_file_name"] for row in missing_rows}
    rows = reconcile(existing, staging, missing)
    info = summary(rows, missing)
    staging_rows = [row for row in rows if row["record_type"] == "TEAM_STAGING_FILE"]
    if len(existing) != 20 or len(staging) != 17 or len(missing) != 10:
        raise ValueError(f"Unexpected governed counts: existing={len(existing)}, staging={len(staging)}, missing={len(missing)}")
    if any(row["recommendation"] not in RECOMMENDATIONS for row in rows):
        raise ValueError("Invalid recommendation emitted")
    write_reconciliation(rows, info)
    write_manifest(staging_rows)
    player_rows = read_csv_rows(PLAYER_RECONCILIATION)
    write_combined([str(item["name"]) for item in existing], player_rows, rows, info)
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
