from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONTROL = ROOT / "csv" / "out" / "control"
RECONCILIATION = CONTROL / "sortable_stats_reconciliation_1981_asof_1981-07-19.csv"
REGISTRY = CONTROL / "abl_source_registry.csv"
EXISTING_DIR = ROOT / "csv" / "abl_statistics"
STAGING_DIR = Path(
    "C:/Users/earld/OneDrive/Documents/Out of the Park Developments/"
    "OOTP Baseball 26/saved_games/Action Baseball League.lg/import_export/"
    "abl_statistics_player_statistics"
)
AS_OF = "1981-07-19"


def normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.strip().lower())


def read_profile(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        usable = [line for line in handle if line.strip() and not line.lstrip().startswith("#")]
    parsed = list(csv.reader(usable))
    headers = parsed[0] if parsed else []
    rows = [row for row in parsed[1:] if any(cell.strip() for cell in row)]
    return {
        "headers": headers, "normalized_headers": {normalize_header(value) for value in headers},
        "row_count": len(rows), "column_count": len(headers),
    }


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def missing_category(filename: str) -> str:
    lower = filename.lower()
    if "abl_staff" in lower:
        return "staff"
    if "team_finan" in lower:
        return "financials"
    if "team_pers_park" in lower:
        return "park/personality"
    if "team_cur_rec_hist" in lower:
        return "team context"
    if any(token in lower for token in ("batting", "pitching", "c_fielding")):
        return "team stats"
    return "other"


def missing_details(filename: str) -> tuple[str, str, str]:
    lower = filename.lower()
    if "abl_staff" in lower:
        return ("Supplies the team-to-manager/staff layer required for management context and later manager signals.", "yes", "P1 - high")
    if "batting_stats" in lower:
        return ("Provides the 24-team offensive totals needed to reconcile and enrich team identity.", "yes", "P0 - critical")
    if "batting_xtra" in lower:
        return ("Adds advanced team batting/baserunning denominators not guaranteed by the player-only capture.", "yes", "P0 - critical")
    if "c_fielding" in lower:
        return ("Provides team catcher-defense and running-control evidence for defensive context.", "yes", "P0 - critical")
    if "pitching_1" in lower:
        return ("Provides core 24-team pitching totals and workload needed for team run-prevention enrichment.", "yes", "P0 - critical")
    if "pitching_2" in lower:
        return ("Provides advanced 24-team pitching rates and components for validated run-prevention profiles.", "yes", "P0 - critical")
    if "team_cur_rec_hist" in lower:
        return ("Provides a supplemental OOTP team-record/history report for reconciliation against raw-game standings; it cannot establish the cutoff.", "no - validation/context only", "P1 - high")
    if "team_finan" in lower:
        return ("Provides payroll, budget, attendance, and financial context for organization and fan stories.", "no - context only", "P2 - context")
    if "team_pers_park" in lower:
        return ("Provides park and team personality context unavailable in a player-statistics-only capture.", "no - context only", "P2 - context")
    return ("Preserves an existing governed sortable-stat contract.", "review", "P2 - review")


def build_missing(reconciliation: list[dict[str, str]], registry: dict[str, dict[str, str]]) -> list[dict[str, object]]:
    rows = []
    for item in reconciliation:
        if item["record_type"] != "EXISTING_NO_MATCH":
            continue
        filename = item["existing_file"]
        rel_path = f"csv/abl_statistics/{filename}"
        reg = registry.get(rel_path, {})
        why, required, priority = missing_details(filename)
        rows.append({
            "existing_file_name": filename,
            "subject_area": reg.get("subject_area", item["subject_area"]),
            "likely_grain": reg.get("likely_grain", item["likely_grain"]),
            "source_family": reg.get("source_family", "sortable_stats"),
            "authority_level": reg.get("authority_level", "supplemental_report_extract"),
            "volatility": reg.get("volatility", "unknown"),
            "why_it_matters": why,
            "required_for_current_state_enrichment": required,
            "likely_ootp_report_family": missing_category(filename),
            "recommended_capture_priority": priority,
        })
    return sorted(rows, key=lambda row: (row["recommended_capture_priority"], row["existing_file_name"]))


NEW_SOURCE_VALUE = {
    "batting_potential": ("Player offensive ceiling and development context; not current performance evidence.", "register_as_new_source", 92),
    "financial_info": ("Player salary, contract, and option context for roster/organization reporting.", "register_as_new_source", 90),
    "individual_pitch_potential": ("Pitch-by-pitch arsenal potential distinct from current pitch ratings.", "register_as_new_source", 92),
    "personality_morale": ("Structured personality and morale context with strict guardrails against invented motives.", "register_as_new_source", 86),
    "pitching_potential": ("Overall pitcher ceiling and development context distinct from current ratings.", "register_as_new_source", 92),
    "popularity_info": ("Local/national popularity context for fan-interest reporting.", "register_as_new_source", 88),
}


def new_key(filename: str) -> str:
    lower = filename.lower()
    for key in NEW_SOURCE_VALUE:
        if key.replace("_", "") in re.sub(r"[^a-z]", "", lower):
            return key
    raise ValueError(f"No governed new-source rule for {filename}")


def build_new_review(reconciliation: list[dict[str, str]], existing_profiles: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    all_existing_columns = set().union(*(profile["normalized_headers"] for profile in existing_profiles.values()))
    rows = []
    for item in reconciliation:
        if item["recommendation"] != "PROMOTE_AS_NEW_SOURCE":
            continue
        filename = item["staging_file"]
        profile = read_profile(STAGING_DIR / filename)
        existing_name = item["existing_file"]
        existing = existing_profiles[existing_name]
        shared = sorted(profile["normalized_headers"] & existing["normalized_headers"])
        unique_original = [
            header for header in profile["headers"]
            if normalize_header(header) not in all_existing_columns
        ]
        key = new_key(filename)
        value, recommendation, confidence = NEW_SOURCE_VALUE[key]
        if not unique_original:
            recommendation = "hold_as_duplicate"
            confidence = 95
            value += " Column review found no fields absent from the existing governed source set."
        rows.append({
            "staging_file_name": filename, "row_count": profile["row_count"],
            "column_count": profile["column_count"], "column_headers": "|".join(profile["headers"]),
            "inferred_subject_area": item["subject_area"], "inferred_grain": item["likely_grain"],
            "overlap_with_existing_sources": f"{existing_name}; {len(shared)} normalized shared columns: {'|'.join(shared)}",
            "unique_columns_not_found_in_existing_sources": "|".join(unique_original),
            "story_engine_value": value, "recommendation": recommendation,
            "confidence_score": confidence,
        })
    return sorted(rows, key=lambda row: row["staging_file_name"])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_capture_manifest(reconciliation: list[dict[str, str]], review_decisions: dict[str, str]) -> list[dict[str, object]]:
    by_name = {row["staging_file"]: row for row in reconciliation if row["record_type"] == "STAGING_FILE"}
    rows = []
    for path in sorted(STAGING_DIR.glob("*.csv")):
        profile = read_profile(path)
        rec = by_name[path.name]
        rows.append({
            "file_name": path.name, "full_path": str(path), "file_size_bytes": path.stat().st_size,
            "modified_timestamp": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
            "row_count": profile["row_count"], "column_count": profile["column_count"],
            "sha256": sha256(path), "reconciliation_status": rec["match_type"],
            "promotion_recommendation": review_decisions.get(path.name, rec["recommendation"]),
        })
    return rows


def write_missing(rows: list[dict[str, object]]) -> None:
    stem = CONTROL / f"missing_sortable_replacements_1981_asof_{AS_OF}"
    fields = list(rows[0])
    write_csv(stem.with_suffix(".csv"), fields, rows)
    write_json(stem.with_suffix(".json"), {"as_of_date": AS_OF, "missing_count": len(rows), "missing_sources": rows})
    lines = ["# Missing Sortable Replacements — 1981 as of 1981-07-19", "",
             f"Missing governed sources: **{len(rows)}**", "",
             "These files remain in the official source set but have no counterpart in the player-statistics staging capture.", "",
             "| Priority | OOTP family | Existing file | Grain | Required for current enrichment | Why it matters |",
             "|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['recommended_capture_priority']} | {row['likely_ootp_report_family']} | `{row['existing_file_name']}` | {row['likely_grain']} | {row['required_for_current_state_enrichment']} | {row['why_it_matters']} |")
    lines += ["", "A separate team/staff/context staging capture is required. Absence here does not authorize retirement or deletion.", ""]
    stem.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def write_new_review(rows: list[dict[str, object]]) -> None:
    stem = CONTROL / f"new_sortable_source_review_1981_asof_{AS_OF}"
    write_csv(stem.with_suffix(".csv"), list(rows[0]), rows)
    write_json(stem.with_suffix(".json"), {"as_of_date": AS_OF, "reviewed_count": len(rows), "sources": rows})
    lines = ["# New Sortable Source Review — 1981 as of 1981-07-19", "",
             f"Proposed new sources reviewed: **{len(rows)}**", "",
             "| Staging file | Rows | Columns | Unique columns | Recommendation | Confidence | Story-engine value |",
             "|---|---:|---:|---|---|---:|---|"]
    for row in rows:
        unique = row["unique_columns_not_found_in_existing_sources"] or "none"
        lines.append(f"| `{row['staging_file_name']}` | {row['row_count']} | {row['column_count']} | `{unique}` | `{row['recommendation']}` | {row['confidence_score']} | {row['story_engine_value']} |")
    lines += ["", "Registration does not authorize promotion or current-state use. Each source remains supplemental and must be tied to a validated capture batch.", ""]
    stem.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def write_manifest(rows: list[dict[str, object]]) -> None:
    stem = CONTROL / f"sortable_capture_manifest_1981_asof_{AS_OF}_player_statistics"
    write_csv(stem.with_suffix(".csv"), list(rows[0]), rows)
    write_json(stem.with_suffix(".json"), {
        "source_family": "sortable_stats", "capture_scope": "player_statistics_subset",
        "intended_as_of_date": AS_OF, "staging_folder": str(STAGING_DIR),
        "file_count": len(rows), "files": rows,
    })
    lines = ["# Sortable Capture Manifest — Player Statistics — 1981 as of 1981-07-19", "",
             f"- Staging folder: `{STAGING_DIR}`", f"- Files: **{len(rows)}**",
             "- Hash: SHA-256", "- Promotion status: **staged; not promoted**", "",
             "| File | Bytes | Modified (UTC) | Rows | Columns | SHA-256 | Reconciliation | Recommendation |",
             "|---|---:|---|---:|---:|---|---|---|"]
    for row in rows:
        lines.append(f"| `{row['file_name']}` | {row['file_size_bytes']} | {row['modified_timestamp']} | {row['row_count']} | {row['column_count']} | `{row['sha256']}` | `{row['reconciliation_status']}` | `{row['promotion_recommendation']}` |")
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_checklist(missing: list[dict[str, object]]) -> None:
    stem = CONTROL / f"next_sortable_capture_checklist_1981_asof_{AS_OF}"
    ordered = sorted(missing, key=lambda row: (row["recommended_capture_priority"], row["likely_ootp_report_family"], row["existing_file_name"]))
    payload = {
        "as_of_date": AS_OF, "second_staging_folder_required": True,
        "required_scope": "team_statistics_staff_and_team_context",
        "promotion_blocked": True, "files_to_capture": ordered,
    }
    write_json(stem.with_suffix(".json"), payload)
    lines = ["# Next Sortable Capture Checklist — 1981 as of 1981-07-19", "",
             "Promotion remains **blocked**. Create a second staging folder for team statistics, staff, and team context. Capture every report below; do not place these directly into `csv/abl_statistics/`.", ""]
    current = None
    for row in ordered:
        if row["recommended_capture_priority"] != current:
            current = row["recommended_capture_priority"]
            lines += [f"## {current}", ""]
        lines.append(f"- [ ] `{row['existing_file_name']}` — OOTP category: **{row['likely_ootp_report_family']}** — {row['why_it_matters']}")
    lines += ["", "## Required completion checks", "",
              "- [ ] The second staging folder contains all ten filenames or documented structurally equivalent exports.",
              "- [ ] Re-run structural reconciliation across both staging folders.",
              "- [ ] Create SHA-256 capture manifests for both folders.",
              "- [ ] Confirm all 20 governed existing source contracts have a reviewed replacement or an explicit retain-old decision.",
              "- [ ] Resolve the six proposed new-source registrations and all overlap holds.",
              "- [ ] Prepare a dry-run promotion plan and request explicit authorization before copying or replacing any source.", ""]
    stem.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    with RECONCILIATION.open("r", encoding="utf-8-sig", newline="") as handle:
        reconciliation = list(csv.DictReader(handle))
    with REGISTRY.open("r", encoding="utf-8-sig", newline="") as handle:
        registry = {row["file_path"]: row for row in csv.DictReader(handle)}
    existing_profiles = {path.name: read_profile(path) for path in EXISTING_DIR.glob("*.csv")}
    missing = build_missing(reconciliation, registry)
    new_review = build_new_review(reconciliation, existing_profiles)
    review_decisions = {row["staging_file_name"]: row["recommendation"] for row in new_review}
    manifest = build_capture_manifest(reconciliation, review_decisions)
    if len(missing) != 10 or len(new_review) != 6 or len(manifest) != 27:
        raise ValueError(f"Unexpected governed counts: missing={len(missing)}, new={len(new_review)}, manifest={len(manifest)}")
    write_missing(missing)
    write_new_review(new_review)
    write_manifest(manifest)
    write_checklist(missing)
    print(json.dumps({
        "missing_existing_sources": len(missing), "new_sources_reviewed": len(new_review),
        "staging_files_hashed": len(manifest), "second_staging_folder_required": True,
        "promotion_blocked": True,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
