from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXISTING = ROOT / "csv" / "abl_statistics"
DEFAULT_STAGING = Path(
    "C:/Users/earld/OneDrive/Documents/Out of the Park Developments/"
    "OOTP Baseball 26/saved_games/Action Baseball League.lg/import_export/"
    "abl_statistics_player_statistics"
)
OUT_DIR = ROOT / "csv" / "out" / "control"
SOURCE_REGISTRY = OUT_DIR / "abl_source_registry.csv"
OUTPUT_STEM = "sortable_stats_reconciliation_1981_asof_1981-07-19"
FIELDS = [
    "record_type", "staging_file", "existing_file", "staging_normalized_name",
    "existing_normalized_name", "staging_row_count", "existing_row_count",
    "staging_column_count", "existing_column_count", "header_jaccard",
    "staging_header_containment", "filename_similarity", "subject_area",
    "likely_grain", "match_type", "confidence_score", "recommendation",
    "duplicate_or_overlap_with", "promotion_reason", "warnings",
]


def normalize_name(name: str) -> str:
    stem = Path(name).stem.lower()
    stem = re.sub(r"^abl_statistics_(player|team)_statistics_+", "", stem)
    stem = re.sub(r"(?:info_)?-?_?sortable_stats_", "", stem)
    tokens = re.findall(r"[a-z0-9]+", stem)
    aliases = {"bat": "batting", "pitch": "pitching", "finan": "financial"}
    return "_".join(aliases.get(token, token) for token in tokens)


def normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.strip().lower())


def read_profile(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        usable = [line for line in handle if line.strip() and not line.lstrip().startswith("#")]
    parsed = list(csv.reader(usable))
    header = parsed[0] if parsed else []
    rows = [row for row in parsed[1:] if any(cell.strip() for cell in row)]
    normalized = [normalize_header(cell) for cell in header]
    return {
        "path": path, "name": path.name, "normalized_name": normalize_name(path.name),
        "header": header, "normalized_header": normalized, "header_set": set(normalized),
        "row_count": len(rows), "column_count": len(header),
        "size_bytes": path.stat().st_size,
        "header_hash": hashlib.sha256("|".join(normalized).encode("utf-8")).hexdigest(),
    }


def subject_area(profile: dict[str, object]) -> str:
    name = str(profile["normalized_name"])
    if "financial" in name:
        return "player_financial_context"
    if "personality" in name or "morale" in name:
        return "player_personality_and_morale"
    if "popularity" in name:
        return "player_popularity"
    if "fielding" in name or "_field_" in f"_{name}_" or "position_ratings" in name:
        return "player_fielding"
    if "pitch" in name:
        return "player_pitching"
    if "bat" in name:
        return "player_batting"
    if "indicative" in name or "misc" in name or name == "default":
        return "player_identity_and_indicative"
    if any(token in name for token in ("staff", "manager")):
        return "team_staff"
    if any(token in name for token in ("team", "c_fielding")):
        return "team_stats_or_context"
    return "unknown"


def likely_grain(profile: dict[str, object]) -> str:
    headers = profile["header_set"]
    rows = int(profile["row_count"])
    if "name" in headers and rows > 100:
        if "pos" in headers and any(token in str(profile["normalized_name"]) for token in ("fielding", "position")):
            return "player-position capture"
        return "player capture"
    if rows in range(20, 31):
        return "team capture"
    return "unknown"


def schema_metrics(new: dict[str, object], old: dict[str, object]) -> tuple[float, float]:
    a, b = new["header_set"], old["header_set"]
    if not a and not b:
        return 1.0, 1.0
    intersection = len(a & b)
    union = len(a | b)
    return (intersection / union if union else 0.0, intersection / len(a) if a else 0.0)


def candidate_score(new: dict[str, object], old: dict[str, object]) -> dict[str, object]:
    jaccard, containment = schema_metrics(new, old)
    filename = SequenceMatcher(None, str(new["normalized_name"]), str(old["normalized_name"])).ratio()
    subject_match = subject_area(new) == subject_area(old)
    grain_match = likely_grain(new) == likely_grain(old)
    column_ratio = min(int(new["column_count"]), int(old["column_count"])) / max(int(new["column_count"]), int(old["column_count"]), 1)
    exact_name = new["name"].lower() == old["name"].lower()
    score = (
        (0.30 if exact_name else 0.0) + 0.20 * filename + 0.25 * jaccard
        + 0.15 * containment + 0.05 * column_ratio
        + (0.03 if subject_match else 0.0) + (0.02 if grain_match else 0.0)
    )
    return {
        "old": old, "jaccard": jaccard, "containment": containment,
        "filename": filename, "score": min(score, 1.0), "exact_name": exact_name,
        "subject_match": subject_match, "column_ratio": column_ratio,
    }


def recognized_sortable(profile: dict[str, object]) -> bool:
    return (
        str(profile["name"]).lower().endswith(".csv")
        and "sortable_stats" in str(profile["name"]).lower()
        and "name" in profile["header_set"]
        and int(profile["row_count"]) > 0
    )


def duplicate_groups(staging: list[dict[str, object]]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for profile in staging:
        groups[str(profile["header_hash"])].append(str(profile["name"]))
    return {name: files for files in groups.values() if len(files) > 1 for name in files}


def reconcile(existing: list[dict[str, object]], staging: list[dict[str, object]]) -> list[dict[str, object]]:
    duplicate_map = duplicate_groups(staging)
    rows: list[dict[str, object]] = []
    matched_existing: set[str] = set()
    for new in staging:
        candidate_pool = existing
        if likely_grain(new).startswith("player"):
            player_existing = [old for old in existing if int(old["row_count"]) > 100]
            if player_existing:
                candidate_pool = player_existing
        scored = sorted((candidate_score(new, old) for old in candidate_pool), key=lambda item: item["score"], reverse=True)
        best = scored[0]
        old = best["old"]
        exact_schema = new["normalized_header"] == old["normalized_header"]
        duplicate_names = [name for name in duplicate_map.get(str(new["name"]), []) if name != new["name"]]
        warnings: list[str] = []
        if int(new["row_count"]) != int(old["row_count"]):
            warnings.append(f"row count changed {old['row_count']} -> {new['row_count']}")
        if duplicate_names:
            match_type = "STAGING_DUPLICATE"
            recommendation = "HOLD_DUPLICATE_OR_OVERLAP"
            reason = "Staging file has an identical normalized header sequence to another staging file."
        elif best["exact_name"] and exact_schema:
            match_type = "EXACT_FILENAME_AND_SCHEMA"
            recommendation = "PROMOTE_REPLACE_EXISTING"
            reason = "Exact filename and column sequence match; changed row count is consistent with a newer capture."
            matched_existing.add(str(old["name"]))
        elif best["exact_name"]:
            match_type = "EXACT_FILENAME_SCHEMA_CHANGED"
            recommendation = "HOLD_UNKNOWN_REVIEW"
            reason = "Filename matches but schema changed; replacement requires explicit schema review."
            matched_existing.add(str(old["name"]))
            warnings.append("exact filename does not preserve the existing schema")
        elif best["jaccard"] >= 0.88 and best["column_ratio"] >= 0.90:
            match_type = "RENAMED_STRUCTURAL_MATCH"
            recommendation = "PROMOTE_REPLACE_EXISTING"
            reason = "Different filename but near-identical schema and grain indicate a likely renamed export."
            matched_existing.add(str(old["name"]))
        elif "potential" in str(new["normalized_name"]) and recognized_sortable(new):
            match_type = "NEW_SOURCE_CANDIDATE"
            recommendation = "PROMOTE_AS_NEW_SOURCE"
            reason = "Potential/scouting view is semantically distinct from current ratings despite shared identity and pitch/rating columns."
        elif (best["containment"] >= 0.65 or best["filename"] >= 0.65) and best["subject_match"]:
            match_type = "OVERLAPPING_VIEW"
            recommendation = "HOLD_DUPLICATE_OR_OVERLAP"
            reason = "Most staging columns already occur in an existing broader source for the same subject."
        elif recognized_sortable(new) and subject_area(new) != "unknown":
            match_type = "NEW_SOURCE_CANDIDATE"
            recommendation = "PROMOTE_AS_NEW_SOURCE"
            reason = "Recognized player sortable-stat export with a distinct schema and subject role."
        elif recognized_sortable(new):
            match_type = "UNKNOWN_SORTABLE"
            recommendation = "HOLD_UNKNOWN_REVIEW"
            reason = "Looks like a sortable-stat export, but subject/grain could not be classified safely."
        else:
            match_type = "NOT_ABL_SORTABLE_STATS"
            recommendation = "IGNORE_NOT_ABL_SORTABLE_STATS"
            reason = "File does not satisfy the ABL sortable-stat naming/content checks."
        confidence = 100.0 if match_type == "EXACT_FILENAME_AND_SCHEMA" else round(float(best["score"]) * 100, 1)
        rows.append({
            "record_type": "STAGING_FILE", "staging_file": new["name"], "existing_file": old["name"],
            "staging_normalized_name": new["normalized_name"], "existing_normalized_name": old["normalized_name"],
            "staging_row_count": new["row_count"], "existing_row_count": old["row_count"],
            "staging_column_count": new["column_count"], "existing_column_count": old["column_count"],
            "header_jaccard": round(float(best["jaccard"]), 4),
            "staging_header_containment": round(float(best["containment"]), 4),
            "filename_similarity": round(float(best["filename"]), 4),
            "subject_area": subject_area(new), "likely_grain": likely_grain(new),
            "match_type": match_type, "confidence_score": confidence,
            "recommendation": recommendation,
            "duplicate_or_overlap_with": "|".join(duplicate_names) if duplicate_names else (old["name"] if recommendation == "HOLD_DUPLICATE_OR_OVERLAP" else ""),
            "promotion_reason": reason, "warnings": "; ".join(warnings),
        })
    for old in existing:
        if str(old["name"]) not in matched_existing:
            rows.append({
                "record_type": "EXISTING_NO_MATCH", "staging_file": "", "existing_file": old["name"],
                "staging_normalized_name": "", "existing_normalized_name": old["normalized_name"],
                "staging_row_count": "", "existing_row_count": old["row_count"],
                "staging_column_count": "", "existing_column_count": old["column_count"],
                "header_jaccard": "", "staging_header_containment": "", "filename_similarity": "",
                "subject_area": subject_area(old), "likely_grain": likely_grain(old),
                "match_type": "EXISTING_SOURCE_WITH_NO_NEW_MATCH", "confidence_score": "",
                "recommendation": "HOLD_UNKNOWN_REVIEW", "duplicate_or_overlap_with": "",
                "promotion_reason": "Existing source has no replacement candidate in this staging capture.",
                "warnings": "Do not infer deletion or retirement from absence in a player-statistics staging folder.",
            })
    return rows


def summary(rows: list[dict[str, object]], existing_count: int, staging_count: int,
            unregistered_existing: list[str], registered_missing: list[str]) -> dict[str, object]:
    staging_rows = [row for row in rows if row["record_type"] == "STAGING_FILE"]
    counts = Counter(str(row["match_type"]) for row in staging_rows)
    recommendations = Counter(str(row["recommendation"]) for row in staging_rows)
    broader = any(str(row["subject_area"]).startswith("team_") for row in staging_rows)
    safe = (
        recommendations["HOLD_UNKNOWN_REVIEW"] == 0
        and recommendations["HOLD_DUPLICATE_OR_OVERLAP"] == 0
        and not any(row["record_type"] == "EXISTING_NO_MATCH" for row in rows)
        and not unregistered_existing and not registered_missing
    )
    return {
        "existing_files_inspected": existing_count, "staging_files_inspected": staging_count,
        "exact_filename_and_schema_matches": counts["EXACT_FILENAME_AND_SCHEMA"],
        "exact_filename_schema_changed": counts["EXACT_FILENAME_SCHEMA_CHANGED"],
        "likely_renamed_matches": counts["RENAMED_STRUCTURAL_MATCH"],
        "new_source_candidates": recommendations["PROMOTE_AS_NEW_SOURCE"],
        "duplicates_or_overlaps": recommendations["HOLD_DUPLICATE_OR_OVERLAP"],
        "unknown_manual_review": recommendations["HOLD_UNKNOWN_REVIEW"],
        "ignored_not_abl_sortable": recommendations["IGNORE_NOT_ABL_SORTABLE_STATS"],
        "existing_sources_with_no_new_match": sum(row["record_type"] == "EXISTING_NO_MATCH" for row in rows),
        "promotion_safe": safe,
        "capture_scope": "broader sortable-stats batch" if broader else "player-statistics subset only",
        "unregistered_existing_files": unregistered_existing,
        "registered_sources_missing_from_existing_folder": registered_missing,
        "warnings": [
            "No source files were promoted or modified.",
            "The staging path contains no team/staff/financial team reports; it is not a complete replacement for csv/abl_statistics.",
            "PROMOTE_AS_NEW_SOURCE remains a proposal until batch metadata, schema contracts, and overlap decisions are approved.",
        ],
    }


def write_outputs(rows: list[dict[str, object]], info: dict[str, object], existing_dir: Path, staging_dir: Path) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"{OUTPUT_STEM}.csv"
    json_path = OUT_DIR / f"{OUTPUT_STEM}.json"
    md_path = OUT_DIR / f"{OUTPUT_STEM}.md"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    payload = {"existing_folder": str(existing_dir), "staging_folder": str(staging_dir), "summary": info, "reconciliation": rows}
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    sections = {
        "Exact filename and schema matches": lambda r: r["match_type"] == "EXACT_FILENAME_AND_SCHEMA",
        "Likely renamed matches": lambda r: r["match_type"] == "RENAMED_STRUCTURAL_MATCH",
        "Existing sources with no new match": lambda r: r["record_type"] == "EXISTING_NO_MATCH",
        "Staging files with no existing replacement match": lambda r: r["record_type"] == "STAGING_FILE" and r["match_type"] in {"NEW_SOURCE_CANDIDATE", "UNKNOWN_SORTABLE", "OVERLAPPING_VIEW", "STAGING_DUPLICATE"},
        "Likely new source candidates": lambda r: r["recommendation"] == "PROMOTE_AS_NEW_SOURCE",
        "Duplicates or overlaps": lambda r: r["recommendation"] == "HOLD_DUPLICATE_OR_OVERLAP",
        "Unknown or manual review": lambda r: r["recommendation"] == "HOLD_UNKNOWN_REVIEW",
    }
    lines = [
        "# Sortable-Stats Reconciliation — 1981 as of 1981-07-19", "",
        f"- Existing sortable-stat CSVs inspected: **{info['existing_files_inspected']}**",
        f"- Staging capture files inspected: **{info['staging_files_inspected']}**",
        f"- Exact filename/schema matches: **{info['exact_filename_and_schema_matches']}**",
        f"- Likely renamed matches: **{info['likely_renamed_matches']}**",
        f"- New-source candidates: **{info['new_source_candidates']}**",
        f"- Duplicate/overlap holds: **{info['duplicates_or_overlaps']}**",
        f"- Unknown/manual-review staging files: **{info['unknown_manual_review']}**",
        f"- Existing sources with no new match: **{info['existing_sources_with_no_new_match']}**",
        f"- Capture scope: **{info['capture_scope']}**",
        f"- Promotion safe: **{'YES' if info['promotion_safe'] else 'NO'}**", "",
        "Matching prioritizes normalized header structure, column coverage, grain, and subject area. Filename similarity is supporting evidence only.", "",
    ]
    for title, predicate in sections.items():
        selected = [row for row in rows if predicate(row)]
        lines += [f"## {title}", ""]
        if not selected:
            lines += ["None.", ""]
            continue
        lines += ["| Staging | Existing/proposed match | Type | Confidence | Recommendation | Reason |", "|---|---|---|---:|---|---|"]
        for row in selected:
            lines.append(f"| `{row['staging_file'] or '—'}` | `{row['existing_file'] or '—'}` | `{row['match_type']}` | {row['confidence_score'] or '—'} | `{row['recommendation']}` | {row['promotion_reason']} |")
        lines.append("")
    lines += ["## Recommended promotion plan", "",
              "1. Do not promote the folder as a complete replacement; it is player-only and omits the ten existing team/staff/context reports.",
              "2. After a capture batch manifest and checksums exist, review exact filename/schema matches as replacement candidates.",
              "3. Approve genuinely distinct player reports as new registered sources only after column-level overlap review.",
              "4. Keep subset/overlap views on hold unless a curated consumer needs a distinct contract.",
              "5. Capture the missing team/staff/context report family separately before any full sortable-stats promotion.", "",
              "## Warnings", ""]
    lines += [f"- {warning}" for warning in info["warnings"]]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "markdown": str(md_path), "json": str(json_path), **info}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconcile a staged OOTP sortable-stats capture without promoting files.")
    parser.add_argument("--existing", type=Path, default=DEFAULT_EXISTING)
    parser.add_argument("--staging", type=Path, default=DEFAULT_STAGING)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.existing.is_dir():
        raise NotADirectoryError(args.existing)
    if not args.staging.is_dir():
        raise NotADirectoryError(args.staging)
    existing_paths = sorted(args.existing.glob("*.csv"))
    staging_paths = sorted(args.staging.glob("*.csv"))
    if not SOURCE_REGISTRY.exists():
        raise FileNotFoundError(f"Governed source registry not found: {SOURCE_REGISTRY}")
    with SOURCE_REGISTRY.open("r", encoding="utf-8-sig", newline="") as handle:
        registered = {
            row["file_path"] for row in csv.DictReader(handle)
            if row["source_family"] == "sortable_stats"
        }
    observed = {path.relative_to(ROOT).as_posix() for path in existing_paths}
    unregistered_existing = sorted(observed - registered)
    registered_missing = sorted(registered - observed)
    existing = [read_profile(path) for path in existing_paths]
    staging = [read_profile(path) for path in staging_paths]
    rows = reconcile(existing, staging)
    info = summary(rows, len(existing), len(staging), unregistered_existing, registered_missing)
    write_outputs(rows, info, args.existing, args.staging)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
