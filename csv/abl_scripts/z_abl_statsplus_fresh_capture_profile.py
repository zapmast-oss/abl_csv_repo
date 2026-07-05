#!/usr/bin/env python3
"""Profile a fresh StatsPlus staging capture without promoting source files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONTROL = REPO / "csv" / "out" / "control"
LEGACY_INVENTORY = CONTROL / "statsplus_legacy_inventory.csv"
RAW_DIR = REPO / "csv" / "ootp_csv"
SORTABLE_DIR = REPO / "csv" / "abl_statistics"
AUTHORITY_DOC = REPO / "docs" / "ABL_STATSPLUS_AUTHORITY_RULES.md"

FAMILY_KEYWORDS = [
    ("Grand Tournament of Champions", ("gtoc", "grand tournament", "grand champ")),
    ("historical fan interest", ("historical fan", "fan interest history")),
    ("playoff odds by division", ("playoff odds div", "odds division")),
    ("playoff odds by league", ("playoff odds league", "odds league")),
    ("front office/coaches", ("front office", "coaches", "coach staff")),
    ("best batting game", ("best batting game", "batting game")),
    ("best pitching game", ("best pitching game", "pitching game")),
    ("player baserunning", ("player baserunning", "player base running")),
    ("team baserunning", ("team baserunning", "team base running")),
    ("player batting", ("player batting",)),
    ("player pitching", ("player pitching", "player p itching")),
    ("player fielding", ("player fielding",)),
    ("team batting by division", ("team batting div",)),
    ("team batting by league", ("team batting league",)),
    ("team pitching by division", ("team pitching div",)),
    ("team pitching by league", ("team pitching league",)),
    ("team fielding by division", ("team fielding div",)),
    ("team fielding by league", ("team fielding league",)),
    ("injury summary", ("injury", "injuries")),
    ("league standings", ("standings", "league table")),
    ("BaseRuns", ("base runs", "baseruns")),
    ("ELO ratings", ("elo",)),
    ("team WAR", ("team war",)),
    ("team age", ("team age", "age data")),
    ("fan data", ("fan data", "fan interest")),
    ("financials", ("financial", "payroll", "budget")),
    ("owner info", ("owner",)),
]

MODEL_FAMILIES = {"playoff odds by division", "playoff odds by league", "BaseRuns", "ELO ratings", "team WAR"}
STATSP_SPECIFIC = MODEL_FAMILIES | {
    "historical fan interest", "fan data", "financials", "owner info", "injury summary",
    "best batting game", "best pitching game", "Grand Tournament of Champions", "team baserunning",
    "player baserunning", "team age",
}
CROSSCHECK_ONLY = {
    "league standings", "team batting by division", "team batting by league", "team pitching by division",
    "team pitching by league", "team fielding by division", "team fielding by league", "player batting",
    "player pitching", "player fielding", "front office/coaches",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile and reconcile a fresh StatsPlus staging capture.")
    parser.add_argument("staging_folder", type=Path, help="Fresh StatsPlus staging folder to inspect recursively.")
    parser.add_argument("--season", type=int, default=1981)
    parser.add_argument("--as-of-date", default="1981-07-19", help="Expected cutoff in YYYY-MM-DD form.")
    parser.add_argument("--output-dir", type=Path, default=CONTROL)
    parser.add_argument("--legacy-inventory", type=Path, default=LEGACY_INVENTORY)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--sortable-dir", type=Path, default=SORTABLE_DIR)
    parser.add_argument("--authority-rules", type=Path, default=AUTHORITY_DOC)
    return parser.parse_args()


def read_text(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace"), "utf-8-replace"


def csv_info(path: Path) -> dict:
    content, encoding = read_text(path)
    rows = list(csv.reader(content.splitlines()))
    headers = rows[0] if rows else []
    data = [row for row in rows[1:] if any(cell.strip() for cell in row)]
    return {"headers": headers, "row_count": len(data), "column_count": len(headers), "encoding": encoding, "sample_rows": data[:3]}


def norm(value: str) -> str:
    value = re.sub(r"<[^>]+>|&nbsp;", " ", str(value).lower())
    return " ".join(re.findall(r"[a-z0-9]+", value))


def filename_tokens(path_or_name: str) -> set[str]:
    name = Path(path_or_name).stem
    name = re.sub(r"^\d{2}_", "", name)
    return set(norm(name).split()) - {"abl", "statsplus", "csv", "sortable", "stats"}


def header_set(headers: list[str]) -> set[str]:
    return {norm(x) for x in headers if norm(x)}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


def family_from_name(name: str) -> str:
    cleaned = norm(name)
    for family, terms in FAMILY_KEYWORDS:
        if any(term in cleaned for term in terms):
            return family
    return "unknown"


def infer_grain(family: str, headers: list[str], extension: str) -> str:
    if extension == ".txt":
        return "text/deep-dive notes"
    if family in {"front office/coaches"}:
        return "staff/front office"
    if family.startswith("player"):
        return "player"
    if family.startswith("best "):
        return "game"
    if family in {"historical fan interest", "Grand Tournament of Champions"}:
        return "historical/franchise"
    if family in {"playoff odds by division", "playoff odds by league"}:
        return "model/projection"
    normalized = header_set(headers)
    if {"game id", "date"} <= normalized:
        return "game"
    if "division" in normalized and "team" not in normalized:
        return "division"
    return "team" if family != "unknown" else "unknown"


def legacy_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if row.get("likely_source_role") == "StatsPlus legacy source"]


def legacy_headers(row: dict) -> list[str]:
    try:
        return json.loads(row.get("column_headers", "[]"))
    except json.JSONDecodeError:
        return []


def match_score(fresh: dict, legacy: dict) -> dict:
    fheaders = header_set(fresh.get("headers", []))
    lheaders = header_set(legacy_headers(legacy))
    hs = jaccard(fheaders, lheaders)
    ns = jaccard(filename_tokens(fresh["file_name"]), filename_tokens(legacy["file_name"]))
    try:
        lc = int(legacy.get("column_count") or 0)
    except ValueError:
        lc = 0
    fc = fresh.get("column_count", 0)
    cs = 1.0 if lc == fc and lc else max(0.0, 1.0 - abs(lc - fc) / max(lc, fc, 1))
    name_family = family_from_name(fresh["file_name"])
    fs = 1.0 if name_family == legacy.get("inferred_table_family") else 0.0
    score = 0.60 * hs + 0.15 * ns + 0.15 * cs + 0.10 * fs
    return {"score": round(score, 4), "header_similarity": round(hs, 4), "filename_similarity": round(ns, 4), "column_similarity": round(cs, 4)}


def authority_class(family: str, extension: str) -> tuple[str, str]:
    if extension == ".txt":
        return "HISTORICAL_REFERENCE_ONLY", "Deep Dive text is design/reference material, not factual authority."
    if family == "league standings":
        return "RAW_OOTP_GOVERNS", "StatsPlus standings are crosscheck-only."
    if family in CROSSCHECK_ONLY:
        return "CROSSCHECK_ONLY", "Raw OOTP or promoted sortable stats govern overlapping observed fields."
    if family in STATSP_SPECIFIC:
        return "STATSPLUS_SPECIFIC_AFTER_PROMOTION", "StatsPlus may govern its specific fields after cutoff/schema validation and explicit promotion."
    return "HOLD_REVIEW", "No field-level authority is established for this table."


def schema_library(folder: Path) -> list[dict]:
    output = []
    if not folder.is_dir():
        return output
    for path in folder.rglob("*.csv"):
        try:
            info = csv_info(path)
            output.append({"path": str(path.relative_to(REPO)), "headers": info["headers"], "column_count": info["column_count"]})
        except Exception:
            continue
    return output


def best_schema_overlap(headers: list[str], library: list[dict]) -> tuple[str, float]:
    target = header_set(headers)
    ranked = [(jaccard(target, header_set(item["headers"])), item["path"]) for item in library]
    if not ranked:
        return "", 0.0
    score, path = max(ranked)
    return path, round(score, 4)


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else ["status"])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def md_table(headers: list[str], rows: list[list]) -> str:
    clean = lambda value: str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(clean(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    staging = args.staging_folder.resolve()
    try:
        date.fromisoformat(args.as_of_date)
    except ValueError as exc:
        raise SystemExit(f"Invalid --as-of-date: {args.as_of_date}") from exc
    for required, label in ((staging, "staging folder"), (args.legacy_inventory, "legacy inventory"),
                            (args.authority_rules, "authority rules")):
        if not required.exists():
            raise SystemExit(f"Missing {label}: {required}")
    if not staging.is_dir():
        raise SystemExit(f"Staging path is not a directory: {staging}")
    resolved_output = args.output_dir.resolve()
    if staging == resolved_output or resolved_output in staging.parents or staging in resolved_output.parents:
        raise SystemExit("The staging folder and generated control-output folder must not contain one another.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = f"{args.season}_asof_{args.as_of_date}"
    legacy = legacy_rows(args.legacy_inventory)
    raw_schemas = schema_library(args.raw_dir)
    sortable_schemas = schema_library(args.sortable_dir)
    all_files = sorted((path for path in staging.rglob("*") if path.is_file()), key=lambda p: str(p).lower())
    digest_groups: dict[str, list[Path]] = defaultdict(list)
    manifest = []
    file_data = []

    for path in all_files:
        extension = path.suffix.lower()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        digest_groups[digest].append(path)
        readable = True
        warning = []
        info = {"headers": [], "row_count": "", "column_count": "", "encoding": "", "sample_rows": []}
        preview = []
        try:
            if extension == ".csv":
                info = csv_info(path)
            elif extension == ".txt":
                content, encoding = read_text(path)
                info["encoding"] = encoding
                preview = [line.strip() for line in content.splitlines() if line.strip()][:10]
            else:
                warning.append("Unsupported table type; inventory only.")
        except Exception as exc:
            readable = False
            warning.append(f"Read error: {exc}")
        item = {
            "source_file": str(path), "relative_path": str(path.relative_to(staging)), "file_name": path.name,
            "file_extension": extension, "file_size": path.stat().st_size,
            "modified_timestamp": datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds"),
            "sha256": digest, "readable": "yes" if readable else "no", "row_count": info["row_count"],
            "column_count": info["column_count"], "column_headers": json.dumps(info["headers"], ensure_ascii=False),
            "first_non_empty_lines": json.dumps(preview, ensure_ascii=False), "encoding": info["encoding"],
            "warnings": " | ".join(warning),
        }
        manifest.append(item)
        file_data.append({**item, **info, "path": path})

    for item in manifest:
        peers = [str(path.relative_to(staging)) for path in digest_groups[item["sha256"]] if str(path) != item["source_file"]]
        if peers:
            item["warnings"] = " | ".join(filter(None, [item["warnings"], "Exact duplicate of: " + "; ".join(peers)]))

    profiles = []
    fresh_matches: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for item in file_data:
        ranked = sorted(((match_score(item, old), old) for old in legacy), key=lambda pair: pair[0]["score"], reverse=True)
        match, old = ranked[0] if ranked else ({"score": 0, "header_similarity": 0, "filename_similarity": 0, "column_similarity": 0}, {})
        inferred = old.get("inferred_table_family") if match["score"] >= 0.58 else family_from_name(item["file_name"])
        if item["file_extension"] == ".txt":
            inferred = "Deep Dive 25 notes"
        grain = infer_grain(inferred, item.get("headers", []), item["file_extension"])
        authority, reason = authority_class(inferred, item["file_extension"])
        raw_path, raw_overlap = best_schema_overlap(item.get("headers", []), raw_schemas)
        sortable_path, sortable_overlap = best_schema_overlap(item.get("headers", []), sortable_schemas)
        duplicate_count = len(digest_groups[item["sha256"]]) - 1
        if duplicate_count:
            recommendation = "HOLD_DUPLICATE_OR_OVERLAP"
        elif item["readable"] == "no" or inferred == "unknown":
            recommendation = "HOLD_UNKNOWN_REVIEW"
        elif authority in {"RAW_OOTP_GOVERNS", "CROSSCHECK_ONLY"}:
            recommendation = "CROSSCHECK_ONLY"
        elif authority == "HISTORICAL_REFERENCE_ONLY":
            recommendation = "HISTORICAL_REFERENCE_ONLY"
        else:
            recommendation = "PROFILE_FOR_POSSIBLE_PROMOTION"
        match_type = "NO_LEGACY_MATCH"
        if old and match["score"] >= 0.58:
            same_name = norm(item["file_name"]) == norm(old["file_name"])
            match_type = "EXACT_FILENAME_STRUCTURAL_MATCH" if same_name else "RENAMED_STRUCTURAL_MATCH" if match["score"] >= 0.72 and match["header_similarity"] >= 0.75 else "LIKELY_LEGACY_MATCH"
            fresh_matches[old["relative_path"]].append((item, match))
        profiles.append({
            "source_file": item["source_file"], "relative_path": item["relative_path"], "file_name": item["file_name"],
            "file_extension": item["file_extension"], "readable": item["readable"], "row_count": item["row_count"],
            "column_count": item["column_count"], "inferred_table_family": inferred, "inferred_grain": grain,
            "legacy_match_file": old.get("relative_path", "") if match["score"] >= 0.58 else "",
            "legacy_match_type": match_type, "match_confidence": match["score"], "header_similarity": match["header_similarity"],
            "schema_drift": "yes" if old and match["score"] >= 0.58 and header_set(item.get("headers", [])) != header_set(legacy_headers(old)) else "no",
            "raw_schema_overlap_file": raw_path, "raw_schema_overlap_score": raw_overlap,
            "sortable_schema_overlap_file": sortable_path, "sortable_schema_overlap_score": sortable_overlap,
            "authority_class": authority, "authority_reason": reason, "story_value": "high" if inferred in STATSP_SPECIFIC else "medium" if inferred != "unknown" else "hold",
            "promotion_recommendation": recommendation, "warnings": item["warnings"],
        })

    reconciliation = []
    matched_fresh = set()
    for old in legacy:
        candidates = []
        for item in file_data:
            candidates.append((match_score(item, old), item))
        candidates.sort(key=lambda pair: pair[0]["score"], reverse=True)
        score, fresh = candidates[0] if candidates else ({"score": 0, "header_similarity": 0, "filename_similarity": 0, "column_similarity": 0}, None)
        if fresh is None or score["score"] < 0.58:
            status = "MISSING_LEGACY_TABLE"
            fresh_name = ""
            added = removed = ""
            drift = "not_applicable"
        else:
            matched_fresh.add(fresh["relative_path"])
            same_name = norm(fresh["file_name"]) == norm(old["file_name"])
            status = "EXACT_FILENAME_MATCH" if same_name else "RENAMED_STRUCTURAL_MATCH" if score["score"] >= 0.72 and score["header_similarity"] >= 0.75 else "LIKELY_MATCH_REVIEW"
            fresh_name = fresh["relative_path"]
            fresh_headers, old_headers = header_set(fresh.get("headers", [])), header_set(legacy_headers(old))
            added = "|".join(sorted(fresh_headers - old_headers))
            removed = "|".join(sorted(old_headers - fresh_headers))
            drift = "yes" if added or removed else "no"
        reconciliation.append({
            "legacy_file": old["relative_path"], "legacy_family": old["inferred_table_family"], "legacy_rows": old["row_count"],
            "legacy_columns": old["column_count"], "fresh_file": fresh_name, "reconciliation_status": status,
            "confidence_score": score["score"], "header_similarity": score["header_similarity"],
            "schema_drift": drift, "columns_added": added, "columns_removed": removed,
            "recommendation": "REVIEW_SCHEMA_DRIFT" if drift == "yes" else "REVIEW_MISSING_CAPTURE" if status == "MISSING_LEGACY_TABLE" else "MATCHED_PENDING_CUTOFF_VALIDATION",
        })
    for profile in profiles:
        if profile["relative_path"] not in matched_fresh and profile["file_extension"] == ".csv":
            reconciliation.append({
                "legacy_file": "", "legacy_family": "", "legacy_rows": "", "legacy_columns": "",
                "fresh_file": profile["relative_path"], "reconciliation_status": "NEW_FRESH_TABLE",
                "confidence_score": profile["match_confidence"], "header_similarity": profile["header_similarity"],
                "schema_drift": "not_applicable", "columns_added": "", "columns_removed": "",
                "recommendation": "REVIEW_AS_NEW_STATSP_SOURCE",
            })

    counts = {
        "files": len(all_files), "csv_files": sum(item["file_extension"] == ".csv" for item in manifest),
        "txt_files": sum(item["file_extension"] == ".txt" for item in manifest),
        "readable_files": sum(item["readable"] == "yes" for item in manifest),
        "missing_legacy_tables": sum(row["reconciliation_status"] == "MISSING_LEGACY_TABLE" for row in reconciliation),
        "renamed_matches": sum(row["reconciliation_status"] == "RENAMED_STRUCTURAL_MATCH" for row in reconciliation),
        "new_tables": sum(row["reconciliation_status"] == "NEW_FRESH_TABLE" for row in reconciliation),
        "schema_drift_tables": sum(row["schema_drift"] == "yes" for row in reconciliation),
        "exact_duplicate_files": sum(len(group) - 1 for group in digest_groups.values() if len(group) > 1),
    }
    authority_counts = dict(Counter(row["authority_class"] for row in profiles))
    high_value = [row for row in profiles if row["story_value"] == "high"]
    held = [row for row in profiles if row["promotion_recommendation"].startswith("HOLD")]

    manifest_stem = output_dir / f"statsplus_fresh_capture_manifest_{slug}"
    profile_stem = output_dir / f"statsplus_fresh_capture_profile_{slug}"
    reconciliation_stem = output_dir / f"statsplus_legacy_vs_fresh_reconciliation_{slug}"
    authority_stem = output_dir / f"statsplus_authority_crosscheck_{slug}"
    enrichment_stem = output_dir / f"statsplus_story_enrichment_plan_{slug}"
    write_csv(manifest_stem.with_suffix(".csv"), manifest)
    write_json(manifest_stem.with_suffix(".json"), {"staging_folder": str(staging), "season": args.season, "as_of_date": args.as_of_date, "counts": counts, "files": manifest})
    write_csv(profile_stem.with_suffix(".csv"), profiles)
    write_json(profile_stem.with_suffix(".json"), {"staging_folder": str(staging), "season": args.season, "as_of_date": args.as_of_date, "counts": counts, "profiles": profiles})
    write_csv(reconciliation_stem.with_suffix(".csv"), reconciliation)
    write_json(reconciliation_stem.with_suffix(".json"), {"legacy_inventory": str(args.legacy_inventory), "counts": counts, "reconciliation": reconciliation})

    manifest_md = f"""# StatsPlus Fresh-Capture Manifest

**Staging folder:** `{staging}`  
**Target:** {args.season} as of {args.as_of_date}  
**Status:** Profile only; no files promoted.

{md_table(["Measure", "Count"], [[key.replace('_',' ').title(), value] for key, value in counts.items()])}

## Files

{md_table(["Relative path", "Type", "Bytes", "Rows", "Cols", "Readable", "Warnings"], [[row['relative_path'], row['file_extension'], row['file_size'], row['row_count'], row['column_count'], row['readable'], row['warnings']] for row in manifest])}
"""
    manifest_stem.with_suffix(".md").write_text(manifest_md, encoding="utf-8")
    profile_md = f"""# StatsPlus Fresh-Capture Profile

**Target:** {args.season} as of {args.as_of_date}  
**Cutoff caution:** Structural matching does not prove that table values reach the target date.

{md_table(["File", "Family", "Grain", "Legacy match", "Confidence", "Drift", "Authority", "Recommendation"], [[row['relative_path'], row['inferred_table_family'], row['inferred_grain'], row['legacy_match_type'], row['match_confidence'], row['schema_drift'], row['authority_class'], row['promotion_recommendation']] for row in profiles])}

High-value candidates: {len(high_value)}. Held/review files: {len(held)}. Promotion remains unauthorized.
"""
    profile_stem.with_suffix(".md").write_text(profile_md, encoding="utf-8")
    reconciliation_md = f"""# StatsPlus Legacy-vs-Fresh Reconciliation

{md_table(["Legacy table", "Family", "Fresh table", "Status", "Confidence", "Drift", "Recommendation"], [[row['legacy_file'], row['legacy_family'], row['fresh_file'], row['reconciliation_status'], row['confidence_score'], row['schema_drift'], row['recommendation']] for row in reconciliation])}

Missing legacy tables, new tables, renamed tables, duplicate views, and schema drift require explicit review. A structural match does not establish current-date compatibility.
"""
    reconciliation_stem.with_suffix(".md").write_text(reconciliation_md, encoding="utf-8")

    conflicts = []
    for row in profiles:
        if row["authority_class"] in {"RAW_OOTP_GOVERNS", "CROSSCHECK_ONLY"}:
            conflicts.append({"file": row["relative_path"], "family": row["inferred_table_family"], "authority_outcome": row["authority_class"], "raw_overlap": row["raw_schema_overlap_file"], "sortable_overlap": row["sortable_schema_overlap_file"], "rule": row["authority_reason"]})
    authority_payload = {
        "season": args.season, "as_of_date": args.as_of_date, "authority_rules": str(args.authority_rules),
        "authority_counts": authority_counts, "crosscheck_tables": conflicts,
        "rules": {"raw_ootp": "completed games, scores, logs, schedule state, and current proof",
                  "sortable": "governed current player/team enrichment",
                  "statsplus": "StatsPlus-specific fields after validation and promotion",
                  "conflicts": "report both values; never silently resolve"},
        "promotion_safe": False,
    }
    write_json(authority_stem.with_suffix(".json"), authority_payload)
    authority_md = f"""# StatsPlus Authority Crosscheck

**Target:** {args.season} as of {args.as_of_date}  
**Promotion safe:** No—profiling is not promotion approval.

{md_table(["Authority class", "Tables"], [[key, value] for key, value in sorted(authority_counts.items())])}

## Crosscheck-only/overlapping tables

{md_table(["File", "Family", "Outcome", "Closest raw schema", "Closest sortable schema"], [[row['file'], row['family'], row['authority_outcome'], row['raw_overlap'], row['sortable_overlap']] for row in conflicts]) if conflicts else 'None detected.'}

Raw OOTP governs current game/score/record proof. Sortable stats govern overlapping current enrichment. StatsPlus governs only validated StatsPlus-specific fields after explicit promotion. Conflicts must be reported, not silently resolved.
"""
    authority_stem.with_suffix(".md").write_text(authority_md, encoding="utf-8")

    enrichment_rows = []
    for row in profiles:
        if row["story_value"] != "high":
            continue
        family = row["inferred_table_family"]
        signal = {
            "playoff odds by division": "division-race probability and pressure", "playoff odds by league": "league qualification probability",
            "ELO ratings": "team strength and momentum", "BaseRuns": "expected-performance gap", "team WAR": "team strength",
            "historical fan interest": "historical fan-pressure context", "fan data": "fan-pressure context",
            "financials": "resource/results pressure", "owner info": "organization context", "injury summary": "injury burden",
            "team baserunning": "team baserunning", "player baserunning": "player baserunning",
            "best batting game": "game-performance discovery", "best pitching game": "game-performance discovery",
            "Grand Tournament of Champions": "tournament/history echo", "team age": "organization age profile",
        }.get(family, "StatsPlus enrichment")
        enrichment_rows.append({"file": row["relative_path"], "family": family, "candidate_signal": signal,
                                "use_state": "candidate_after_promotion", "authority_class": row["authority_class"],
                                "required_validation": "cutoff, schema, identity, duplicate, and field-semantic validation"})
    enrichment_payload = {"season": args.season, "as_of_date": args.as_of_date, "sources": enrichment_rows,
                          "disabled_until_promotion": True,
                          "editorial_rule": "StatsPlus enriches where to look; raw game proof establishes what happened."}
    write_json(enrichment_stem.with_suffix(".json"), enrichment_payload)
    enrichment_md = f"""# StatsPlus Story-Enrichment Plan

No source in this plan is enabled before promotion.

{md_table(["File", "Family", "Candidate signal", "State", "Validation required"], [[row['file'], row['family'], row['candidate_signal'], row['use_state'], row['required_validation']] for row in enrichment_rows]) if enrichment_rows else 'No high-value enrichment source was identified.'}

StatsPlus supplies model/context enrichment. Raw OOTP remains the proof layer. Manager tendencies remain disabled without validated same-cutoff staff identity and fields.
"""
    enrichment_stem.with_suffix(".md").write_text(enrichment_md, encoding="utf-8")

    print(json.dumps({"staging_folder": str(staging), "outputs_written": 13, **counts, "promotion_safe": False}, indent=2))


if __name__ == "__main__":
    main()
