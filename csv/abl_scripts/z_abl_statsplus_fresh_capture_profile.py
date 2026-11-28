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
INTENTIONAL_EXCLUSIONS = {
    6: {"table": "standings", "reason": "Current standings are governed through raw OOTP current-state authority.",
        "status": "intentionally excluded from Feed 3 promotion"},
    27: {"table": "Grand Tournament of Champions history", "reason": "The table does not change until postseason.",
         "status": "intentionally excluded from this current refresh"},
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
    canonical_rows = sorted("\x1f".join(cell.strip() for cell in row) for row in data)
    row_multiset_hash = hashlib.sha256("\x1e".join(canonical_rows).encode("utf-8")).hexdigest()
    return {"headers": headers, "row_count": len(data), "column_count": len(headers), "encoding": encoding,
            "sample_rows": data[:3], "data_rows": data, "row_multiset_hash": row_multiset_hash}


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


def table_number(name: str) -> int | None:
    match = re.match(r"^(\d{2})_", Path(name).name)
    return int(match.group(1)) if match else None


def infer_source_type(file_name: str, extension: str, family: str) -> str:
    cleaned = norm(file_name)
    if extension == ".txt":
        return "generated output" if any(x in cleaned for x in ("prompt", "report", "output")) else "user-curated table"
    if "statsplus" in cleaned:
        return "StatsPlus export"
    if "transactions personnel" in cleaned or family in {"owner info", "front office/coaches"}:
        return "OOTP-derived table"
    if any(x in cleaned for x in ("combined", "curated", "supplemental")):
        return "user-curated table"
    return "unknown/review"


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
    fn, ln = table_number(fresh["file_name"]), table_number(legacy["file_name"])
    number_agreement = 1.0 if fn is not None and fn == ln else 0.0
    # Numbering is supporting identity evidence, never sufficient alone; family or structural evidence is also required.
    number_bonus = 0.35 if number_agreement and (fs or hs >= 0.25) else 0.0
    score = min(1.0, 0.60 * hs + 0.15 * ns + 0.15 * cs + 0.10 * fs + number_bonus)
    return {"score": round(score, 4), "header_similarity": round(hs, 4), "filename_similarity": round(ns, 4),
            "column_similarity": round(cs, 4), "number_agreement": number_agreement}


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


def load_raw_records(raw_dir: Path) -> dict[str, tuple[int, int]]:
    with (raw_dir / "teams.csv").open(encoding="utf-8-sig", newline="") as handle:
        teams = list(csv.DictReader(handle))
    with (raw_dir / "team_record.csv").open(encoding="utf-8-sig", newline="") as handle:
        records = {row["team_id"]: row for row in csv.DictReader(handle)}
    output = {}
    for team in teams:
        if team.get("league_id") != "200" or team.get("level") != "1" or team["team_id"] not in records:
            continue
        record = (int(records[team["team_id"]]["w"]), int(records[team["team_id"]]["l"]))
        for key in (team.get("abbr", ""), team.get("name", ""), f"{team.get('name','')} {team.get('nickname','')}"):
            if key:
                output[norm(key)] = record
    return output


def resolve_team_record(value: str, raw_records: dict[str, tuple[int, int]]) -> tuple[str, tuple[int, int]] | None:
    cleaned = norm(value)
    if cleaned in raw_records:
        return cleaned, raw_records[cleaned]
    tokens = cleaned.split()
    for key, record in raw_records.items():
        if len(key) <= 4 and (key in tokens or cleaned.endswith(key)):
            return key, record
    return None


def current_record_crosschecks(file_data: list[dict], profiles: list[dict], raw_records: dict[str, tuple[int, int]]) -> list[dict]:
    family_by_path = {row["relative_path"]: row["inferred_table_family"] for row in profiles}
    record_families = {"league standings", "playoff odds by division", "playoff odds by league", "BaseRuns", "ELO ratings", "team WAR"}
    checks = []
    for item in file_data:
        if family_by_path.get(item["relative_path"]) not in record_families:
            continue
        headers = item.get("headers", [])
        normalized = {norm(header): header for header in headers}
        exact = {header.strip().lower(): header for header in headers}
        team_header = exact.get("team") or normalized.get("team")
        w_header, l_header = exact.get("w"), exact.get("l")
        record_header = next((original for cleaned, original in normalized.items() if cleaned in {"record", "record rec"}), None)
        if not team_header or (not (w_header and l_header) and not record_header):
            continue
        for values in item.get("data_rows", []):
            row = dict(zip(headers, values))
            resolved = resolve_team_record(row.get(team_header, ""), raw_records)
            if not resolved:
                checks.append({"file": item["relative_path"], "family": family_by_path.get(item["relative_path"], "unknown"),
                               "team_value": row.get(team_header, ""), "feed3_record": "", "raw_record": "",
                               "outcome": "HOLD_IDENTITY_AMBIGUOUS"})
                continue
            team_key, raw_record = resolved
            try:
                if w_header and l_header:
                    fresh_record = (int(float(row[w_header])), int(float(row[l_header])))
                else:
                    match = re.search(r"(\d+)\s*-\s*(\d+)", row.get(record_header, ""))
                    if not match:
                        raise ValueError("record not parseable")
                    fresh_record = (int(match.group(1)), int(match.group(2)))
                outcome = "CROSSCHECK_AGREES" if fresh_record == raw_record else "CROSSCHECK_CONFLICT"
                fresh_text = f"{fresh_record[0]}-{fresh_record[1]}"
            except (ValueError, TypeError):
                outcome, fresh_text = "HOLD_SCHEMA_DRIFT", row.get(record_header or w_header, "")
            checks.append({"file": item["relative_path"], "family": family_by_path.get(item["relative_path"], "unknown"),
                           "team_value": row.get(team_header, ""), "resolved_team_key": team_key,
                           "feed3_record": fresh_text, "raw_record": f"{raw_record[0]}-{raw_record[1]}", "outcome": outcome})
    return checks


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
    row_multiset_groups: dict[str, list[Path]] = defaultdict(list)
    manifest = []
    file_data = []

    for path in all_files:
        extension = path.suffix.lower()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        digest_groups[digest].append(path)
        readable = True
        warning = []
        info = {"headers": [], "row_count": "", "column_count": "", "encoding": "", "sample_rows": [],
                "data_rows": [], "row_multiset_hash": ""}
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
        if info.get("row_multiset_hash"):
            row_multiset_groups[info["row_multiset_hash"]].append(path)
        item = {
            "source_file": str(path), "relative_path": str(path.relative_to(staging)), "file_name": path.name,
            "file_extension": extension, "file_size": path.stat().st_size,
            "modified_timestamp": datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds"),
            "sha256": digest, "readable": "yes" if readable else "no", "row_count": info["row_count"],
            "column_count": info["column_count"], "column_headers": json.dumps(info["headers"], ensure_ascii=False),
            "first_non_empty_lines": json.dumps(preview, ensure_ascii=False), "encoding": info["encoding"],
            "row_multiset_hash": info.get("row_multiset_hash", ""),
            "warnings": " | ".join(warning),
        }
        manifest.append(item)
        file_data.append({**item, **info, "path": path})

    for item in manifest:
        peers = [str(path.relative_to(staging)) for path in digest_groups[item["sha256"]] if str(path) != item["source_file"]]
        if peers:
            item["warnings"] = " | ".join(filter(None, [item["warnings"], "Exact duplicate of: " + "; ".join(peers)]))
        row_peers = [str(path.relative_to(staging)) for path in row_multiset_groups.get(item["row_multiset_hash"], [])
                     if str(path) != item["source_file"] and path.read_bytes() != Path(item["source_file"]).read_bytes()]
        if row_peers:
            item["warnings"] = " | ".join(filter(None, [item["warnings"], "Same row multiset in different order/encoding as: " + "; ".join(row_peers)]))

    profiles = []
    fresh_matches: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for item in file_data:
        ranked = sorted(((match_score(item, old), old) for old in legacy), key=lambda pair: pair[0]["score"], reverse=True)
        match, old = ranked[0] if ranked else ({"score": 0, "header_similarity": 0, "filename_similarity": 0, "column_similarity": 0}, {})
        inferred = old.get("inferred_table_family") if match["score"] >= 0.58 else family_from_name(item["file_name"])
        if item["file_extension"] == ".txt":
            inferred = "Deep Dive 25 notes"
        grain = infer_grain(inferred, item.get("headers", []), item["file_extension"])
        source_type = infer_source_type(item["file_name"], item["file_extension"], inferred)
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
            "column_count": item["column_count"], "numbered_table": table_number(item["file_name"]),
            "inferred_table_family": inferred, "inferred_source_type": source_type, "inferred_grain": grain,
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
        old_number = table_number(old["file_name"])
        candidates = []
        for item in file_data:
            candidates.append((match_score(item, old), item))
        candidates.sort(key=lambda pair: pair[0]["score"], reverse=True)
        score, fresh = candidates[0] if candidates else ({"score": 0, "header_similarity": 0, "filename_similarity": 0, "column_similarity": 0}, None)
        excluded_absent = old_number in INTENTIONAL_EXCLUSIONS and not any(table_number(item["file_name"]) == old_number for item in file_data)
        if excluded_absent or fresh is None or score["score"] < 0.58:
            status = "INTENTIONALLY_EXCLUDED" if excluded_absent else "MISSING_LEGACY_TABLE"
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
            "legacy_table_number": old_number, "legacy_file": old["relative_path"], "legacy_family": old["inferred_table_family"], "legacy_rows": old["row_count"],
            "legacy_columns": old["column_count"], "fresh_file": fresh_name, "reconciliation_status": status,
            "confidence_score": score["score"], "header_similarity": score["header_similarity"],
            "schema_drift": drift, "columns_added": added, "columns_removed": removed,
            "recommendation": "INTENTIONAL_EXCLUSION_ACCEPTED" if status == "INTENTIONALLY_EXCLUDED" else "REVIEW_SCHEMA_DRIFT" if drift == "yes" else "REVIEW_MISSING_CAPTURE" if status == "MISSING_LEGACY_TABLE" else "MATCHED_PENDING_CUTOFF_VALIDATION",
        })
    for profile in profiles:
        if profile["relative_path"] not in matched_fresh and profile["file_extension"] == ".csv":
            reconciliation.append({
                "legacy_table_number": "", "legacy_file": "", "legacy_family": "", "legacy_rows": "", "legacy_columns": "",
                "fresh_file": profile["relative_path"], "reconciliation_status": "NEW_FRESH_TABLE",
                "confidence_score": profile["match_confidence"], "header_similarity": profile["header_similarity"],
                "schema_drift": "not_applicable", "columns_added": "", "columns_removed": "",
                "recommendation": "REVIEW_AS_NEW_STATSP_SOURCE",
            })

    record_crosschecks = current_record_crosschecks(file_data, profiles, load_raw_records(args.raw_dir))
    record_outcomes = Counter(row["outcome"] for row in record_crosschecks)

    counts = {
        "files": len(all_files), "csv_files": sum(item["file_extension"] == ".csv" for item in manifest),
        "txt_files": sum(item["file_extension"] == ".txt" for item in manifest),
        "readable_files": sum(item["readable"] == "yes" for item in manifest),
        "unreadable_files": sum(item["readable"] == "no" for item in manifest),
        "numbered_tables_found": len({table_number(item["file_name"]) for item in file_data if table_number(item["file_name"]) is not None}),
        "intentionally_excluded_tables": sum(row["reconciliation_status"] == "INTENTIONALLY_EXCLUDED" for row in reconciliation),
        "missing_legacy_tables": sum(row["reconciliation_status"] == "MISSING_LEGACY_TABLE" for row in reconciliation),
        "renamed_matches": sum(row["reconciliation_status"] == "RENAMED_STRUCTURAL_MATCH" for row in reconciliation),
        "new_tables": sum(row["reconciliation_status"] == "NEW_FRESH_TABLE" for row in reconciliation),
        "schema_drift_tables": sum(row["schema_drift"] == "yes" for row in reconciliation),
        "exact_duplicate_files": sum(len(group) - 1 for group in digest_groups.values() if len(group) > 1),
        "reordered_overlap_files": sum(len(group) - 1 for group in row_multiset_groups.values() if len(group) > 1),
        "record_crosschecks": len(record_crosschecks),
        "record_crosscheck_conflicts": record_outcomes.get("CROSSCHECK_CONFLICT", 0),
        "record_identity_holds": record_outcomes.get("HOLD_IDENTITY_AMBIGUOUS", 0),
    }
    authority_counts = dict(Counter(row["authority_class"] for row in profiles))
    high_value = [row for row in profiles if row["story_value"] == "high"]
    held = [row for row in profiles if row["promotion_recommendation"].startswith("HOLD")]
    crosscheck = [row for row in profiles if row["promotion_recommendation"] == "CROSSCHECK_ONLY"]
    same_legacy = [row for row in reconciliation if row["reconciliation_status"] in {"EXACT_FILENAME_MATCH", "RENAMED_STRUCTURAL_MATCH", "LIKELY_MATCH_REVIEW"}]
    capture_coherent = (counts["files"] > 0 and counts["unreadable_files"] == 0 and counts["missing_legacy_tables"] == 0
                        and counts["record_crosscheck_conflicts"] == 0 and counts["record_identity_holds"] == 0)
    ready_for_dry_run = capture_coherent and not held

    manifest_stem = output_dir / f"statsplus_fresh_capture_manifest_{slug}"
    profile_stem = output_dir / f"statsplus_fresh_capture_profile_{slug}"
    reconciliation_stem = output_dir / f"statsplus_legacy_vs_fresh_reconciliation_{slug}"
    authority_stem = output_dir / f"statsplus_authority_crosscheck_{slug}"
    enrichment_stem = output_dir / f"statsplus_story_enrichment_plan_{slug}"
    exclusion_stem = output_dir / f"statsplus_intentional_exclusions_{slug}"
    write_csv(manifest_stem.with_suffix(".csv"), manifest)
    write_json(manifest_stem.with_suffix(".json"), {"staging_folder": str(staging), "season": args.season, "as_of_date": args.as_of_date, "counts": counts, "files": manifest})
    write_csv(profile_stem.with_suffix(".csv"), profiles)
    write_json(profile_stem.with_suffix(".json"), {"staging_folder": str(staging), "season": args.season,
               "as_of_date": args.as_of_date, "counts": counts, "capture_coherent": capture_coherent,
               "ready_for_separate_dry_run_promotion_task": ready_for_dry_run,
               "promotion_authorized": False, "profiles": profiles})
    write_csv(reconciliation_stem.with_suffix(".csv"), reconciliation)
    write_json(reconciliation_stem.with_suffix(".json"), {"legacy_inventory": str(args.legacy_inventory),
               "intentional_exclusions": INTENTIONAL_EXCLUSIONS, "counts": counts, "reconciliation": reconciliation})

    exclusions = [{"table_number": number, **details} for number, details in sorted(INTENTIONAL_EXCLUSIONS.items())]
    exclusion_payload = {"season": args.season, "as_of_date": args.as_of_date,
                         "exclusions_correctly_marked": all(any(row["legacy_table_number"] == number and row["reconciliation_status"] == "INTENTIONALLY_EXCLUDED" for row in reconciliation) for number in INTENTIONAL_EXCLUSIONS),
                         "exclusions": exclusions}
    write_json(exclusion_stem.with_suffix(".json"), exclusion_payload)
    exclusion_stem.with_suffix(".md").write_text(
        "# StatsPlus Intentional Exclusions\n\n"
        f"**Target:** {args.season} as of {args.as_of_date}\n\n"
        + md_table(["Table", "Family", "Reason", "Status"],
                   [[row["table_number"], row["table"], row["reason"], row["status"]] for row in exclusions])
        + "\n\nThese omissions are expected and are not capture errors.\n", encoding="utf-8")

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

{md_table(["#", "File", "Source type", "Family", "Grain", "Legacy match", "Confidence", "Drift", "Authority", "Recommendation"], [[row['numbered_table'], row['relative_path'], row['inferred_source_type'], row['inferred_table_family'], row['inferred_grain'], row['legacy_match_type'], row['match_confidence'], row['schema_drift'], row['authority_class'], row['promotion_recommendation']] for row in profiles])}

## Assessment

- Fresh files inspected: {counts['files']}.
- Readable tables/files: {counts['readable_files']}; unreadable: {counts['unreadable_files']}.
- Numbered tables found: {counts['numbered_tables_found']}.
- Intentionally excluded: tables 6 and 27 ({counts['intentionally_excluded_tables']} total).
- Unexpected missing legacy tables: {counts['missing_legacy_tables']}.
- Same/likely legacy tables: {len(same_legacy)}; renamed structural matches: {counts['renamed_matches']}.
- New fresh tables: {counts['new_tables']}.
- Exact duplicate files: {counts['exact_duplicate_files']}; reordered overlaps: {counts['reordered_overlap_files']}.
- Schema drift tables: {counts['schema_drift_tables']}.
- High-value story-engine sources: {len(high_value)}.
- Crosscheck-only sources: {len(crosscheck)}; hold/review sources: {len(held)}.
- Capture coherent: **{'Yes' if capture_coherent else 'No'}**.
- Ready for a separate dry-run promotion task: **{'Yes' if ready_for_dry_run else 'No'}**.

Promotion remains unauthorized. A dry run must retain table-level authority and schema-drift decisions.
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
        "current_record_crosscheck_counts": dict(record_outcomes),
        "current_record_crosschecks": record_crosschecks,
        "rules": {"raw_ootp": "completed games, scores, logs, schedule state, and current proof",
                  "sortable": "governed current player/team enrichment",
                  "statsplus": "StatsPlus-specific fields after validation and promotion",
                  "conflicts": "report both values; never silently resolve"},
        "promotion_safe": False, "ready_for_separate_dry_run_promotion_task": ready_for_dry_run,
    }
    write_json(authority_stem.with_suffix(".json"), authority_payload)
    authority_md = f"""# StatsPlus Authority Crosscheck

**Target:** {args.season} as of {args.as_of_date}  
**Promotion safe now:** No—profiling is not promotion approval.  
**Ready for separate dry-run promotion task:** {'Yes' if ready_for_dry_run else 'No'}.

{md_table(["Authority class", "Tables"], [[key, value] for key, value in sorted(authority_counts.items())])}

## Raw OOTP current-record crosscheck

{md_table(["Outcome", "Rows"], [[key, value] for key, value in sorted(record_outcomes.items())]) if record_outcomes else 'No Feed 3 record-bearing table was detected.'}

Conflicts or identity holds: {sum(value for key, value in record_outcomes.items() if key != 'CROSSCHECK_AGREES')}.

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

    print(json.dumps({"staging_folder": str(staging), "outputs_written": 15, **counts,
                      "capture_coherent": capture_coherent,
                      "ready_for_separate_dry_run_promotion_task": ready_for_dry_run,
                      "promotion_safe_now": False}, indent=2))


if __name__ == "__main__":
    main()
