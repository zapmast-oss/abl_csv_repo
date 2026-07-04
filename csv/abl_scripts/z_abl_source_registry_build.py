from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "csv" / "out" / "docs" / "abl_data_catalog.csv"
OUT_DIR = ROOT / "csv" / "out" / "control"
FIELDS = [
    "source_id", "source_name", "source_family", "file_path", "row_count",
    "column_count", "likely_grain", "subject_area", "authority_level",
    "volatility", "can_drive_current_state", "can_support_current_state",
    "can_support_historical_context", "can_support_editorial_context",
    "as_of_detection_method", "primary_key_guess", "join_key_guess",
    "validation_rules", "curated_target", "notes",
]


def truth(value: bool) -> str:
    return "yes" if value else "no"


def source_family(path: str) -> str:
    lower = path.lower()
    if lower.startswith("csv/ootp_csv/"):
        return "ootp_csv"
    if lower.startswith("csv/abl_statistics/"):
        return "sortable_stats"
    if lower.startswith("csv/out/almanac/"):
        return "historical_almanac"
    if lower.startswith("csv/out/") or lower.startswith("out/") or lower.startswith("csv/abl_csv/"):
        return "generated_output"
    if ("story_dictionary" in lower or "story_menu" in lower or "story_candidates" in lower
            or lower.startswith("csv/config/")):
        return "editorial_config"
    if lower.startswith("docs/") or lower.startswith("csv/docs/"):
        return "documentation"
    return "unknown"


def authority(family: str) -> str:
    return {
        "ootp_csv": "system_of_record_extract",
        "sortable_stats": "supplemental_report_extract",
        "generated_output": "derived_output",
        "historical_almanac": "historical_context",
        "editorial_config": "editorial_config",
        "documentation": "documentation",
    }.get(family, "unknown")


def subject(path: str) -> str:
    name = Path(path).stem.lower()
    rules = [
        (("game", "schedule", "matchup", "series"), "games_and_schedule"),
        (("standings", "record", "division", "conference", "wild_card"), "standings_and_races"),
        (("manager", "coach", "staff"), "staff_and_management"),
        (("financial", "finan", "salary", "contract", "market", "fan", "pers_park", "park"), "financial_and_team_context"),
        (("pitch", "bullpen", "rotation", "ace"), "pitching"),
        (("bat", "hitting", "offense", "run_creation"), "batting"),
        (("field", "defense", "catcher"), "fielding_and_defense"),
        (("rating",), "ratings"),
        (("indicative", "misc_info"), "player_indicative_and_misc"),
        (("player", "roster", "rookie", "prospect", "war", "leader"), "players_and_rosters"),
        (("team",), "teams"),
        (("transaction", "trade", "injury"), "transactions_and_availability"),
        (("story", "menu", "candidate", "dictionary"), "editorial_story_config"),
        (("almanac", "champion", "history", "flashback"), "historical_context"),
    ]
    for tokens, label in rules:
        if any(token in name for token in tokens):
            return label
    return "reference_or_other"


def volatility(path: str, family: str) -> str:
    name = Path(path).stem.lower()
    if family == "historical_almanac":
        return "static"
    if family in {"editorial_config", "documentation"}:
        return "slow-changing"
    if family == "generated_output":
        return "changes every export"
    if family == "sortable_stats":
        if any(token in name for token in ("_stats", "pitching_", "batting_", "fielding_", "cur_rec", "finan")) and "ratings" not in name:
            return "changes every export"
        if any(token in name for token in ("ratings", "indicative", "misc_info", "staff", "pers_park")):
            return "slow-changing"
        return "unknown"
    if family == "ootp_csv":
        if any(token in name for token in ("games", "record", "stats", "roster", "contract", "injury", "streak", "value", "projected", "events", "messages")):
            return "changes every export"
        if any(token in name for token in ("continents", "nations", "languages", "states", "cities")):
            return "static"
        return "slow-changing"
    return "unknown"


def current_permissions(path: str, family: str) -> tuple[bool, bool, str]:
    name = Path(path).name.lower()
    if family == "ootp_csv":
        driver = name in {"games.csv", "games_score.csv", "game_logs.csv"}
        if driver:
            method = "Filter league/game type and completed rows; detect maximum played game date and completed games per team."
        elif any(token in name for token in ("history", "career")):
            method = "Filter explicit season/league fields; inherit capture batch and never infer latest date from this file alone."
        else:
            method = "Tie file checksum to a validated raw OOTP capture batch whose games.csv proves the cutoff."
        return driver, True, method
    if family == "sortable_stats":
        return False, True, "Tie file checksum to a sortable-stats capture batch; compare team/player coverage to the validated raw batch."
    if family == "historical_almanac":
        return False, False, "Detect season/league from path and columns; historical only."
    if family == "generated_output":
        return False, False, "Read producer manifest and source lineage; never use this output to prove current state."
    if family in {"editorial_config", "documentation"}:
        return False, False, "Version-controlled editorial/documentation file; no baseball as-of authority."
    return False, False, "No reliable as-of detection method established."


def historical_support(path: str, family: str) -> bool:
    lower = path.lower()
    return family == "historical_almanac" or any(token in lower for token in ("history", "career", "champion", "archive", "flashback"))


def editorial_support(family: str) -> bool:
    return family in {"editorial_config", "documentation"}


def keys(path: str, catalog_keys: str, area: str) -> tuple[str, str]:
    primary = catalog_keys.replace("|", ";") if catalog_keys else "inspect_schema"
    join_by_area = {
        "games_and_schedule": "game_id;team_id;league_id;date",
        "standings_and_races": "team_id;league_id;season",
        "staff_and_management": "manager_id;coach_id;team_id",
        "financial_and_team_context": "team_id;park_id;season",
        "pitching": "player_id;team_id;league_id;season",
        "batting": "player_id;team_id;league_id;season",
        "fielding_and_defense": "player_id;team_id;position;season",
        "ratings": "player_id or Name;team_id",
        "player_indicative_and_misc": "player_id or Name;team_id",
        "players_and_rosters": "player_id;team_id;league_id",
        "teams": "team_id;league_id;division_id",
        "transactions_and_availability": "player_id;team_id;date",
        "historical_context": "season;league_id;team_id;player_id",
        "editorial_story_config": "story_id;coverage_label",
    }
    return primary, join_by_area.get(area, "inspect_schema")


def validations(path: str, family: str, area: str) -> str:
    checks = ["file exists", "header/schema check", "row-count check", "duplicate-key check"]
    if family == "ootp_csv":
        checks += ["capture-batch checksum", "league/team referential integrity"]
        if area == "games_and_schedule":
            checks += ["played-date monotonicity", "games-per-team reconciliation", "score/result completeness"]
    elif family == "sortable_stats":
        checks += ["capture-batch checksum", "24-team or player-universe coverage as applicable", "join names/IDs to raw dimensions", "reject mixed capture dates"]
    elif family == "generated_output":
        checks += ["producer manifest present", "lineage and as-of metadata present", "rebuildability check"]
    elif family == "historical_almanac":
        checks += ["season/league path-column agreement", "historical row uniqueness"]
    return "; ".join(checks)


def curated_target(area: str, family: str) -> str:
    if family in {"generated_output", "editorial_config", "documentation"}:
        return "none_direct"
    mapping = {
        "games_and_schedule": "curated_current_games or curated_current_schedule",
        "standings_and_races": "curated_current_standings",
        "staff_and_management": "curated_current_staff",
        "financial_and_team_context": "curated_current_team_context",
        "pitching": "curated_current_player_pitching or curated_current_team_stats",
        "batting": "curated_current_player_batting or curated_current_team_stats",
        "fielding_and_defense": "curated_current_team_stats",
        "ratings": "curated_current_player_ratings",
        "player_indicative_and_misc": "curated_current_player_indicative",
        "players_and_rosters": "curated_current_player_batting; curated_current_player_pitching; curated_current_staff",
        "teams": "curated_current_team_context",
        "transactions_and_availability": "curated_current_team_context",
        "historical_context": "curated_historical_context",
    }
    return mapping.get(area, "curated_reference")


def make_row(path: str, catalog_row: dict[str, str] | None = None) -> dict[str, str]:
    catalog_row = catalog_row or {}
    family = source_family(path)
    area = subject(path)
    drive, support, detection = current_permissions(path, family)
    primary, join = keys(path, catalog_row.get("key_columns", ""), area)
    sid = f"{family}_{hashlib.sha1(path.encode('utf-8')).hexdigest()[:12]}"
    note = "Catalog-derived CSV registration." if catalog_row else "Repository-inspected documentation registration."
    return {
        "source_id": sid, "source_name": Path(path).stem, "source_family": family,
        "file_path": path, "row_count": catalog_row.get("row_count", ""),
        "column_count": catalog_row.get("column_count", ""),
        "likely_grain": catalog_row.get("likely_grain", "document") if catalog_row else "document",
        "subject_area": area, "authority_level": authority(family),
        "volatility": volatility(path, family),
        "can_drive_current_state": truth(drive), "can_support_current_state": truth(support),
        "can_support_historical_context": truth(historical_support(path, family)),
        "can_support_editorial_context": truth(editorial_support(family)),
        "as_of_detection_method": detection, "primary_key_guess": primary,
        "join_key_guess": join, "validation_rules": validations(path, family, area),
        "curated_target": curated_target(area, family), "notes": note,
    }


def write_outputs(rows: list[dict[str, str]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "abl_source_registry.csv"
    json_path = OUT_DIR / "abl_source_registry.json"
    md_path = OUT_DIR / "abl_source_registry.md"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    json_path.write_text(json.dumps({"source_count": len(rows), "sources": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    families = Counter(row["source_family"] for row in rows)
    authorities = Counter(row["authority_level"] for row in rows)
    lines = [
        "# ABL Source Registry", "",
        f"Registered sources: **{len(rows)}**", "",
        "Generated from the authoritative CSV catalog plus repository documentation inspection. This registry classifies authority; it does not promote any source to current state.", "",
        "## Counts by source family", "",
        "| Source family | Count |", "|---|---:|",
    ]
    lines += [f"| `{key}` | {value} |" for key, value in sorted(families.items())]
    lines += ["", "## Counts by authority", "", "| Authority | Count |", "|---|---:|"]
    lines += [f"| `{key}` | {value} |" for key, value in sorted(authorities.items())]
    lines += ["", "## Registry", "", "| ID | Family | Authority | Drive current | Support current | Path |", "|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| `{row['source_id']}` | `{row['source_family']}` | `{row['authority_level']}` | {row['can_drive_current_state']} | {row['can_support_current_state']} | `{row['file_path']}` |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"source_count": len(rows), "csv": str(csv_path), "markdown": str(md_path), "json": str(json_path)}, indent=2))


def main() -> int:
    if not CATALOG.exists():
        raise FileNotFoundError(f"Authoritative catalog not found: {CATALOG}")
    with CATALOG.open("r", encoding="utf-8-sig", newline="") as handle:
        catalog_rows = list(csv.DictReader(handle))
    rows = [make_row(row["file_path"], row) for row in catalog_rows]
    registered = {row["file_path"] for row in rows}
    docs = sorted(
        path.relative_to(ROOT).as_posix()
        for base in (ROOT / "docs", ROOT / "csv" / "docs")
        if base.exists()
        for path in base.rglob("*.md")
    )
    rows.extend(make_row(path) for path in docs if path not in registered)
    rows.sort(key=lambda row: row["file_path"])
    write_outputs(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

