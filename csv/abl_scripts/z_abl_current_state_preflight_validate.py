from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import date, datetime
from pathlib import Path

from abl_path_policy import validate_output_paths


ROOT = Path(__file__).resolve().parents[2]
CONTROL = ROOT / "csv" / "out" / "control"
RAW = ROOT / "csv" / "ootp_csv"
SORTABLE = ROOT / "csv" / "abl_statistics"
DRIVERS = ("games.csv", "games_score.csv", "game_logs.csv")
STALE_FILES_1981 = [
    "csv/out/star_schema/monday_1981_standings_by_division.csv (32-game snapshot)",
    "csv/out/star_schema/fact_team_reporting_1981_weekly_change.csv (32-game snapshot)",
    "csv/out/star_schema/fact_player_batting.csv (preserved early snapshot)",
    "csv/out/star_schema/fact_player_pitching.csv (preserved early snapshot)",
    "csv/out/csv_out/z_ABL_Manager_Tendencies.csv (62-game snapshot)",
    "csv/out/csv_out/z_ABL_Division_Leverage.csv (62-game snapshot)",
    "csv/out/csv_out/z_ABL_Rotation_Stability.csv (62-game snapshot)",
]


def parse_iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected YYYY-MM-DD date, got {value!r}") from exc


def parse_game_date(value: str) -> date:
    """Parse OOTP dates, which may omit leading zeroes."""

    return datetime.strptime(value, "%Y-%m-%d").date()


def select_completed_games(
    rows: list[dict[str, str]], season: int, as_of_date: date, league_id: str = "200"
) -> list[dict[str, str]]:
    """Select completed regular-season games no later than the named cutoff."""

    selected: list[dict[str, str]] = []
    for row in rows:
        if row.get("league_id") != league_id or row.get("game_type") != "0" or row.get("played") != "1":
            continue
        game_date = parse_game_date(row["date"])
        if game_date.year == season and game_date <= as_of_date:
            selected.append(row)
    return selected


def completed_games_for_season(
    rows: list[dict[str, str]], season: int, league_id: str = "200"
) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("league_id") == league_id
        and row.get("game_type") == "0"
        and row.get("played") == "1"
        and parse_game_date(row["date"]).year == season
    ]


def output_stem(output_dir: Path, season: int, as_of_date: date) -> Path:
    return output_dir / f"current_state_preflight_{season}_target_{as_of_date.isoformat()}"


def resolve_repo_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def read_ids(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        return {row["game_id"] for row in csv.DictReader(handle)}


def add(
    checks: list[dict[str, str]],
    name: str,
    value: object,
    status: str,
    details: str,
) -> None:
    checks.append({
        "check_name": name,
        "value": str(value),
        "status": status,
        "details": details,
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate ABL current-state inputs through an explicit as-of cutoff."
    )
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--as-of-date", type=parse_iso_date, required=True)
    parser.add_argument("--newsroom-date", type=parse_iso_date)
    parser.add_argument("--league-id", default="200")
    parser.add_argument("--output-dir", type=Path, default=CONTROL)
    args = parser.parse_args()
    if args.as_of_date.year != args.season:
        parser.error("--as-of-date year must match --season")
    if args.newsroom_date and args.newsroom_date < args.as_of_date:
        parser.error("--newsroom-date cannot precede --as-of-date")
    return args


def main() -> int:
    args = parse_args()
    target = args.as_of_date
    newsroom = args.newsroom_date or target
    output_dir = resolve_repo_path(args.output_dir)
    stem = output_stem(output_dir, args.season, target)
    outputs = tuple(stem.with_suffix(ext) for ext in (".csv", ".json", ".md"))
    validate_output_paths(outputs, ROOT)
    stale_files = STALE_FILES_1981 if args.season == 1981 else []

    checks: list[dict[str, str]] = []
    missing = [name for name in DRIVERS if not (RAW / name).exists()]
    source_latest: date | None = None
    record_reconciliation: bool | None = None
    if missing:
        verdict = "NOT_READY_NO_CURRENT_STATE_DRIVER"
        add(
            checks,
            "driver_files",
            "|".join(missing),
            "FAIL",
            "Approved current-state driver files are missing.",
        )
        completed: list[dict[str, str]] = []
        counts: Counter[str] = Counter()
        earliest = latest = None
        teams: dict[str, str] = {}
        score_coverage = log_coverage = False
    else:
        with (RAW / "games.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            games = list(csv.DictReader(handle))
        source_completed = completed_games_for_season(games, args.season, args.league_id)
        source_dates = [parse_game_date(row["date"]) for row in source_completed]
        source_latest = max(source_dates) if source_dates else None
        completed = select_completed_games(games, args.season, target, args.league_id)
        dates = [parse_game_date(row["date"]) for row in completed]
        earliest = min(dates) if dates else None
        latest = max(dates) if dates else None
        counts = Counter()
        for row in completed:
            counts[row["home_team"]] += 1
            counts[row["away_team"]] += 1
        completed_ids = {row["game_id"] for row in completed}
        score_ids = read_ids(RAW / "games_score.csv")
        log_ids = read_ids(RAW / "game_logs.csv")
        score_coverage = completed_ids.issubset(score_ids)
        log_coverage = completed_ids.issubset(log_ids)
        with (RAW / "teams.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            teams = {
                row["team_id"]: f"{row['name']} {row['nickname']}"
                for row in csv.DictReader(handle)
                if row["league_id"] == args.league_id
                and row["level"] == "1"
                and row.get("allstar_team") == "0"
            }

        add(checks, "driver_files", "|".join(DRIVERS), "PASS", "All approved drivers found and readable.")
        add(
            checks,
            "games_score_completed_game_coverage",
            score_coverage,
            "PASS" if score_coverage else "FAIL",
            f"Covered {len(completed_ids & score_ids)} of {len(completed_ids)} completed ABL game IDs through {target.isoformat()}.",
        )
        add(
            checks,
            "game_logs_completed_game_coverage",
            log_coverage,
            "PASS" if log_coverage else "FAIL",
            f"Covered {len(completed_ids & log_ids)} of {len(completed_ids)} completed ABL game IDs through {target.isoformat()}.",
        )

        if source_latest is not None and source_latest > target:
            add(
                checks,
                "team_record_reconciliation",
                "not_applicable_later_source_capture",
                "SKIP",
                "team_record.csv reflects a later source capture; it cannot validate an earlier as-of cutoff.",
            )
        else:
            with (RAW / "team_record.csv").open("r", encoding="utf-8-sig", newline="") as handle:
                official_records = {
                    row["team_id"]: (int(row["g"]), int(row["w"]), int(row["l"]))
                    for row in csv.DictReader(handle)
                    if row["team_id"] in teams
                }
            computed_wl: dict[str, Counter[str]] = {
                team_id: Counter() for team_id in teams
            }
            for row in completed:
                home, away = row["home_team"], row["away_team"]
                home_runs, away_runs = int(row["runs1"]), int(row["runs0"])
                winner, loser = (home, away) if home_runs > away_runs else (away, home)
                computed_wl[winner]["w"] += 1
                computed_wl[loser]["l"] += 1
            record_reconciliation = all(
                official_records.get(team_id)
                == (counts[team_id], computed_wl[team_id]["w"], computed_wl[team_id]["l"])
                for team_id in teams
            )
            add(
                checks,
                "team_record_reconciliation",
                record_reconciliation,
                "PASS" if record_reconciliation else "FAIL",
                "Computed G-W-L matches team_record.csv because the source capture does not extend beyond the requested cutoff.",
            )

        counts_valid = (
            len(counts) == 24
            and set(counts) == set(teams)
            and max(counts.values()) - min(counts.values()) <= 2
        ) if counts else False
        if not dates:
            verdict = "NOT_READY_NO_DATE_DETECTED"
        elif latest != target:
            verdict = "NOT_READY_MISSING_TARGET_DATE"
        elif not counts_valid or not score_coverage or not log_coverage or record_reconciliation is False:
            verdict = "NOT_READY_INCONSISTENT_GAME_COUNTS"
        else:
            verdict = "READY_FOR_CURRENT_RUN"

    min_games = min(counts.values()) if counts else ""
    max_games = max(counts.values()) if counts else ""
    all_teams = len(counts) == 24 and set(counts) == set(teams)
    add(checks, "source_latest_completed_game_date", source_latest.isoformat() if source_latest else "", "PASS" if source_latest else "FAIL", "Latest completed regular-season date present in the source capture before cutoff filtering.")
    add(checks, "earliest_completed_game_date", earliest.isoformat() if earliest else "", "PASS" if earliest else "FAIL", "Earliest completed regular-season game within the requested season and cutoff.")
    add(checks, "latest_completed_game_date", latest.isoformat() if latest else "", "PASS" if latest == target else "FAIL", f"Filtered evidence must end on the requested cutoff {target.isoformat()}.")
    add(checks, "completed_regular_season_games", len(completed), "PASS" if completed else "FAIL", f"Completed league-{args.league_id} regular-season games with date <= {target.isoformat()}.")
    add(checks, "teams_represented", len(counts), "PASS" if all_teams else "FAIL", "Expected all 24 ABL major-league teams.")
    add(checks, "min_games_per_team", min_games, "PASS" if counts else "FAIL", "Minimum completed games across ABL teams at the cutoff.")
    add(checks, "max_games_per_team", max_games, "PASS" if counts and int(max_games) - int(min_games) <= 2 else "FAIL", "A two-game spread is accepted for a partial-day cutoff and normal schedule balance.")
    sortable_files = sorted(path.name for path in SORTABLE.glob("*.csv"))
    add(checks, "sortable_support_files", len(sortable_files), "PASS" if len(sortable_files) == 20 else "WARN", "Supplemental only; cannot prove current date.")
    add(checks, "accepted_schema_drift", 2, "PASS", "Staff and team batting 13-column schemas accepted as forward standards.")
    add(checks, "stale_derivatives_excluded", len(stale_files), "PASS", "Excluded from current-state proof.")
    add(checks, "final_verdict", verdict, "PASS" if verdict == "READY_FOR_CURRENT_RUN" else "FAIL", "Story work is safe to resume only on READY_FOR_CURRENT_RUN.")

    output_dir.mkdir(parents=True, exist_ok=True)
    with outputs[0].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check_name", "value", "status", "details"])
        writer.writeheader()
        writer.writerows(checks)
    team_rows = [
        {"team_id": team_id, "team_name": teams.get(team_id, "unknown"), "games_played": count}
        for team_id, count in sorted(counts.items(), key=lambda item: int(item[0]))
    ]
    payload = {
        "season": args.season,
        "league_id": args.league_id,
        "newsroom_date": newsroom.isoformat(),
        "target_as_of_date": target.isoformat(),
        "source_latest_completed_game_date": source_latest.isoformat() if source_latest else None,
        "earliest_completed_game_date": earliest.isoformat() if earliest else None,
        "latest_completed_game_date": latest.isoformat() if latest else None,
        "completed_regular_season_games": len(completed),
        "min_games_per_team": min_games,
        "max_games_per_team": max_games,
        "all_24_teams_represented": all_teams,
        "current_state_driver_files_used": [f"csv/ootp_csv/{name}" for name in DRIVERS if (RAW / name).exists()],
        "driver_cross_checks": {
            "games_score_complete": score_coverage,
            "game_logs_complete": log_coverage,
            "team_record_reconciliation": record_reconciliation,
        },
        "games_by_team": team_rows,
        "sortable_support_files_available": sortable_files,
        "stale_derivative_files_excluded": stale_files,
        "verdict": verdict,
        "story_work_safe_to_resume": verdict == "READY_FOR_CURRENT_RUN",
        "checks": checks,
    }
    outputs[1].write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        f"# ABL Current-State Preflight — Target {target.isoformat()}",
        "",
        f"- Season: `{args.season}`",
        f"- Newsroom date: `{newsroom.isoformat()}`",
        f"- Target as-of date: `{target.isoformat()}`",
        f"- Latest source date before filtering: `{source_latest.isoformat() if source_latest else 'not detected'}`",
        f"- Latest included game: `{latest.isoformat() if latest else 'not detected'}`",
        f"- Completed ABL regular-season games through cutoff: **{len(completed)}**",
        f"- Games per team: **{min_games}–{max_games}**",
        f"- All 24 teams represented: **{'YES' if all_teams else 'NO'}**",
        f"- Verdict: **{verdict}**",
        "",
        f"Only completed regular-season games dated on or before {target.isoformat()} are included. Later source rows are excluded.",
        "",
        "## Current-state drivers used",
        "",
    ]
    lines += [f"- `csv/ootp_csv/{name}`" for name in DRIVERS if (RAW / name).exists()]
    lines += ["", "## Games per team", "", "| Team ID | Team | Games |", "|---:|---|---:|"]
    lines += [f"| {row['team_id']} | {row['team_name']} | {row['games_played']} |" for row in team_rows]
    lines += [
        "",
        "## Team-record cross-check",
        "",
        "- Skipped because `team_record.csv` reflects a later source capture than this cutoff."
        if record_reconciliation is None
        else f"- Reconciliation: **{'PASS' if record_reconciliation else 'FAIL'}**",
        "",
        "## Stale derivatives excluded from current-state proof",
        "",
    ]
    lines += [f"- `{item}`" for item in stale_files]
    lines += ["", "## Final verdict", "", f"**{verdict}**", ""]
    outputs[2].write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "season": args.season,
        "target_as_of_date": target.isoformat(),
        "source_latest_game_date": source_latest.isoformat() if source_latest else None,
        "latest_included_game_date": latest.isoformat() if latest else None,
        "completed_regular_season_games": len(completed),
        "verdict": verdict,
        "outputs": [str(path) for path in outputs],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
