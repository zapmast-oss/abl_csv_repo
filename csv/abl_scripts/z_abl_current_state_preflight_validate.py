from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONTROL = ROOT / "csv" / "out" / "control"
RAW = ROOT / "csv" / "ootp_csv"
SORTABLE = ROOT / "csv" / "abl_statistics"
OUT_STEM = CONTROL / "current_state_preflight_1981_target_1981-07-19"
TARGET = datetime.strptime("1981-07-19", "%Y-%m-%d").date()
NEWSROOM = "1981-07-20"
DRIVERS = ("games.csv", "games_score.csv", "game_logs.csv")
STALE_FILES = [
    "csv/out/star_schema/monday_1981_standings_by_division.csv (32-game snapshot)",
    "csv/out/star_schema/fact_team_reporting_1981_weekly_change.csv (32-game snapshot)",
    "csv/out/star_schema/fact_player_batting.csv (preserved early snapshot)",
    "csv/out/star_schema/fact_player_pitching.csv (preserved early snapshot)",
    "csv/out/csv_out/z_ABL_Manager_Tendencies.csv (62-game snapshot)",
    "csv/out/csv_out/z_ABL_Division_Leverage.csv (62-game snapshot)",
    "csv/out/csv_out/z_ABL_Rotation_Stability.csv (62-game snapshot)",
]


def parse_date(value: str):
    return datetime.strptime(value, "%Y-%m-%d").date()


def read_ids(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        return {row["game_id"] for row in csv.DictReader(handle)}


def add(checks: list[dict[str, str]], name: str, value: object, status: str, details: str) -> None:
    checks.append({"check_name": name, "value": str(value), "status": status, "details": details})


def main() -> int:
    checks: list[dict[str, str]] = []
    missing = [name for name in DRIVERS if not (RAW / name).exists()]
    if missing:
        verdict = "NOT_READY_NO_CURRENT_STATE_DRIVER"
        add(checks, "driver_files", "|".join(missing), "FAIL", "Approved current-state driver files are missing.")
        completed = []; counts = Counter(); earliest = latest = None; teams: dict[str, str] = {}
        score_coverage = log_coverage = False
    else:
        with (RAW / "games.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            games = list(csv.DictReader(handle))
        completed = [row for row in games if row["league_id"] == "200" and row["game_type"] == "0" and row["played"] == "1"]
        dates = [parse_date(row["date"]) for row in completed]
        earliest = min(dates) if dates else None; latest = max(dates) if dates else None
        counts: Counter[str] = Counter()
        for row in completed:
            counts[row["home_team"]] += 1; counts[row["away_team"]] += 1
        completed_ids = {row["game_id"] for row in completed}
        score_ids = read_ids(RAW / "games_score.csv"); log_ids = read_ids(RAW / "game_logs.csv")
        score_coverage = completed_ids.issubset(score_ids); log_coverage = completed_ids.issubset(log_ids)
        with (RAW / "teams.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            teams = {
                row["team_id"]: f"{row['name']} {row['nickname']}"
                for row in csv.DictReader(handle)
                if row["league_id"] == "200" and row["level"] == "1" and row.get("allstar_team") == "0"
            }
        add(checks, "driver_files", "|".join(DRIVERS), "PASS", "All approved drivers found and readable.")
        add(checks, "games_score_completed_game_coverage", score_coverage, "PASS" if score_coverage else "FAIL", f"Covered {len(completed_ids & score_ids)} of {len(completed_ids)} completed ABL game IDs.")
        add(checks, "game_logs_completed_game_coverage", log_coverage, "PASS" if log_coverage else "FAIL", f"Covered {len(completed_ids & log_ids)} of {len(completed_ids)} completed ABL game IDs.")
        if not dates:
            verdict = "NOT_READY_NO_DATE_DETECTED"
        elif latest < TARGET:
            verdict = "NOT_READY_MISSING_TARGET_DATE"
        elif set(counts) != set(teams) or len(counts) != 24 or max(counts.values()) - min(counts.values()) > 1 or not score_coverage or not log_coverage:
            verdict = "NOT_READY_INCONSISTENT_GAME_COUNTS"
        else:
            verdict = "READY_FOR_CURRENT_RUN"
    min_games = min(counts.values()) if counts else ""
    max_games = max(counts.values()) if counts else ""
    all_teams = len(counts) == 24 and set(counts) == set(teams)
    add(checks, "earliest_completed_game_date", earliest.isoformat() if earliest else "", "PASS" if earliest else "FAIL", "Earliest completed league-200 regular-season game.")
    add(checks, "latest_completed_game_date", latest.isoformat() if latest else "", "PASS" if latest and latest >= TARGET else "FAIL", f"Target is {TARGET.isoformat()}; July 20 games are not required.")
    add(checks, "completed_regular_season_games", len(completed), "PASS" if completed else "FAIL", "Completed league-200 regular-season games in games.csv.")
    add(checks, "teams_represented", len(counts), "PASS" if all_teams else "FAIL", "Expected all 24 ABL major-league teams.")
    add(checks, "min_games_per_team", min_games, "PASS" if counts else "FAIL", "Minimum completed games across ABL teams.")
    add(checks, "max_games_per_team", max_games, "PASS" if counts and max_games - min_games <= 1 else "FAIL", "A one-game spread is accepted as normal schedule balance.")
    sortable_files = sorted(path.name for path in SORTABLE.glob("*.csv"))
    add(checks, "sortable_support_files", len(sortable_files), "PASS" if len(sortable_files) == 20 else "WARN", "Supplemental only; cannot prove current date.")
    add(checks, "accepted_schema_drift", 2, "PASS", "Staff and team batting 13-column schemas accepted as forward standards.")
    add(checks, "stale_derivatives_excluded", len(STALE_FILES), "PASS", "Excluded from current-state proof.")
    add(checks, "final_verdict", verdict, "PASS" if verdict == "READY_FOR_CURRENT_RUN" else "FAIL", "Story work is safe to resume only on READY_FOR_CURRENT_RUN.")
    CONTROL.mkdir(parents=True, exist_ok=True)
    with OUT_STEM.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check_name", "value", "status", "details"]); writer.writeheader(); writer.writerows(checks)
    team_rows = [
        {"team_id": team_id, "team_name": teams.get(team_id, "unknown"), "games_played": count}
        for team_id, count in sorted(counts.items(), key=lambda item: int(item[0]))
    ]
    payload = {
        "season": 1981, "newsroom_date": NEWSROOM, "target_as_of_date": TARGET.isoformat(),
        "earliest_completed_game_date": earliest.isoformat() if earliest else None,
        "latest_completed_game_date": latest.isoformat() if latest else None,
        "completed_regular_season_games": len(completed), "min_games_per_team": min_games,
        "max_games_per_team": max_games, "all_24_teams_represented": all_teams,
        "current_state_driver_files_used": [f"csv/ootp_csv/{name}" for name in DRIVERS if (RAW / name).exists()],
        "driver_cross_checks": {"games_score_complete": score_coverage, "game_logs_complete": log_coverage},
        "games_by_team": team_rows, "sortable_support_files_available": sortable_files,
        "accepted_schema_drift": [
            "abl_staff.csv: accepted 13-column schema; staff IDs resolved downstream",
            "batting_stats.csv: accepted 13-column schema; removed baserunning fields unavailable or sourced elsewhere",
        ],
        "limited_or_disabled_enrichments": [
            "Unresolved staff-name-to-ID matches remain null.",
            "CS and SB% require another validated source.",
            "BatR, wSB, UBR, and BsR are unavailable; dependent enrichments remain disabled.",
        ],
        "stale_derivative_files_excluded": STALE_FILES,
        "verdict": verdict, "story_work_safe_to_resume": verdict == "READY_FOR_CURRENT_RUN",
        "checks": checks,
    }
    OUT_STEM.with_suffix(".json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# ABL Current-State Preflight — Target 1981-07-19", "",
             f"- Newsroom date: `{NEWSROOM}`", f"- Target as-of date: `{TARGET.isoformat()}`",
             f"- Earliest completed game: `{earliest.isoformat() if earliest else 'not detected'}`",
             f"- Latest completed game: `{latest.isoformat() if latest else 'not detected'}`",
             f"- Completed ABL regular-season games: **{len(completed)}**",
             f"- Games per team: **{min_games}–{max_games}**",
             f"- All 24 teams represented: **{'YES' if all_teams else 'NO'}**",
             f"- Verdict: **{verdict}**",
             f"- Story work safe to resume: **{'YES' if verdict == 'READY_FOR_CURRENT_RUN' else 'NO'}**", "",
             "July 20 is the newsroom date. July 19 is the completed-game cutoff; missing July 20 games are not an error.", "",
             "## Current-state drivers used", ""]
    lines += [f"- `csv/ootp_csv/{name}`" for name in DRIVERS if (RAW / name).exists()]
    lines += ["", "## Games per team", "", "| Team ID | Team | Games |", "|---:|---|---:|"]
    lines += [f"| {row['team_id']} | {row['team_name']} | {row['games_played']} |" for row in team_rows]
    lines += ["", "## Sortable support", "", f"All **{len(sortable_files)}** promoted sortable CSVs are available as supplemental support. They do not prove the date cutoff.", "",
              "## Accepted schema drift", "",
              "- Staff report: 13-column forward schema; IDs supplied by downstream coach lookup where possible.",
              "- Team batting report: 13-column forward schema; WAR comes from batting-extra and removed baserunning fields remain limited.", "",
              "## Limited or disabled enrichment", "",
              "- Unresolved staff IDs remain null.", "- CS and SB% need another validated source.",
              "- BatR, wSB, UBR, and BsR enrichments remain disabled.", "",
              "## Stale derivatives excluded from current-state proof", ""]
    lines += [f"- `{item}`" for item in STALE_FILES]
    lines += ["", "## Final verdict", "", f"**{verdict}**", "",
              "The raw driver set reaches July 19, represents all teams, has a normal one-game schedule spread, and reconciles completed game IDs across score and log files." if verdict == "READY_FOR_CURRENT_RUN" else "Current story generation must remain blocked until the failed checks are corrected.", ""]
    OUT_STEM.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "earliest_game_date": earliest.isoformat() if earliest else None,
        "latest_game_date": latest.isoformat() if latest else None,
        "min_games_per_team": min_games, "max_games_per_team": max_games,
        "all_teams_represented": all_teams, "verdict": verdict,
        "story_work_safe_to_resume": verdict == "READY_FOR_CURRENT_RUN",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
