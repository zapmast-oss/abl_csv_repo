"""Admit a stable OOTP export as an immutable input for a newsroom cycle."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from uuid import uuid4
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

from abl_path_policy import validate_output_path


LEAGUE_ID = "200"
LIVE_RELATIVE = Path("Documents/Out of the Park Developments/OOTP Baseball 26/saved_games/Action Baseball League.lg/import_export/csv")


class ExportError(ValueError):
    """The capture cannot yet support a new editorial edition."""


def default_live_dir() -> Path:
    return Path(os.environ.get("OneDrive") or Path.home() / "OneDrive") / LIVE_RELATIVE


def rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle, escapechar="\\")


def iso_date(value: str) -> str:
    return date(*map(int, value.split("-"))).isoformat()


def inventory(directory: Path, expected: set[str]) -> dict[str, str]:
    actual = {p.name for p in directory.glob("*.csv") if p.is_file()}
    if actual != expected:
        raise ExportError(f"{directory}: expected 72 immediate CSVs; missing {sorted(expected-actual)}, extra {sorted(actual-expected)}")
    hashes = {}
    for name in sorted(actual):
        path = directory / name
        if path.is_symlink() or not path.stat().st_size:
            raise ExportError(f"Empty or linked input: {path}")
        with path.open("rb") as handle:
            hashes[name] = hashlib.file_digest(handle, "sha256").hexdigest()
    return hashes


def source_digest(hashes: dict[str, str]) -> str:
    data = "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items()))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def registered_sources(repo: Path) -> dict[str, str]:
    registry = repo / "csv/out/control/abl_source_registry.csv"
    registered = {Path(r["file_path"]).name: r["source_id"] for r in rows(registry)
                  if Path(r["file_path"]).parent.as_posix() == "csv/ootp_csv"}
    if len(registered) != 72:
        raise ExportError(f"Expected 72 registered promoted input names, found {len(registered)} in {registry}")
    return registered


def prior_snapshot(root: Path) -> tuple[Path, dict]:
    # Link only to previously captured bytes; a live export can be rewritten in place.
    folders = [d for d in root.iterdir() if d.is_dir() and len(d.name) == 64]
    for folder in sorted(folders, key=lambda p: p.stat().st_mtime_ns, reverse=True):
        manifest = folder / "manifest.json"
        if manifest.is_file():
            return folder, json.loads(manifest.read_text(encoding="utf-8"))["hashes"]
    return root, {}


def capture(source: Path, root: Path, hashes: dict[str, str], source_id: str, registry: dict) -> Path:
    final = root / source_id
    root.mkdir(parents=True, exist_ok=True)
    if final.exists():
        if inventory(final, set(hashes)) != hashes:
            raise ExportError(f"Stored immutable snapshot failed a SHA-256 recheck: {final}")
        if inventory(source, set(hashes)) != hashes:
            raise ExportError("Live export changed during snapshot lookup; retry when stable")
        return final
    previous, old_hashes = prior_snapshot(root)
    # Inherit workspace ACLs so a later run under the operator account can recover.
    staging = root / (".capture-" + uuid4().hex)
    staging.mkdir()
    try:
        for name, digest in hashes.items():
            if old_hashes.get(name) == digest:
                try:
                    os.link(previous / name, staging / name)
                    continue
                except OSError:
                    pass  # Hard links are an optimization. A checked copy is equivalent.
            shutil.copyfile(source / name, staging / name)
        if inventory(staging, set(hashes)) != hashes or inventory(source, set(hashes)) != hashes:
            raise ExportError("Export changed during capture or copy verification failed; retry when stable")
        manifest = {"schema": 1, "source_id": source_id, "original_directory": str(source),
                    "captured_utc": datetime.now(timezone.utc).isoformat(), "hashes": hashes,
                    "registry_ids": registry}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        try:
            staging.rename(final)
        except FileExistsError:
            if inventory(final, set(hashes)) != hashes:
                raise ExportError(f"Concurrent snapshot mismatch: {final}")
        return final
    finally:
        # Temp directory created by this call, distinct from committed archives.
        if staging.exists() and staging.resolve().parent == root.resolve():
            shutil.rmtree(staging)


def schedule(directory: Path) -> dict:
    leagues = [r for r in rows(directory / "leagues.csv") if r["league_id"] == LEAGUE_ID]
    if len(leagues) != 1:
        raise ExportError("Expected exactly one league 200 configuration")
    season = int(leagues[0]["season_year"])
    league_date = iso_date(leagues[0]["current_date"])
    teams = {r["team_id"]: r for r in rows(directory / "teams.csv")
             if r["league_id"] == LEAGUE_ID and r["level"] == "1" and r["allstar_team"] == "0"}
    if len(teams) != 24:
        raise ExportError(f"Expected 24 ABL franchises; found {len(teams)}")
    games, completed = {}, []
    for game in rows(directory / "games.csv"):
        if game["league_id"] != LEAGUE_ID or game["game_type"] != "0":
            continue
        when = iso_date(game["date"])
        if int(when[:4]) != season:
            continue
        game["date"] = when
        if game["away_team"] not in teams or game["home_team"] not in teams or game["home_team"] == game["away_team"]:
            raise ExportError(f"Unexpected participants in ABL game {game['game_id']}")
        gid = game["game_id"]
        if gid in games or game["played"] not in ("0", "1"):
            raise ExportError(f"Duplicate game or invalid completion flag: {gid}")
        games[gid] = game
        if game["played"] == "1":
            if when > league_date or int(game["innings"]) < 1 or game["runs0"] == game["runs1"]:
                raise ExportError(f"Invalid completed game {gid} on {when}")
            completed.append(game)
    if not completed:
        raise ExportError(f"No completed ABL regular-season games in season {season}")
    completed.sort(key=lambda game: (game["date"], int(game["game_id"])))
    cutoff = completed[-1]["date"]
    final_day = [g for g in games.values() if g["date"] == cutoff]
    remaining = [g["game_id"] for g in final_day if g["played"] != "1"]
    return {"season": season, "league_date": league_date, "cutoff": cutoff, "teams": teams, "games": completed,
            "day": {"scheduled": len(final_day), "completed": len(final_day)-len(remaining),
                    "remaining": len(remaining), "remaining_game_ids": remaining,
                    "status": "PARTIAL" if remaining else "COMPLETE"}}


def confirm_results(raw: Path, state: dict) -> list[dict]:
    complete = {g["game_id"]: g for g in state["games"]}
    totals = defaultdict(Counter)
    keys = set()
    for line in rows(raw / "games_score.csv"):
        gid = line["game_id"]
        if gid not in complete:
            continue
        team, inning, runs = int(line["team"]), int(line["inning"]), int(line["score"])
        key = gid, team, inning
        if key in keys or runs < 0 or team not in (0, 1) or not 1 <= inning <= int(complete[gid]["innings"]):
            raise ExportError(f"Invalid or duplicate inning: {key}")
        keys.add(key)
        totals[gid][team] += runs
    for gid, g in complete.items():
        if gid not in totals or set(totals[gid]) != {0, 1} or [totals[gid][i] for i in (0, 1)] != [int(g["runs0"]), int(g["runs1"])]:
            raise ExportError(f"Inning lines disagree with final score for game {gid}")

    first_half, second_half, final_line = {}, {}, {}
    with (raw / "game_logs.csv").open(encoding="utf-8-sig") as log:
        for line in log:
            parts = line.rstrip().split(",", 3)
            if len(parts) != 4:
                continue
            gid, kind, _, body = parts
            if gid not in complete:
                continue
            if kind == "4":
                final_line[gid] = body
            elif kind == "1" and body.startswith('"Top of the 1st -'):
                first_half[gid] = body
            elif kind == "1" and body.startswith('"Bottom of the 1st -'):
                second_half[gid] = body
    exceptions = []
    for gid, g in complete.items():
        away, home = [state["teams"][g[k]] for k in ("away_team", "home_team")]
        errors = []
        for club, half in ((away, first_half), (home, second_half)):
            full = f"{club['name'].strip()} {club['nickname'].strip()} batting"
            if full not in half.get(gid, ""):
                errors.append("batting club " + full)
        score = f"{away['name'].strip()} {g['runs0']} - {home['name'].strip()} {g['runs1']}"
        if score not in final_line.get(gid, ""):
            errors.append("final line " + score)
        if errors:
            exceptions.append({"game_id": gid, "date": g["date"], "excluded_log_reason": ", ".join(errors)})
    return exceptions


def confirm_player_game_totals(raw: Path, state: dict, reconciliation=None) -> None:
    """Detect a mixed export where the schedule advanced ahead of the player tables."""
    from abl_player_games import BATTING, PITCHING, player_game_rows

    games = {game["game_id"]: game for game in state["games"]}
    for filename, fields in ((BATTING, ("r", "h")), (PITCHING, ("r", "ha", "gs"))):
        totals = defaultdict(Counter)
        for row in player_game_rows(raw, filename, state["season"], games, reconciliation):
            gid = row["game_id"]
            team = row["team_id"]
            if team not in (games[gid]["away_team"], games[gid]["home_team"]):
                raise ExportError(f"{filename}: player game line has unexpected team {team}, game {gid}")
            totals[gid, team].update({key: int(row[key]) for key in fields})
        for gid, game in games.items():
            for side, team in enumerate((game["away_team"], game["home_team"])):
                if "gs" in fields:
                    expected = (int(game[f"runs{1-side}"]), int(game[f"hits{1-side}"]), 1)
                else:
                    expected = (int(game[f"runs{side}"]), int(game[f"hits{side}"]))
                actual = totals.get((gid, team))
                if actual is None or tuple(actual[key] for key in fields) != expected:
                    raise ExportError(f"{filename}: completed {gid}, team {team}, expected {dict(zip(fields, expected))}, got {actual}")


def verify_export(repo: Path, preflight_dir: Path, live_dir: Path | None = None, *, snapshot_root: Path | None = None) -> dict:
    """Return JSON-serializable verified state. Only immutable snapshots feed reporting."""
    from abl_player_games import reconcile_batting

    repo = Path(repo).resolve()
    source = (Path(live_dir) if live_dir else default_live_dir()).resolve()
    promoted = repo / "csv/ootp_csv"
    registry = registered_sources(repo)
    hashes = inventory(source, set(registry))
    baseline = inventory(promoted, set(registry))
    differing = sorted(k for k in registry if hashes[k] != baseline[k])
    source_id = source_digest(hashes)
    root = validate_output_path(snapshot_root or repo / "output/newsroom/snapshots", repo)
    out = validate_output_path(preflight_dir, repo)
    if not root.is_relative_to(repo) or not out.is_relative_to(repo) or root.is_relative_to(source) or out.is_relative_to(source):
        raise ExportError("Snapshot and verification outputs must be in the repository outside the live export")
    raw = capture(source, root, hashes, source_id, registry)
    state = schedule(raw)
    if differing:
        base = schedule(promoted)
        historical_replay = live_dir is not None and root != (repo / "output/newsroom/snapshots").resolve()
        if not historical_replay and (state["season"], state["cutoff"], state["league_date"]) < (base["season"], base["cutoff"], base["league_date"]):
            raise ExportError(f"Source regresses behind promoted league date/cutoff {base['league_date']} / {base['cutoff']}")
        if not historical_replay and state["season"] == base["season"] and {g["game_id"] for g in base["games"]}-{g["game_id"] for g in state["games"]}:
            raise ExportError("Some previously completed games in the promoted snapshot are absent from this live export")
        for name in registry:
            with (raw / name).open("rb") as incoming, (promoted / name).open("rb") as reference:
                if incoming.readline().rstrip() != reference.readline().rstrip():
                    raise ExportError(f"Changed OOTP header requires review: {name}")
    exceptions = confirm_results(raw, state)
    reconciliation = reconcile_batting(raw, state, root, exceptions)
    confirm_player_game_totals(raw, state, reconciliation)
    args = [sys.executable, "-B", str(repo / "csv/abl_scripts/z_abl_current_state_preflight_validate.py"),
            "--season", str(state["season"]), "--as-of-date", state["cutoff"], "--newsroom-date", state["league_date"],
            "--output-dir", str(out), "--raw-dir", str(raw), "--json-only"]
    run = subprocess.run(args, capture_output=True, text=True, check=False)
    if run.returncode != 0:
        raise ExportError(f"Existing ABL preflight failed: {run.stderr[-1500:]}")
    receipt = out / f"current_state_preflight_{state['season']}_target_{state['cutoff']}.json"
    verified = json.loads(receipt.read_text(encoding="utf-8"))
    if verified["verdict"] != "READY_FOR_CURRENT_RUN" or verified["completed_regular_season_games"] != len(state["games"]):
        raise ExportError(f"Current-state preflight returned {verified['verdict']} for {state['cutoff']}: {receipt}")
    state.update(hashes=hashes, source_id=source_id, raw_dir=str(raw), live_dir=str(source),
                 preflight=verified, core_files=list(verified["current_state_driver_files_used"]),
                 promoted_matches=72-len(differing), promoted_different_files=differing, log_exceptions=exceptions)
    if reconciliation:
        state["player_game_reconciliation"] = reconciliation
        (out / "player-game-reconciliation.json").write_text(
            json.dumps(reconciliation, indent=2) + "\n", encoding="utf-8")
    return state
