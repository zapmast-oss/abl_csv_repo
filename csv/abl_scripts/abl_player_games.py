"""Validate ABL player-game rows and account for evidenced suspension placeholders."""

from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal
from hashlib import sha256
import csv
import json
import re

from abl_newsroom_verify import ExportError, LEAGUE_ID, inventory, iso_date, rows, source_digest


BATTING = "players_game_batting.csv"
PITCHING = "players_game_pitching_stats.csv"
AT_BATS = "players_at_bat_batting_stats.csv"
IDENTITY = {"year", "league_id", "level_id", "game_id", "team_id", "player_id", "split_id", "position", "stint"}
GAME_IDENTITY = ("game_id", "league_id", "game_type", "home_team", "away_team", "time", "starter0", "starter1")
SEASON_FIELDS = ("g", "gs", "pa", "ab", "h", "d", "t", "hr", "r", "rbi", "k", "bb", "ibb",
                 "gdp", "sh", "sf", "hp", "ci", "sb", "cs", "pitches_seen")
HALF = re.compile(r"^(Top|Bottom) of the (\d+)(?:st|nd|rd|th) - ")
BATTER = re.compile(r"/players/player_(\d+)\.html")


def row_key(row):
    return row["game_id"], row["team_id"], row["player_id"]


def row_digest(row):
    return sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def scoped_rows(raw, filename, season, ids):
    for line, row in enumerate(rows(raw / filename), 2):
        if row["game_id"] in ids and (row["league_id"], row["level_id"], row["year"]) == (LEAGUE_ID, "1", str(season)):
            yield line, row


def player_game_rows(raw, filename, season, game_ids, reconciliation=None):
    """Yield unique completed-game records, checking each excluded row against its receipt."""
    ids = {str(gid) for gid in game_ids}
    evidence = {}
    if filename == BATTING and reconciliation:
        if reconciliation["source_id"] != raw.name:
            raise ExportError("Player-game reconciliation belongs to a different capture.")
        for entry in reconciliation["entries"]:
            if entry["game_id"] in ids:
                for role in ("retained", "auxiliary"):
                    item = entry[role]
                    evidence[item["csv_row"]] = (item["sha256"], role == "auxiliary")
    seen = set()
    for line, row in scoped_rows(raw, filename, season, ids):
        if line in evidence:
            expected, omit = evidence.pop(line)
            if row_digest(row) != expected:
                raise ExportError("The reconciled batting row failed its SHA-256 recheck.")
            if omit:
                continue
        key = row_key(row)
        if key in seen:
            raise ExportError(f"{filename}: unexplained duplicate player-game key {key}")
        seen.add(key)
        yield row
    if evidence:
        raise ExportError("The reconciled batting row is missing from the requested games.")


def snapshot_receipt(folder):
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest["source_id"] != folder.name or source_digest(manifest["hashes"]) != folder.name:
        raise ExportError(f"Unverified snapshot receipt for {folder.name}")
    return manifest


def unfinished_snapshots(raw, root, games):
    current = datetime.fromisoformat(snapshot_receipt(raw)["captured_utc"])
    candidates = []
    for folder in root.iterdir():
        if folder == raw or not re.fullmatch(r"[a-f0-9]{64}", folder.name) or not folder.is_dir():
            continue
        if folder.is_symlink() or not (folder / "manifest.json").is_file():
            continue
        receipt = snapshot_receipt(folder)
        when = datetime.fromisoformat(receipt["captured_utc"])
        if when < current:
            candidates.append((when, folder, receipt["hashes"]))
    found = {}
    for _, folder, hashes in sorted(candidates, reverse=True):
        pending = games.keys() - found.keys()
        if not pending:
            break
        path = folder / "games.csv"
        with path.open("rb") as stream:
            if sha256(stream.read()).hexdigest() != hashes["games.csv"]:
                raise ExportError(f"Earlier schedule snapshot failed verification: {folder.name}")
        matches = {}
        for earlier in rows(path):
            gid = earlier["game_id"]
            if gid not in pending or earlier["played"] != "0":
                continue
            final = games[gid]
            if (0 < int(earlier["innings"]) < int(final["innings"])
                    and iso_date(earlier["date"]) == final["date"]
                    and all(earlier[k] == final[k] for k in GAME_IDENTITY)
                    and all(int(earlier[k]) <= int(final[k]) for k in ("runs0", "runs1", "hits0", "hits1"))):
                if gid in matches:
                    raise ExportError(f"Ambiguous earlier schedule for game {gid}")
                matches[gid] = earlier
        if matches:
            if inventory(folder, set(hashes)) != hashes:
                raise ExportError(f"Earlier OOTP snapshot failed verification: {folder.name}")
            found.update({gid: (folder, earlier, hashes) for gid, earlier in matches.items()})
    return found


def events_for(raw, keys):
    events = defaultdict(list)
    for line, row in enumerate(rows(raw / AT_BATS), 2):
        key = row_key(row)
        if key in keys:
            events[key].append({"csv_row": line, "inning": int(row["inning"]), "sha256": row_digest(row)})
    return events


def log_appearances(raw, games, keys):
    """Index batting announcements in a completed, score-checked game log."""
    observed, halves, seen = defaultdict(list), {}, set()
    with (raw / "game_logs.csv").open(encoding="utf-8-sig") as stream:
        for text in stream:
            gid = text.split(",", 1)[0]
            if gid not in games:
                continue
            _, kind, line, body = next(csv.reader([text], escapechar="\\"))
            if (gid, line) in seen:
                raise ExportError(f"Repeated play-by-play line in game {gid}")
            seen.add((gid, line))
            if kind == "1" and (half := HALF.match(body)):
                halves[gid] = games[gid]["away_team" if half[1] == "Top" else "home_team"], int(half[2])
            if kind == "2" and body.startswith(("Batting:", "Pinch Hitting:")) and (pid := BATTER.search(body)):
                if gid not in halves:
                    raise ExportError(f"Missing preceding inning header in game {gid}")
                team, inning = halves[gid]
                key = (gid, team, pid[1])
                if key in keys:
                    observed[key].append({"game_log_line": int(line), "inning": inning})
    return observed


def resolve_pair(pair, prior, events_before, events_after, announcements, prior_innings):
    """The sole admitted pattern retains an exact earlier starter line."""
    if len(pair) != 2 or len(prior) != 1 or len({line for line, _ in pair}) != 2:
        raise ExportError("Expected one prior batting row and two distinct current records.")
    _, earlier = prior[0]
    matching = [(line, row) for line, row in pair if row == earlier]
    if len(matching) != 1:
        raise ExportError("Exactly one current row must match the earlier batting line.")
    retained = matching[0]
    auxiliary = next((line, row) for line, row in pair if line != retained[0])
    extra = auxiliary[1]
    if ((earlier["g"], earlier["gs"]) != ("1", "1") or int(earlier["pa"]) <= 0
            or extra.keys() != earlier.keys()
            or any(extra[k] != earlier[k] for k in IDENTITY)
            or (extra["g"], extra["gs"]) != ("1", "0")
            or any(Decimal(extra[k]) != 0 for k in extra.keys() - IDENTITY - {"g", "gs"})):
        raise ExportError("Additional row contains unexplained batting, baserunning, or lineup information.")
    before = Counter(event["sha256"] for event in events_before)
    after = Counter(event["sha256"] for event in events_after)
    innings = Counter(event["inning"] for event in events_after)
    if (before != after or len(events_after) != int(earlier["pa"])
            or innings != Counter(call["inning"] for call in announcements)
            or any(not 1 <= inning <= prior_innings for inning in innings)):
        raise ExportError("The two captures and completed play-by-play do not corroborate the batting line.")
    return retained, auxiliary


def season_discrepancies(raw, state, report):
    season = state["season"]
    games = {game["game_id"]: game for game in state["games"]}
    reconciled_players, reconciled_teams = defaultdict(Counter), defaultdict(Counter)
    for row in player_game_rows(raw, BATTING, season, games, report):
        team, player = row["team_id"], row["player_id"]
        stats = {field: int(row[field]) for field in SEASON_FIELDS}
        reconciled_teams[team].update(stats)
        reconciled_players[team, player].update(stats)
    club_games = Counter(game[side] for game in games.values() for side in ("home_team", "away_team"))
    for team, count in club_games.items():
        reconciled_teams[team]["g"] = count

    differences = []
    coverage = {}
    for filename, split, native_key, reconciled in (
        ("players_career_batting_stats.csv", "1", lambda row: (row["team_id"], row["player_id"]), reconciled_players),
        ("team_batting_stats.csv", "0", lambda row: row["team_id"], reconciled_teams),
    ):
        native = defaultdict(Counter)
        for row in rows(raw / filename):
            key = native_key(row)
            if (row["league_id"], row["year"], row["level_id"], row["split_id"]) == (
                LEAGUE_ID, str(season), "1", split
            ) and row["team_id"] in state["teams"] and row.get("game_id", "0") == "0":
                native[key].update({field: int(row[field]) for field in SEASON_FIELDS})
        for key in sorted(reconciled.keys() | native.keys()):
            team, player = key if isinstance(key, tuple) else (key, None)
            comparable = SEASON_FIELDS
            if player is not None:
                # The game table can leave batter pitch counts unpopulated throughout a season.
                comparable = SEASON_FIELDS[:-1]
            fields = {}
            for field in comparable:
                official = native.get(key, {}).get(field)
                derived = reconciled.get(key, {}).get(field)
                if official != derived:
                    fields[field] = {"ootp": official, "reconciled": derived}
            if fields:
                differences.append({"native_table": filename, "team_id": team, "player_id": player, "fields": fields})
        if split == "1":
            coverage = {
                "native_nonzero": sum(bool(line["pitches_seen"]) for line in native.values()),
                "derived_nonzero": sum(bool(line["pitches_seen"]) for line in reconciled.values()),
                "differing": sum(native.get(key, {}).get("pitches_seen") != reconciled.get(key, {}).get("pitches_seen")
                                 for key in native.keys() | reconciled.keys()),
            }
    return differences, coverage


def reconcile_batting(raw, state, root, log_exceptions):
    games = {game["game_id"]: game for game in state["games"]}
    counts = Counter(row_key(row) for _, row in scoped_rows(raw, BATTING, state["season"], games))
    repeated_keys = {key for key, count in counts.items() if count > 1}
    if not repeated_keys:
        return None
    repeated_games = {gid: games[gid] for gid, _, _ in repeated_keys}
    if repeated_games.keys() & {item["game_id"] for item in log_exceptions}:
        raise ExportError("A repeated player-game line lacks a verified completed game log.")
    repeats = defaultdict(list)
    for line, row in scoped_rows(raw, BATTING, state["season"], repeated_games):
        key = row_key(row)
        if key in repeated_keys:
            repeats[key].append((line, row))
    prior = unfinished_snapshots(raw, root, repeated_games)
    after = events_for(raw, repeats)
    calls = log_appearances(raw, repeated_games, repeats)
    current_hashes = snapshot_receipt(raw)["hashes"]
    report = {"schema": 1, "rule": "unfinished-game-zero-batting-placeholder", "source_id": raw.name,
              "season": state["season"], "cutoff": state["cutoff"], "entries": []}
    for key, pair in sorted(repeats.items()):
        gid, team, player = key
        if gid not in prior:
            raise ExportError(f"{BATTING}: repeated key {key} has no verified earlier unfinished capture.")
        folder, partial, hashes = prior[gid]
        earlier = [(line, row) for line, row in scoped_rows(folder, BATTING, state["season"], {gid}) if row_key(row) == key]
        before = events_for(folder, {key})[key]
        try:
            retained, extra = resolve_pair(pair, earlier, before, after[key], calls[key], int(partial["innings"]))
        except (ValueError, ArithmeticError) as error:
            raise ExportError(f"{BATTING}: cannot reconcile repeated key {key}: {error}") from error
        report["entries"].append({
            "game_id": gid, "team_id": team, "player_id": player, "game_date": games[gid]["date"],
            "prior_source_id": folder.name, "prior_innings": int(partial["innings"]),
            "final_innings": int(games[gid]["innings"]), "prior_csv_row": earlier[0][0],
            "retained": {"csv_row": retained[0], "sha256": row_digest(retained[1])},
            "auxiliary": {"csv_row": extra[0], "sha256": row_digest(extra[1])},
            "prior_pa_rows": before, "completed_pa_rows": after[key], "play_by_play": calls[key],
            "prior_file_hashes": {name: hashes[name] for name in ("games.csv", BATTING, AT_BATS)},
            "completed_file_hashes": {name: current_hashes[name] for name in ("games.csv", BATTING, AT_BATS, "game_logs.csv")},
        })
    report["season_discrepancies"], report["player_pitch_count_coverage"] = season_discrepancies(raw, state, report)
    return report
