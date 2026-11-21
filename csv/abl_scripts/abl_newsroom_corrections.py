"""Identify revised player lines and inning scores in already reported ABL games."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path


PLAYER_TABLES = ("players_game_batting.csv", "players_game_pitching_stats.csv")
ROW_IDS = {"league_id", "year", "level_id", "game_id", "team_id", "player_id"}


def _manifest_hashes(directory: Path) -> dict:
    manifest = directory / "manifest.json"
    if not manifest.is_file():
        return {}
    return json.loads(manifest.read_text(encoding="utf-8"))["hashes"]


def _line_groups(path: Path, game_ids: set[int], season: int) -> dict:
    """Stable record identity plus a multiset, so CSV order has no editorial meaning."""
    grouped = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            gid = int(row["game_id"])
            if gid not in game_ids:
                continue
            if path.name == "games_score.csv":
                key = (gid, int(row["team"]), int(row["inning"]))
                fields = (("score", row["score"]),)
            else:
                if row["league_id"] != "200" or int(row["year"]) != season or row["level_id"] != "1":
                    continue
                key = (gid, int(row["team_id"]), int(row["player_id"]))
                fields = tuple(sorted((k, v) for k, v in row.items() if k not in ROW_IDS))
            grouped[key].append(fields)
    return grouped


def _same_value(before: str, after: str) -> bool:
    if before == after:
        return True
    try:
        return Decimal(before) == Decimal(after)
    except (InvalidOperation, ValueError, TypeError):
        return False


def _describe(key, filename, before, after):
    gid, a, b = key
    if filename == "games_score.csv":
        reference = f"[games_score.csv game_id={gid} team={a} inning={b}]"
        subject = f"Inning {b}, {'visitor' if a == 0 else 'home'} scoring"
    else:
        reference = f"[{filename} game_id={gid} team_id={a} player_id={b}]"
        role = "batting" if filename == PLAYER_TABLES[0] else "pitching"
        subject = f"Team #{a}, player #{b}, {role} line"
    if before is None or after is None:
        row = dict(after if before is None else before)
        highlights = ("score", "ab", "h", "rbi", "outs", "er", "r", "k")
        numbers = ", ".join(f"{field} {row[field]}" for field in highlights if field in row)
        change = "added" if before is None else "removed"
        return f"{subject} {change} ({numbers}). {reference}"
    lhs, rhs = dict(before), dict(after)
    changes = []
    for field in sorted(lhs.keys() | rhs.keys()):
        previous, current = lhs.get(field, "absent"), rhs.get(field, "absent")
        if not _same_value(previous, current):
            changes.append(f"{field} {previous} → {current}")
    if changes:
        return f"{subject} revised ({', '.join(changes)}). {reference}"
    return None


def changed_lines(old_dir: Path, new_dir: Path, common_game_ids, season: int) -> dict[int, list[str]]:
    """Compare only completed games shared by the previous and new verified editions."""
    old_dir, new_dir = Path(old_dir), Path(new_dir)
    valid_games = {int(gid) for gid in common_game_ids}
    changes = defaultdict(list)
    if not valid_games:
        return {}
    old_hashes, new_hashes = _manifest_hashes(old_dir), _manifest_hashes(new_dir)
    for filename in (*PLAYER_TABLES, "games_score.csv"):
        if filename in old_hashes and old_hashes[filename] == new_hashes.get(filename):
            continue
        old = _line_groups(old_dir / filename, valid_games, int(season))
        new = _line_groups(new_dir / filename, valid_games, int(season))
        for key in sorted(old.keys() | new.keys()):
            before, after = sorted(old.get(key, ())), sorted(new.get(key, ()))
            if before == after:
                continue
            # Remove identical occurrences before matching repeated rows with the same key.
            for line in before.copy():
                if line in after:
                    before.remove(line)
                    after.remove(line)
            for index in range(max(len(before), len(after))):
                prior = before[index] if index < len(before) else None
                current = after[index] if index < len(after) else None
                description = _describe(key, filename, prior, current)
                if description:
                    changes[key[0]].append(description)
    return dict(changes)
