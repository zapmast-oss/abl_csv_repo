"""Exercise the reconciliation guard and optionally replay both sealed 1113 captures."""

import csv
import json
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
import shutil
import sys
import unittest
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "csv/abl_scripts"), str(ROOT / "tools/eab")]

from abl_newsroom_verify import ExportError, inventory
from abl_player_games import BATTING, PITCHING, player_game_rows, reconcile_batting, resolve_pair, row_digest


EARLY = "2acae29146076f6298fcafecf8afc73fb47bfdc8e8362ea73a5c38454767d02d"
FINAL = "461bc955c6db88b42b31da7c12c95fc8e1d15796ddd1d26de6f51af30296118d"


@contextmanager
def workspace():
    root = (ROOT / "tmp").resolve()
    root.mkdir(exist_ok=True)
    folder = root / f"rg-{uuid4().hex[:8]}"
    folder.mkdir()
    try:
        yield folder
    finally:
        if folder.resolve().parent == root:
            shutil.rmtree(folder)


class PlayerGameTests(unittest.TestCase):
    def setUp(self):
        self.starter = dict(year="1981", league_id="200", level_id="1", game_id="42", team_id="1",
                            player_id="7", split_id="0", position="3", stint="0",
                            g="1", gs="1", pa="2", ab="2", h="1", r="0", sb="0", wpa="0.0080")
        self.extra = dict(self.starter, gs="0", pa="0", ab="0", h="0", wpa="0.0000")
        self.events = [{"sha256": "a", "inning": 1}, {"sha256": "b", "inning": 3}]
        self.calls = [{"inning": 1}, {"inning": 3}]

    def test_requires_an_exact_starter_and_corroboration(self):
        starter, extra = (7, self.starter), (3, self.extra)
        result = resolve_pair([extra, starter], [(2, self.starter)], self.events,
                              self.events[::-1], self.calls, 5)
        self.assertEqual(result, (starter, extra))

    def test_ambiguous_additional_rows_fail_closed(self):
        for change in ({"g": "0"}, {"gs": "1"}, {"pa": "1"}, {"r": "1"}, {"sb": "1"},
                       {"position": "4"}, {"stint": "1"}, {"wpa": "0.0001"}):
            with self.subTest(change=change), self.assertRaises(ExportError):
                resolve_pair([(2, self.starter), (3, dict(self.extra, **change))],
                             [(2, self.starter)], self.events, self.events, self.calls, 5)
        for lines in ([(3, self.extra)], [(2, self.starter), (3, self.starter)],
                      [(2, self.starter), (3, self.extra), (4, self.extra)]):
            with self.subTest(lines=lines), self.assertRaises(ExportError):
                resolve_pair(lines, [(2, self.starter)], self.events, self.events, self.calls, 5)

    def test_incomplete_or_contradictory_evidence_fails_closed(self):
        pair = [(2, self.starter), (3, self.extra)]
        cases = (
            ([], self.events, self.events, self.calls, 5),
            ([(2, dict(self.starter, h="0"))], self.events, self.events, self.calls, 5),
            ([(2, self.starter)], self.events, self.events[:1], self.calls, 5),
            ([(2, self.starter)], self.events, self.events, self.calls[:1], 5),
            ([(2, self.starter)], self.events, self.events, self.calls, 2),
            ([(2, self.starter)], self.events, [dict(self.events[0], sha256="c"), self.events[1]], self.calls, 5),
        )
        for index, args in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(ExportError):
                resolve_pair(pair, *args)

    def test_reader_retains_legitimate_zero_pa_and_rechecks_both_hashes(self):
        runner = dict(self.extra, player_id="8", r="1", sb="1")
        with workspace() as folder:

            def write(filename, *lines):
                with (folder / filename).open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, list(self.starter))
                    writer.writeheader()
                    writer.writerows(lines)

            report = {"source_id": folder.name, "entries": [
                {"game_id": "42", "retained": {"csv_row": 2, "sha256": row_digest(self.starter)},
                 "auxiliary": {"csv_row": 3, "sha256": row_digest(self.extra)}}]}
            write(BATTING, self.starter, self.extra, runner)
            self.assertEqual(list(player_game_rows(folder, BATTING, 1981, {42}, report)), [self.starter, runner])
            with self.assertRaisesRegex(ExportError, "unexplained duplicate"):
                list(player_game_rows(folder, BATTING, 1981, {42}))
            for starter, extra in ((dict(self.starter, h="0"), self.extra),
                                   (self.starter, dict(self.extra, sb="1"))):
                write(BATTING, starter, extra, runner)
                with self.assertRaisesRegex(ExportError, "SHA-256"):
                    list(player_game_rows(folder, BATTING, 1981, {42}, report))
            write(BATTING, self.starter)
            with self.assertRaisesRegex(ExportError, "missing"):
                list(player_game_rows(folder, BATTING, 1981, {42}, report))
            write(PITCHING, self.starter, self.starter)
            with self.assertRaisesRegex(ExportError, "unexplained duplicate"):
                list(player_game_rows(folder, PITCHING, 1981, {42}, report))


def replay():
    from abl_eab_data import load_world
    from abl_eab_release import Publisher, audit_manifest, catalog_for, file_hash
    from abl_newsroom import run
    from abl_newsroom_slate import game_lines
    from abl_newsroom_verify import verify_export

    sealed = ROOT / "output/newsroom/snapshots"
    old, final = sealed / EARLY, sealed / FINAL
    raw = ROOT / "csv/ootp_csv"

    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))

    checksum = {folder: read(folder / "manifest.json")["hashes"] for folder in (old, final)}
    checksum[raw] = checksum[final]
    for folder, hashes in checksum.items():
        assert len(hashes) == 72 and inventory(folder, set(hashes)) == hashes, folder

    with workspace() as work:
        archive = work / "nr"
        snapshots = archive / "snapshots"
        publisher = Publisher(output=work / "web", newsroom=archive, ledger=work / "ledger", assets=work / "assets")

        first = run(archive, old)
        first_seal = archive / "editions" / first["edition"] / "files.json"
        assert first["source_id"] == EARLY
        old_seal = file_hash(first_seal)
        before, outcome = publisher.publish(old, snapshots=snapshots)
        assert outcome == "published" and before["finals"] == 1450
        old_home = publisher.dist / "releases" / before["id"] / "index.html"
        old_home_hash = file_hash(old_home)
        first_state = verify_export(ROOT, work / "p1", old, snapshot_root=snapshots)
        first_world = load_world(ROOT, first_state, work / "assets")
        assert first_world.games[1113].score is None and 1113 not in first_world.boxes
        assert not first_state.get("player_game_reconciliation")
        print("August 19 snapshot: 1,450 finals. Game 1113 remained unfinished.", flush=True)

        latest = run(archive, final)
        assert latest["source_id"] == FINAL and latest["status"] == "completed"
        intel = archive / "editions" / latest["edition"]
        verification = read(intel / "verification.json")
        report = verification["player_game_reconciliation"]
        assert read(intel / "preflight/player-game-reconciliation.json") == report
        assert len(report["entries"]) == 2
        assert {entry["prior_source_id"] for entry in report["entries"]} == {EARLY}
        assert {entry["player_id"]: [call["game_log_line"] for call in entry["play_by_play"]]
                for entry in report["entries"]} == {"19": [40, 185], "1547": [66, 148]}
        assert len(verification["log_exceptions"]) == len(first_state["log_exceptions"]) == 6

        state = verify_export(ROOT, work / "p2", final, snapshot_root=snapshots)
        with unittest.TestCase().assertRaisesRegex(ExportError, "no verified earlier unfinished capture"):
            reconcile_batting(Path(state["raw_dir"]), state, work, [])
        with unittest.TestCase().assertRaisesRegex(ExportError, "lacks a verified completed game log"):
            reconcile_batting(Path(state["raw_dir"]), state, snapshots, [{"game_id": "1113"}])
        world = load_world(ROOT, state, work / "assets")
        current, outcome = publisher.publish(final, snapshots=snapshots)
        assert outcome == "published" and current["finals"] == world.finals == 1571
        assert current["league_date"] == "1981-08-31" and current["cutoff"] == "1981-08-30"
        assert world.player_game_reconciliation == report
        journal = (publisher.logs / "publications.jsonl").read_text(encoding="utf-8").splitlines()
        assert [entry["report"] for line in journal if (entry := json.loads(line))["status"] == "player-game reconciliation"] == [report]

        wins, losses = Counter(), Counter()
        for game in state["games"]:
            away, home = int(game["away_team"]), int(game["home_team"])
            ra, rh = int(game["runs0"]), int(game["runs1"])
            winner, loser = (away, home) if ra > rh else (home, away)
            wins[winner] += 1
            losses[loser] += 1
            box = world.boxes[int(game["game_id"])]
            for side, club in enumerate((away, home)):
                batters = [line for _, line in box[club, "bat"]]
                assert (sum(line["r"] for line in batters), sum(line["h"] for line in batters)) == (
                    int(game[f"runs{side}"]), int(game[f"hits{side}"]))
        assert len(world.teams) == 24 and sum(wins.values()) == sum(losses.values()) == 1571
        assert all((club["w"], club["l"]) == (wins[tid], losses[tid]) for tid, club in world.teams.items())
        assert world.games[1113].score == (5, 3)

        game = next(g for g in state["games"] if g["game_id"] == "1113")
        news_book = {1113: {"home_id": int(game["home_team"]), "away_id": int(game["away_team"]),
                            "date": game["date"]}}
        by_player, _ = game_lines(Path(state["raw_dir"]), news_book, 1981, report)
        for pid, tid, strikeouts, season_games in ((19, 23, 1, 82), (1547, 19, 0, 92)):
            eab_lines = [line for batter, line in world.boxes[1113][tid, "bat"] if batter == pid]
            newsroom_lines = [record[-1] for record in by_player["bat"][pid]]
            for lines in (eab_lines, newsroom_lines):
                assert len(lines) == 1
                assert tuple(lines[0][key] for key in ("g", "pa", "ab", "h", "k")) == (1, 2, 2, 0, strikeouts)
            assert world.batting[pid, tid]["g"] == season_games
        zero_pa = [line for box in world.boxes.values() for (_tid, kind), lines in box.items()
                   if kind == "bat" for _, line in lines if line["pa"] == 0]
        assert zero_pa and any(line["r"] or line["sb"] for line in zero_pa)
        print(f"August 30 snapshot: all 24 standings match 1,571 finals. Both batting lines and the "
              f"1,571 EAB box-score R/H totals agree. Retained {len(zero_pa):,} zero-PA lines.", flush=True)

        root = publisher.dist / "releases" / current["id"]
        game_html = (root / "games" / f"{game['date']}.html").read_text(encoding="utf-8")
        player_html = (root / "players/index.html").read_text(encoding="utf-8")
        for name in ("Brent Keyser", "Chris Sullivan"):
            assert name in game_html and name in player_html
        assert "Differences from OOTP season totals" in player_html
        differences = {(item["team_id"], item["player_id"]): [(key, tally["ootp"], tally["reconciled"])
                       for key, tally in item["fields"].items()] for item in report["season_discrepancies"]}
        assert differences == {
            ("19", "1547"): [("g", 93, 92)],
            ("23", "19"): [("g", 83, 82)],
            ("19", None): [("gs", 1178, 1179), ("pa", 5044, 5046), ("ab", 4350, 4352)],
            ("23", None): [("gs", 1178, 1179), ("pa", 5104, 5106), ("ab", 4475, 4477), ("k", 792, 793)],
        }
        assert report["player_pitch_count_coverage"] == {"native_nonzero": 525, "derived_nonzero": 0, "differing": 525}
        audit_manifest(publisher.dist, current["id"])
        assert catalog_for(publisher.dist)["current"] == current["id"]
        assert publisher.publish(final, snapshots=snapshots) == (current, "unchanged")
        assert file_hash(first_seal) == old_seal and file_hash(old_home) == old_home_hash
        audit_manifest(publisher.dist, before["id"])
        assert world.history == first_world.history and world.historical_players == first_world.historical_players
        assert max(world.history) == 1980
        for folder, hashes in checksum.items():
            assert inventory(folder, set(hashes)) == hashes, folder
        print("Through 1980: historical standings, results and player totals are identical. "
              "Both archives and all 72 original hashes passed. Unchanged rebuild made no revision.", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--replay"]:
        replay()
    else:
        unittest.main()
