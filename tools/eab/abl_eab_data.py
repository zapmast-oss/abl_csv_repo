"""Build the public baseball record from an admitted OOTP export."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from fractions import Fraction
from itertools import product
from pathlib import Path
import re
import struct
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "csv/abl_scripts"), str(ROOT / "scripts")]
from abl_newsroom_verify import LEAGUE_ID, iso_date, rows
from abl_player_games import player_game_rows


CODE = re.compile(r"[A-Z0-9]{2,5}\Z")
IMAGE_NAME = re.compile(r"[a-z0-9][a-z0-9_-]*\.png\Z", re.I)
BAT = ("pa", "ab", "h", "d", "t", "hr", "r", "rbi", "bb", "hp", "sf", "sb")
PITCH = ("outs", "ha", "r", "er", "bb", "k", "hra", "w", "l", "s", "gs")


@dataclass(frozen=True)
class Game:
    id: int
    day: str
    time: int
    away: int
    home: int
    score: tuple[int, int] | None
    innings: int
    attendance: int

    def side(self, team):
        return self.away if team == self.home else self.home


@dataclass
class World:
    season: int
    league_date: str
    cutoff: str
    source_id: str
    teams: dict
    divisions: dict
    games: dict
    days: dict
    by_team: dict
    series: dict
    players: dict
    batting: dict
    pitching: dict
    boxes: dict
    history: dict
    historical_players: dict
    races: dict
    photos: dict
    logos: dict
    player_game_reconciliation: dict | None = None

    @property
    def finals(self):
        return sum(g.score is not None for g in self.games.values())

    @property
    def next_day(self):
        return next((day for day, games in self.days.items()
                     if day >= self.league_date and any(g.score is None for g in games)), None)


def reference(raw, filename, key):
    return {int(row[key]): row for row in rows(raw / filename)}


def read_clubs(raw, verified):
    parks = reference(raw, "parks.csv", "park_id")
    standings = reference(raw, "team_record.csv", "team_id")
    conferences = {row["sub_league_id"]: row for row in rows(raw / "sub_leagues.csv")
                   if row["league_id"] == LEAGUE_ID}
    divisions = {(row["sub_league_id"], row["division_id"]): row["name"].strip()
                 for row in rows(raw / "divisions.csv") if row["league_id"] == LEAGUE_ID}
    teams, groups = {}, defaultdict(list)
    for key, team in verified["teams"].items():
        tid = int(key)
        group = (team["sub_league_id"], team["division_id"])
        conf = conferences[group[0]]
        park = parks[int(team["park_id"])]
        abbr = team["abbr"].strip()
        name = f"{team['name'].strip()} {team['nickname'].strip()}"
        if not CODE.fullmatch(abbr) or abbr in {club["code"] for club in teams.values()}:
            raise ValueError("The export needs 24 unique, URL-safe franchise codes.")
        teams[tid] = {
            "id": tid, "code": abbr, "city": team["name"].strip(), "name": name,
            "writer": name + " beat", "conference": conf["abbr"], "division": divisions[group],
            "park": park["name"].strip(), "capacity": int(park["capacity"] or 0),
            "logo": team["logo_file_name"], "colors": (team["background_color_id"], team["text_color_id"]),
            "w": int(standings[tid]["w"]), "l": int(standings[tid]["l"]),
        }
        groups[conf["abbr"], divisions[group]].append(tid)
    if len(teams) != 24 or len(groups) != 6 or any(len(group) != 4 for group in groups.values()):
        raise ValueError("The ABL export must contain six four-club divisions.")
    return teams, dict(sorted(groups.items(), key=lambda pair: (pair[0][0] != "NBC",
                              ("Eastern Division", "Central Division", "Western Division").index(pair[0][1]))))


def record(team):
    total = team["w"] + team["l"]
    return Fraction(team["w"], total) if total else Fraction()


def games_behind(lead, club):
    return (lead["w"] - club["w"] + club["l"] - lead["l"]) / 2


def read_games(raw, verified, teams):
    admitted = {int(game["game_id"]): game for game in verified["games"]}
    games = {}
    for row in rows(raw / "games.csv"):
        if row["league_id"] != LEAGUE_ID or row["game_type"] != "0":
            continue
        when = iso_date(row["date"])
        if int(when[:4]) != verified["season"]:
            continue
        gid = int(row["game_id"])
        away, home = int(row["away_team"]), int(row["home_team"])
        if gid in games or away == home or away not in teams or home not in teams:
            raise ValueError(f"Invalid ABL schedule row {gid}")
        has_final = gid in admitted
        if has_final != (row["played"] == "1") or (has_final and when > verified["cutoff"]):
            raise ValueError(f"Unadmitted final {gid}")
        score = (int(row["runs0"]), int(row["runs1"])) if has_final else None
        games[gid] = Game(gid, when, int(row["time"] or 0), away, home, score,
                          int(row["innings"] or 0), int(row["attendance"] or 0) if has_final else 0)
    if not admitted.keys() <= games.keys():
        raise ValueError("A verified final is missing from the schedule.")
    expected = Counter()
    lost = Counter()
    days, clubs = defaultdict(list), defaultdict(list)
    for game in sorted(games.values(), key=lambda g: (g.day, g.time, g.id)):
        days[game.day].append(game)
        clubs[game.away].append(game)
        clubs[game.home].append(game)
        if game.score:
            a, h = game.score
            winner, loser = (game.away, game.home) if a > h else (game.home, game.away)
            expected[winner] += 1
            lost[loser] += 1
    if any((expected[tid], lost[tid]) != (club["w"], club["l"]) for tid, club in teams.items()):
        raise ValueError("Official standings do not reconcile with the verified finals.")
    return games, dict(days), dict(clubs)


def game_series(days):
    """Consecutive dates at the same venue. An incomplete extra stays separate."""
    matchups = defaultdict(list)
    for slate in days.values():
        for game in slate:
            if game.score is not None or game.innings == 0:
                matchups[game.away, game.home].append(game)
    series = {}
    for matchup in matchups.values():
        runs = []
        for game in matchup:
            if not runs or date.fromisoformat(game.day) - date.fromisoformat(runs[-1][-1].day) > timedelta(days=1):
                runs.append([])
            runs[-1].append(game)
        for run in runs:
            if len(run) > 1:
                for game in run:
                    series[game.id] = (run[0].day, run[-1].day, len(run))
    return series


def stats(raw, season, games, names, reconciliation=None):
    bat, pitch, boxes = {}, {}, defaultdict(dict)
    appeared = set()
    finals = {gid for gid, game in games.items() if game.score is not None}
    for filename, numbers, target, kind in (
        ("players_game_batting.csv", (*BAT, "g", "k"), bat, "bat"),
        ("players_game_pitching_stats.csv", PITCH, pitch, "pitch"),
    ):
        for row in player_game_rows(raw, filename, season, finals, reconciliation):
            game = games[int(row["game_id"])]
            person, club = int(row["player_id"]), int(row["team_id"])
            if club not in (game.away, game.home):
                raise ValueError(f"Unmatched ABL player-game row {(game.id, person, club)}")
            line = {k: int(row[k] or 0) for k in numbers}
            if kind == "pitch" and int(row["bf"] or 0) == 0:
                continue
            totals = target.setdefault((person, club), Counter())
            totals.update(line)
            if kind == "pitch":
                totals["g"] += 1
            boxes[game.id].setdefault((club, kind), []).append((person, line))
            appeared.add(person)
    if not appeared <= names.keys():
        raise ValueError("A participating player has no stable identity.")
    return bat, pitch, dict(boxes)


def ops(line):
    base = line["ab"] + line["bb"] + line["hp"] + line["sf"]
    if not base or not line["ab"]:
        return None
    return ((line["h"] + line["bb"] + line["hp"]) / base +
            (line["h"] + line["d"] + 2 * line["t"] + 3 * line["hr"]) / line["ab"])


def era(line):
    return line["er"] * 27 / line["outs"] if line["outs"] else None


def races(teams, divisions, season):
    ranked = {d: sorted(ids, key=lambda t: (-record(teams[t]), teams[t]["code"])) for d, ids in divisions.items()}
    result = {}
    for conf in ("NBC", "ABC"):
        groups = [ids for (abbr, _), ids in ranked.items() if abbr == conf]
        options = [[t for t in ids if record(teams[t]) == record(teams[ids[0]])] for ids in groups]
        wc_options = []
        for winners in product(*options):
            order = sorted((t for ids in groups for t in ids if t not in winners),
                           key=lambda t: (-record(teams[t]), teams[t]["code"]))
            wc_options.append((winners, order))
        matchups = []
        if len(wc_options) == 1:
            winners, order = wc_options[0]
            seeds = sorted(winners, key=lambda t: -record(teams[t]))
            if len({record(teams[t]) for t in seeds}) == 3 and record(teams[order[0]]) > record(teams[order[1]]):
                one, two, three = seeds
                four = order[0]
                matchups = ([(one, three), (two, four)] if season >= 1981 and
                            teams[one]["division"] == teams[four]["division"] else [(one, four), (two, three)])
        result[conf] = {"options": wc_options, "dcs": matchups}
    return ranked, result


def team_history(repo, raw, season, teams):
    history = defaultdict(lambda: {"teams": {}, "scores": []})
    facts = {(int(row["year"]), int(row["team_id"])): row for row in rows(raw / "team_history.csv")
             if row["league_id"] == LEAGUE_ID and int(row["year"]) < season and int(row["team_id"]) in teams}
    for row in rows(raw / "team_history_record.csv"):
        year, tid = int(row["year"]), int(row["team_id"])
        if row["league_id"] == LEAGUE_ID and (year, tid) in facts:
            fact = facts[year, tid]
            history[year]["teams"][tid] = {
                "city": fact["name"].strip(), "name": f"{fact['name'].strip()} {fact['nickname'].strip()}",
                "w": int(row["w"]), "l": int(row["l"]), "place": int(row["pos"]),
                "playoffs": fact["made_playoffs"] == "1", "champion": fact["won_playoffs"] == "1",
            }
    for year, data in history.items():
        source = repo / f"csv/out/almanac/{year}/games_{year}_league200.csv"
        if not source.is_file():
            continue
        aliases = defaultdict(set)
        for tid, club in data["teams"].items():
            aliases[club["city"].casefold()].add(tid)
            aliases[club["name"].casefold()].add(tid)
        seen, skipped = set(), 0
        for g in rows(source):
            away, home = aliases[g["away_team_name"].strip().casefold()], aliases[g["home_team_name"].strip().casefold()]
            when = date.fromisoformat(g["game_date"])
            if (g["league_id"] != LEAGUE_ID or when.year != year or len(away) != 1 or len(home) != 1
                    or away == home or g["game_id"] in seen):
                skipped += 1
                continue
            a, h = int(g["away_runs"]), int(g["home_runs"])
            if a < 0 or h < 0:
                raise ValueError("A historical score is negative.")
            seen.add(g["game_id"])
            data["scores"].append((when.isoformat(), g["game_id"], next(iter(away)), next(iter(home)), a, h))
        data["scores"].sort()
        data["skipped"] = skipped
    return dict(sorted(history.items(), reverse=True))


def historical_player_stats(raw, history, names):
    """Use closed seasons and regular-season major-league totals, keyed by OOTP ID."""
    archive = defaultdict(dict)
    for kind, filename, fields in (
        ("batting", "players_career_batting_stats.csv", BAT),
        ("pitching", "players_career_pitching_stats.csv", PITCH),
    ):
        by_year, seen = defaultdict(dict), set()
        for row in rows(raw / filename):
            year = int(row["year"])
            if (year not in history or (row["league_id"], row["level_id"], row["split_id"], row["game_id"])
                    != (LEAGUE_ID, "1", "1", "0")):
                continue
            player, club = int(row["player_id"]), int(row["team_id"])
            if club not in history[year]["teams"] or player not in names:
                continue
            key = year, player, club
            if key in seen:
                raise ValueError(f"Duplicate historical regular-season player line {key}")
            seen.add(key)
            totals, clubs = by_year[year].setdefault(player, (Counter(), set()))
            totals.update({field: int(row[field] or 0) for field in (*fields, "g")})
            clubs.add(club)
        for year, players in by_year.items():
            archive[year][kind] = players
    return dict(archive)


def clean_png(path, limit=3_000_000):
    """Retain image chunks only, omitting editor text and other ancillary metadata."""
    if not path.is_file() or path.stat().st_size > limit:
        return None
    blob = path.read_bytes()
    if not blob.startswith(b"\x89PNG\r\n\x1a\n"):
        return None
    pos, kept = 8, [blob[:8]]
    while pos + 12 <= len(blob):
        size, kind = struct.unpack_from(">I4s", blob, pos)
        end = pos + size + 12
        if end > len(blob):
            return None
        if kind in (b"IHDR", b"PLTE", b"tRNS", b"IDAT", b"IEND"):
            kept.append(blob[pos:end])
        if kind == b"IEND":
            return b"".join(kept)
        pos = end
    return None


def cover_art(asset_dir, teams, season, cutoff):
    logos, photos = {}, {}
    for tid, club in teams.items():
        name = club["logo"]
        if not IMAGE_NAME.fullmatch(name):
            continue
        content = clean_png(asset_dir / "team_logos" / name)
        if content:
            logos[tid] = content
        stem = Path(name).stem
        pattern = re.compile(re.escape(stem) + r"_(\d{4}-\d{2}-\d{2})\.png\Z")
        dated = []
        for path in (asset_dir / "team_pictures").glob(f"{stem}_{season}-*.png"):
            candidate = pattern.fullmatch(path.name)
            if candidate and candidate[1] <= cutoff:
                dated.append((candidate[1], path))
        if dated:
            when, path = max(dated)
            content = clean_png(path)
            if content:
                photos[tid] = (when, content)
    league = clean_png(asset_dir / "league_logos/action_baseball_league.png")
    if league:
        logos[0] = league
    return logos, photos


def load_world(repo, verified, asset_dir):
    raw = Path(verified["raw_dir"])
    teams, divisions = read_clubs(raw, verified)
    games, days, by_team = read_games(raw, verified, teams)
    players = {int(row["player_id"]): f"{row['first_name']} {row['last_name']}".strip()
               for row in rows(raw / "players.csv")}
    reconciliation = verified.get("player_game_reconciliation")
    batting, pitching, boxes = stats(raw, verified["season"], games, players, reconciliation)
    divisions, postseason = races(teams, divisions, verified["season"])
    logos, photos = cover_art(asset_dir, teams, verified["season"], verified["cutoff"])
    history = team_history(repo, raw, verified["season"], teams)
    player_history = historical_player_stats(raw, history, players)
    return World(verified["season"], verified["league_date"], verified["cutoff"], verified["source_id"],
                 teams, divisions, games, days, by_team, game_series(days), players, batting, pitching,
                 boxes, history, player_history, postseason, photos, logos, reconciliation)
