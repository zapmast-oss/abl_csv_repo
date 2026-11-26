"""Franchise reporting for one verified ABL batch. All functions are read-only."""
from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter, defaultdict
from datetime import date
from fractions import Fraction
from pathlib import Path

from abl_player_games import player_game_rows
from z_abl_current_state_preflight_validate import select_completed_games


POSITIONS = {1: "P", 2: "C", 3: "1B", 4: "2B", 5: "3B", 6: "SS",
             7: "LF", 8: "CF", 9: "RF", 10: "DH"}
BATTING = ("h", "ab", "pa", "hr", "rbi", "bb", "k")
PITCHING = ("outs", "er", "ha", "gs", "k", "bb", "cg", "s")


def rows(directory, name):
    with (directory / f"{name}.csv").open(encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f)


def day(value):
    if isinstance(value, date):
        return value
    return date(*map(int, value.split("-")))


def full_name(row):
    return f"{row['name'].strip()} {row['nickname'].strip()}"


def person(players, player_id):
    row = players.get(player_id)
    if row is None:
        return f"player #{player_id}"
    return f"{row['first_name']} {row['last_name']} (#{player_id})"


def innings(outs):
    whole, remainder = divmod(outs, 3)
    if not remainder:
        return str(whole)
    return f"{whole} {remainder}/3" if whole else f"{remainder}/3"


def record(games, team_id):
    wins = losses = ties = runs = allowed = 0
    for g in games:
        home = g["home_id"] == team_id
        scored, conceded = (g["runs_home"], g["runs_away"]) if home else (g["runs_away"], g["runs_home"])
        wins += scored > conceded
        losses += scored < conceded
        ties += scored == conceded
        runs += scored
        allowed += conceded
    return {"wins": wins, "losses": losses, "ties": ties, "runs": runs, "allowed": allowed}


def wl(line):
    text = f"{line['wins']}-{line['losses']}"
    return text + (f"-{line['ties']}" if line["ties"] else "")


def percentage(line):
    return Fraction(line["wins"], max(1, line["wins"] + line["losses"]))


def games_margin(value):
    return f"{value:g} {'game' if value == 1 else 'games'}"


def race(team_id, members, records, teams):
    ranked = sorted(members, key=lambda t: (-percentage(records[t]), t))
    leader = records[ranked[0]]
    leaders = [t for t in ranked if percentage(records[t]) == percentage(leader)]
    own = records[team_id]
    def gap(a, b):
        return ((a["wins"] - b["wins"]) + (b["losses"] - a["losses"])) / 2
    gb = gap(leader, own)
    if team_id in leaders:
        if len(leaders) > 1:
            rest = ", ".join(teams[t]["name"] for t in leaders if t != team_id)
            description = f"shares first place by winning percentage with {rest}"
        else:
            opponent = ranked[1]
            description = f"leads {teams[opponent]['name']} by {games_margin(gap(own, records[opponent]))}"
    else:
        description = f"trails {' and '.join(teams[t]['name'] for t in leaders)} by {games_margin(gb)}"
    return {"gb": gb, "leaders": sorted(leaders), "description": description}


def game_records(raw_dir, cutoff, teams):
    season = select_completed_games(list(rows(raw_dir, "games")), cutoff.year, cutoff)
    games = {}
    for row in season:
        home, away = int(row["home_team"]), int(row["away_team"])
        if home not in teams or away not in teams:
            continue
        gid = int(row["game_id"])
        if gid in games:
            raise ValueError(f"Repeated ABL regular-season game ID {gid}")
        runs_away, runs_home = int(row["runs0"]), int(row["runs1"])
        when = day(row["date"]).isoformat()
        games[gid] = {
            "game_id": gid, "date": when, "home_id": home, "away_id": away,
            "runs_away": runs_away, "runs_home": runs_home, "innings": int(row["innings"]),
            "summary": f"{teams[away]['name']} {runs_away}, {teams[home]['name']} {runs_home}",
            "source": f"games.csv game_id={gid} date={when} league_id=200 game_type=0 played=1",
        }
    return games


def player_lines(raw_dir, games, year, new_ids, reconciliation=None):
    totals, recent = {}, {}
    for kind, source, fields in [("bat", "players_game_batting", BATTING),
                                 ("pit", "players_game_pitching_stats", PITCHING)]:
        totals[kind] = defaultdict(lambda: defaultdict(Counter))
        recent[kind] = defaultdict(dict)
        for row in player_game_rows(raw_dir, source + ".csv", year, games, reconciliation):
            gid, team, pid = map(int, (row["game_id"], row["team_id"], row["player_id"]))
            game = games[gid]
            if team not in (game["home_id"], game["away_id"]):
                raise ValueError(f"{source}.csv assigns player {pid} to unrelated team {team} in game {gid}")
            stats = {key: int(row[key]) for key in fields}
            totals[kind][team][pid].update(stats)
            if gid in new_ids:
                recent[kind][team, gid][pid] = stats
    return totals, recent


def trade_records(raw_dir, teams, players, cutoff):
    trades = defaultdict(list)
    for row in rows(raw_dir, "trade_history"):
        when = day(row["date"])
        if when.year != cutoff.year or when > cutoff:
            continue
        mid = int(row["message_id"])
        sides = [int(row[f"team_id_{side}"]) for side in range(2)]
        assets = [[int(row[f"player_id_{side}_{i}"]) for i in range(10)
                   if int(row[f"player_id_{side}_{i}"]) > 0] for side in range(2)]
        fallback = hashlib.sha256(repr((when.isoformat(), sides, assets, row["summary"])).encode()).hexdigest()[:20]
        key = f"trade:{mid}" if mid > 0 else f"trade:{fallback}"
        for side, club in enumerate(sides):
            if club not in teams:
                continue
            partner = sides[1 - side]
            other = teams[partner]["name"] if partner in teams else f"team #{partner}"
            incoming, outgoing = assets[1 - side], assets[side]
            facts = []
            if incoming:
                facts.append("received " + ", ".join(person(players, p) for p in incoming))
            if outgoing:
                facts.append("sent " + ", ".join(person(players, p) for p in outgoing))
            source = f"trade_history.csv date={when} team_id_0={sides[0]} team_id_1={sides[1]} message_id={mid}"
            movement = " and ".join(facts)
            detail = f"Trade with {other}." if not movement else f"In a trade with {other}, {teams[club]['name']} {movement}."
            trades[club].append({"id": key, "date": when.isoformat(), "partner_id": partner,
                                 "incoming_ids": incoming, "outgoing_ids": outgoing,
                                 "summary": detail.rstrip("."), "source": source})
    return {club: sorted(items, key=lambda x: (x["date"], x["id"])) for club, items in trades.items()}


def player_card(pid, players):
    row = players[pid]
    return {"player_id": pid, "name": person(players, pid), "age": int(row["age"]),
            "position": POSITIONS.get(int(row["position"]), f"position code {row['position']}"),
            "team_id": int(row["team_id"]),
            "injury_flag": int(row["injury_is_injured"]), "day_to_day_flag": int(row["injury_dtd_injury"])}


def farm_watch(club, affiliates, players, year):
    candidates = []
    for pid, row in players.items():
        tid = int(row["team_id"])
        if tid not in affiliates or int(row["retired"]) or int(row["age"]) > 25:
            continue
        draft_year, pick = int(row["draft_year"]), int(row["draft_overall_pick"])
        if int(row["draft_team_id"]) != club or not (year - 3 <= draft_year <= year and pick > 0):
            continue
        card = player_card(pid, players)
        card.update(draft_year=draft_year, draft_pick=pick, affiliate=full_name(affiliates[tid]))
        candidates.append(card)
    return sorted(candidates, key=lambda p: (p["draft_pick"], -p["draft_year"], p["player_id"]))[:3]


def history_notes(club, history, historical_records):
    seasons = sorted((y for t, y in history if t == club))
    if not seasons:
        return ["Franchise history awaits a prior-season league-200 record."]
    titles = [str(y) for y in seasons if history[club, y]["won_playoffs"] == "1"]
    berths = [str(y) for y in seasons if history[club, y]["made_playoffs"] == "1"]
    last = historical_records.get((club, seasons[-1]))
    facts = [f"Seasons {seasons[0]}–{seasons[-1]} cover {len(seasons)} recorded campaigns. "
             f"Tournament appearances ({len(berths)}) — {', '.join(berths) or 'none'}. "
             f"League titles ({len(titles)}) — {', '.join(titles) or 'none'}. "
             f"[team_history.csv, team_id={club}, league_id=200, year={seasons[0]}–{seasons[-1]}]"]
    if last:
        facts.append(f"The club finished {last['w']}-{last['l']} in {last['year']}, division place {last['pos']}. "
                     f"[team_history_record.csv, team_id={club}, league_id=200, year={last['year']}]")
    return facts


def highlighted_performance(club, game, recent, players):
    gid = game["game_id"]
    hitters = recent["bat"].get((club, gid), {})
    notable = []
    if hitters:
        pid, line = max(hitters.items(), key=lambda item: (
            item[1]["hr"], item[1]["rbi"], item[1]["h"], -item[0]))
        if line["h"] or line["rbi"]:
            notable.append(f"{person(players, pid)} went {line['h']}-for-{line['ab']} with "
                           f"{line['hr']} HR and {line['rbi']} RBI. "
                           f"[players_game_batting.csv, game_id={gid}, team_id={club}, player_id={pid}]")
    pitchers = recent["pit"].get((club, gid), {})
    if pitchers:
        pid, line = max(pitchers.items(), key=lambda item: (item[1]["outs"], item[1]["k"], -item[0]))
        notable.append(f"{person(players, pid)} worked {innings(line['outs'])} {'inning' if line['outs'] == 3 else 'innings'}, "
                       f"allowed {line['er']} earned {'run' if line['er'] == 1 else 'runs'} and struck out {line['k']}. "
                       f"[players_game_pitching_stats.csv, game_id={gid}, team_id={club}, player_id={pid}]")
    return " ".join(notable)


def season_leaders(totals, club, year, cutoff, players):
    hits = totals["bat"][club]
    pitches = totals["pit"][club]
    notes = []
    for dataset, table, stat, label in [(hits, "players_game_batting", "h", "hits"),
                                         (pitches, "players_game_pitching_stats", "outs", "pitching outs")]:
        if not dataset:
            continue
        most = max(line[stat] for line in dataset.values())
        leaders = sorted((p for p in dataset if dataset[p][stat] == most))
        leaders_text = ", ".join(person(players, p) for p in leaders)
        scopes = f"[{table}.csv joined to played games.csv, team_id={club}, year={year}, through {cutoff}]"
        notes.append(f"{leaders_text} share the club lead in {label} ({most}). {scopes}")
        if len(leaders) == 1:
            line = dataset[leaders[0]]
            if stat == "h":
                notes[-1] = f"{leaders_text} leads the club with {most} hits in {line['ab']} at-bats, including {line['hr']} home runs. {scopes}"
            elif line["outs"]:
                notes[-1] = (f"{leaders_text} leads the staff in innings pitched ({innings(most)}), with "
                             f"{line['k']} strikeouts, {line['bb']} walks and a {27 * line['er'] / most:.2f} ERA. {scopes}")
    return notes


def prior_coverage(cutoff, teams):
    """Identify local archives by their recorded date and club, distinguish publication receipts."""
    repo = Path(__file__).resolve().parents[2]
    production = repo / "csv/out/story/production"
    candidates = []
    for path in [*production.glob("*.md"), *repo.glob("*bewell*.md"), *repo.glob("*showcase*.md")]:
        dates = re.findall(r"(\d{4})[-_](\d{2})[-_](\d{2})", path.stem)
        if not dates:
            continue
        published_for = max(date(*map(int, parts)) for parts in dates)
        if published_for <= cutoff:
            candidates.append((published_for, path.relative_to(repo).as_posix(), path.read_text(encoding="utf-8-sig")))
    candidates.sort(reverse=True)
    receipt_dir = repo / "csv/out/story/manifests"
    receipts = []
    for path in sorted(receipt_dir.glob("*publication_checkpoint*.md")):
        body = path.read_text(encoding="utf-8-sig")
        match = re.search(r"\*\*Newsroom date:\*\*\s*(\d{4}-\d{2}-\d{2})", body)
        if match and day(match[1]) <= cutoff:
            channels = re.findall(r"^\|\s*([^|]+?)\s*\|\s*made\s*\|", body, re.M)
            if channels:
                receipts.append(f"The league's {match[1]} publication checkpoint records completed coverage for "
                                f"{', '.join(x.strip() for x in channels)}. [{path.relative_to(repo).as_posix()}]")
    result = {}
    for club, team in teams.items():
        references = []
        for _, path, body in candidates:
            if full_name(team).casefold() in body.casefold():
                references.append(path)
                if len(references) == 2:
                    break
        result[club] = (receipts[-1:] +
                        [f"Earlier internal coverage naming {full_name(team)} is filed at {p}." for p in references])
    return result


def next_opponent(raw_dir, cutoff, teams):
    scheduled = defaultdict(list)
    for g in rows(raw_dir, "games"):
        if g["league_id"] != "200" or g["game_type"] != "0" or g["played"] != "0":
            continue
        when = day(g["date"])
        visitor, home = int(g["away_team"]), int(g["home_team"])
        if when < cutoff or when.year != cutoff.year or home not in teams or visitor not in teams:
            continue
        gid = int(g["game_id"])
        for team, opponent in [(visitor, home), (home, visitor)]:
            scheduled[team].append((when, int(g["time"]), gid, opponent, home))
    return {t: min(games) for t, games in scheduled.items()}


def build_beats(raw_dir: Path, cutoff: str, new_game_ids: set[int], snapshot_date: str | None = None,
                *, reconciliation=None) -> dict[int, dict]:
    """Read a frozen, validated raw directory. Return a JSON-serializable notebook per club."""
    raw_dir, cutoff = Path(raw_dir), day(cutoff)
    new_ids = {int(x) for x in new_game_ids}
    league = next(row for row in rows(raw_dir, "leagues") if row["league_id"] == "200")
    snapshot = day(snapshot_date or league["current_date"])
    allteams = {int(row["team_id"]): row for row in rows(raw_dir, "teams")}
    teams = {club: t for club, t in allteams.items() if t["league_id"] == "200"
             and t["level"] == "1" and t["allstar_team"] == "0"}
    if len(teams) != 24:
        raise ValueError(f"Expected 24 ABL franchises, found {len(teams)}")
    conferences = {row["sub_league_id"]: row["abbr"] for row in rows(raw_dir, "sub_leagues") if row["league_id"] == "200"}
    divisions = {(row["sub_league_id"], row["division_id"]): row["name"] for row in rows(raw_dir, "divisions") if row["league_id"] == "200"}
    games = game_records(raw_dir, cutoff, teams)
    if new_ids - games.keys():
        raise ValueError(f"Unverified regular-season game IDs {sorted(new_ids - games.keys())}")
    views = defaultdict(list)
    for g in sorted(games.values(), key=lambda g: (g["date"], g["game_id"])):
        for t in (g["home_id"], g["away_id"]):
            views[t].append(g)
    records = {t: record(views[t], t) for t in teams}
    previous_records = {t: record([g for g in views[t] if g["game_id"] not in new_ids], t) for t in teams}
    totals, recent = player_lines(raw_dir, games, cutoff.year, new_ids, reconciliation)
    players = {int(row["player_id"]): row for row in rows(raw_dir, "players")}
    active = defaultdict(list)
    for row in rows(raw_dir, "players_roster_status"):
        club, pid = int(row["team_id"]), int(row["player_id"])
        if club in teams and row["is_active"] == "1":
            active[club].append(player_card(pid, players))
    affiliates = defaultdict(dict)
    for row in rows(raw_dir, "team_affiliations"):
        club, affiliate = int(row["team_id"]), int(row["affiliated_team_id"])
        if club in teams:
            affiliates[club][affiliate] = allteams[affiliate]
    # Roster and transaction news may advance on an idle day between game cutoffs.
    trades = trade_records(raw_dir, teams, players, snapshot)
    innings_scores = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for row in rows(raw_dir, "games_score"):
        gid = int(row["game_id"])
        if gid in new_ids:
            innings_scores[gid][int(row["inning"])][int(row["team"])] += int(row["score"])
    past_coverage = prior_coverage(cutoff, teams)
    fixture = next_opponent(raw_dir, cutoff, teams)
    def seasons(table):
        return {(int(x["team_id"]), int(x["year"])): x for x in rows(raw_dir, table)
                if x["league_id"] == "200" and int(x["year"]) < cutoff.year}
    history, history_rec = seasons("team_history"), seasons("team_history_record")
    output = {}
    for club, team in teams.items():
        division_key = team["sub_league_id"], team["division_id"]
        conf = conferences[division_key[0]]
        division = f"{conf} {divisions[division_key]}"
        group = [t for t, other in teams.items() if (other["sub_league_id"], other["division_id"]) == division_key]
        standing = race(club, group, records, teams)
        previous = race(club, group, previous_records, teams)
        members = [g for g in views[club] if g["game_id"] in new_ids]
        name = full_name(team)
        own = records[club]
        opponents = sorted(group, key=lambda t: (-percentage(records[t]), t))
        rivals = ", ".join(f"{teams[t]['name']} {wl(records[t])}" for t in opponents)
        source = f"games.csv, league_id=200, game_type=0, played=1, year={cutoff.year}, through {cutoff}, team_id={club}"
        division_source = (f"teams.csv and games.csv league_id=200, completed regular season through {cutoff}, "
                           f"division team IDs {','.join(map(str, sorted(group)))}")
        race_note = f"{team['name']} is {wl(own)} and {standing['description']} in the {division}. [{division_source}]"
        recent_games = views[club][-7:]
        last = record(recent_games, club)
        ids = ", ".join(str(g["game_id"]) for g in recent_games)
        form = (f"Over its last {len(recent_games)} games the club went {wl(last)}, scoring {last['runs']} runs and "
                f"allowing {last['allowed']}. Season totals are {own['runs']} scored and {own['allowed']} allowed. "
                f"[{source}. Recent game IDs {ids or 'none'}]")
        stories = [{"key": f"{club}:{cutoff.year}:division", "question": f"Where does {team['name']} stand against its division opponents?",
                    "evidence": race_note}, {"key": f"{club}:{cutoff.year}:runs",
                    "question": f"Will the next series change {team['name']}'s recent scoring margin?",
                    "evidence": form}]
        focus = []
        priority = 0
        headline = f"{team['name']} holds at {wl(own)}"
        headline_game = None
        for g in members:
            diff = abs(g["runs_home"] - g["runs_away"])
            team_win = (g["runs_home"] > g["runs_away"]) == (g["home_id"] == club) if diff else False
            opp = g["away_id"] if g["home_id"] == club else g["home_id"]
            label = "beat" if team_win else "fell to"
            if not diff:
                label = "tied"
            rf, ra = (g["runs_home"], g["runs_away"]) if g["home_id"] == club else (g["runs_away"], g["runs_home"])
            score = f"{max(rf,ra)}-{min(rf,ra)}"
            fact = f"{team['name']} {label} {teams[opp]['name']} {score}"
            rank = 1
            if diff <= 1 or g["innings"] > 9:
                rank = 2
            if diff >= 10 or min(rf, ra) == 0:
                rank = 3
            if g["innings"] >= 13 or max(rf, ra) >= 16:
                rank = 4
            if rank >= priority:
                headline = fact
                headline_game = g
                priority = rank
            focus.append(f"{g['date']} — {fact} in {g['innings']} innings. [{g['source']}] "
                         + highlighted_performance(club, g, recent, players))
            if rank >= 3:
                line = innings_scores[g["game_id"]]
                side = int(g["home_id"] == club)
                if [sum(x[i] for x in line.values()) for i in range(2)] != [g["runs_away"], g["runs_home"]]:
                    raise ValueError(f"Inning totals disagree in feature game {g['game_id']}")
                biggest = max((v[side] for v in line.values()), default=0)
                if biggest:
                    frames = ", ".join(str(i) for i,v in sorted(line.items()) if v[side] == biggest)
                    focus.append(f"{team['name']} scored {biggest} of its {rf} {'run' if rf == 1 else 'runs'} in inning "
                                 f"{frames}." if len(frames.split(','))==1 else
                                 f"{team['name']} scored {biggest} {'run' if biggest == 1 else 'runs'} apiece in innings {frames}.")
                    focus[-1] += f" [games_score.csv game_id={g['game_id']} team={side}, inning={frames}]"
                else:
                    focus.append(f"{teams[opp]['name']} kept {team['name']} scoreless across all "
                                 f"{g['innings']} innings. [games_score.csv game_id={g['game_id']} team={side}]")
        moved = (previous["leaders"] != standing["leaders"] and
                 all(previous_records[t]["wins"] + previous_records[t]["losses"] for t in group))
        if members and moved and club in set(previous["leaders"] + standing["leaders"]):
            # The first result of a season cannot signify the loss of an existing lead.
            if all(previous_records[t]["wins"] + previous_records[t]["losses"] for t in group):
                priority = 4
                race_note += (f" The division lead changed from {', '.join(teams[t]['name'] for t in previous['leaders'])}"
                              f" to {', '.join(teams[t]['name'] for t in standing['leaders'])} across the newly included games. [{division_source}]")
        stories[0]["evidence"] = race_note
        rivalry_notes = []
        for opponent in opponents:
            if opponent == club:
                continue
            head = [g for g in views[club] if opponent in (g["home_id"], g["away_id"])]
            if head:
                h2h = record(head, club)
                rivalry_notes.append(f"Against {full_name(teams[opponent])}, {team['name']} is {wl(h2h)} "
                                      f"with {h2h['runs']} runs scored and {h2h['allowed']} allowed. "
                                      f"[games.csv {cutoff.year} completed regular season, team IDs {club}/{opponent}, "
                                      f"latest game #{head[-1]['game_id']}, {head[-1]['date']}]")
        if club in fixture:
            due, _, gid, opponent, home = fixture[club]
            label = full_name(teams[opponent])
            site = "hosts" if club == home else "visits"
            preview = f"{team['name']} {site} {label} in the next listed game on {due}. [games.csv game_id={gid}, played=0]"
            rivalry_notes.append(preview)
            if opponent in group:
                stories.append({"key":f"{club}:{cutoff.year}:rival:{opponent}", "question":f"What will the next {team['name']}–{teams[opponent]['name']} game do to the division race?",
                                "evidence":preview+" "+race_note})
        roster = sorted(active[club], key=lambda x: (x["position"], x["name"], x["player_id"]))
        prospects = farm_watch(club, affiliates[club], players, cutoff.year)
        roster_source = f"players_roster_status.csv team_id={club} is_active=1 joined to players.csv player_id, export league date {snapshot}"
        roster_notes = [f"The {snapshot} export lists {len(roster)} active players. [{roster_source}]"]
        for pos in POSITIONS.values():
            names = [p["name"] for p in roster if p["position"] == pos]
            if names:
                roster_notes.append(f"{pos} — {', '.join(names)}.")
        tagged = [p["name"] for p in roster if p["injury_flag"]]
        if tagged:
            roster_notes.append(f"Active players carrying an injury flag on the {snapshot} snapshot: "
                                f"{', '.join(tagged)}. Game-day availability requires a dated roster update. "
                                f"[players.csv injury_is_injured=1, {roster_source}]")
        farm_notes = []
        for p in prospects:
            farm_notes.append(f"{p['name']}, {p['age']}, {p['position']} at {p['affiliate']} — drafted in {p['draft_year']} "
                              f"with overall pick {p['draft_pick']}. [players.csv player_id={p['player_id']}, team_affiliations.csv "
                              f"team_id={club} affiliated_team_id={p['team_id']}, teams.csv, export {snapshot}]")
        if prospects:
            farm_notes.insert(0,"Development watch follows up to three affiliate players age 25 or younger drafted by the club in the current or previous three draft years, ordered by earliest overall pick.")
        else:
            farm_notes.append(f"No affiliate player met this watch's age and draft filters in the {snapshot} export. [players.csv, team_affiliations.csv team_id={club}]")
        deals = trades.get(club, [])
        trade_notes = [f"{x['date']} — {x['summary']}. [{x['source']}]" for x in deals[-3:]]
        if not deals:
            trade_notes.append(f"No dated trade involving team #{club} appears in this season's trade_history.csv through {snapshot}.")
        else:
            last_deal = deals[-1]
            for pid in last_deal["incoming_ids"][:1]:
                b = totals["bat"][club].get(pid,{})
                p = totals["pit"][club].get(pid,{})
                usage = (f"{person(players,pid)} has {b.get('pa',0)} batting PA and {innings(p.get('outs',0))} pitching innings "
                         f"for {team['name']} in completed major-league games this season. "
                         f"[players_game_batting.csv and players_game_pitching_stats.csv, player_id={pid}, {source}]")
                stories.append({"key":f"{club}:{cutoff.year}:{last_deal['id']}:{pid}",
                                "question":f"What role will {person(players,pid)} earn with {team['name']}?",
                                "evidence": usage + f" [{last_deal['source']}]"})
                trade_notes.append(usage)
        section_data = [
            ("Reporting focus",focus or [f"No newly admitted completed result for {team['name']} in this cycle. [{source}]"]),
            ("Division and recent form",[race_note, f"Division records — {rivals}. [{division_source}]", form]),
            ("Division opponents and next fixture",rivalry_notes),
            ("Season performances",season_leaders(totals,club,cutoff.year,cutoff,players)),
            ("Franchise record",history_notes(club,history,history_rec)),
            ("Active roster",roster_notes),("Organization and draft watch",farm_notes),
            ("Recent front-office record",trade_notes),("Continuing questions",[s["question"]+" "+s["evidence"] for s in stories]),
            ("Prior coverage",past_coverage[club])]
        source_names = ["games.csv","teams.csv","leagues.csv","divisions.csv","sub_leagues.csv","players.csv","players_game_batting.csv",
                        "players_game_pitching_stats.csv","team_history.csv","team_history_record.csv","trade_history.csv",
                        "team_affiliations.csv","players_roster_status.csv"]
        brief = (f"{headline}. [{headline_game['source']}] " if headline_game else "") + race_note
        output[club] = {"team_id":club, "team":name, "abbr":team["abbr"], "division":division, "conference":conf,
                        "wins":own["wins"], "losses":own["losses"], "gb":standing["gb"], "priority":priority,
                        "division_leaders_change":bool(moved), "division_leaders":standing["leaders"],
                        "intensity":("reference","brief","analysis","analysis","feature")[priority],
                        "headline":headline,"brief":brief,"sections":[{"title":title,"paragraphs":parts} for title,parts in section_data],
                        "stories":stories,"games":members,"game_ids":[g["game_id"] for g in members],"transactions":deals,
                        "roster_ids":sorted(p["player_id"] for p in roster),"roster":roster,"prospects":prospects,
                        "sources":source_names,"recent_game_ids":[g["game_id"] for g in recent_games]}
    return output
