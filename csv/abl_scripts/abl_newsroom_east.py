"""Read-only reporting on the 1981 NBC East. Each investigation has an evidence gate."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import timedelta
from itertools import groupby
from pathlib import Path

from abl_newsroom_beats import day, full_name, game_records, innings, percentage, record, rows, wl


MIAMI, ATLANTA, CHARLOTTE, TAMPA = 1, 3, 6, 8
EAST = {MIAMI, ATLANTA, CHARLOTTE, TAMPA}


def written_date(value):
    d = day(value)
    return f"{d:%B} {d.day}"


def gap(leader, chaser):
    return ((leader["wins"] - chaser["wins"]) + (chaser["losses"] - leader["losses"])) / 2


def counted(n, singular, plural=None):
    small = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
    quantity = small[n] if 0 <= n < len(small) else str(n)
    return f"{quantity} {singular if n == 1 else plural or singular + 's'}"


def piece(title, sections, evidence, trigger):
    return {"title": title, "sections": sections, "evidence": evidence, "trigger": trigger}


class EastDesk:
    def __init__(self, raw, cutoff, beats):
        self.raw = Path(raw)
        self.cutoff = day(cutoff)
        self.beats = beats
        self.clubs = {int(r["team_id"]): r for r in rows(self.raw, "teams")
                      if r["league_id"] == "200" and r["level"] == "1" and r["allstar_team"] == "0"}
        self.games = game_records(self.raw, self.cutoff, self.clubs)
        self.played = defaultdict(list)
        for g in sorted(self.games.values(), key=lambda x: (x["date"], x["game_id"])):
            for club in (g["home_id"], g["away_id"]):
                self.played[club].append(g)
        self.lines = {t: record(self.played[t], t) for t in EAST}
        self.order = sorted(EAST, key=lambda t: (-percentage(self.lines[t]), t))
        self.calendar = defaultdict(list)
        for r in rows(self.raw, "games"):
            if r["league_id"] != "200" or r["game_type"] != "0":
                continue
            when = day(r["date"])
            if when.year != self.cutoff.year:
                continue
            gid, home, road = int(r["game_id"]), int(r["home_team"]), int(r["away_team"])
            for t, opp in ((home, road), (road, home)):
                if t in EAST and opp in self.clubs:
                    self.calendar[t].append((when, int(r["time"]), gid, opp, home))
        for games in self.calendar.values():
            games.sort()
        self.players = {int(r["player_id"]): r for r in rows(self.raw, "players")}
        self.stats = defaultdict(list)
        for kind, filename, fields in (
            ("bat", "players_game_batting", ("g", "gs", "ab", "h", "hr", "rbi")),
            ("pit", "players_game_pitching_stats", ("g", "gs", "outs", "er", "k", "bb")),
        ):
            for r in rows(self.raw, filename):
                if r["league_id"] != "200" or r["level_id"] != "1" or int(r["year"]) != self.cutoff.year:
                    continue
                gid, team = int(r["game_id"]), int(r["team_id"])
                g = self.games.get(gid)
                if not g:
                    continue
                if team not in (g["home_id"], g["away_id"]):
                    raise ValueError(f"Unrelated team {team} in {filename}.csv game {gid}")
                self.stats[kind, team, int(r["player_id"])].append(
                    (gid, day(g["date"]), Counter({k: int(r[k]) for k in fields})))
        self.history = {(int(r["team_id"]), int(r["year"])): r for r in rows(self.raw, "team_history")
                        if r["league_id"] == "200" and int(r["year"]) < self.cutoff.year}
        self.old = {(int(r["team_id"]), int(r["year"])): r for r in rows(self.raw, "team_history_record")
                    if r["league_id"] == "200" and int(r["year"]) < self.cutoff.year}

    def name(self, pid):
        p = self.players[pid]
        return f"{p['first_name']} {p['last_name']}"

    def summed(self, kind, team, player, *, start=None, before=None, ids=None):
        total = Counter()
        for gid, when, line in self.stats[kind, team, player]:
            if (start is None or when >= start) and (before is None or when <= before) and (ids is None or gid in ids):
                total.update(line)
        return total

    def leading_arm(self, club, game_ids):
        arms = [(pid, self.summed("pit", t, pid, ids=game_ids)) for kind, t, pid in self.stats
                if kind == "pit" and t == club]
        return max(arms, key=lambda x: (x[1]["outs"], -x[0]), default=(None, Counter()))

    def against(self, club, rivals):
        return [g for g in self.played[club] if (g["home_id"] if g["away_id"] == club else g["away_id"]) in rivals]

    def next_series(self, club):
        for _, group in groupby(self.calendar[club], key=lambda x: (x[3], x[4])):
            series = list(group)
            remaining = [g for g in series if g[2] not in self.games and g[0] >= self.cutoff]
            if remaining:
                return series[0][2], remaining
        return None, []

    def trades(self, team):
        return {x["id"]: x for x in self.beats[team]["transactions"] if day(x["date"]) <= self.cutoff}

    def game_source(self, text):
        return f"games.csv, league_id=200, game_type=0, played=1, season {self.cutoff.year} through {self.cutoff}. {text}"

    def charlotte(self):
        club = CHARLOTTE
        year = self.cutoff.year - 1
        history = [self.history.get((club, y)) for y in range(1972, year + 1)]
        if (not history or not all(history) or [int(r["year"]) for r in history if r["made_playoffs"] == "1"] != [year]
                or (club, year) not in self.old or (club, year - 1) not in self.old
                or self.order[:2] != [CHARLOTTE, MIAMI] or self.beats[CHARLOTTE]["division_leaders"] != [CHARLOTTE]):
            return None
        old, earlier = self.old[club, year], self.old[club, year - 1]
        if old["pos"] != "1":
            return None
        own = self.lines[club]
        inside = record(self.against(club, EAST - {club}), club)
        outside = record(self.against(club, self.clubs.keys() - EAST), club)
        per = {t: record(self.against(club, {t}), club) for t in EAST - {club}}
        if min(r["wins"] - r["losses"] for r in per.values()) <= 0 or inside["runs"] <= inside["allowed"]:
            return None
        month = self.cutoff.replace(day=1)
        monthly = record([g for g in self.played[club] if day(g["date"]) >= month], club)
        paragraphs = [
            f"Charlotte spent its first eight ABL seasons outside the tournament. In {year - 1} the Colonels went "
            f"{earlier['w']}-{earlier['l']}. A year later they went {old['w']}-{old['l']} and won the East. "
            f"Now they are {wl(own)}, {gap(own, self.lines[MIAMI]):g} games ahead of Miami. The best evidence that Charlotte "
            "can defend its title comes from the clubs that would have to take it away.",
            f"The Colonels have gone {wl(inside)} against the three teams trying to catch them and {wl(outside)} against "
            f"the rest of the league. They have outscored their neighbors {inside['runs']}-{inside['allowed']}. "
            f"Charlotte is {wl(per[MIAMI])} against Miami, "
            f"{wl(per[ATLANTA])} against Atlanta and {wl(per[TAMPA])} against Tampa Bay.",
        ]
        division_ids = {g["game_id"] for g in self.against(club, EAST - {club})}
        p, arm = self.leading_arm(club, division_ids)
        if not arm["outs"] or not arm["gs"]:
            return None
        paragraphs.append(f"{self.name(p)} has worked {innings(arm['outs'])} innings against East opponents. "
                          f"In {counted(arm['gs'], 'start')} he has allowed {counted(arm['er'], 'earned run')} and struck out {arm['k']}. "
                          f"That comes to a {arm['er'] * 27 / arm['outs']:.2f} ERA against the clubs closest to home.")
        _, fixtures = self.next_series(club)
        close = f"The Colonels are {wl(monthly)} in {month:%B}, with {monthly['runs']} runs scored and {monthly['allowed']} allowed."
        if fixtures and fixtures[0][3] == TAMPA:
            n = len(fixtures)
            close += (f" The schedule gives them {counted(n, 'game')} {'at Tampa Bay' if fixtures[0][4] == TAMPA else 'at home against Tampa Bay'}, "
                      f"starting {written_date(fixtures[0][0])}. Of Charlotte's three division opponents, "
                      f"the Storm have already allowed the Colonels {per[TAMPA]['runs']} runs in "
                      f"{len(self.against(club, {TAMPA}))} meetings.")
        return piece("How Charlotte built a lead it can defend", [("A second pennant race", paragraphs),
                     ("Back inside the division", [close])], [
                         self.game_source("team_id=6, grouped by opponents 1, 3, 8 and all other clubs; current calendar month."),
                         f"players.csv and players_game_pitching_stats.csv, team_id=6, player_id={p}, opponents 1, 3, 8.",
                         f"team_history.csv and team_history_record.csv, league_id=200, team_id=6, years 1972–{year}.",
                         "games.csv, scheduled Charlotte series game IDs " + ", ".join(str(g[2]) for g in fixtures) + "."],
                     f"first-berth-{year}:sole-leader-6:winning-against-all-three")

    def miami(self):
        ids = ["trade:19852", "trade:19868", "trade:19877"]
        trades = self.trades(MIAMI)
        if not all(i in trades for i in ids):
            return None
        deals = [trades[i] for i in ids]
        if [t["incoming_ids"] for t in deals] != [[671], [8655], [1849]]:
            raise ValueError("Miami transaction evidence and incoming identities disagree")
        first = day(deals[0]["date"])
        if self.cutoff < day(deals[-1]["date"]) + timedelta(days=6):
            return None
        own = self.lines[MIAMI]
        if own["wins"] <= own["losses"] or own["runs"] >= own["allowed"]:
            return None
        gs = self.played[MIAMI]
        close = record([g for g in gs if abs(g["runs_home"] - g["runs_away"]) == 1], MIAMI)
        wide = record([g for g in gs if abs(g["runs_home"] - g["runs_away"]) >= 5], MIAMI)
        since = record([g for g in gs if day(g["date"]) >= first], MIAMI)
        before = record([g for g in gs if day(g["date"]) < first], MIAMI)
        bats = []
        for deal in deals[1:]:
            p = deal["incoming_ids"][0]
            when, former = day(deal["date"]), deal["partner_id"]
            prior = self.summed("bat", former, p, before=when)
            now = self.summed("bat", MIAMI, p, start=when)
            if not prior["gs"] or not now["gs"]:
                return None
            bats.append(f"{self.name(p)} had {counted(prior['gs'], 'start')} and went {prior['h']}-for-{prior['ab']} with "
                        f"{self.clubs[former]['name']}. Since the {written_date(when)} trade he has made "
                        f"{counted(now['gs'], 'start')} for Miami, going {now['h']}-for-{now['ab']} with {counted(now['hr'], 'home run')}.")
        p = deals[0]["incoming_ids"][0]
        arm = self.summed("pit", MIAMI, p, start=first)
        if not arm["outs"]:
            return None
        pen = (f"{self.name(p)}, the pitcher acquired from {self.clubs[deals[0]['partner_id']]['name']} on {written_date(first)}, "
               f"has allowed {counted(arm['er'], 'earned run')} in {innings(arm['outs'])} innings for Miami.")
        return piece("What Miami bought in three July trades", [
            ("The cost of a bad loss", [
                f"The Hurricanes have won {own['wins']} games and lost {own['losses']}, even though opponents have outscored them "
                f"{own['allowed']}-{own['runs']}. The margins help explain how. Miami is {wl(close)} in one-run games. "
                f"In decisions by at least five runs it is {wl(wide)}, with {wide['runs']} runs scored and {wide['allowed']} allowed. "
                "A lopsided loss does lasting damage to the run differential. It still costs one game in the standings."]),
            ("Who has actually played", [bats[0] + " " + bats[1], pen + " " +
                f"The Hurricanes were {wl(before)} before the first of those trades. From {written_date(first)} through "
                f"{written_date(self.cutoff)} they went {wl(since)}, scoring {since['runs']} and allowing {since['allowed']}. "
                "The two hitters have played enough to change the lineup. How much credit to give the newcomers for the club's record remains open."])],
            [self.game_source(f"team_id=1, margins exactly 1 or at least 5; records before and from {first}."),
             *(d["source"] for d in deals),
             "players.csv, players_game_batting.csv and players_game_pitching_stats.csv, league_id=200, level_id=1. "
             f"Players 671, 8655, 1849. Compare sending-club games through each trade date and Miami games beginning that date through {self.cutoff}."],
            "july-deals-19852-19868-19877:winning-with-negative-differential")

    def atlanta(self):
        club = ATLANTA
        anchor, fixtures = self.next_series(club)
        if (not fixtures or fixtures[0][3:] != (MIAMI, ATLANTA) or fixtures[0][0] > self.cutoff + timedelta(days=4)
                or self.order.index(MIAMI) >= self.order.index(ATLANTA)
                or self.beats[CHARLOTTE]["division_leaders"] != [CHARLOTTE]):
            return None
        versus = self.against(club, {MIAMI})
        meetings = record(versus, club)
        own = self.lines[club]
        if meetings["wins"] <= meetings["losses"] or own["wins"] >= own["losses"]:
            return None
        date_split = self.cutoff.replace(month=7, day=1)
        start = record([g for g in self.played[club] if day(g["date"]) < date_split], club)
        recent = record([g for g in self.played[club] if day(g["date"]) >= date_split], club)
        if recent["wins"] <= recent["losses"] or start["losses"] <= start["wins"]:
            return None
        matchup_ids = {g["game_id"] for g in versus}
        pid, vs = self.leading_arm(club, matchup_ids)
        if vs["gs"] < 2:
            return None
        arm = self.summed("pit", club, pid)
        hitters = [(p, self.summed("bat", t, p)) for kind, t, p in list(self.stats) if kind == "bat" and t == club]
        if not hitters:
            return None
        bat, line = max(hitters, key=lambda x: (x[1]["hr"], x[1]["h"], -x[0]))
        who, slugger = self.name(pid), self.name(bat)
        lead = (f"Miami comes to Atlanta on {written_date(fixtures[0][0])} with {counted(len(fixtures), 'game')} "
                f"on the schedule here." if fixtures[0][2] == anchor else
                f"Miami and Atlanta have {counted(len(fixtures), 'game')} left in their series in Atlanta.")
        last = versus[-1]
        road_side = last["away_id"] == club
        last_own = last["runs_away"] if road_side else last["runs_home"]
        last_other = last["runs_home"] if road_side else last["runs_away"]
        last_sentence = (f"Their last meeting went {last['innings']} innings on {written_date(last['date'])}. "
                         f"{'Atlanta' if last_own > last_other else 'Miami'} won {max(last_own,last_other)}-{min(last_own,last_other)}."
                         if last_own != last_other else f"Their last meeting ended in a tie on {written_date(last['date'])}.")
        return piece("Atlanta has already shown Miami what it can do", [
            ("The opponent Miami has to get past", [f"{lead} The Kings are {wl(own)}, but they have beaten the Hurricanes "
                f"{counted(meetings['wins'], 'time')} in {len(versus)} meetings and outscored them {meetings['runs']}-{meetings['allowed']}. "
                f"{last_sentence}",
                f"{who} has given the Kings {innings(vs['outs'])} innings against the Hurricanes, allowing "
                f"{counted(vs['er'], 'earned run')} and striking out {vs['k']}. In {counted(arm['gs'], 'start')} this season "
                f"he has pitched {innings(arm['outs'])} innings. If Miami sees him again, it will face the pitcher who has "
                "worked the most innings against it for Atlanta this year."]),
            ("What the Kings have at stake", [f"Atlanta went {wl(start)} through June. Since July 1 it has gone {wl(recent)}. "
                f"{slugger} has contributed {counted(line['hr'], 'home run')} and {line['rbi']} RBI this season. "
                f"Over the stretch since July began, the whole club has scored {recent['runs']} runs and allowed {recent['allowed']}. "
                "The better record gives the Kings something of their own to defend while Miami plays for position."])],
            [self.game_source("team_id=3, record against team_id=1, and season split at July 1."),
             "games.csv, scheduled Atlanta home series IDs " + ", ".join(str(g[2]) for g in fixtures) + ".",
             f"players_game_pitching_stats.csv, player_id={pid}, team_id=3. Miami game IDs "
             + ", ".join(str(g) for g, _, _ in self.stats['pit', club, pid] if g in matchup_ids) + ".",
             f"players_game_batting.csv and players.csv, team_id=3, player_id={bat}, filtered to validated season games."],
            f"atlanta-hosts-miami:{anchor}:winning-matchup")

    def tampa(self):
        club = TAMPA
        deal = self.trades(club).get("trade:19798")
        if not deal:
            return None
        if deal["incoming_ids"] != [6782] or deal["outgoing_ids"] != [1656]:
            raise ValueError("Tampa Bay's Dallas trade has contradictory player identities")
        since, partner = day(deal["date"]), deal["partner_id"]
        p, former = deal["incoming_ids"][0], deal["outgoing_ids"][0]
        # A poor start is the reporting occasion. The months-long comparison supplies context.
        candidates = []
        for gid, when, line in self.stats["pit", club, p]:
            if when < max(since, self.cutoff - timedelta(days=4)) or line["gs"] != 1 or line["bb"] < 5:
                continue
            game = self.games[gid]
            own, opp = ((game["runs_home"], game["runs_away"]) if game["home_id"] == club else
                        (game["runs_away"], game["runs_home"]))
            if opp - own >= 8:
                candidates.append((opp - own, gid, own, opp, line))
        if not candidates:
            return None
        _, gid, scored, conceded, outing = max(candidates, key=lambda x: (x[0], x[1]))
        game = self.games[gid]
        opponent = game["away_id"] if game["home_id"] == club else game["home_id"]
        hitters = [(pid, self.summed("bat", t, pid)) for k, t, pid in list(self.stats) if k == "bat" and t == club]
        bat, year = max(hitters, key=lambda x: (x[1]["h"], x[1]["rbi"], -x[0]), default=(None, Counter()))
        if not year["ab"]:
            return None
        appearance = self.summed("bat", club, bat, ids={gid})
        if not appearance["ab"]:
            return None
        before = self.summed("pit", partner, p, before=since)
        after = self.summed("pit", club, p, start=since)
        outgoing = self.summed("pit", partner, former, start=since)
        if not all(x["outs"] for x in (before, after, outgoing)):
            return None
        recent = self.played[club][-7:]
        losing = record(recent, club)
        hitname, name, former_name = self.name(bat), self.name(p), self.name(former)
        avg = f"{year['h'] / year['ab']:.3f}".lstrip("0")
        open_text = (f"{hitname} went {appearance['h']}-for-{appearance['ab']} in Tampa Bay's {conceded}-{scored} loss "
                     f"to {self.clubs[opponent]['name']} on {written_date(game['date'])}. Through {written_date(self.cutoff)} "
                     f"he is batting {avg} with {counted(year['hr'], 'home run')} and {year['rbi']} RBI. "
                     f"His {counted(year['h'], 'hit')} lead the Storm. The harder numbers to turn around belong to the whole club. "
                     f"Tampa Bay went {wl(losing)} over its last {len(recent)}, scoring {losing['runs']} runs and allowing {losing['allowed']}.")
        game_text = (f"{name} walked {counted(outing['bb'], 'hitter')} in {innings(outing['outs'])} innings against {self.clubs[opponent]['name']} "
                     f"and was charged with {counted(outing['er'], 'earned run')}. The walks have followed him since the "
                     f"{written_date(since)} trade. For {self.clubs[partner]['name']} he walked {before['bb']} in "
                     f"{innings(before['outs'])} innings. For Tampa Bay he has walked {after['bb']} in {innings(after['outs'])}.")
        next_text = (f"The pitcher the Storm sent the other way, {former_name}, has allowed {counted(outgoing['er'], 'earned run')} "
                     f"in {innings(outgoing['outs'])} innings over {counted(outgoing['gs'], 'start')} for {self.clubs[partner]['name']}. "
                     f"For Tampa Bay the immediate question is "
                     f"whether {name.split()[-1]} can get through a lineup without handing it so many baserunners.")
        _, fixtures = self.next_series(club)
        if fixtures and fixtures[0][3:] == (CHARLOTTE, TAMPA):
            against = record(self.against(club, {CHARLOTTE}), club)
            next_text += (f" Charlotte comes to town on {written_date(fixtures[0][0])}. The Colonels have already scored "
                          f"{against['allowed']} times in {len(self.against(club, {CHARLOTTE}))} games against the Storm.")
        return piece(f"{hitname} keeps hitting while Tampa Bay pays for the walks", [
            ("What one hitter can do", [open_text]), ("Two pitchers after a trade", [game_text, next_text])], [
             self.game_source(f"team_id=8, game_id={gid}; last seven IDs " + ", ".join(str(g["game_id"]) for g in recent) + "."),
             deal["source"], f"players.csv and players_game_batting.csv, player_id={bat}, team_id=8, season and game_id={gid}.",
             f"players_game_pitching_stats.csv, player_ids={p},{former}, team_ids={club},{partner}, through {self.cutoff}; "
             f"split at trade date {since}, starter line game_id={gid}.",
             "games.csv, scheduled Tampa Bay series game IDs " + ", ".join(str(g[2]) for g in fixtures) + "."],
            f"tampa-starter-command:trade-19798:game-{gid}")

    def division(self):
        if self.order[:2] != [CHARLOTTE, MIAMI] or self.beats[CHARLOTTE]["division_leaders"] != [CHARLOTTE]:
            return None
        first, second = self.lines[CHARLOTTE], self.lines[MIAMI]
        c_anchor, c_schedule = self.next_series(CHARLOTTE)
        m_anchor, m_schedule = self.next_series(MIAMI)
        if (not c_schedule or not m_schedule or c_schedule[0][3:] != (TAMPA, TAMPA)
                or m_schedule[0][3:] != (ATLANTA, ATLANTA) or c_schedule[0][0] != m_schedule[0][0]):
            return None
        direct = [g for g in self.calendar[MIAMI] if g[3] == CHARLOTTE and g[2] not in self.games and g[0] >= self.cutoff]
        if not direct or direct[0][0] < c_schedule[-1][0]:
            return None
        split = day(self.trades(MIAMI)["trade:19852"]["date"]) if "trade:19852" in self.trades(MIAMI) else None
        if not split:
            return None
        earlier = {t: record([g for g in self.played[t] if day(g["date"]) < split], t) for t in (CHARLOTTE, MIAMI)}
        old_gap = gap(earlier[CHARLOTTE], earlier[MIAMI])
        now_gap = gap(first, second)
        if now_gap <= 0 or old_gap <= now_gap:
            return None
        east = {t: record(self.against(t, EAST - {t}), t) for t in (CHARLOTTE, MIAMI)}
        outside = {t: record(self.against(t, self.clubs.keys() - EAST), t) for t in (CHARLOTTE, MIAMI)}
        if percentage(east[CHARLOTTE]) <= percentage(east[MIAMI]) or percentage(outside[MIAMI]) <= percentage(outside[CHARLOTTE]):
            return None
        storm = record(self.against(TAMPA, {CHARLOTTE}), TAMPA)
        kings = record(self.against(ATLANTA, {MIAMI}), ATLANTA)
        current = f"Charlotte's {wl(first)} record leaves Miami ({wl(second)}) {now_gap:g} games back."
        lead = (f"Charlotte and Miami will next meet on {written_date(direct[0][0])} in {self.clubs[direct[0][4]]['name']}. "
                f"There is already a race to determine what that series will be worth. {current} "
                f"The margin stood at {old_gap:g} games before play on {written_date(split)}.")
        force = (f"Miami has done better outside the East, going {wl(outside[MIAMI])} to Charlotte's {wl(outside[CHARLOTTE])}. "
                 f"Inside it the order reverses. The Colonels are {wl(east[CHARLOTTE])} and the Hurricanes {wl(east[MIAMI])}. "
                 "Returning to division play puts Miami back in the part of its schedule where it has lost ground.")
        dates = (f"On {written_date(c_schedule[0][0])}, Charlotte is scheduled at Tampa Bay and Miami at Atlanta. "
                 f"The Kings are {wl(kings)} against the Hurricanes. The Storm are {wl(storm)} against the Colonels. "
                 "Miami could win in Atlanta and still gain nothing that day if Charlotte wins too.")
        number = len(direct)
        close = (f"The schedule still has {counted(number, 'meeting')} between the two leaders. "
                 "In each of those games a Miami win cuts a full game from the gap and a Charlotte win adds one. "
                 "Until then, Atlanta and Tampa Bay have as much say in how close this race gets as either contender.")
        refs = [self.game_source(f"team_ids=1,3,6,8. Standings before {split} and at cutoff; division and outside-division splits."),
                "games.csv, scheduled Charlotte and Miami series IDs "
                + ", ".join(str(g[2]) for g in c_schedule + m_schedule) + ".",
                "games.csv, scheduled remaining Charlotte–Miami game IDs " + ", ".join(str(g[2]) for g in direct) + "."]
        return piece("Before Charlotte and Miami meet, the rest of the East gets a say", [
            ("The race before the race", [lead, force]), ("What Atlanta and Tampa Bay can change", [dates, close])],
            refs, f"return-to-east:{c_anchor}:{m_anchor}:charlotte-ahead")


def build_east(raw_dir, cutoff, new_game_ids, beats, snapshot_date=None):
    """Develop supported investigations. Empty slots deliberately yield to the wider editorial desk."""
    if day(cutoff).year != 1981 or not EAST <= beats.keys():
        return {"teams": {}, "division": None}
    desk = EastDesk(raw_dir, cutoff, beats)
    if set(map(int, new_game_ids)) - desk.games.keys():
        raise ValueError("East articles received unverified completed game IDs")
    for team in EAST:
        line = desk.lines[team]
        if ((beats[team]["wins"], beats[team]["losses"]) != (line["wins"], line["losses"])
                or beats[team]["team"] != full_name(desk.clubs[team])):
            raise ValueError(f"NBC East articles disagree with the verified franchise notebook for team {team}")
    if day(snapshot_date or cutoff) < desk.cutoff:
        raise ValueError("NBC East source snapshot predates its game cutoff")
    features = {CHARLOTTE: desk.charlotte(), MIAMI: desk.miami(), ATLANTA: desk.atlanta(), TAMPA: desk.tampa()}
    return {"teams": {t: v for t, v in features.items() if v}, "division": desk.division()}
