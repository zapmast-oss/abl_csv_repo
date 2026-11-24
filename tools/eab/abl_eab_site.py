"""Static pages and content-addressed assets for the public EAB edition."""

from collections import Counter, defaultdict
from datetime import date
from html import escape, unescape
from pathlib import Path
from posixpath import dirname, join, normpath, relpath
import re
from urllib.parse import urlsplit, urlunsplit

from abl_eab_analytics import BEACON_TAG, beacon
from abl_eab_coverage import coverage_block
from abl_eab_data import era, games_behind, ops, record
from abl_eab_editorial import article_pdf, identity


HERE = Path(__file__).parent
NAME = "Experiencing Action Baseball"
NAV = (("Today", "index.html"), ("Newsroom", "newsroom/index.html"), ("Teams", "teams/index.html"),
       ("Standings", "standings/index.html"), ("Games", "games/index.html"), ("Players", "players/index.html"),
       ("History", "history/index.html"), ("Fan Culture", "fans/index.html"),
       ("Sim Baseball Vision", "sbv/index.html"), ("About", "about/index.html"))
TODAY_LINK = re.compile(r'(<a(?: aria-current="page")? href=")([^"]+)(">Today</a>)')


def pretty(when, *, year=True):
    day = date.fromisoformat(when)
    return f"{day:%B} {day.day}" + (f", {day.year}" if year else "")


def publication_time(entry):
    timestamp = entry["published_at"]
    label = timestamp.replace("T", " ").replace("Z", " UTC")
    return f'<time datetime="{escape(timestamp)}">Published {escape(label)}</time>'


def wl(club):
    return f"{club['w']}–{club['l']}"


def baseball_rate(value, places=3):
    if value is None:
        return "—"
    text = f"{value:.{places}f}"
    return text[1:] if places == 3 and text.startswith("0.") else text


def ip(outs):
    return f"{outs//3}.{outs%3}"


def gb(lead, club):
    value = games_behind(lead, club)
    return "—" if value == 0 else f"{value:g}"


def href(page, target):
    if target.startswith("https://"):
        return target
    path, separator, fragment = target.partition("#")
    return relpath(path, dirname(page) or ".") + (separator + fragment if separator else "")


def a(page, target, label, cls=""):
    styling = f' class="{cls}"' if cls else ""
    return f'<a{styling} href="{escape(href(page, target), quote=True)}">{escape(str(label))}</a>'


def img(page, target, alt="", cls="", lazy=True):
    return (f'<img class="{cls}" src="{escape(href(page, target), quote=True)}" alt="{escape(alt, quote=True)}"'
            + (' loading="lazy"' if lazy else "") + ">")


def heading(title, detail="", eyebrow=""):
    return ('<header class="page-title">' + (f'<p class="eyebrow">{escape(eyebrow)}</p>' if eyebrow else "")
            + f"<h1>{escape(title)}</h1>" + (f"<p>{escape(detail)}</p>" if detail else "") + "</header>")


def section(title, body, action="", cls="", *, anchor=""):
    identity = f' id="{anchor}"' if anchor else ""
    return f'<section class="{cls}"{identity}><div class="section-head"><h2>{escape(title)}</h2>{action}</div>{body}</section>'


def grid(items, cls=""):
    return f'<div class="grid {cls}">' + "".join(items) + "</div>"


def card(title, text, link="", meta=""):
    return (f'<article class="tile">{f"<small>{escape(meta)}</small>" if meta else ""}<h3>{escape(title)}</h3>'
            f'<p>{escape(text)}</p>{link}</article>')


def search(label, target, placeholder):
    return (f'<label class="search">{escape(label)}<input type="search" data-filter="{target}" '
            f'placeholder="{escape(placeholder, quote=True)}"></label><p class="sr-only" role="status" data-count="{target}"></p>')


def table(headers, rows, *, key="", attrs=None, label=""):
    top = "".join(f'<th scope="col">{escape(field)}</th>' for field in headers)
    parts = []
    for i, cells in enumerate(rows):
        props = attrs[i] if attrs else ""
        parts.append(f"<tr{props}>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    element_id = f' id="{key}"' if key else ""
    caption = f'<caption class="sr-only">{escape(label)}</caption>' if label else ""
    return f'<div class="table-wrap"><table{element_id}>{caption}<thead><tr>{top}</tr></thead><tbody>{"".join(parts)}</tbody></table></div>'


class Site:
    def __init__(self, staging, world, entry, catalog, *, dist=None, analytics_token=None):
        self.stage = staging
        self.world = world
        self.entry = entry
        self.catalog = catalog
        self.dist = dist
        self.beacon = beacon(analytics_token)
        self.reading_routes = {}
        self.prefix = f"releases/{entry['id']}"
        self.styles = self.asset((HERE / "site.css").read_bytes(), "css")
        self.script = self.asset((HERE / "site.js").read_bytes(), "js")
        self.logos = {tid: self.asset(blob, "png") for tid, blob in world.logos.items()}
        self.photos = {tid: (when, self.asset(blob, "png")) for tid, (when, blob) in world.photos.items()}
        self.current_issue = None
        self.almanacs = {}

    def put(self, path, data):
        path = self.stage / path
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, str):
            path.write_text(data, encoding="utf-8", newline="\n")
        else:
            path.write_bytes(data)

    def asset(self, data, extension):
        route = f"assets/{identity(data)[:24]}.{extension}"
        if not (self.stage / route).exists():
            self.put(route, data)
        return route

    def route(self, path):
        return self.prefix + "/" + path

    def team_route(self, tid):
        return self.route(f"teams/{self.world.teams[tid]['code'].lower()}/index.html")

    def game_route(self, game):
        return self.route(f"games/{game.day}.html#g-{game.id}")

    def team(self, page, tid, field="name"):
        return a(page, self.team_route(tid), self.world.teams[tid][field])

    def logo(self, page, tid, cls="club-logo"):
        return img(page, self.logos[tid], cls=cls) if tid in self.logos else ""

    def frame(self, page, title, body, tab="", *, evergreen=False):
        logo = self.logo(page, 0, "brand-mark")
        home = "index.html" if evergreen else self.route("index.html")
        nav = []
        for label, dest in NAV:
            if label == "Today":
                target = dest
            elif not evergreen:
                target = self.route(dest)
            elif label == "History":
                target = "index.html#editions"
                label = "Publication timeline"
            else:
                continue
            current = ' aria-current="page"' if tab == label else ""
            nav.append(f'<a{current} href="{href(page, target)}">{label}</a>')
        baseball_label = "ABL baseball date " if page == home else ""
        datebar = (f'{self.world.season} season <span>·</span> {baseball_label}{pretty(self.world.league_date)}'
                   if not evergreen else "The ABL almanac")
        if not evergreen:
            datebar += f'<span class="cutoff">Finals through {pretty(self.world.cutoff, year=False)}</span>'
            if page == home:
                revision_id = self.entry["id"].rsplit("_", 1)[-1]
                datebar += f'<span id="publication-details">Publication ID {escape(revision_id)}</span>'
            datebar += a(page, "index.html#editions", "Edition timeline", "edition-link")
        return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Follow the fictional Action Baseball League through its verified games, clubs and history.">
<title>{escape(title)} | {NAME}</title>
<link rel="stylesheet" href="{href(page, self.styles)}"><script defer src="{href(page, self.script)}"></script></head>
<body><a class="skip" href="#content">Skip to content</a>
<header class="masthead"><div class="site-width masthead-inner"><a class="brand" href="{href(page, home)}">{logo}<span>{NAME}</span></a>
<div class="masthead-aside">Action Baseball League</div></div></header>
<nav class="navigation" aria-label="Main navigation"><div class="site-width nav-inner">{''.join(nav)}</div></nav>
<div class="edition-bar"><div class="site-width">{datebar}</div></div>
<main class="site-width" id="content">{body}</main>
<footer><div class="site-width footer-inner"><div><strong>{NAME}</strong><p>Because baseball matters.</p></div>
<div>{a(page, "index.html", "Current edition")}{a(page, "index.html#editions", "Publication timeline")}</div>
<p>Real. Fictional. ⚾.</p></div></footer>{self.beacon}</body></html>"""

    def page(self, path, title, body, tab="", *, absolute=False, evergreen=False):
        route = path if absolute else self.route(path)
        self.put(route, self.frame(route, title, body, tab, evergreen=evergreen))

    def matchup(self, page, game, *, clubs=False):
        if clubs:
            return f"{self.team(page, game.away)} at {self.team(page, game.home)}"
        return a(page, self.game_route(game), f"{self.world.teams[game.away]['code']} at {self.world.teams[game.home]['code']}")

    def games_table(self, page, games, *, include_day=True):
        items = []
        for game in games:
            if game.score is not None:
                state = a(page, self.game_route(game), f"Final · {game.score[0]}–{game.score[1]}", "score-link")
            else:
                label = "Unfinished" if game.innings else (f"{game.time//100:02}:{game.time%100:02}" if game.time else "Scheduled")
                state = a(page, self.game_route(game), label)
            cells = [self.matchup(page, game, clubs=True), state]
            if include_day:
                cells.insert(0, a(page, self.game_route(game), pretty(game.day, year=False)))
            items.append(cells)
        return table((["Date"] if include_day else []) + ["Away at home", "Result or listed start"], items) if items else "<p>There are no games in this view.</p>"

    def standings_table(self, page, ids):
        lead = self.world.teams[ids[0]]
        lines = [[self.team(page, tid), str(c["w"]), str(c["l"]), baseball_rate(float(record(c))), gb(lead, c)]
                 for tid in ids if (c := self.world.teams[tid])]
        return table(["Club", "W", "L", "Pct", "GB"], lines, label="Division standings")

    def standings(self):
        page = self.route("standings/index.html")
        world = self.world
        body = heading("Standings & races", f"Official ABL regular-season records through {pretty(world.cutoff)}.")
        for conf, fullname in (("NBC", "National Baseball Conference"), ("ABC", "American Baseball Conference")):
            divisions = grid([f'<section class="race"><h3>{escape(div)}</h3>{self.standings_table(page, ids)}</section>'
                              for (abbr, div), ids in world.divisions.items() if abbr == conf], "three")
            body += section(fullname, divisions)
        body += section("The postseason picture", "<p>Each conference sends three division winners and one wild card. Eight clubs qualify in all. This view uses current winning percentage and carries ties forward.</p>")
        notes = ("In the 1981-forward Division Championship Series, seeds 1 and 4 meet unless they share a division. "
                 "When they do, seed 1 faces 3 and seed 2 faces 4."
                 if world.season >= 1981 else "In this season, seed 1 faces seed 4 and seed 2 faces seed 3 in the Division Championship Series.")
        body += f"<p>{notes}</p>"
        for conf, race in world.races.items():
            options = []
            for winners, contenders in race["options"]:
                lead = world.teams[contenders[0]]
                rows = [[self.team(page, tid), wl(world.teams[tid]), gb(lead, world.teams[tid])] for tid in contenders]
                options.append('<section class="race"><h3>Division leaders</h3><p>' + ", ".join(self.team(page, tid) for tid in winners)
                               + "</p>" + table(["Wild-card order", "W–L", "GB"], rows) + "</section>")
            text = "<p>Each table shows one possible division-leader assignment for the tied race.</p>" if len(options) > 1 else ""
            matchups = race["dcs"]
            if matchups:
                matchup_lines = [f"<li>{self.team(page, local)} vs. {self.team(page, visitor)}</li>" for local, visitor in matchups]
                text += '<p><strong>Matchups by current standing</strong></p><ul>' + "".join(matchup_lines) + "</ul>"
            else:
                text += "<p>Placement is open while ties remain.</p>"
            body += section(f"{conf} wild card", text + grid(options, "two"))
        body += '<p class="note">Games back is computed from wins and losses. Standings and postseason scenarios use OOTP records. StatsPlus model estimates are separate from league results.</p>'
        self.page("standings/index.html", "Standings & races", body, "Standings")

    def feature(self):
        world = self.world
        group, teams = min(world.divisions.items(), key=lambda item: games_behind(
            world.teams[item[1][0]], world.teams[item[1][1]]))
        first, other = teams[:2]
        leader, chaser = world.teams[first], world.teams[other]
        gap = games_behind(leader, chaser)
        label = f"{group[0]} {group[1].replace('ern Division', '')}"
        if record(leader) == record(chaser):
            title = f"First place is shared in the {label}"
            copy = (f"{leader['city']} and {chaser['city']} both stand at {wl(leader)}." if wl(leader) == wl(chaser)
                    else f"{leader['city']} and {chaser['city']} share the same winning percentage.")
        else:
            title = f"{leader['city']} leads by {gap:g} in the {label}" if gap else f"{leader['city']} leads the {label} on winning percentage"
            copy = f"{chaser['city']} starts at {wl(chaser)} with {leader['city']} at {wl(leader)}."
        matchup = next((g for g in world.by_team[first] if g.score is None and g.day >= world.league_date
                        and g.side(first) == other), None)
        art = matchup.home if matchup else first
        return title, copy, matchup, art

    def box_score(self, page, game):
        output = []
        for tid in (game.away, game.home):
            box = self.world.boxes.get(game.id, {})
            bats = box.get((tid, "bat"), [])
            arms = box.get((tid, "pitch"), [])
            bat_rows = [[self.player_link(page, pid, "batting"), str(r["ab"]), str(r["r"]), str(r["h"]),
                         str(r["rbi"]), str(r["hr"])] for pid, r in bats]
            arm_rows = [[self.player_link(page, pid, "pitching"), ip(r["outs"]), str(r["ha"]), str(r["er"]),
                         str(r["bb"]), str(r["k"])] for pid, r in arms]
            output.append(f"<section><h4>{self.team(page, tid)}</h4>" + table(["Batter", "AB", "R", "H", "RBI", "HR"], bat_rows)
                          + table(["Pitcher", "IP", "H", "ER", "BB", "K"], arm_rows) + "</section>")
        return '<details class="box"><summary>Box score</summary>' + grid(output, "two") + "</details>"

    def daily(self):
        world = self.world
        page = self.route("games/index.html")
        days = list(world.days)
        latest = [d for d in days if d <= world.cutoff]
        upcoming = [d for d in days if d >= world.league_date and any(g.score is None for g in world.days[d])]
        body = heading("Game Center", f"{world.season} regular season · {world.finals:,} verified finals through {pretty(world.cutoff)}.")
        body += grid([
            card("Latest scores", pretty(latest[-1]), a(page, self.route(f"games/{latest[-1]}.html"), "See finals", "text-link")),
            card("Next scheduled slate", pretty(upcoming[0]) if upcoming else "The current schedule has no later fixtures.",
                 a(page, self.route(f"games/{upcoming[0]}.html"), "See the schedule", "text-link") if upcoming else ""),
            card("Historical scores", "Explore the closed seasons from the ABL almanac.",
                 a(page, self.route("history/index.html"), "Open the almanac", "text-link")),
        ], "three")
        body += section("Games by date", search("Find a game date", "game-dates", "Try August 20 or 1981-08-20")
                        + '<ul id="game-dates" class="date-grid">' + "".join(
            f'<li data-item data-search="{d}">{a(page, self.route(f"games/{d}.html"), pretty(d, year=False))}'
            f'<span>{sum(g.score is not None for g in world.days[d])} final / {len(world.days[d])} listed</span></li>'
            for d in days) + "</ul>")
        self.page("games/index.html", "Game Center", body, "Games")
        for i, (day, games) in enumerate(world.days.items()):
            local = f"games/{day}.html"
            page = self.route(local)
            previous = a(page, self.route(f"games/{days[i-1]}.html"), "Previous date") if i else ""
            following = a(page, self.route(f"games/{days[i+1]}.html"), "Next date") if i + 1 < len(days) else ""
            body = heading(pretty(day), f"{sum(g.score is not None for g in games)} final · {len(games)} listed games", "Game Center")
            body += f'<nav class="subnav" aria-label="Game dates">{previous}{a(page,self.route("games/index.html"),"All dates")}{following}</nav>'
            for game in games:
                park = world.teams[game.home]["park"]
                if game.score is not None:
                    line = f"Final · {game.score[0]}–{game.score[1]} · {game.innings} innings"
                else:
                    clock = f" · {game.time//100:02}:{game.time%100:02} listed" if game.time else ""
                    line = "Unfinished game" + clock if game.innings else "Scheduled" + clock
                series = ""
                if game.id in world.series:
                    start, end, count = world.series[game.id]
                    dates = pretty(start, year=False) + ("–" + pretty(end, year=False) if end != start else "")
                    series = f' <span class="subtle">· {count} listed at this venue, {escape(dates)}</span>'
                body += (f'<section id="g-{game.id}" class="game"><div class="game-heading"><div><h2>{self.matchup(page, game, clubs=True)}</h2>'
                         f'<p>{escape(park)}{series}</p></div><strong>{escape(line)}</strong></div>')
                if game.score is not None:
                    body += self.box_score(page, game)
                for article in self.all_articles():
                    if game.id in article.get("games", []) and article["cutoff"] >= game.day:
                        body += f'<p class="related">Coverage {self.reading_link(page, article["path"], article["title"])}</p>'
                body += "</section>"
            self.page(local, pretty(day) + " scores & schedule", body, "Games")

    def scores_strip(self, page, games):
        cards = []
        for game in games:
            teams = (game.away, game.home)
            rows = "".join(f"<span>{self.logo(page, tid, 'score-mark')}<b>{escape(self.world.teams[tid]['code'])}</b>"
                           f'<strong>{game.score[i]}</strong></span>' for i, tid in enumerate(teams))
            cards.append(f'<a class="score-card" href="{href(page, self.game_route(game))}">{rows}<small>Final</small></a>')
        return '<div class="score-strip">' + "".join(cards) + "</div>"

    def home(self):
        world = self.world
        page = self.route("index.html")
        title, detail, appointment, art = self.feature()
        if appointment:
            visitor, home = world.teams[appointment.away], world.teams[appointment.home]
            detail += f" {visitor['city']} visits {home['city']} on {pretty(appointment.day, year=False)}."
        if art in self.photos:
            when, picture = self.photos[art]
            name = world.teams[art]["name"]
            media = (f'<figure class="hero-art">{img(page, picture, name + " team portrait", lazy=False)}'
                     f'<figcaption>{escape(name)} · {pretty(when)}</figcaption></figure>')
        else:
            media = '<div class="hero-art league-art">' + self.logo(page, 0, "cover-mark") + "</div>"
        slate_link = (a(page, self.game_route(appointment), "See the matchup", "button secondary")
                      if appointment else a(page, self.route("games/index.html"), "See upcoming games", "button secondary"))
        body = f"""<section class="hero"><div class="hero-message">
<p class="eyebrow">The ABL Today <span>·</span> {pretty(world.league_date)}</p>
<h1>{escape(title)}</h1><p class="hero-deck">{escape(detail)}</p>
<div class="buttons">{a(page, self.route("standings/index.html"), "Explore the races", "button")}{slate_link}</div>
</div>{media}</section>"""
        body += self.special_coverage(page, featured=True)
        body += (f'<div class="edition-credit"><p>A fictional baseball world, played in OOTP and preserved game by game. '
                 f'{a(page, self.route("about/index.html"), "See it for yourself.")}</p>'
                 f'{publication_time(self.entry)}</div>')
        finals = [g for g in world.days[world.cutoff] if g.score is not None]
        body += section(f"Scores · {pretty(world.cutoff, year=False)}", self.scores_strip(page, finals),
                        a(page, self.route(f"games/{world.cutoff}.html"), "All scores & box scores", "text-link"))
        body += '<div class="home-columns"><div>'
        recaps = [item for item in self.current_issue["articles"] if not item.get("feature_through")] if self.current_issue else []
        stories = []
        if recaps:
            for item in recaps[:3]:
                stories.append(card(item["title"], item["dek"], self.reading_link(page, item["path"], "Read the story", "text-link"), item["byline"]))
        else:
            for game in sorted(finals, key=lambda g: -abs(g.score[0] - g.score[1]))[:3]:
                a_runs, h_runs = game.score
                win, lose = (game.away, game.home) if a_runs > h_runs else (game.home, game.away)
                line = f"{world.teams[win]['city']} beat {world.teams[lose]['city']} {max(game.score)}–{min(game.score)}"
                value = wl(world.teams[win])
                stories.append(card(line, f"{world.teams[win]['name']} stand at {value} through {pretty(game.day, year=False)}.",
                                    a(page, self.game_route(game), "Read the box score", "text-link"), "Around the league"))
        body += section("From the Newsroom" if recaps else "Around the league", grid(stories, "story-grid"),
                        a(page, self.route("newsroom/index.html"), "Browse coverage", "text-link"))
        if world.next_day:
            upcoming = [g for g in world.days[world.next_day] if g.score is None]
            body += section(f"Coming up · {pretty(world.next_day, year=False)}",
                            self.games_table(page, upcoming, include_day=False),
                            a(page, self.route("games/index.html"), "Full calendar", "text-link"))
        body += "</div><aside>"
        races = []
        for (conf, div), ids in world.divisions.items():
            lead, chase = (world.teams[tid] for tid in ids[:2])
            margin = games_behind(lead, chase)
            races.append(f'<li><b>{escape(conf + " " + div.replace("ern Division", ""))}</b><span>'
                         + (self.team(page, ids[0], "city") + " / " + self.team(page, ids[1], "city") + " tied"
                            if record(lead) == record(chase) else self.team(page, ids[0], "city")
                            + (f" +{margin:g}" if margin else " · Pct")) + "</span></li>")
        body += section("Race board", '<ul class="race-list">' + "".join(races) + "</ul>",
                        a(page, self.route("standings/index.html"), "Standings", "text-link"), "sidebar-block")
        hitters, pitchers = [], []
        for game in finals:
            for (team, kind), lines in world.boxes[game.id].items():
                for person, row in lines:
                    if kind == "bat" and (row["hr"] or row["h"] >= 3):
                        hitters.append(((row["hr"], row["h"], row["rbi"]), person, team,
                                        f'{row["h"]}-for-{row["ab"]}, {row["hr"]} HR, {row["rbi"]} RBI', game))
                    elif kind == "pitch" and row["outs"] >= 15:
                        rank = (-row["r"], row["outs"], row["k"], -row["ha"])
                        line = f'{ip(row["outs"])} IP, {row["ha"]} H, {row["r"]} R, {row["k"]} K'
                        pitchers.append((rank, person, team, line, game))

        for title, kind, performers, target, link, anchor in (
            ("At the plate", "batting", hitters, "players/index.html", "League leaders", ""),
            ("On the mound", "pitching", pitchers, "players/pitching.html", "Pitching stats", "on-the-mound"),
        ):
            notable = []
            for _, person, team, result, game in sorted(performers, reverse=True, key=lambda item: item[:3])[:3]:
                notable.append(f'<li>{self.player_link(page, person, kind)} · {self.team(page, team, "code")}<br>'
                               f'{a(page, self.game_route(game), result)}</li>')
            if notable:
                body += section(title, '<ul class="performance-list">' + "".join(notable) + "</ul>",
                                a(page, self.route(target), link, "text-link"), "sidebar-block", anchor=anchor)
        body += "</aside></div>"
        beat_stories = [item for item in self.all_articles() if item["kind"] == "beat"]
        linked = [self.team_tile(page, tid, beat=True) for tid in sorted(world.teams, key=lambda t: world.teams[t]["name"])]
        body += section("On the beat", self.article_cards(page, beat_stories[:4]) or grid(linked[:8], "four"),
                        a(page, self.route("newsroom/index.html") + "#franchise-beats", "All 24 beats", "text-link"))
        self.page("index.html", "The ABL Today", body, "Today")

    def all_articles(self):
        seen = set()
        for issue in reversed(self.catalog["issues"]):
            for article in issue["articles"]:
                if article["path"] not in seen:
                    seen.add(article["path"])
                    yield article

    def special_coverage(self, page, *, featured=False):
        reports = [item for item in self.all_articles() if item.get("feature_through")
                   and (not featured or item["league_date"] <= self.world.league_date <= item["feature_through"])]
        if not reports:
            return ""
        cards = [card(item["title"], "Read the pre-simulation race boards, wild-card picture, trade deadline analysis and day-by-day ABL Advance guide.",
                      self.reading_link(page, item["path"], "Read the coverage", "text-link"), item["byline"])
                 for item in reports]
        return section("Special Coverage", grid(cards, "story-grid"))

    def team_tile(self, page, tid, *, beat=False):
        club = self.world.teams[tid]
        destination = self.team_route(tid) + ("#beat" if beat else "")
        name = club["writer"] if beat else club["name"]
        return (f'<a data-item class="team-tile" href="{href(page,destination)}">{self.logo(page, tid)}'
                f'<span><b>{escape(name)}</b><small>{club["conference"]} · {wl(club)}</small></span></a>')

    def teams(self):
        world = self.world
        page = self.route("teams/index.html")
        body = heading("The 24 clubs", f"Meet the franchises and browse each club’s beat from the {world.season} season.")
        for (conf, div), ids in world.divisions.items():
            body += section(f"{conf} {div}", grid([self.team_tile(page, tid) for tid in ids], "four"))
        self.page("teams/index.html", "The 24 clubs", body, "Teams")
        for tid, club in world.teams.items():
            code = club["code"].lower()
            local = f"teams/{code}/index.html"
            page = self.route(local)
            division = world.divisions[club["conference"], club["division"]]
            leader = world.teams[division[0]]
            gap = games_behind(leader, club)
            first = record(leader) == record(club)
            if first:
                position = "Shares first" if record(leader) == record(world.teams[division[1]]) else "First place"
            else:
                position = f"{gap:g} games back" if gap else "Level on games back"
            body = (f'<header class="club-head">{self.logo(page, tid, "large-logo")}<div>'
                    f'<p class="eyebrow">{escape(club["conference"] + " " + club["division"])}</p>'
                    f'<h1>{escape(club["name"])}</h1><p>{wl(club)} · {position} · {escape(club["park"])}</p></div></header>')
            body += f'<nav class="subnav" aria-label="Club sections">{a(page,self.route(f"teams/{code}/stats.html"),"Club statistics")}'
            body += f'{a(page,self.route("standings/index.html"),"Division standings")}{a(page,self.route("fans/index.html")+f"#{code}","Ballpark attendance")}</nav>'
            completed = [g for g in world.by_team[tid] if g.score is not None]
            scheduled = [g for g in world.by_team[tid] if g.score is None and g.day >= world.league_date]
            sides = ["<div>" + section("Recent results", self.games_table(page, list(reversed(completed[-5:])))) + "</div>",
                     "<div>" + section("Upcoming schedule", self.games_table(page, scheduled[:6])) + "</div>"]
            body += grid(sides, "two")
            items = [item for item in self.all_articles() if tid in item["teams"]]
            recap = (f"The latest final on {pretty(completed[-1].day, year=False)} was "
                     f"{world.teams[completed[-1].away]['city']} {completed[-1].score[0]}, "
                     f"{world.teams[completed[-1].home]['city']} {completed[-1].score[1]}." if completed else "")
            journal = self.article_cards(page, items)
            body += section(club["writer"],
                            f"<p>{escape(recap)}</p>" + (journal or "<p>This beat has results, upcoming games and franchise records. Its article archive is open for future coverage.</p>"),
                            anchor="beat")
            history = []
            for year, season_data in world.history.items():
                if old := season_data["teams"].get(tid):
                    finish = "Champion" if old["champion"] else ("Tournament" if old["playoffs"] else "Regular season")
                    history.append([a(page, self.almanacs[year], year), escape(old["name"]), wl(old), finish])
            if history:
                body += section("Franchise history", table(["Season", "Franchise", "W–L", "Finish"], history),
                                a(page, self.route("history/index.html"), "League almanac", "text-link"))
            if tid in self.photos:
                when, picture = self.photos[tid]
                body += section("Team portrait", f'<figure class="team-photo">{img(page,picture,club["name"]+" team portrait")}'
                                f'<figcaption>OOTP team portrait · {pretty(when)}</figcaption></figure>')
            self.page(local, club["name"], body, "Teams")
            self.team_stats(tid, code)

    def player_link(self, page, pid, kind):
        return a(page, self.route(f"players/{kind}.html#p-{pid}"), self.world.players[pid])

    def stat_line(self, page, pid, line, kind, clubs="", *, link=True):
        label = self.player_link(page, pid, kind) if link else escape(self.world.players[pid])
        if kind == "batting":
            data = (line["g"], line["pa"], line["h"], line["hr"], line["rbi"],
                    baseball_rate(line["h"] / line["ab"] if line["ab"] else None), baseball_rate(ops(line)))
        else:
            data = (line["g"], line["gs"], ip(line["outs"]), line["w"], line["l"], line["s"], line["k"],
                    baseball_rate(era(line), 2))
        return [label, clubs, *map(str, data)]

    def stat_headers(self, kind, use_club=True):
        fields = (["G", "PA", "H", "HR", "RBI", "AVG", "OPS"] if kind == "batting"
                  else ["G", "GS", "IP", "W", "L", "SV", "SO", "ERA"])
        return ["Player"] + (["Club"] if use_club else []) + fields

    def team_stats(self, tid, code):
        world = self.world
        local = f"teams/{code}/stats.html"
        page = self.route(local)
        club = world.teams[tid]
        body = heading(club["name"] + " statistics", f"Regular-season games completed through {pretty(world.cutoff)}.")
        body += f'<nav class="subnav">{a(page,self.team_route(tid),"Club home")}{a(page,self.route("players/index.html"),"League leaders")}</nav>'
        for kind, source in (("batting", world.batting), ("pitching", world.pitching)):
            names = [(pid, line) for (pid, team), line in source.items() if team == tid]
            names.sort(key=lambda entry: -entry[1]["pa" if kind == "batting" else "outs"])
            records = []
            total = Counter()
            for pid, line in names:
                stat = self.stat_line(page, pid, line, kind)
                records.append([stat[0], *stat[2:]])
                total.update(line)
            if kind == "batting":
                average = baseball_rate(total["h"] / total["ab"] if total["ab"] else None)
                overall = f'{total["h"]} hits · {total["hr"]} HR · {average} AVG · {baseball_rate(ops(total))} OPS'
            else:
                overall = f'{ip(total["outs"])} IP · {total["k"]} SO · {baseball_rate(era(total), 2)} ERA'
            body += section("Team " + kind, f'<p class="stat-total">{overall}</p>' + table(self.stat_headers(kind, False), records))
        self.page(local, club["name"] + " statistics", body, "Teams")

    def players(self):
        world = self.world
        page = self.route("players/index.html")
        body = heading("Players & statistics", f"Regular-season lines through {pretty(world.cutoff)}. Derived from completed OOTP player-game records.")
        sections = []
        for kind, source, amount in (("batting", world.batting, "pa"), ("pitching", world.pitching, "outs")):
            directory = defaultdict(Counter)
            team_ids = defaultdict(set)
            for (pid, tid), line in source.items():
                directory[pid].update(line)
                team_ids[pid].add(tid)
            title = kind.capitalize()
            if kind == "batting":
                eligible = [pid for pid, line in directory.items() if line["pa"] >= 300 and ops(line) is not None]
                leading = sorted(eligible, key=lambda pid: (-ops(directory[pid]), world.players[pid]))[:10]
                typ, requirement = "OPS", "300 plate appearances"
            else:
                eligible = [pid for pid, line in directory.items() if line["outs"] >= 270]
                leading = sorted(eligible, key=lambda pid: (era(directory[pid]), world.players[pid]))[:10]
                typ, requirement = "ERA", "90 innings"
            lead_lines = [[self.player_link(page, pid, kind),
                           baseball_rate(ops(directory[pid])) if kind == "batting" else baseball_rate(era(directory[pid]), 2)]
                          for pid in leading]
            intro = f'<p>Display minimum {requirement}.</p>' + (table(["Player", typ], lead_lines) if lead_lines else
                                                               '<p>The current sample is below that minimum.</p>')
            sections.append(f'<section class="race"><h2>{title} leaders · {typ}</h2>{intro}'
                            + a(page, self.route(f"players/{kind}.html"), f"All individual {kind}", "text-link") + "</section>")
            local = f"players/{kind}.html"
            route = self.route(local)
            formatted, attrs = [], []
            for pid in sorted(directory, key=lambda pid: (-directory[pid][amount], world.players[pid], pid)):
                teams = sorted(team_ids[pid])
                team_links = " / ".join(self.team(route, tid, "code") for tid in teams)
                formatted.append(self.stat_line(route, pid, directory[pid], kind, team_links, link=False))
                team_query = " ".join(world.teams[t]["name"] for t in teams)
                attrs.append(f' id="p-{pid}" data-item data-search="{escape(team_query, quote=True)}"')
            helptext = (f"{pretty(world.cutoff)} · Season lines combine appearances by player ID. Individual club stints are on the club statistics pages.")
            player_body = (heading(f"Individual {kind}", helptext) + f'<nav class="subnav">{a(route, self.route("players/index.html"),"League leaders")}</nav>'
                           + search("Find a player or club", "player-table", "Player name or team")
                           + table(self.stat_headers(kind), formatted, key="player-table", attrs=attrs, label=f"Individual {kind}"))
            self.page(local, f"Individual {kind}", player_body, "Players")
        body += grid(sections, "two")
        body += section("By team", grid([card(club["name"], f"{club['conference']} · {club['division']}",
            a(page,self.route(f"teams/{club['code'].lower()}/stats.html"), "Batting & pitching", "text-link"))
            for club in sorted(world.teams.values(), key=lambda c: c["name"])], "four"))
        body += '<p class="note">OPS combines on-base percentage and slugging. ERA uses earned runs and recorded outs. Innings after the decimal are outs. The lines on these pages are derived from OOTP.</p>'
        body += '<p>' + a(page, self.route("history/index.html"), "Browse historical player and club records", "text-link") + '</p>'
        self.page("players/index.html", "Players & statistics", body, "Players")

    def reading_link(self, page, target, label, cls=""):
        return a(page, self.reading_routes.get(target, target), label, cls)

    def newsroom_reading_copies(self):
        if not self.catalog["issues"]:
            return
        pages = {issue["path"] for issue in self.catalog["issues"]}
        pages.update(article["path"] for article in self.all_articles())
        copies = {}
        for path in sorted(pages):
            source = self.dist / path
            if not source.resolve().is_relative_to(self.dist.resolve()):
                raise ValueError("Archived newsroom page is outside the publication.")
            html = source.read_text(encoding="utf-8")
            today = TODAY_LINK.search(html)
            if (self.beacon and self.beacon not in html) or (today and today[2] != href(path, "index.html")):
                self.reading_routes[path] = self.route("reading/" + path.removeprefix("releases/"))
                copies[path] = html

        for original, html in copies.items():
            destination = self.reading_routes[original]

            def rebase(match):
                attr, value = match.groups()
                parts = urlsplit(unescape(value))
                if parts.scheme or parts.netloc or not parts.path:
                    return match[0]
                target = normpath(join(dirname(original), parts.path))
                target = self.reading_routes.get(target, target)
                address = urlunsplit(("", "", relpath(target, dirname(destination)), parts.query, parts.fragment))
                return f'{attr}="{escape(address, quote=True)}"'

            html = BEACON_TAG.sub("", html)
            html = re.sub(r'(?<=\s)(href|src)="([^"]*)"', rebase, html)
            today_target = href(destination, "index.html")
            html = TODAY_LINK.sub(lambda link: link[1] + today_target + link[3], html, count=1)
            self.put(destination, html.replace("</body>", self.beacon + "</body>", 1))

    def article_cards(self, page, items):
        return grid([f'<article class="tile" data-item data-search="{escape(" ".join(self.world.teams[t]["name"] for t in item["teams"]), quote=True)}">'
                     f'<small>{escape(item["byline"])} · {pretty(item["league_date"],year=False)}</small>'
                     f'<h3>{self.reading_link(page,item["path"],item["title"])}</h3><p>{escape(item["dek"])}</p></article>'
                     for item in items], "story-grid") if items else ""

    @staticmethod
    def coverage_body(sections):
        finished = []
        for label, blocks in sections:
            if label:
                finished.append(f"<h2>{escape(label)}</h2>")
            for block in blocks:
                kind, content = coverage_block(block)
                if kind == "heading":
                    finished.append(f"<h3>{escape(content)}</h3>")
                elif kind == "table":
                    columns, rows = content
                    finished.append(table(columns, [[escape(cell) for cell in row] for row in rows], label=label))
                else:
                    finished.append(f"<p>{escape(content)}</p>")
        return "".join(finished)

    def editorial(self, eligible, warnings):
        existing = {issue["key"] for issue in self.catalog["issues"]}
        versions = {article["digest"]: article for article in self.all_articles()}
        for issue in eligible:
            if issue["key"] in existing:
                continue
            base = self.route(f"newsroom/issues/{issue['key'][:16]}")
            published = {field: issue[field] for field in ("key", "source_id", "edition", "revision", "league_date", "cutoff")}
            published.update(path=base + "/index.html", articles=[])
            for article in sorted(issue["articles"], key=lambda item: not item.lead):
                previous = next((other for other in self.all_articles() if other["slug"] == article.slug
                                 and other["digest"] != article.digest), None)
                if article.digest in versions:
                    published["articles"].append(versions[article.digest])
                    continue
                opening = next(p for _, paragraphs in article.sections for p in paragraphs)
                dek = opening if len(opening) <= 185 else opening[:185].rsplit(" ", 1)[0] + "…"
                route = f"{base}/{article.slug}.html"
                item = dict(path=route, title=article.title, slug=article.slug, digest=article.digest, dek=dek,
                            byline=article.byline, teams=list(article.teams), kind=article.kind, games=list(article.game_ids),
                            league_date=issue["league_date"], cutoff=issue["cutoff"])
                docs = a(route, published["path"], "Read the issue")
                pdf = article_pdf(article, issue, warnings)
                if pdf:
                    item["pdf"] = f"{base}/{article.slug}.pdf"
                    self.put(item["pdf"], pdf)
                if article.feature_through:
                    item["feature_through"] = article.feature_through
                    body = heading(article.title, f"{article.byline} · Data cutoff {pretty(issue['cutoff'])}", "Special Coverage")
                    guide = " · ".join(a(route, self.route(dest), label) for dest, label in (
                        ("games/index.html", "Game calendar"), ("standings/index.html", "Standings"),
                        ("sbv/index.html", "Sim Baseball Vision and ABL Advance")))
                    pdf_link = f'<a href="{href(route, item["pdf"])}" download>Download public PDF</a>'
                    body += f'<p class="related">{pdf_link} · {guide}</p><article class="reading coverage">'
                    body += self.coverage_body(article.sections)
                    related = ""
                else:
                    body = heading(article.title, f"{article.byline} · {pretty(issue['league_date'])}",
                                   "The Baseball Observer" if article.kind == "observer" else "The ABL Newsroom")
                    edition = next((entry["id"] for entry in reversed(self.catalog["editions"])
                                    if entry["source_id"] == issue["source_id"]), self.entry["id"])
                    record = a(route, f"releases/{edition}/games/{issue['cutoff']}.html", "Verified ABL results")
                    body += f'<article class="reading"><p class="note">Games through {pretty(issue["cutoff"])} · {record}</p>'
                    for label, paragraphs in article.sections:
                        if label:
                            body += f"<h2>{escape(label)}</h2>"
                        body += "".join(f"<p>{escape(line)}</p>" for line in paragraphs)
                    related = " · ".join(self.team(route, tid) for tid in article.teams)
                if pdf:
                    docs += " · " + a(route, item["pdf"], "View PDF")
                if previous:
                    docs += " · " + self.reading_link(route, previous["path"], "Earlier version")
                if related:
                    body += f'<p class="related">{related}</p>'
                self.page(route, article.title, body + f'<p>{docs}</p></article>', "Newsroom", absolute=True)
                versions[item["digest"]] = item
                published["articles"].append(item)
            self.catalog["issues"].append(published)
            page = published["path"]
            issue_title = f"The {pretty(issue['league_date'], year=False)} issue"
            if all(item.get("feature_through") for item in published["articles"]):
                published["category"] = "Special Coverage"
                issue_title = f"Special Coverage · {pretty(issue['league_date'], year=False)}"
            body = heading(issue_title,
                           f"Games through {pretty(issue['cutoff'])}.", "ABL Newsroom")
            body += self.article_cards(page, published["articles"])
            self.page(page, "Newsroom issue " + issue["league_date"], body, "Newsroom", absolute=True)
        self.current_issue = next((issue for issue in reversed(self.catalog["issues"])
                                   if issue["source_id"] == self.world.source_id
                                   and any(not item.get("feature_through") for item in issue["articles"])), None)

    def newsroom(self):
        page = self.route("newsroom/index.html")
        current = self.current_issue
        issue_route = self.route("newsroom/current.html")
        body = heading("ABL Newsroom", "League reporting, franchise beats and The Baseball Observer.")
        body += self.special_coverage(page)
        current_title = f"The {pretty(self.world.league_date, year=False)} edition"
        body += section(current_title,
                        self.article_cards(page, current["articles"]) if current else
                        f'<p>The baseball edition is available with finals through {pretty(self.world.cutoff,year=False)}.</p>',
                        a(page, issue_route, "Open current issue", "text-link"))
        items = list(self.all_articles())
        archive = ""
        if self.catalog["issues"]:
            archive = search("Search article archive", "articles", "Headline, writer or club")
            archive += '<div id="articles">' + self.article_cards(page, items) + "</div>"
            archive += '<ul class="list">' + "".join(f'<li>{self.reading_link(page,i["path"],i.get("category") or pretty(i["league_date"]))} · '
                                                   f'Games through {pretty(i["cutoff"],year=False)}</li>'
                                                   for i in reversed(self.catalog["issues"])) + "</ul>"
        else:
            archive = '<p>Published articles and issue revisions will appear here as they enter the public record.</p>'
        body += section("Article & issue archive", archive, a(page, "index.html#editions", "Baseball editions", "text-link"))
        body += section("Franchise beats", grid([self.team_tile(page, tid, beat=True) for tid in self.world.teams], "four"),
                        anchor="franchise-beats")
        self.page("newsroom/index.html", "ABL Newsroom", body, "Newsroom")
        issue = heading(current_title, f"Games through {pretty(self.world.cutoff)}.", "ABL Newsroom")
        if current:
            issue += self.article_cards(issue_route, current["articles"])
            issue += f'<p>{self.reading_link(issue_route, current["path"], "Permanent issue link")}</p>'
        else:
            issue += '<p>This date has no published articles. Explore the recorded games and club beats.</p>'
            issue += grid([
                card("Latest scores", "Finals and player box scores.", a(issue_route, self.route(f"games/{self.world.cutoff}.html"), "Visit Game Center", "text-link")),
                card("Around the league", "Club records and the six division races.", a(issue_route, self.route("standings/index.html"), "Read the standings", "text-link")),
            ], "two")
        self.page("newsroom/current.html", current_title, issue, "Newsroom")

    def almanac_page(self, year, title, body):
        # All almanac pages share the same relative path depth and navigation.
        rendered = self.frame("almanac/season/index.html", title, body, evergreen=True)
        path = f"almanac/{year}-{identity(rendered.encode())[:16]}/index.html"
        self.put(path, rendered)
        return path

    def almanac_players(self, year, data):
        lines = self.world.historical_players.get(year)
        if not lines:
            return ""
        title = f"{year} player records"
        body = heading(title, "Closed regular-season OOTP totals. Player IDs connect club stints across the season.", "ABL Almanac")
        for kind, players in lines.items():
            key = "historical-" + kind
            records, attrs = [], []
            amount = "pa" if kind == "batting" else "outs"
            for pid in sorted(players, key=lambda pid: (-players[pid][0][amount], self.world.players[pid], pid)):
                values, club_ids = players[pid]
                clubs = " / ".join(escape(data["teams"][tid]["name"]) for tid in sorted(club_ids))
                records.append(self.stat_line("", pid, values, kind, clubs, link=False))
                attrs.append(f' id="{kind}-{pid}" data-item')
            body += section(kind.capitalize(), search(f"Find a player or club in {kind}", key, "Player name or club")
                            + table(self.stat_headers(kind), records, key=key, attrs=attrs, label=kind.capitalize()))
        return self.almanac_page(year, title, body)

    def almanac(self):
        world = self.world
        for year, data in world.history.items():
            champs = [team["name"] for team in data["teams"].values() if team["champion"]]
            champion = ", ".join(champs) if champs else "Champion entry unavailable"
            body = heading(f"{year} season almanac", champion, "Action Baseball League history")
            team_rows = [[escape(r["name"]), wl(r), "Champion" if r["champion"] else ("Tournament" if r["playoffs"] else "Regular season")]
                         for r in sorted(data["teams"].values(), key=lambda r: (-record(r), r["name"]))]
            body += section("Club records", table(["Club", "W–L", "Finish"], team_rows))
            if players := self.almanac_players(year, data):
                body += section("Player records", "<p>Regular-season batting and pitching by player and club.</p>",
                                a("almanac/season/index.html", players, f"Browse {year} player records", "text-link"))
            if data["scores"]:
                historic = []
                attrs = []
                dates = set()
                for when, gid, visitor, home, run_a, run_h in data["scores"]:
                    anchor = f' id="day-{when}"' if when not in dates else ""
                    dates.add(when)
                    historic.append([pretty(when, year=False), escape(data["teams"][visitor]["name"]),
                                     escape(data["teams"][home]["name"]), f"{run_a}–{run_h}"])
                    attrs.append(f'{anchor} data-item data-search="{when}"')
                caption = (f'{len(historic):,} matched games from the archived report. This score register includes several parts of '
                           'the calendar. The club records above give official season standings.')
                if data.get("skipped"):
                    caption += f' {data["skipped"]} entries fall outside the matched franchise register.'
                body += section("Archived scores", f"<p>{escape(caption)}</p>"
                                + search("Find a date or club", "historic-scores", "Try September 1 or Dallas")
                                + table(["Date", "Visitor", "Home", "Score (visitor–home)"], historic,
                                        key="historic-scores", attrs=attrs))
            else:
                body += section("Archived scores", "<p>A verified day-by-day score register is unavailable for this season.</p>")
            self.almanacs[year] = self.almanac_page(year, f"{year} season almanac", body)
        page = self.route("history/index.html")
        dates = [card(str(year), next((r["name"] + " · ABL champion" for r in data["teams"].values() if r["champion"]),
                                       "Archived franchise records"),
                      a(page, self.almanacs[year], "Open season", "text-link")) for year, data in world.history.items()]
        body = heading("History & archives", "Return to an earlier league date or explore seasons from the ABL’s first year in 1972.")
        body += section("The publication timeline", "<p>Every EAB edition preserves the standings, games, team records and eligible stories available at that cutoff.</p>",
                        a(page, "index.html#editions", "Browse EAB editions", "text-link"))
        body += section("Seasons & champions", grid(dates, "three"),
                        a(page, self.route("games/index.html"), f"Current {world.season} season", "text-link"))
        body += section("Newsroom archive", "<p>Find coverage by article, issue or franchise.</p>",
                        a(page, self.route("newsroom/index.html"), "Browse the Newsroom", "text-link"))
        self.page("history/index.html", "History & archives", body, "History")

    def fan_culture(self):
        page = self.route("fans/index.html")
        fan = []
        for tid, club in self.world.teams.items():
            values = [g.attendance for g in self.world.by_team[tid] if g.home == tid and g.score is not None and g.attendance > 0]
            fan.append((sum(values) / len(values) if values else 0, len(values), tid, sum(values)))
        attrs, records = [], []
        for average, count, tid, total in sorted(fan, reverse=True):
            club = self.world.teams[tid]
            attrs.append(f' id="{club["code"].lower()}" data-item')
            records.append([self.team(page, tid), escape(club["park"]), str(count), f"{total:,}",
                            f"{average:,.0f}" if count else "—", f'{club["capacity"]:,}' if club["capacity"] else "—"])
        body = heading("Fan Culture", "The communities around the 24 clubs, beginning with what the league records about its ballparks.")
        body += section("Recorded attendance", f"<p>Through {pretty(self.world.cutoff)}. Averages use completed home games with a recorded positive attendance. Capacity describes the current ballpark.</p>"
                        + search("Find a club or ballpark", "attendance", "Club or park")
                        + table(["Club", "Ballpark", "Games reported", "Attendance", "Average", "Capacity"],
                                records, key="attendance", attrs=attrs))
        body += section("Reading the numbers", "<p>Attendance measures the size of a reported crowd. The schedule, available entries and venue capacity give that count context. The current edition’s fan record consists of attendance and ballpark information.</p>")
        self.page("fans/index.html", "Fan Culture", body, "Fan Culture")

    def media_and_about(self, media):
        page = self.route("sbv/index.html")
        body = heading("Sim Baseball Vision", "The commissioner’s broadcasts and commentary offer another way to experience the Action Baseball League.")
        programs = [card(item["title"], item["description"], a(page, item["url"], item["label"], "text-link"))
                    for item in media]
        programs.append(card("ABL Advance", "Follow the recorded games and the upcoming slate as the league advances.",
                             a(page, self.route("games/index.html"), "Browse the game calendar", "text-link")))
        body += grid(programs, "three")
        body += self.special_coverage(page)
        self.page("sbv/index.html", "Sim Baseball Vision", body, "Sim Baseball Vision")
        page = self.route("about/index.html")
        about = heading("See it for yourself.", "Experiencing Action Baseball opens the record of a league whose history is still being played.")
        about += grid([
            card("Action Baseball League", "A fictional professional baseball league established in 1972. Its games are played in Out of the Park Baseball and their results become the league’s history."),
            card("Experiencing Action Baseball", "The public record for those games, the standings, club coverage and earlier editions."),
            card("Sim Baseball Vision", "The commissioner’s personal media operation, with live and recorded programming, commentary and featured games.",
                 a(page, self.route("sbv/index.html"), "Explore programming", "text-link")),
        ], "three")
        about += section("How the record is kept", "<p>EAB opens a new edition from the verified OOTP record. The league date belongs to the simulation. The update time records when that edition was made available. Game results and statistics carry a completed-game cutoff. Articles and issues retain their original dates and earlier versions.</p>",
                         a(page, "index.html#editions", "Visit the publication timeline", "text-link"))
        self.page("about/index.html", "About EAB", about, "About")

    def build(self, eligible, media, warnings):
        self.newsroom_reading_copies()
        self.editorial(eligible, warnings)
        self.almanac()
        self.home()
        self.newsroom()
        self.teams()
        self.standings()
        self.daily()
        self.players()
        self.fan_culture()
        self.media_and_about(media)
