from collections import Counter, defaultdict
from fractions import Fraction
from html import escape
import copy
import json
import os
from pathlib import Path
import sys

from beats import BEATS
from lead import LEADS

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "csv/abl_scripts"))
from abl_newsroom import audit
from abl_newsroom_editorial import audit_desk, checked_review, copy_hash, page
from abl_newsroom_factcheck import ClaimCheck
from abl_newsroom_verify import iso_date, rows
from abl_player_games import BATTING, PITCHING, player_game_rows


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def club_files(edition):
    return {doc["team_id"]: doc for f in (edition / "teams").glob("*.json") if (doc := read(f))}


archive = root / "csv/out/story/newsroom"
cur_pointer = (archive / "current.json").read_bytes()
pointer = json.loads(cur_pointer)
edition = archive / "editions" / pointer["edition"]
coverage = read(edition / "coverage.json")
older = archive / "editions" / coverage["previous_edition"]
desk = Path(pointer["editorial"])
verification = read(edition / "verification.json")
earlier = read(older / "verification.json")
raw = Path(verification["raw_dir"])
audit(edition)
audit(older)
audit_desk(desk, verification["source_id"])
threads = read(desk / "slate.json")["threads"]
before, after = club_files(older), club_files(edition)
assignments = read(edition / "assignments.json")
checker = ClaimCheck(raw, verification["cutoff"], verification["league_date"], after)
window = sorted(coverage["completed_game_ids"])
all_recent = set(window)
assert len(all_recent) == 121 and len(read(edition / "state.json")["games"]) == 1571
assert len(read(older / "state.json")["games"]) == 1450 and len(after) == 24
assert not coverage["corrected_game_ids"]
assert (verification["league_date"], verification["cutoff"]) == ("1981-08-31", "1981-08-30")

out = root / "tools/newsroom/reviews/1981-08-31"
brief = archive / "issues/1981-08-19_2acae29146076f62-special-coverage/fe2087d00dbabd170a1b/league/august-division-challenges-1981.md"
observer = archive / "issues/1981-08-19_2acae29146076f62-observer-conversation/1d3e0572201c81837d55/observer/the-pennant-races-come-home.md"
rules = root / "docs/ABL_POSTSEASON_RULES.md"
handoff = root / "tools/newsroom/reviews/editorial-handoff.md"
asof = {key: verification[key] for key in ("source_id", "league_date", "cutoff")}
divisions = ["NBC Eastern Division", "NBC Central Division", "NBC Western Division",
             "ABC Eastern Division", "ABC Central Division", "ABC Western Division"]
sections = dict(zip(("NBC East", "NBC Central", "NBC West", "ABC East", "ABC Central", "ABC West"), divisions))


def fmt(ids):
    return ",".join(map(str, ids))


def club_name(tid):
    return after[tid]["team"]


def game_record(g):
    when = iso_date(g["date"])
    visitor, home = int(g["away_team"]), int(g["home_team"])
    return dict(date=when, visitor=visitor, home=home,
                away_score=int(g["runs0"]) if g["played"] == "1" else None,
                home_score=int(g["runs1"]) if g["played"] == "1" else None, innings=int(g["innings"]))


games = {gid: game_record(g) for gid, g in checker.games.items()}
first_wl = {int(row["team_id"]): [int(row["w"]), int(row["l"])] for row in rows(Path(earlier["raw_dir"]) / "team_record.csv")
            if int(row["team_id"]) in after}
last_wl = {int(row["team_id"]): [int(row["w"]), int(row["l"])] for row in rows(raw / "team_record.csv")
           if int(row["team_id"]) in after}
standings = {}
for tid, b in after.items():
    past = before[tid]
    recent = sorted(b["game_ids"], key=lambda gid: (games[gid]["date"], int(checker.games[gid]["time"])))
    future = sorted((gid for gid, g in games.items() if g["home_score"] is None and tid in (g["visitor"], g["home"])),
                    key=lambda gid: (games[gid]["date"], int(checker.games[gid]["time"])))
    rec, runs = Counter(), Counter()
    for gid in recent:
        g = games[gid]
        ours, theirs = (g["home_score"], g["away_score"]) if tid == g["home"] else (g["away_score"], g["home_score"])
        rec["w" if ours > theirs else "l"] += 1
        if abs(ours - theirs) == 1:
            rec["one_run_w" if ours > theirs else "one_run_l"] += 1
        runs.update(scored=ours, allowed=theirs)
    assert {gid for gid in window if tid in (games[gid]["home"], games[gid]["visitor"])} == set(recent)
    assert first_wl[tid] == [past["wins"], past["losses"]] and last_wl[tid] == [b["wins"], b["losses"]]
    assert [last_wl[tid][0] - first_wl[tid][0], last_wl[tid][1] - first_wl[tid][1]] == [rec["w"], rec["l"]]
    standings[tid] = dict(name=club_name(tid), division=b["division"], before=[*first_wl[tid], past["gb"]],
                          after=[*last_wl[tid], b["gb"]], window=[rec["w"], rec["l"]],
                          one_run=[rec["one_run_w"], rec["one_run_l"]], runs=dict(runs),
                          completed_game_ids=recent, remaining_game_ids=future)
    assert len(future) == 162 - sum(last_wl[tid])
assert sum(row["window"][0] for row in standings.values()) == sum(row["window"][1] for row in standings.values()) == 121
assert max(standings, key=lambda tid: Fraction(standings[tid]["window"][0], sum(standings[tid]["window"]))) == 19
assert {tid for tid, line in standings.items() if line["window"] == [1, 9]} == {5, 8, 15}
assert {tid for tid, line in standings.items() if len(line["remaining_game_ids"]) == 32} == {1, 11}
assert len([g for g in games.values() if g["home_score"] is None and g["date"] == "1981-08-31"]) == 8
settings = next(row for row in rows(raw / "leagues.csv") if row["league_id"] == "200")
expansion = {key: settings[key] for key in ("rosters_expanded", "roster_expand_date",
                                           "rules_active_roster_limit", "rules_expanded_roster_limit")}
assert list(expansion.values()) == ["0", "1981-9-1", "26", "40"]
postseason = next(row for row in rows(raw / "league_playoffs.csv") if row["league_id"] == "200")
assert postseason["num_wild_cards"] == "1"

def gb(leader, trailer, records):
    a, b = records[leader], records[trailer]
    return (a[0]-b[0]+b[1]-a[1])/2

assert [gb(1, club, last_wl) for club in (12, 10, 9, 5)] == [.5, 2.5, 3.5, 5.5]
assert [gb(13, club, last_wl) for club in (24, 14, 19, 22)] == [0, 4, 4, 6]
assert gb(5, 1, first_wl) == .5 and gb(13, 24, first_wl) == 2
assert sorted((6, 4, 2), key=lambda tid: last_wl[tid][0], reverse=True) == [6, 4, 2]
assert sorted((23, 17, 20), key=lambda tid: last_wl[tid][0], reverse=True) == [23, 17, 20]
assert after[6]["division"] == after[1]["division"] and after[23]["division"] != after[13]["division"] == after[24]["division"]
checkpoints = {}
for day in ("1981-08-23", "1981-08-26", "1981-08-28", "1981-08-29"):
    records = copy.deepcopy(first_wl)
    for gid in window:
        g = games[gid]
        if g["date"] > day:
            continue
        visitor, home = g["visitor"], g["home"]
        win, loss = (visitor, home) if g["away_score"] > g["home_score"] else (home, visitor)
        records[win][0] += 1
        records[loss][1] += 1
    checkpoints[day] = records
assert gb(6, 1, checkpoints["1981-08-26"]) == 5.5
assert gb(4, 12, checkpoints["1981-08-26"]) == 0
assert gb(17, 24, checkpoints["1981-08-29"]) == 0 and gb(24, 13, checkpoints["1981-08-28"]) == 2
history = {int(row["team_id"]): {field: row[field] for field in ("position_in_division", "made_playoffs", "won_playoffs")}
           for row in rows(raw / "team_history.csv") if row["league_id"] == "200" and row["year"] == "1980"}
for row in rows(raw / "team_history_record.csv"):
    if row["league_id"] == "200" and row["year"] == "1980":
        history[int(row["team_id"])]["w_l"] = [int(row["w"]), int(row["l"])]
assert all(history[tid]["position_in_division"] == "1" for tid in (2, 6, 12, 21))
assert history[5]["won_playoffs"] == "1"
assert [history[tid]["w_l"][0] for tid in (1, 2, 19)] == [73, 107, 62]

player_ids = {pid for spec in LEADS for pid in spec["players"]} | {pid for beat in BEATS for pid in beat[4]}
names = {int(row["player_id"]): f'{row["first_name"]} {row["last_name"]}' for row in rows(raw / "players.csv")
         if int(row["player_id"]) in player_ids}
player_lines = {}
totals = defaultdict(Counter)
for file, fields in ((BATTING, ("g", "ab", "pa", "h", "hr", "rbi", "bb", "k", "sb")),
                     (PITCHING, ("gs", "outs", "ha", "r", "er", "k", "bb", "s", "w"))):
    data = []
    for row in player_game_rows(raw, file, 1981, all_recent, verification.get("player_game_reconciliation")):
        pid, tid, gid = (int(row[k]) for k in ("player_id", "team_id", "game_id"))
        if pid in player_ids:
            line = {k: int(row[k]) for k in fields}
            totals[file, tid, pid].update(line)
            data.append(dict(player=names[pid], player_id=pid, team_id=tid, game_id=gid, **line))
    player_lines[file] = data

for file, tid, pid, numbers in [
    (BATTING, 6, 7208, dict(h=17)), (BATTING, 8, 7350, dict(h=12, bb=10, sb=6)),
    (BATTING, 5, 4648, dict(h=14, bb=6)), (BATTING, 2, 2025, dict(h=17, sb=4)),
    (BATTING, 10, 8897, dict(hr=4)), (BATTING, 22, 3201, dict(h=21, rbi=15)),
    (BATTING, 15, 2144, dict(h=11, ab=35, bb=6)), (BATTING, 16, 9363, dict(rbi=14)),
    (BATTING, 16, 1057, dict(rbi=12)), (BATTING, 24, 6611, dict(h=17)),
    (BATTING, 18, 2955, dict(hr=7, rbi=13)), (PITCHING, 7, 10161, dict(outs=43, gs=2, er=1)),
    (PITCHING, 7, 9599, dict(outs=44, gs=2, er=3)), (PITCHING, 20, 1, dict(outs=73, er=1)),
    (PITCHING, 20, 3081, dict(s=6)), (PITCHING, 23, 2383, dict(outs=75, er=3, gs=3)),
    (PITCHING, 18, 1815, dict(er=0, s=4)), (PITCHING, 12, 854, dict(s=4)),
    (BATTING, 1, 1849, dict(rbi=12)),
]:
    assert all(totals[file, tid, pid][stat] == value for stat, value in numbers.items()), (tid, pid, numbers)

individual = {(line["game_id"], line["player_id"]): line for line in player_lines[BATTING]}
individual.update({(line["game_id"], line["player_id"]): line for line in player_lines[PITCHING]})
for gid, pid, counts in [
    (1490, 1005, dict(rbi=5)), (1525, 1849, dict(h=3, hr=1, rbi=4)), (1525, 5456, dict(k=10)),
    (1526, 1054, dict(h=5)), (1526, 4545, dict(rbi=3)), (1538, 4545, dict(rbi=3)),
    (1480, 5873, dict(outs=24, r=0)), (1551, 5873, dict(outs=24, ha=4, r=0)),
    (1455, 8125, dict(outs=24, r=0, k=13)), (1480, 2756, dict(outs=24, r=0)),
    (1540, 2756, dict(outs=24, er=2)), (1494, 1440, dict(outs=24, r=0, ha=2)),
    (1565, 1440, dict(outs=23, er=1)), (1566, 5838, dict(hr=1, rbi=3)),
    (1493, 4584, dict(h=4)), (1566, 4928, dict(hr=2)), (1568, 1, dict(outs=27, ha=5, r=0)),
    (1567, 3201, dict(h=4)), (1567, 4859, dict(outs=23, r=0)), (1568, 3032, dict(outs=16, er=1, k=6)),
    (1569, 8032, dict(h=3, hr=1)), (1498, 6418, dict(outs=27, ha=6, r=0)),
    (1486, 8217, dict(h=3)), (1498, 8217, dict(h=4)), (1570, 6733, dict(outs=23, k=8)),
    (1516, 7584, dict(outs=25, k=10, r=0)), (1558, 7393, dict(rbi=4)),
    (1571, 6052, dict(outs=23, r=2)), (1560, 949, dict(outs=24)), (1572, 1733, dict(outs=17, k=7)),
    (1572, 6735, dict(outs=22, k=8)), (1547, 2955, dict(hr=1)), (1559, 2955, dict(hr=2)),
]:
    assert all(individual[gid, pid][field] == value for field, value in counts.items()), (gid, pid, counts)

log_refs = {
    1476: "bottom of 14th, lines 659–671, Olguin reaches, steals second, takes third on error and scores on Wilson's single",
    1540: "bottom of 14th, lines 550–566, Thompson scores on Nishimoto's sacrifice fly",
    1561: "bottom of ninth, lines 425–443, Watters singles with two out to score Postell",
    1569: "top of ninth, lines 438–459, Juan Grant makes final out with bases loaded",
}
trade_refs = {
    1: ["trade_history.csv, message_id=19877, date=1981-07-29, team IDs 1,2, player ID 1849.",
        "team_history_record.csv, team ID 1 for 1980, 73 wins and 89 losses."],
    6: ["team_history.csv and team_history_record.csv, team ID 6 for 1980 NBC East division title."],
    21: ["trade_history.csv, message_id=19868, date=1981-07-28, team IDs 21,1, player IDs 3032,8655."],
    5: ["team_history.csv and team_history_record.csv, team ID 5 for 1980 ABL champion."],
    2: ["team_history.csv and team_history_record.csv, team ID 2 for 1980, 107 wins and first in the NBC West."],
    12: ["team_history.csv, team IDs 12,5 for the 1980 NBC Central winner and Grand Series champion."],
    19: ["team_history_record.csv, team ID 19 for 1980, 62 wins and 100 losses."],
}
trade_refs[21].append("team_history.csv, team ID 21 for 1980 ABC East first place.")
key_claims = {
    "one-game-is-the-new-margin": [
        ([1561], "Pat Watters singles with two outs in the ninth to score Dwayne Postell and end the Charlotte sweep."),
        ([1455, 1467, 1479, 1491, 1527, 1539, 1551, 1563],
         "Chicago wins three of four against Detroit and Dallas sweeps the Dukes."),
        ([1476, 1560, 1571, 1572], "Wilson ends the 14-inning game in Seattle. Denver wins the last two there while Las Vegas holds first."),
    ],
    "august-31-division-and-wild-card-board": [
        ([1113, 1462, 1474, 1486, 1498, 1534, 1546, 1558, 1570], "St. Louis wins five against Houston and three against Cincinnati."),
        ([1525, 1537, 1549, 1561], "Miami sweeps Charlotte and closes to a game and a half."),
        ([1560, 1571, 1572], "Las Vegas ends a game up and Denver and Seattle tie for the wild card."),
    ],
    "miami-makes-four-count": [([1561], "Watters singles in the ninth to complete the sweep over Charlotte."),
                              ([1525], "De Ruijter drives in all four Miami runs in the opener.")],
    "minneapolis-one-run-games": [([1540], "Nishimoto's fly in the 14th scores Ricky Thompson.")],
    "seattle-what-a-one-run-game-costs": [([1476], "Adam Wilson singles in the winning run in the 14th.")],
    "nashville-even-with-everyone": [([1569], "Juan Grant makes the final out with the bases loaded.")],
}


def gb_text(value):
    return f"{value:g}"


def board(division):
    clubs = sorted((tid for tid in after if after[tid]["division"] == division),
                   key=lambda tid: (-Fraction(last_wl[tid][0], sum(last_wl[tid])), tid))
    pieces = []
    for tid in clubs:
        record = standings[tid]
        w0, l0, gb0 = record["before"]
        w1, l1, gb1 = record["after"]
        w, l = record["window"]
        pieces.append(f"{record['name']} {w0}–{l0} ({gb_text(gb0)} GB) → {w1}–{l1} ({gb_text(gb1)} GB) [{w}–{l}]")
    return pieces


def relative(path):
    return os.path.relpath(path, out).replace(os.sep, "/")


articles = copy.deepcopy(LEADS)
for tid, slug, title, summary, players, focus, schedule, logs, paragraphs in BEATS:
    articles.append(dict(slug=slug, kind="beat", byline=assignments[str(tid)]["assignment"], title=title,
                         summary=summary, teams=[tid], focus=focus, sched=schedule, players=players,
                         logs=logs, sections=[("", paragraphs)]))

assert len(articles) == 26 and {article["teams"][0] for article in articles if article["kind"] == "beat"} == set(after)
assert len({article["slug"] for article in articles}) == len(articles)
baseline = {path: path.read_bytes() for path in (archive / "current.json", edition/"files.json", older/"files.json",
                                                  desk/"files.json", brief, observer, root/"output/eab/dist/index.html")}
copies, validation = {}, {}
for article in articles:
    slug, kind, tids = article["slug"], article["kind"], article["teams"]
    main = tids[0]
    focus = sorted(set(article["focus"]) | {gid for ids, _ in key_claims.get(slug, []) for gid in ids})
    claimed = set(focus)
    owned_slate = sorted({gid for tid in tids for gid in standings[tid]["remaining_game_ids"]})
    cited_schedule = sorted(set(article["sched"]))
    assert claimed <= all_recent and set(cited_schedule) <= set(owned_slate)
    proof = [
        f"team_record.csv, teams.csv and divisions.csv, team IDs {fmt(tids)}. Compare the sealed August 19 and August 30 standings. See the club entries in the companion challenge ledger.",
        f"games.csv, completed game IDs {fmt(focus)}. Game dates, scores, opponents and inning counts. The companion challenge ledger lists the entire comparable window by club.",
        f"games.csv, scheduled game IDs {fmt(cited_schedule)}. The companion ledger carries the full remaining schedule and venue for each club.",
    ]
    if article["players"]:
        proof.append(f"players.csv, players_game_batting.csv and players_game_pitching_stats.csv, player IDs {fmt(article['players'])}. Retain the reconciled game lines for each player and check the stated August 20–30 or single-game statistic against the companion ledger.")
    for gid in article["logs"]:
        proof.append(f"game_logs.csv, game_id={gid}, {log_refs[gid]}. Cross-check the innings and scoring in games_score.csv, game_id={gid}.")
    if kind == "beat":
        proof.extend(trade_refs.get(main, []))
    if kind == "league":
        proof.append("trade_history.csv, message_id=19901, date=1981-07-31, team IDs 20,12, player ID 854. Compare saves with the player pitching lines.")
        proof.append("trade_history.csv, message_id=19877, date=1981-07-29, team IDs 1,2, player ID 1849. Compare RBI with the player batting lines.")
    if kind != "beat" or main in (1, 5, 12, 13, 18, 19, 21, 22, 24):
        proof.append("league_playoffs.csv, league_id=200. Apply docs/ABL_POSTSEASON_RULES.md for wild-card calculations and 1981 DCS protection.")
    if kind != "beat" or main in (7, 8, 15):
        proof.append("leagues.csv, league_id=200, roster_expand_date=1981-9-1, rosters_expanded=0, active_roster_limit=26, expanded_roster_limit=40. Compare future exports of players_roster_status.csv and box scores before naming any call-up.")
    if kind != "beat" or main in (1, 10, 12, 23):
        proof.append("Sealed August 20 Special Coverage briefing and The Pennant Races Come Home. Use the forecast and schedule as pre-challenge evidence. Compare results to the August 30 cutoff.")
    if kind == "league":
        article["sections"] = [(heading, ([*board(sections[heading]), *paragraphs] if heading in sections else paragraphs))
                               for heading, paragraphs in article["sections"]]
    thread_ids = [f"{tid}:1981:race" for tid in tids]
    if kind == "beat":
        thread_ids += [key for key in threads if any(key in (f"{main}:1981:player:{pid}:bat", f"{main}:1981:player:{pid}:pit")
                                                     for pid in article["players"])]
    completed_claims = [dict(games=ids, claim=claim) for ids, claim in key_claims.get(slug, [])]
    for gid in focus:
        game = games[gid]
        completed_claims.append(dict(games=[gid], claim=f"{game['date']}. {club_name(game['visitor'])} {game['away_score']}, {club_name(game['home'])} {game['home_score']}."))
    doc = dict(**asof, slug=slug, title=article["title"], kind=kind, byline=article["byline"], team_ids=tids,
               story_ids=thread_ids, lead=kind=="observer", sections=article["sections"], evidence=proof,
               claim_games=sorted(claimed | set(cited_schedule)), claim_map=completed_claims,
               review=dict(decision="pending", reviewer="", copy_sha256="", question=article["summary"],
                           promise=article["title"], sequence="Read the window chronologically and verify against the dated selectors.",
                           stakes=article["summary"], ending=article["sections"][-1][1][-1]))
    digest = copy_hash(doc)
    # Verify exact draft through the review validator without storing an approval.
    sample = copy.deepcopy(doc)
    sample["review"].update(decision="approve", reviewer="In-memory validation only", copy_sha256=digest)
    validation[slug] = dict(copy_sha256=digest, **checked_review(sample, verification, checker, threads))
    try:
        checked_review(doc, verification, checker, threads)
    except ValueError as error:
        assert "approval" in str(error)
    else:
        raise AssertionError(f"Pending copy unexpectedly qualifies for release: {slug}")
    copies[slug] = doc

out.mkdir(exist_ok=True)
for slug, doc in copies.items():
    write(out / f"{slug}.json", doc)
proof = dict(**asof, baseline_source_id=earlier["source_id"], finals_before=1450, finals_after=1571,
             postseason=dict(wild_cards_per_conference=1, rule=relative(rules), nbc_division_seeds=[6,4,2],
                             nbc_wild_card=1, nbc_dcs=[[6,2],[4,1]], abc_division_seeds=[23,17,20],
                             abc_wild_card_tie=[13,24], abc_other_dcs=[17,20]),
             standings=standings, intervening_standings=checkpoints, club_history_1980=history,
             finals={gid: games[gid] for gid in window},
             scheduled={gid: g for gid, g in games.items() if g["home_score"] is None}, player_game_lines=player_lines,
             player_window_totals=[dict(table=file, team_id=tid, player_id=pid, name=names[pid], stats=dict(stats))
                                   for (file,tid,pid), stats in sorted(totals.items())],
             roster_expansion=expansion, article_checks=validation)
write(out / "challenge-evidence.json", proof)

h = escape
a = lambda href, label: f'<a href="{h(href, quote=True)}">{h(label)}</a>'
body = ('<header><p class="date">ABL Newsroom · August 31, 1981 · Games through August 30</p>'
        '<h1>After the August Division Challenges</h1><p>Proposed edition for commissioner review. '
        '1,571 verified finals · 26 manuscripts · Approval pending</p></header>')
body += ("<p>" + a("challenge-evidence.json", "Reporting ledger") + " · " +
         a(relative(brief), "August 20 briefing") + " · " + a(relative(observer), "The Pennant Races Come Home") +
         " · " + a(relative(rules), "Postseason rules") + " · " + a(relative(handoff), "Editorial handoff") + "</p>")

def lineup(title, selected, anchor):
    return (f'<section id="{anchor}"><h2>{h(title)}</h2><ul>' + "".join(
        f'<li>{a("#"+item["slug"], item["title"])}<br>{h(item["summary"])}</li>'
        for item in selected) + "</ul></section>")

body += lineup("The Observer and The Desk", articles[:2], "league-coverage")
body += '<h2>Club beats</h2><div class="divisions">' + "".join(
    lineup(division, [item for item in articles if item["kind"] == "beat" and after[item["teams"][0]]["division"] == division],
           division.lower().replace(" division","").replace("ern","").replace(" ","-"))
    for division in divisions) + "</div>"
for item in articles:
    slug = item["slug"]
    doc = copies[slug]
    article_body = "".join((f"<h3>{h(heading)}</h3>" if heading else "") +
                           "".join(f"<p>{h(p)}</p>" for p in paragraphs)
                           for heading, paragraphs in doc["sections"])
    evidence = "".join(f"<li>{h(ref)}</li>" for ref in doc["evidence"])
    body += (f'<article id="{slug}"><hr><p class="date">{h(doc["byline"])}</p><h2>{h(doc["title"])}</h2>'
             f'{article_body}<details><summary>Reporting record</summary><ul>{evidence}</ul>'
             f'<p>Exact draft fingerprint <code>{validation[slug]["copy_sha256"]}</code></p></details>'
             f'<p class="meta">{a(slug+".json", "Manuscript and evidence")}</p></article>')
target = out / "index.html"
target.write_text(page("August 31 editorial review", body), encoding="utf-8")
assert all(path.read_bytes() == original for path, original in baseline.items())
assert (archive / "current.json").read_bytes() == cur_pointer and pointer["publication"] is None
print("26 pending manuscripts passed source, editing and correction-map checks.")
print("1450 prior finals, 121 additions, 1571 cumulative finals. All 24 standings and schedule totals agree.")
print("Previous editions, sealed priority manuscripts and EAB homepage were unchanged.")
print(target)
