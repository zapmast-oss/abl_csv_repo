import datetime
from textwrap import dedent

def main():
    today = datetime.date.today()
    header_date = today.strftime("%B %d, %Y")

    text = dedent(f"""\
    [b][size=150]Around the ABL – Weekly Notebook[/size][/b]
    [i]Season 1981 snapshot – standings, streaks, stars and storylines. (Generated {header_date})[/i]

    [b]Top of the mountain[/b]

    (Use this section to highlight the top teams in the standings and their run differentials.)

    • Example: [b]Las Vegas Gamblers (24–8, .750, +59)[/b] – short note here about why they’re on top.
    • Example: [b]Charlotte Colonels (22–10, .688, +50)[/b] – note about their dominance in the NBC East.
    • Example: [b]Dallas Rustlers (19–13, .594, +28)[/b] – note about their push in the NBC Central.
    • Add 1–3 more contenders based on the current standings.

    [b]Power rankings snapshot (forum formula)[/b]

    (Summarize your A+B+C+D power rankings here – from the forum formula script.)

    1) [b]Team[/b] – total power [b]000.0[/b]
    2) [b]Team[/b] – [b]000.0[/b]
    3) [b]Team[/b] – [b]000.0[/b]
    4–6) [b]Team, Team, Team[/b] – all bunched together around [b]000.0[/b]
    7–10) Fill in as needed from the power ranking output.

    [b]Heating up & cooling off – streaks[/b]

    (Use the output from your streaks script.)

    • Hottest team: [b]Team[/b], on a [b]X-game winning streak[/b].
    • Other hot teams: list clubs on 2–3 game winning streaks.
    • Coldest team: [b]Team[/b], on a [b]X-game losing streak[/b].
    • Any others on mini-skids worth mentioning.

    [b]Bats of the week – league’s hottest hitters[/b]

    (Use the top hitters from abl_player_leaders.py.)

    • [b]Player (TEAM)[/b] – AVG .000, HR, RBI, OPS 0.000 (PA NNN).
    • [b]Player (TEAM)[/b] – AVG .000, HR, RBI, OPS 0.000 (PA NNN).
    • [b]Player (TEAM)[/b] – AVG .000, HR, RBI, OPS 0.000 (PA NNN).
    • Add 2–3 more blurbs from the leaderboard.

    [b]Arms of the week – rotation aces[/b]

    (Use the top pitchers from abl_player_leaders.py.)

    • [b]Pitcher (TEAM)[/b] – W–L, ERA 0.00 in NN.N IP, K, WHIP 0.00.
    • [b]Pitcher (TEAM)[/b] – W–L, ERA 0.00 in NN.N IP, K, WHIP 0.00.
    • [b]Pitcher (TEAM)[/b] – W–L, ERA 0.00 in NN.N IP, K, WHIP 0.00.
    • Add 2–3 more blurbs.

    [b]Fun with speed & strikeouts[/b]

    (Use abl_fun_leaders.py results.)

    [b]Top base stealers[/b]  
    • [b]Runner (TEAM)[/b]: SB, CS, SB% 0.000, AVG .000.
    • [b]Runner (TEAM)[/b]: SB, CS, SB% 0.000, AVG .000.
    • [b]Runner (TEAM)[/b]: SB, CS, SB% 0.000, AVG .000.
    • Add 1–2 more from the list.

    [b]Strikeout artists[/b]  
    • [b]Pitcher (TEAM)[/b]: K in NN.N IP, K/9 0.0, ERA 0.00.
    • [b]Pitcher (TEAM)[/b]: K in NN.N IP, K/9 0.0, ERA 0.00.
    • [b]Pitcher (TEAM)[/b]: K in NN.N IP, K/9 0.0, ERA 0.00.
    • Add 1–2 more.

    [b]News & notes[/b]

    [b]Recent trades[/b]  
    (Copy or summarize the ‘Recent Trades’ section from abl_news_notes.py here.)

    • Example: Team traded Player A to Team B for Player C (and picks / salary retention if needed).
    • Add a couple more if you’ve had an active week.

    [b]Recent injuries[/b]  
    (Copy the ‘Recent Injuries’ lines from abl_news_notes.py.)

    • Example: Team – Player (out X days)
    • List 3–5 notable injuries.

    [b]Looking ahead[/b]

    (Use this to tease your featured matchup – e.g., Chicago Fire at Miami Hurricanes.)

    Mention why the matchup matters:
    • Standings implications (division race, wildcard race).
    • Hot players involved (from your team reports).
    • Manager matchup flavor (from abl_manager_matchup.py).

    Wrap with a line inviting readers/viewers to check out the stream or follow along.
    """)

    print(text)

if __name__ == "__main__":
    main()
