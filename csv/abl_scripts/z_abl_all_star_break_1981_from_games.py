from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


LEAGUE_ID = "200"
SEASON = 1981

SCRIPT_PATH = Path(__file__).resolve()
CSV_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[2]
RAW_DIR = CSV_ROOT / "ootp_csv"
OUT_DIR = CSV_ROOT / "out"

GAMES_CSV = RAW_DIR / "games.csv"
TEAMS_CSV = RAW_DIR / "teams.csv"
SUB_LEAGUES_CSV = RAW_DIR / "sub_leagues.csv"
DIVISIONS_CSV = RAW_DIR / "divisions.csv"

LEAGUE_REPORT = OUT_DIR / "league_report_abl_1981_asb_0715.md"
STORY_PACK = OUT_DIR / "abl_1981_act3_story_pack.md"
BALANCE_REPORT = OUT_DIR / "abl_1981_nbc_vs_abc_balance.md"
VIDEO_OUTLINE = OUT_DIR / "video_outline_abl_1981_second_half_preview.md"

CONTENDER_ORDER = [
    "CHA",
    "DAL",
    "SF",
    "CHI",
    "DET",
    "MIA",
    "PHO",
    "LA",
    "LV",
    "HOU",
    "BOS",
    "SEA",
    "DEN",
    "PHI",
    "NY",
    "CIN",
]

STORY_HOOKS = {
    "PHI": "Philadelphia Fury are the only ABL franchise never to make the playoffs.",
    "CHI": "Chicago Fire have made five playoff appearances with no title.",
    "DET": "Detroit Dukes are defending champions.",
    "CHA": "Charlotte Colonels made their first playoff appearance last year and lost to Chicago in six.",
    "LV": "Las Vegas made the playoffs in 1972, missed for years, returned last year, and is now a frontrunner.",
    "DAL": "Dallas's only winning season was the 1977 championship run.",
    "NY": "New York is in life after Mike Shoulders.",
    "CIN": "Cincinnati has Scott Reis, R-E-I-S, chasing a title.",
}


@dataclass
class Team:
    team_id: str
    abbr: str
    name: str
    conference_id: str
    conference: str
    conference_abbr: str
    division_id: str
    division: str


@dataclass
class Game:
    game_id: str
    date: datetime
    home_team: str
    away_team: str
    away_runs: int
    home_runs: int


@dataclass
class Record:
    team: Team
    wins: int = 0
    losses: int = 0
    runs_scored: int = 0
    runs_allowed: int = 0
    home_wins: int = 0
    home_losses: int = 0
    road_wins: int = 0
    road_losses: int = 0
    results: list[tuple[datetime, str]] = field(default_factory=list)
    division_rank: int | None = None
    division_gb: float = 0.0
    wildcard_rank: int | None = None
    wildcard_gb: float = 0.0

    @property
    def games(self) -> int:
        return self.wins + self.losses

    @property
    def pct(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def run_diff(self) -> int:
        return self.runs_scored - self.runs_allowed

    @property
    def record(self) -> str:
        return f"{self.wins}-{self.losses}"

    @property
    def home_record(self) -> str:
        return f"{self.home_wins}-{self.home_losses}"

    @property
    def road_record(self) -> str:
        return f"{self.road_wins}-{self.road_losses}"

    @property
    def last10(self) -> str:
        recent = sorted(self.results, key=lambda item: item[0])[-10:]
        if not recent:
            return "TODO_DERIVE"
        wins = sum(1 for _, result in recent if result == "W")
        return f"{wins}-{len(recent) - wins}"

    @property
    def streak(self) -> str:
        recent = sorted(self.results, key=lambda item: item[0], reverse=True)
        if not recent:
            return "TODO_DERIVE"
        first = recent[0][1]
        count = 0
        for _, result in recent:
            if result != first:
                break
            count += 1
        return f"{first}{count}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_date(raw: str) -> datetime:
    for fmt in ("%Y-%m-%d", "%Y-%m-%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            pass
    # OOTP often writes dates as 1981-7-12.
    return datetime.strptime(raw, "%Y-%m-%d")


def fmt_gb(value: float) -> str:
    if abs(value) < 0.001:
        return "-"
    if value == int(value):
        return f"{int(value)}.0"
    return f"{value:.1f}"


def fmt_pct(value: float) -> str:
    return f"{value:.3f}".replace("0.", ".")


def standings_sort(record: Record) -> tuple[float, int, int, str]:
    return (-record.pct, -record.wins, record.losses, record.team.abbr)


def gb_behind(leader: Record, team: Record) -> float:
    return ((leader.wins - team.wins) + (team.losses - leader.losses)) / 2


def load_metadata() -> tuple[dict[str, Team], dict[tuple[str, str], str], dict[tuple[str, str], tuple[str, str]]]:
    sub_leagues: dict[tuple[str, str], tuple[str, str]] = {}
    for row in read_csv(SUB_LEAGUES_CSV):
        if row["league_id"] == LEAGUE_ID:
            sub_leagues[(row["league_id"], row["sub_league_id"])] = (row["name"], row["abbr"])

    divisions: dict[tuple[str, str, str], str] = {}
    for row in read_csv(DIVISIONS_CSV):
        if row["league_id"] == LEAGUE_ID:
            divisions[(row["league_id"], row["sub_league_id"], row["division_id"])] = row["name"]

    teams: dict[str, Team] = {}
    for row in read_csv(TEAMS_CSV):
        if row["league_id"] != LEAGUE_ID:
            continue
        if row.get("allstar_team") == "1":
            continue
        conf_name, conf_abbr = sub_leagues[(row["league_id"], row["sub_league_id"])]
        div_name = divisions[(row["league_id"], row["sub_league_id"], row["division_id"])]
        full_name = f"{row['name']} {row['nickname']}".strip()
        teams[row["team_id"]] = Team(
            team_id=row["team_id"],
            abbr=row["abbr"],
            name=full_name,
            conference_id=row["sub_league_id"],
            conference=conf_name,
            conference_abbr=conf_abbr,
            division_id=row["division_id"],
            division=div_name,
        )
    return teams, divisions, sub_leagues


def load_games(teams: dict[str, Team]) -> tuple[list[Game], datetime]:
    games: list[Game] = []
    for row in read_csv(GAMES_CSV):
        if row["league_id"] != LEAGUE_ID or row["played"] != "1" or row["game_type"] != "0":
            continue
        date = parse_date(row["date"])
        if date.year != SEASON:
            continue
        if row["home_team"] not in teams or row["away_team"] not in teams:
            continue
        games.append(
            Game(
                game_id=row["game_id"],
                date=date,
                home_team=row["home_team"],
                away_team=row["away_team"],
                away_runs=int(row["runs0"]),
                home_runs=int(row["runs1"]),
            )
        )
    if not games:
        raise RuntimeError("No completed 1981 regular-season ABL games found in games.csv")
    latest_date = max(game.date for game in games)
    return [game for game in games if game.date <= latest_date], latest_date


def build_records(teams: dict[str, Team], games: list[Game]) -> dict[str, Record]:
    records = {team_id: Record(team=team) for team_id, team in teams.items()}
    for game in sorted(games, key=lambda item: (item.date, int(item.game_id))):
        away = records[game.away_team]
        home = records[game.home_team]

        away.runs_scored += game.away_runs
        away.runs_allowed += game.home_runs
        home.runs_scored += game.home_runs
        home.runs_allowed += game.away_runs

        away_won = game.away_runs > game.home_runs
        if away_won:
            away.wins += 1
            away.road_wins += 1
            away.results.append((game.date, "W"))
            home.losses += 1
            home.home_losses += 1
            home.results.append((game.date, "L"))
        else:
            home.wins += 1
            home.home_wins += 1
            home.results.append((game.date, "W"))
            away.losses += 1
            away.road_losses += 1
            away.results.append((game.date, "L"))
    return records


def assign_ranks(records: dict[str, Record]) -> None:
    division_groups: dict[tuple[str, str], list[Record]] = {}
    conference_groups: dict[str, list[Record]] = {}
    for record in records.values():
        division_groups.setdefault((record.team.conference_id, record.team.division_id), []).append(record)
        conference_groups.setdefault(record.team.conference_id, []).append(record)

    division_winners: set[str] = set()
    for group in division_groups.values():
        ordered = sorted(group, key=standings_sort)
        leader = ordered[0]
        division_winners.add(leader.team.team_id)
        for idx, record in enumerate(ordered, start=1):
            record.division_rank = idx
            record.division_gb = gb_behind(leader, record)

    for group in conference_groups.values():
        wildcard_pool = [record for record in group if record.team.team_id not in division_winners]
        ordered = sorted(wildcard_pool, key=standings_sort)
        if not ordered:
            continue
        leader = ordered[0]
        for idx, record in enumerate(ordered, start=1):
            record.wildcard_rank = idx
            record.wildcard_gb = gb_behind(leader, record)


def make_table(rows: list[list[str]]) -> str:
    return "\n".join(" | ".join(row) for row in rows)


def record_line(record: Record) -> str:
    return (
        f"{record.team.abbr} | {record.team.name} | {record.record} | {fmt_pct(record.pct)} | "
        f"{fmt_gb(record.division_gb)} | {record.run_diff:+d} | {record.last10} | "
        f"{record.streak} | {record.home_record} | {record.road_record}"
    )


def division_notes(division_records: list[Record]) -> str:
    ordered = sorted(division_records, key=standings_sort)
    leader = ordered[0]
    challengers = [record for record in ordered[1:] if record.division_gb <= 6]
    if not challengers:
        return f"- The {leader.team.name} have separation in the {leader.team.conference_abbr} {leader.team.division}."
    names = ", ".join(f"{record.team.abbr} {fmt_gb(record.division_gb)} GB" for record in challengers[:3])
    return f"- The {leader.team.name} lead the {leader.team.conference_abbr} {leader.team.division}; chase group: {names}."


def wildcard_notes(conference_records: list[Record]) -> str:
    pool = [record for record in conference_records if record.wildcard_rank is not None]
    ordered = sorted(pool, key=lambda record: (record.wildcard_rank or 999, standings_sort(record)))
    if not ordered:
        return "- TODO_DERIVE wildcard picture."
    top = ", ".join(record.team.abbr for record in ordered[:4])
    conference_abbr = conference_records[0].team.conference_abbr
    if conference_abbr == "NBC":
        return f"- The NBC wildcard field is crowded behind the division leaders, led by {top}."
    if conference_abbr == "ABC":
        return f"- The ABC wildcard field is led by {top} entering the second half."
    return f"- The {conference_abbr} wildcard field is led by {top} entering the second half."


def write_league_report(records: dict[str, Record], latest_date: datetime) -> None:
    lines: list[str] = [
        "# Action Baseball League - 1981 All-Star Break Report",
        f"Generated from raw OOTP games.csv through {latest_date:%Y-%m-%d}.",
        "",
        "Source: csv/ootp_csv/games.csv",
        "Rule: completed 1981 regular-season ABL games only.",
        "",
        "## Standings By Conference And Division",
    ]

    by_conf: dict[str, list[Record]] = {}
    by_division: dict[tuple[str, str], list[Record]] = {}
    for record in records.values():
        by_conf.setdefault(record.team.conference_id, []).append(record)
        by_division.setdefault((record.team.conference_id, record.team.division_id), []).append(record)

    for conf_id in sorted(by_conf):
        conf_name = by_conf[conf_id][0].team.conference
        lines.extend(["", f"### {conf_name}"])
        div_ids = sorted({record.team.division_id for record in by_conf[conf_id]})
        for div_id in div_ids:
            division_records = by_division[(conf_id, div_id)]
            div_name = division_records[0].team.division
            lines.extend(
                [
                    "",
                    f"#### {div_name}",
                    "ABBR | Team | W-L | PCT | GB | RD | L10 | STRK | Home | Road",
                    "--- | --- | --- | --- | --- | --- | --- | --- | --- | ---",
                ]
            )
            for record in sorted(division_records, key=standings_sort):
                lines.append(record_line(record))

    lines.extend(["", "## Wildcard Standings"])
    for conf_id in sorted(by_conf):
        conf_name = by_conf[conf_id][0].team.conference
        lines.extend(
            [
                "",
                f"### {conf_name}",
                "WC | ABBR | Team | W-L | PCT | GB | RD | L10 | STRK",
                "--- | --- | --- | --- | --- | --- | --- | --- | ---",
            ]
        )
        pool = [record for record in by_conf[conf_id] if record.wildcard_rank is not None]
        for record in sorted(pool, key=lambda item: item.wildcard_rank or 999):
            lines.append(
                f"{record.wildcard_rank} | {record.team.abbr} | {record.team.name} | {record.record} | "
                f"{fmt_pct(record.pct)} | {fmt_gb(record.wildcard_gb)} | {record.run_diff:+d} | "
                f"{record.last10} | {record.streak}"
            )

    lines.extend(["", "## Division Race Notes"])
    for key in sorted(by_division):
        lines.append(division_notes(by_division[key]))

    lines.extend(["", "## Wildcard Race Notes"])
    for conf_id in sorted(by_conf):
        lines.append(wildcard_notes(by_conf[conf_id]))

    LEAGUE_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def act3_question(record: Record) -> str:
    questions = {
        "CHA": "Can Charlotte turn last year's first taste of October into a real title push?",
        "DAL": "Is Dallas building another one-year lightning strike like 1977, or something sturdier?",
        "SF": "Can San Francisco convert contender shape into a clean second-half path?",
        "CHI": "Can Chicago finally turn repeated playoff trips into a championship ending?",
        "DET": "Can the defending champs find enough second-half gear to protect the crown?",
        "MIA": "Does Miami have enough consistency to stay in the NBC race?",
        "PHO": "Can Phoenix keep enough pressure on the wildcard field to matter into September?",
        "LA": "Can Los Angeles make its second-half talent show up in the standings?",
        "LV": "Can Las Vegas carry frontrunner pressure after years outside the picture?",
        "HOU": "Can Houston make the ABC race run through the Central?",
        "BOS": "Can Boston keep its floor high enough to survive the wildcard churn?",
        "SEA": "Can Seattle make the ABC West more than a Las Vegas story?",
        "DEN": "Can Denver turn its form into a real October lane?",
        "PHI": "Can Philadelphia finally end the franchise's playoff drought?",
        "NY": "Who are the Aces in life after Mike Shoulders?",
        "CIN": "Can Scott Reis, R-E-I-S, drag Cincinnati into a title chase?",
    }
    return questions.get(record.team.abbr, "TODO_VERIFY Act 3 question.")


def write_story_pack(records: dict[str, Record], latest_date: datetime) -> None:
    by_abbr = {record.team.abbr: record for record in records.values()}
    lines = [
        "# ABL 1981 Act 3 Story Pack",
        f"Raw-games snapshot through {latest_date:%Y-%m-%d}.",
        "",
        "Canon hooks:",
        "- NBC has won 9 of 10 All-Star Games.",
        "- NBC has won 6 of 9 championships.",
        "- Central question: Can the ABC get off the schneid?",
        "",
        "## 16 Roads To October",
    ]
    for abbr in CONTENDER_ORDER:
        record = by_abbr[abbr]
        wc = f"WC {record.wildcard_rank}, {fmt_gb(record.wildcard_gb)} GB" if record.wildcard_rank else "Division leader"
        hook = STORY_HOOKS.get(abbr, "TODO_VERIFY franchise hook.")
        lines.extend(
            [
                "",
                f"### {record.team.name} ({record.team.abbr})",
                f"- Record: {record.record} ({fmt_pct(record.pct)})",
                f"- Division standing: {record.team.conference_abbr} {record.team.division}, "
                f"rank {record.division_rank}, {fmt_gb(record.division_gb)} GB",
                f"- Wildcard position: {wc}",
                f"- Run differential: {record.run_diff:+d}",
                f"- Current form: last 10 {record.last10}, streak {record.streak}, home {record.home_record}, road {record.road_record}",
                f"- Story hook: {hook}",
                f"- Act 3 question: {act3_question(record)}",
                "- EB angle placeholder: TODO_VERIFY",
            ]
        )
    STORY_PACK.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_balance_report() -> None:
    lines = [
        "# ABL 1981 NBC vs ABC Balance",
        "",
        "- 1981 All-Star Game: NBC 7, ABC 0",
        "- NBC dominance: 9 of 10 All-Star Games",
        "- NBC dominance: 6 of 9 championships",
        "- Main question: Can the ABC get off the schneid?",
        "",
        "## ABC Hopes",
        "Las Vegas, Houston, Seattle, Denver, Cincinnati, Boston, Philadelphia, New York",
        "",
        "## NBC Threats",
        "Charlotte, Dallas, San Francisco, Chicago, Detroit, Miami, Phoenix, Los Angeles",
    ]
    BALANCE_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_video_outline() -> None:
    lines = [
        "# ABL 1981 Second Half Preview Video Outline",
        "",
        "## Open",
        "- Set the table at the All-Star break.",
        "",
        "## All-Star Game Recap",
        "- 1981 All-Star Game: NBC 7, ABC 0",
        "",
        "## NBC Dominance / ABC Challenge",
        "- NBC has won 9 of 10 All-Star Games.",
        "- NBC has won 6 of 9 championships.",
        "- Main question: Can the ABC get off the schneid?",
        "",
        "## 16 Roads To October",
        "- NBC: CHA, DAL, SF, CHI, DET, MIA, PHO, LA",
        "- ABC: LV, HOU, BOS, SEA, DEN, PHI, NY, CIN",
        "",
        "## Division Race Map",
        "- Use league_report_abl_1981_asb_0715.md standings.",
        "",
        "## Wildcard Map",
        "- Use league_report_abl_1981_asb_0715.md wildcard standings.",
        "",
        "## Hotter Than July Setup",
        "- TODO_VERIFY segment title beats and featured teams.",
        "",
        "## Close",
        "- Reset the second-half stakes and October paths.",
    ]
    VIDEO_OUTLINE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    teams, _, _ = load_metadata()
    games, latest_date = load_games(teams)
    records = build_records(teams, games)
    assign_ranks(records)

    write_league_report(records, latest_date)
    write_story_pack(records, latest_date)
    write_balance_report()
    write_video_outline()

    by_abbr = {record.team.abbr: record for record in records.values()}
    print("ALL_STAR_BREAK_1981_FROM_GAMES: wrote outputs")
    print(f"- {LEAGUE_REPORT}")
    print(f"- {STORY_PACK}")
    print(f"- {BALANCE_REPORT}")
    print(f"- {VIDEO_OUTLINE}")
    print("")
    print("Verification:")
    for abbr in ("CHA", "DAL", "LV", "HOU"):
        record = by_abbr[abbr]
        print(f"{record.team.name} ({abbr}) record: {record.record}")
    print(f"latest completed regular-season date used: {latest_date:%Y-%m-%d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
