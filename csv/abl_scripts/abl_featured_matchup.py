from datetime import datetime
from pathlib import Path
import pandas as pd

# ---------------- CONFIG ---------------------------------------

DEFAULT_SEASON_YEAR = 1981
ABL_LEAGUE_ID = 200
REGULAR_SEASON_SPLIT_ID = 1

# ---------------- FILES (same folder as this script) ------------

TEAMS_CSV = "teams.csv"
TEAM_RECORD_CSV = "team_record.csv"
PLAYERS_CSV = "players.csv"
PROJ_SP_CSV = "projected_starting_pitchers.csv"
TEAM_BATTING_STATS_CSV = "team_batting_stats.csv"
TEAM_PITCHING_STATS_CSV = "team_pitching_stats.csv"
PLAYERS_GAME_PITCHING_STATS_CSV = "players_game_pitching_stats.csv"
PLAYER_BAT_WAR_CSV = "players_career_batting_stats.csv"
PLAYER_PIT_WAR_CSV = "players_career_pitching_stats.csv"
GAMES_CSV = "games.csv"

# ---------------- LOAD BASE DATA --------------------------------

teams = pd.read_csv(TEAMS_CSV)
team_record = pd.read_csv(TEAM_RECORD_CSV)
players = pd.read_csv(PLAYERS_CSV)
proj_sp = pd.read_csv(PROJ_SP_CSV)
team_bat = pd.read_csv(TEAM_BATTING_STATS_CSV)
team_pitch = pd.read_csv(TEAM_PITCHING_STATS_CSV)
player_game_pitch = pd.read_csv(PLAYERS_GAME_PITCHING_STATS_CSV)
player_bat_war = pd.read_csv(PLAYER_BAT_WAR_CSV)
player_pit_war = pd.read_csv(PLAYER_PIT_WAR_CSV)
games = pd.read_csv(GAMES_CSV)

# Map league abbreviations (ABL only) to team ids for quick lookup
abl_abbr_rows = teams[teams["league_id"] == ABL_LEAGUE_ID][["team_id", "abbr"]].dropna(subset=["abbr"])
ABL_TEAM_ABBR = {
    str(row["abbr"]).strip().upper(): int(row["team_id"])
    for _, row in abl_abbr_rows.iterrows()
}
TEAM_ID_TO_ABBR = {
    int(row["team_id"]): str(row["abbr"]).strip().upper()
    for _, row in abl_abbr_rows.iterrows()
}

# Filled later once we know the season year
abl = None
pitcher_stats = {}
current_season_year = DEFAULT_SEASON_YEAR
best_war_players = {}

# ---------------- STATUS: HOT / COLD / OK -----------------------

def classify_status(row):
    win_pct = row["win_pct"]
    rd = row["run_diff"]

    if win_pct >= 0.600 or rd >= 30:
        return "HOT"
    elif win_pct <= 0.400 or rd <= -30:
        return "COLD"
    else:
        return "OK"


def build_abl(season_year: int) -> pd.DataFrame:
    """Build a dataframe with standings, runs, and statuses for the season."""
    abl_teams = teams[teams["league_id"] == ABL_LEAGUE_ID][[
        "team_id", "name", "nickname", "sub_league_id", "division_id"
    ]].copy()

    merged = team_record.merge(
        abl_teams,
        on="team_id",
        how="inner"
    )

    merged = merged.rename(columns={
        "g": "games",
        "w": "wins",
        "l": "losses",
        "pct": "win_pct"
    })

    merged["team_display"] = merged["name"] + " " + merged["nickname"]

    bat = team_bat[
        (team_bat["league_id"] == ABL_LEAGUE_ID) & (team_bat["year"] == season_year)
    ]
    bat_latest = bat.groupby("team_id", as_index=False)["r"].sum().rename(
        columns={"r": "runs_scored"}
    )

    pit = team_pitch[
        (team_pitch["league_id"] == ABL_LEAGUE_ID) & (team_pitch["year"] == season_year)
    ]
    pit_latest = pit.groupby("team_id", as_index=False)["r"].sum().rename(
        columns={"r": "runs_allowed"}
    )

    merged = merged.merge(bat_latest, on="team_id", how="left")
    merged = merged.merge(pit_latest, on="team_id", how="left")

    merged["run_diff"] = merged["runs_scored"] - merged["runs_allowed"]
    merged["status"] = merged.apply(classify_status, axis=1)
    return merged


def format_outs(outs: float) -> str:
    if pd.isna(outs) or outs <= 0:
        return "0.0"
    outs = int(round(outs))
    innings = outs // 3
    remainder = outs % 3
    return f"{innings}.{remainder}"


def build_pitcher_stats(season_year: int):
    season = player_game_pitch[
        (player_game_pitch["year"] == season_year) &
        (player_game_pitch["league_id"] == ABL_LEAGUE_ID) &
        (player_game_pitch["split_id"] == REGULAR_SEASON_SPLIT_ID)
    ]
    if season.empty:
        return {}

    grouped = season.groupby("player_id", as_index=False).agg({
        "w": "sum",
        "l": "sum",
        "s": "sum",
        "g": "sum",
        "gs": "sum",
        "k": "sum",
        "bb": "sum",
        "er": "sum",
        "outs": "sum"
    })

    stats_map = {}
    for row in grouped.itertuples():
        player_id = int(row.player_id)
        outs = float(row.outs) if not pd.isna(row.outs) else 0.0
        earned = float(row.er) if not pd.isna(row.er) else 0.0
        era = None
        if outs > 0:
            era = (earned * 27) / outs
        stats_map[player_id] = {
            "w": safe_int(row.w),
            "l": safe_int(row.l),
            "s": safe_int(row.s),
            "gs": safe_int(row.gs),
            "g": safe_int(row.g),
            "k": safe_int(row.k),
            "bb": safe_int(row.bb),
            "era": era,
            "outs": outs,
            "ip_str": format_outs(outs),
        }
    return stats_map


def build_best_players(season_year: int):
    best = {}

    def update_best(df: pd.DataFrame, source: str):
        subset = df[
            (df["league_id"] == ABL_LEAGUE_ID) &
            (df["year"] == season_year)
        ].copy()
        if subset.empty:
            return
        subset["war"] = pd.to_numeric(subset["war"], errors="coerce")
        subset = subset.dropna(subset=["war"])
        if subset.empty:
            return
        grouped = subset.groupby(["team_id", "player_id"], as_index=False)["war"].sum()
        if grouped.empty:
            return
        idx = grouped.groupby("team_id")["war"].idxmax()
        top = grouped.loc[idx]
        for row in top.itertuples():
            war_val = float(row.war)
            team_id = int(row.team_id)
            player_id = int(row.player_id)
            current = best.get(team_id)
            if current is None or war_val > current["war"]:
                best[team_id] = {
                    "player_id": player_id,
                    "war": war_val,
                    "source": source
                }

    update_best(player_bat_war, "batting")
    update_best(player_pit_war, "pitching")
    return best


# ---------------- HELPERS TO GRAB TEAMS & PITCHERS --------------

def safe_int(value):
    if pd.isna(value):
        return 0
    return int(value)


def get_team_row(team_id: int):
    row = abl[abl["team_id"] == team_id]
    if row.empty:
        raise ValueError(f"No team with team_id={team_id} in ABL records.")
    return row.iloc[0]


def get_team_sp_id(team_id: int):
    row = proj_sp[proj_sp["team_id"] == team_id]
    if row.empty:
        return None
    row = row.iloc[0]
    # Use starter_0 as the primary projected starter
    return row["starter_0"]


def get_player_name(player_id):
    if pd.isna(player_id):
        return "TBD"
    row = players[players["player_id"] == player_id]
    if row.empty:
        return f"Player {int(player_id)}"
    row = row.iloc[0]
    return f"{row['first_name']} {row['last_name']}"


def build_schedule_choices(game_date: str | None):
    if not game_date:
        return []
    day_games = games[
        (games["league_id"] == ABL_LEAGUE_ID) &
        (games["date"] == game_date)
    ]
    choices = []
    for row in day_games.itertuples():
        try:
            away_id = int(row.away_team)
            home_id = int(row.home_team)
        except (TypeError, ValueError):
            continue
        try:
            away_team = get_team_row(away_id)
            home_team = get_team_row(home_id)
        except ValueError:
            continue
        away_abbr = TEAM_ID_TO_ABBR.get(away_id, f"Team {away_id}")
        home_abbr = TEAM_ID_TO_ABBR.get(home_id, f"Team {home_id}")
        desc = (
            f"{away_abbr} @ {home_abbr} – "
            f"{away_team['team_display']} at {home_team['team_display']}"
        )
        choices.append((away_id, home_id, desc))
    return choices


def team_line(row, label):
    # Handle GB safely
    gb_val = row.get("gb", None)
    if pd.isna(gb_val):
        gb_str = "N/A"
    else:
        gb_str = f"{gb_val:.1f}"

    # Handle run_diff safely
    rd_val = row.get("run_diff", None)
    if pd.isna(rd_val):
        rd_str = "N/A"
    else:
        rd_str = f"{int(rd_val)}"

    return (
        f"{label} - {row['team_display']}: "
        f"{int(row['wins'])}-{int(row['losses'])} "
        f"({row['win_pct']:.3f}), "
        f"GB {gb_str}, "
        f"run diff {rd_str}, "
        f"status {row['status']}"
    )


def format_matchup(away_id: int, home_id: int, idx: int) -> str:
    away_team = get_team_row(away_id)
    home_team = get_team_row(home_id)
    away_abbr = TEAM_ID_TO_ABBR.get(int(away_id), "AWAY")
    home_abbr = TEAM_ID_TO_ABBR.get(int(home_id), "HOME")

    away_sp_id = get_team_sp_id(away_id)
    home_sp_id = get_team_sp_id(home_id)
    away_sp_name = get_player_name(away_sp_id)
    home_sp_name = get_player_name(home_sp_id)

    lines = [
        f"--- Featured Matchup #{idx} ---",
        team_line(away_team, "AWAY"),
        team_line(home_team, "HOME"),
        "",
        "Probable starters:",
        format_pitcher_line(away_abbr, away_sp_id, away_sp_name),
        format_pitcher_line(home_abbr, home_sp_id, home_sp_name),
        "",
        "Top WAR players:",
        format_best_player_line(away_id, away_abbr),
        format_best_player_line(home_id, home_abbr),
    ]
    return "\n".join(lines)


def format_pitcher_line(team_label: str, player_id, pitcher_name: str) -> str:
    if player_id is None or pd.isna(player_id):
        return f"{pitcher_name} ({team_label}) – stats unavailable"
    try:
        key = int(player_id)
    except (TypeError, ValueError):
        return f"{pitcher_name} ({team_label}) – stats unavailable"

    stats = pitcher_stats.get(key)
    if stats is None:
        return f"{pitcher_name} ({team_label}) – stats unavailable"

    era = stats["era"]
    era_txt = f"{era:.2f}" if era is not None else "N/A"
    return (
        f"{pitcher_name} ({team_label}) – {stats['w']}-{stats['l']}, "
        f"{era_txt} ERA, {stats['ip_str']} IP, {stats['k']} K"
    )


def format_best_player_line(team_id: int, team_label: str) -> str:
    info = best_war_players.get(team_id)
    if not info:
        return f"{team_label} WAR leader: data unavailable"
    name = get_player_name(info["player_id"])
    war_val = info["war"]
    source = info["source"]
    return (
        f"{team_label} WAR leader ({source}): "
        f"{name} – {war_val:.1f} WAR"
    )


# ---------------- INTERACTION -----------------------------------

def prompt_season_year() -> int:
    while True:
        raw = input(f"Season year [{DEFAULT_SEASON_YEAR}]: ").strip()
        if not raw:
            return DEFAULT_SEASON_YEAR
        try:
            return int(raw)
        except ValueError:
            print("Please enter a numeric season year.")


def prompt_game_date(prompt_text: str = "Game date (YYYY-MM-DD, blank to skip schedule lookup): ") -> str | None:
    while True:
        raw = input(prompt_text).strip()
        if not raw:
            return None
        try:
            year_str, month_str, day_str = raw.split("-")
            dt = datetime(int(year_str), int(month_str), int(day_str))
            return f"{dt.year}-{dt.month}-{dt.day}"
        except ValueError:
            print("Please enter the date as YYYY-MM-DD.")


def prompt_team_id(prompt: str, allow_blank: bool = False):
    while True:
        raw = input(prompt).strip()
        if not raw:
            if allow_blank:
                return None
            print("Team id is required.")
            continue
        try:
            return int(raw)
        except ValueError:
            abbr_key = raw.upper()
            team_id = ABL_TEAM_ABBR.get(abbr_key)
            if team_id is not None:
                return team_id
            print("Unknown team. Enter a numeric id or an ABL abbreviation (e.g., MIA).")


def collect_schedule_matchups():
    entries = []
    while True:
        game_date = prompt_game_date(
            "Add matchups from date (YYYY-MM-DD, blank when done with schedule dates): "
        )
        if not game_date:
            break
        schedule_choices = build_schedule_choices(game_date)
        if not schedule_choices:
            print(f"No ABL games found on {game_date}.")
            continue
        print(f"Games on {game_date}:")
        for idx, (_, _, desc) in enumerate(schedule_choices, start=1):
            print(f"  {idx}. {desc}")
        selected = prompt_schedule_selection(schedule_choices)
        if not selected:
            print(f"No games selected for {game_date}.")
        else:
            entries.extend(selected)
            print(f"Added {len(selected)} matchup(s) from {game_date}.\n")
    return entries


def collect_manual_matchups():
    entries = []
    print("Enter any additional featured matchups manually.")
    print("Press Enter at the AWAY prompt when you're done.\n")

    while True:
        away_id = prompt_team_id("Away team (ID or abbr, blank to finish): ", allow_blank=True)
        if away_id is None:
            break
        home_id = prompt_team_id("Home team (ID or abbr): ")
        entries.append((away_id, home_id))
        print(f"Added matchup: away {away_id} @ home {home_id}\n")
    return entries


def prompt_schedule_selection(schedule_choices):
    if not schedule_choices:
        return []
    while True:
        raw = input("Select matchups by number (comma-separated), 'all' for every game, or Enter to skip: ").strip()
        if not raw:
            return []
        if raw.lower() == "all":
            return [(away, home) for away, home, _ in schedule_choices]
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if not parts:
            return []
        try:
            indices = [int(p) for p in parts]
        except ValueError:
            print("Please enter numbers like 1,3,5 or 'all'.")
            continue
        invalid = [i for i in indices if i < 1 or i > len(schedule_choices)]
        if invalid:
            print(f"Numbers out of range: {', '.join(str(i) for i in invalid)}")
            continue
        seen = set()
        selected = []
        for idx in indices:
            if idx in seen:
                continue
            seen.add(idx)
            away_id, home_id, _ = schedule_choices[idx - 1]
            selected.append((away_id, home_id))
        return selected


def write_reports(blocks, season_year: int):
    if not blocks:
        print("No successful matchups to export.")
        return

    base_name = f"abl_featured_matchups_{season_year}"
    text_content = "\n\n".join(blocks)

    text_path = Path(f"{base_name}.txt")
    rtf_path = Path(f"{base_name}.rtf")
    pdf_path = Path(f"{base_name}.pdf")

    text_path.write_text(text_content, encoding="utf-8")
    rtf_path.write_text(build_rtf_document(text_content), encoding="utf-8")

    pdf_lines = []
    for block in blocks:
        pdf_lines.extend(block.splitlines())
        pdf_lines.append("")
    if pdf_lines and pdf_lines[-1] == "":
        pdf_lines.pop()
    write_simple_pdf(pdf_path, pdf_lines)

    print(f"Saved reports: {text_path.name}, {rtf_path.name}, {pdf_path.name}")


def build_rtf_document(content: str) -> str:
    def escape(text: str) -> str:
        return (
            text.replace("\\", r"\\")
            .replace("{", r"\{")
            .replace("}", r"\}")
        )

    escaped = escape(content).replace("\n", r"\line " + "\n")
    return r"{\rtf1\ansi " + escaped + "}"


def write_simple_pdf(path: Path, lines):
    if not lines:
        lines = ["No featured matchups."]

    max_lines_per_page = 48
    page_chunks = [lines[i:i + max_lines_per_page] for i in range(0, len(lines), max_lines_per_page)]
    if not page_chunks:
        page_chunks = [["No featured matchups."]]

    page_ids = []
    content_ids = []
    next_id = 3  # 1 catalog, 2 pages
    for _ in page_chunks:
        page_ids.append(next_id)
        next_id += 1
        content_ids.append(next_id)
        next_id += 1
    font_id = next_id

    objects = []
    objects.append((1, "<< /Type /Catalog /Pages 2 0 R >>"))

    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects.append((2, f"<< /Type /Pages /Count {len(page_ids)} /Kids [{kids}] >>"))

    for page_id, content_id in zip(page_ids, content_ids):
        body = (
            "<< /Type /Page /Parent 2 0 R "
            "/MediaBox [0 0 612 792] "
            f"/Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>"
        )
        objects.append((page_id, body))

    for chunk, content_id in zip(page_chunks, content_ids):
        content_lines = [
            "BT",
            "/F1 12 Tf",
            "14 TL",
            "72 770 Td"
        ]
        for line in chunk:
            text = line if line.strip() else " "
            content_lines.append(f"({pdf_escape(text)}) Tj")
            content_lines.append("T*")
        content_lines.append("ET")
        stream = "\n".join(content_lines)
        stream_bytes = stream.encode("latin-1", errors="ignore")
        body = f"<< /Length {len(stream_bytes)} >>\nstream\n{stream}\nendstream"
        objects.append((content_id, body))

    font_body = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    objects.append((font_id, font_body))

    objects.sort(key=lambda item: item[0])

    with path.open("wb") as fh:
        fh.write(b"%PDF-1.4\n")
        offsets = {}
        for obj_id, body in objects:
            offsets[obj_id] = fh.tell()
            fh.write(f"{obj_id} 0 obj\n".encode("latin-1"))
            fh.write(to_latin1(body))
            fh.write(b"\nendobj\n")

        start_xref = fh.tell()
        total_objects = objects[-1][0]
        fh.write(f"xref\n0 {total_objects + 1}\n".encode("latin-1"))
        fh.write(b"0000000000 65535 f \n")
        for obj_id in range(1, total_objects + 1):
            pos = offsets.get(obj_id, 0)
            fh.write(f"{pos:010d} 00000 n \n".encode("latin-1"))

        fh.write(
            to_latin1(
                f"trailer\n<< /Size {total_objects + 1} /Root 1 0 R >>\n"
                f"startxref\n{start_xref}\n%%EOF"
            )
        )


def pdf_escape(text: str) -> str:
    sanitized = (
        text.replace("–", "-")
            .replace("—", "--")
            .replace("\\", "\\\\")
            .replace("(", "\\(")
            .replace(")", "\\)")
    )
    return sanitized.encode("latin-1", "replace").decode("latin-1")


def to_latin1(text: str) -> bytes:
    return text.encode("latin-1", "replace")


def main():
    global abl
    global pitcher_stats
    global current_season_year
    global best_war_players
    season_year = prompt_season_year()
    current_season_year = season_year
    abl = build_abl(season_year)
    pitcher_stats = build_pitcher_stats(season_year)
    best_war_players = build_best_players(season_year)

    matchups = []
    matchups.extend(collect_schedule_matchups())
    matchups.extend(collect_manual_matchups())
    if not matchups:
        print("No matchups entered. Exiting.")
        return

    report_blocks = []
    for idx, (away_id, home_id) in enumerate(matchups, start=1):
        try:
            block = format_matchup(away_id, home_id, idx)
            print(block)
            report_blocks.append(block)
        except ValueError as exc:
            print(f"Skipping matchup #{idx}: {exc}")
        print()
    write_reports(report_blocks, season_year)


if __name__ == "__main__":
    main()
