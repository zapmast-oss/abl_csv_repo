import argparse
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

import pandas as pd

from z_abl_almanac_html_helpers import parse_schedule_grid


def _norm_key(value: str) -> str:
    """
    Normalize a team / city string for fuzzy matching:
    - drop parenthetical state (e.g., "Atlanta(Georgia)" -> "Atlanta")
    - strip non-alphanumeric characters
    - lowercase
    """
    import re

    if value is None:
        return ""
    text = str(value)
    # drop anything in parentheses
    text = re.sub(r"\(.*?\)", "", text)
    # strip to alphanumerics
    text = re.sub(r"[^a-z0-9]+", "", text.lower())
    return text


def _build_team_lookup(dim_path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Build lookup dicts:
    - key -> team_abbr
    - key -> pretty_label ("City Team Name")

    Keys include various combinations of Team Name / City / Abbr so that
    schedule-grid headers like "Atlanta" or "Atlanta Kings" map cleanly.
    """
    if not dim_path.exists():
        logging.warning("dim_team_park not found at %s; using raw team labels", dim_path)
        return {}, {}

    dim = pd.read_csv(dim_path)
    required = {"Team Name", "Abbr", "City"}
    if not required.issubset(set(dim.columns)):
        logging.warning(
            "dim_team_park missing expected columns %s; got %s",
            sorted(required),
            list(dim.columns),
        )
        return {}, {}

    abbr_map: Dict[str, str] = {}
    label_map: Dict[str, str] = {}

    for _, row in dim.iterrows():
        team_name = str(row["Team Name"])
        abbr = str(row["Abbr"])
        city_full = "" if pd.isna(row["City"]) else str(row["City"])
        city_short = city_full.split("(")[0].strip() if city_full else ""
        label = f"{city_short} {team_name}".strip() if city_short else team_name

        candidates = {
            team_name,
            abbr,
            city_full,
            city_short,
            f"{city_short} {team_name}".strip(),
            f"{team_name} ({abbr})",
        }

        for cand in candidates:
            if not cand:
                continue
            key = _norm_key(cand)
            if not key:
                continue
            # first come, first served to avoid noisy overwrites
            abbr_map.setdefault(key, abbr)
            label_map.setdefault(key, label)

    return abbr_map, label_map


def _attach_team_metadata(df: pd.DataFrame, dim_path: Path) -> pd.DataFrame:
    """
    Attach canonical team_abbr / team_label columns using dim_team_park.
    If lookup fails or dim is missing, fall back to team_raw.
    """
    abbr_map, label_map = _build_team_lookup(dim_path)
    if not abbr_map:
        df = df.copy()
        df["team_abbr"] = df["team_raw"]
        df["team_label"] = df["team_raw"]
        return df

    abbrs = []
    labels = []
    for raw in df["team_raw"].astype(str):
        key = _norm_key(raw)
        abbr = abbr_map.get(key, raw)
        label = label_map.get(key, raw)
        abbrs.append(abbr)
        labels.append(label)

    df = df.copy()
    df["team_abbr"] = abbrs
    df["team_label"] = labels
    return df


def _build_schedule_context_markdown(df: pd.DataFrame) -> str:
    """
    Turn a parsed, team-annotated schedule grid into a Markdown fragment for EB.

    Rules for "brutal stretch":
    - Look at every 15-day window between Opening Day and the last regular-season game.
    - If a team plays at least 14 games in such a window (no more than 1 off-day),
      that window counts as a brutal stretch for that club.
    - For each team, keep only its single worst window.
    - Report the top 8 brutal stretches league-wide by games played, then by fewest
      off-days, then by earliest start date.
    """
    if df.empty:
        raise ValueError("Empty schedule grid dataframe")

    df_played = df[df["played"]]
    if df_played.empty:
        raise ValueError("No played games in schedule grid")

    opening_date = df_played["date"].min()
    final_date = df_played["date"].max()

    all_dates = pd.date_range(opening_date, final_date, freq="D").date

    per_date = df.groupby("date")["played"].sum().reset_index()
    per_date["played"] = per_date["played"].astype(int)
    per_date["games_any_team"] = per_date["played"]
    per_date = per_date[(per_date["date"] >= opening_date) & (per_date["date"] <= final_date)]
    per_date["has_any_game"] = per_date["games_any_team"] > 0

    league_game_days = int(per_date["has_any_game"].sum())
    league_off_days = int((~per_date["has_any_game"]).sum())

    # Detect All-Star break from July blank run (3 consecutive no-game days)
    july = per_date[per_date["date"].apply(lambda d: d.month == 7)].copy()
    july = july.sort_values("date")
    july_off = july[~july["has_any_game"]]["date"].tolist()

    derby_date = asg_date = extra_off_date = None
    if july_off:
        run = [july_off[0]]
        for d in july_off[1:]:
            if (d - run[-1]).days == 1:
                run.append(d)
            else:
                if len(run) >= 3:
                    break
                run = [d]
        if len(run) >= 3:
            derby_date, asg_date, extra_off_date = run[0], run[1], run[2]

    teams = sorted(df["team_abbr"].unique() if "team_abbr" in df.columns else df["team_raw"].unique())
    team_key = "team_abbr" if "team_abbr" in df.columns else "team_raw"

    # Team-level rest profile
    records: list[dict[str, Any]] = []
    for team in teams:
        tdf = df[(df[team_key] == team) & (df["played"])]
        dates_played = sorted({d for d in tdf["date"] if opening_date <= d <= final_date})
        off_days = [d for d in all_dates if d not in dates_played]
        records.append(
            {
                "team": team,
                "games": len(dates_played),
                "off_days": len(off_days),
            }
        )

    team_df = pd.DataFrame(records)
    most_rest = team_df.sort_values("off_days", ascending=False).head(3)
    least_rest = team_df.sort_values("off_days", ascending=True).head(3)

    # Brutal stretches (see docstring for rule)
    WINDOW_DAYS = 15
    MIN_GAMES = 14
    brutal_rows: list[dict[str, Any]] = []

    for team in teams:
        tdf = df[df[team_key] == team]
        played_map: dict[Any, bool] = {}
        for d, played in zip(tdf["date"], tdf["played"]):
            if opening_date <= d <= final_date:
                played_map[d] = bool(played)

        dates_seq = list(all_dates)
        vals = [1 if played_map.get(d, False) else 0 for d in dates_seq]
        n = len(dates_seq)
        if n < WINDOW_DAYS:
            continue

        window_sum = sum(vals[:WINDOW_DAYS])
        best_games = window_sum
        best_start_idx = 0

        for start in range(1, n - WINDOW_DAYS + 1):
            window_sum += vals[start + WINDOW_DAYS - 1] - vals[start - 1]
            if window_sum > best_games:
                best_games = window_sum
                best_start_idx = start

        if best_games >= MIN_GAMES:
            start_date = dates_seq[best_start_idx]
            end_date = dates_seq[best_start_idx + WINDOW_DAYS - 1]
            brutal_rows.append(
                {
                    "team": team,
                    "games": int(best_games),
                    "window_days": WINDOW_DAYS,
                    "off_days": WINDOW_DAYS - int(best_games),
                    "start_date": start_date,
                    "end_date": end_date,
                }
            )

    brutal_df = pd.DataFrame(brutal_rows)
    if not brutal_df.empty:
        brutal_df = brutal_df.sort_values(
            ["games", "off_days", "start_date"], ascending=[False, True, True]
        ).head(8)

    def fmt_date(d) -> str:
        return f"{d.strftime('%B')} {d.day}, {d.year}"

    lines: list[str] = []

    lines.append("#### Schedule Overview")
    lines.append("")
    lines.append(f"- Opening Day: {fmt_date(opening_date)}")
    lines.append(f"- Final Day: {fmt_date(final_date)}")
    lines.append(f"- League-wide game days: {league_game_days}")
    lines.append(f"- League-wide full off-days (no games at all): {league_off_days}")
    lines.append("")

    lines.append("#### Rest Distribution")
    lines.append("")
    if not team_df.empty:
        lines.append("Most rest (top 3 by off-days):")
        for _, row in most_rest.iterrows():
            lines.append(
                f"- {row['team']}: {int(row['off_days'])} off-days, {int(row['games'])} games"
            )
        lines.append("")

        lines.append("Least rest (bottom 3 by off-days):")
        for _, row in least_rest.iterrows():
            lines.append(
                f"- {row['team']}: {int(row['off_days'])} off-days, {int(row['games'])} games"
            )
        lines.append("")
    else:
        lines.append("(No team-level rest data available.)")
        lines.append("")

    lines.append("#### All-Star Break")
    lines.append("")
    if derby_date and asg_date and extra_off_date:
        lines.append(f"- HR Derby: {fmt_date(derby_date)}")
        lines.append(f"- All-Star Game: {fmt_date(asg_date)}")
        lines.append(f"- Extra off-day: {fmt_date(extra_off_date)}")
    else:
        lines.append("(All-Star break could not be detected from the schedule grid.)")
    lines.append("")

    lines.append("#### Brutal Stretches")
    lines.append("")
    if brutal_df is not None and not brutal_df.empty:
        lines.append(
            f"Defined as any {WINDOW_DAYS}-day window with at least {MIN_GAMES} games "
            f"(no more than {WINDOW_DAYS - MIN_GAMES} off-day)."
        )
        lines.append("")
        for _, row in brutal_df.iterrows():
            lines.append(
                f"- {row['team']}: {int(row['games'])} games in {int(row['window_days'])} days "
                f"({fmt_date(row['start_date'])}–{fmt_date(row['end_date'])}, "
                f"{int(row['off_days'])} off-day{'s' if row['off_days'] != 1 else ''})"
            )
    else:
        lines.append(
            f"(No brutal stretches found under the rule: {WINDOW_DAYS} days, at least {MIN_GAMES} games.)"
        )

    return "\n".join(lines)


def build_schedule_context_fragment(season: int, league_id: int) -> Path:
    """
    Orchestrate schedule-grid parsing and Markdown generation for one season/league.

    Input HTML:
      csv/in/almanac_core/{season}/leagues/league_{league_id}_schedule_grid.html

    Output fragment:
      csv/out/eb/eb_schedule_context_{season}_league{league_id}.md
    """
    base = Path("csv")
    html_path = base / "in" / "almanac_core" / str(season) / "leagues" / f"league_{league_id}_schedule_grid.html"
    dim_path = base / "out" / "star_schema" / "dim_team_park.csv"

    logging.info("Parsing schedule grid from %s", html_path)
    df = parse_schedule_grid(html_path, season=season)

    logging.info("Attaching team metadata from %s", dim_path)
    df = _attach_team_metadata(df, dim_path)

    logging.info("Building schedule context markdown")
    md = _build_schedule_context_markdown(df)

    out_dir = base / "out" / "eb"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eb_schedule_context_{season}_league{league_id}.md"
    out_path.write_text(md, encoding="utf-8")

    logging.info("Wrote schedule context fragment to %s", out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build EB schedule-context fragment from almanac schedule_grid HTML."
    )
    parser.add_argument("--season", type=int, required=True, help="Season year, e.g. 1972 or 1980")
    parser.add_argument(
        "--league-id",
        type=int,
        default=200,
        help="League ID (default 200 for ABL)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    out_path = build_schedule_context_fragment(args.season, args.league_id)
    print(f"[OK] EB schedule context fragment written to: {out_path}")


if __name__ == "__main__":
    main()
