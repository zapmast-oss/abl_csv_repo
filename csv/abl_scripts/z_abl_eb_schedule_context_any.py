import argparse
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

import pandas as pd

from z_abl_almanac_html_helpers import parse_schedule_grid


def _norm_key(value: str) -> str:
    import re

    if value is None:
        return ""
    text = str(value)
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"[^a-z0-9]+", "", text.lower())
    return text


def _build_team_lookup(dim_path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
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
            abbr_map.setdefault(key, abbr)
            label_map.setdefault(key, label)

    return abbr_map, label_map


def _attach_team_metadata(df: pd.DataFrame, dim_path: Path) -> pd.DataFrame:
    abbr_map, label_map = _build_team_lookup(dim_path)
    df = df.copy()

    if not abbr_map:
        df["team_abbr"] = pd.NA
        df["team_label"] = pd.NA
        return df

    team_abbrs: list[str | None] = []
    team_labels: list[str | None] = []

    for raw in df["team_raw"].astype(str):
        key = _norm_key(raw)
        if key in abbr_map:
            team_abbrs.append(abbr_map[key])
            team_labels.append(label_map.get(key, abbr_map[key]))
        else:
            team_abbrs.append(None)
            team_labels.append(None)

    df["team_abbr"] = team_abbrs
    df["team_label"] = team_labels
    return df


def _detect_opening_day(per_date: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    # opening day: first played date after a 3-day league-wide gap; spring_start: earliest played date
    played_dates = per_date[per_date["has_any_game"]]["date"].sort_values().tolist()
    if not played_dates:
        raise ValueError("No games in schedule grid")
    spring_start = played_dates[0]

    # find earliest run of >=3 consecutive off-days before mid-season
    off_dates = per_date[~per_date["has_any_game"]]["date"].sort_values().tolist()
    opening_day = spring_start
    run: list[pd.Timestamp] = []
    for d in off_dates:
        if not run:
            run = [d]
            continue
        if (d - run[-1]).days == 1:
            run.append(d)
        else:
            if len(run) >= 3 and run[-1] < played_dates[-1]:
                opening_day = run[-1] + pd.Timedelta(days=1)
                break
            run = [d]
    if len(run) >= 3 and opening_day == spring_start:
        opening_day = run[-1] + pd.Timedelta(days=1)
    return spring_start, opening_day


def _compute_final_day(df_teams: pd.DataFrame, per_date: pd.DataFrame) -> pd.Timestamp:
    total_teams = df_teams["team_abbr"].nunique()
    counts = df_teams[df_teams["played"]].groupby("date")["team_abbr"].nunique()
    full_dates = counts[counts == total_teams].index.sort_values()
    if full_dates.empty:
        return per_date["date"].max()
    return full_dates.max()


def _detect_all_star(per_date: pd.DataFrame) -> tuple[pd.Timestamp | None, pd.Timestamp | None, pd.Timestamp | None]:
    july = per_date[per_date["date"].apply(lambda d: d.month == 7)].copy().sort_values("date")
    july_off = july[~july["has_any_game"]]["date"].tolist()
    if not july_off:
        return None, None, None
    run = [july_off[0]]
    for d in july_off[1:]:
        if (d - run[-1]).days == 1:
            run.append(d)
        else:
            if len(run) >= 3:
                break
            run = [d]
    if len(run) >= 3:
        return run[0], run[1], run[2]
    return None, None, None


def _detect_playoff_spans(df_teams: pd.DataFrame, final_day: pd.Timestamp) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Identify playoff spans using team-count thresholds on post-season dates:
      - DCS: first date with team_count >=5 through the last such date
      - CCS: after DCS, first date with team_count >=3 through the last such date
      - GS:  after CCS, first date with team_count >=2 through the last such date
    Each span must include at least 4 game days; otherwise it is omitted.
    """
    spans: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    games_after = df_teams[(df_teams["played"]) & (df_teams["date"] > final_day)]
    if games_after.empty:
        return spans

    per_date_counts = (
        games_after.groupby("date")["team_abbr"]
        .nunique()
        .reset_index()
        .rename(columns={"team_abbr": "team_count"})
        .sort_values("date")
    )
    dates = per_date_counts["date"].tolist()

    vals = per_date_counts["team_count"].tolist()

    def span_for_threshold(after_idx: int | None, threshold: int) -> tuple[int | None, int | None]:
        idxs = [
            i
            for i, v in enumerate(vals)
            if v >= threshold and (after_idx is None or i > after_idx)
        ]
        if not idxs:
            return None, None
        first = idxs[0]
        last = idxs[-1]
        if last - first + 1 < 4:
            return None, None
        return first, last

    dcs_first, dcs_last = span_for_threshold(None, 5)
    ccs_first = ccs_last = None
    gs_first = gs_last = None

    if dcs_first is not None and dcs_last is not None:
        spans["DCS"] = (dates[dcs_first], dates[dcs_last])
        ccs_first, ccs_last = span_for_threshold(dcs_last, 3)
    else:
        ccs_first, ccs_last = span_for_threshold(None, 3)

    if ccs_first is not None and ccs_last is not None:
        spans["CCS"] = (dates[ccs_first], dates[ccs_last])
        gs_first, gs_last = span_for_threshold(ccs_last, 2)
    else:
        gs_first, gs_last = span_for_threshold(None, 2)

    if gs_first is not None and gs_last is not None:
        spans["GS"] = (dates[gs_first], dates[gs_last])

    return spans


def _fmt_date(d: pd.Timestamp | None) -> str:
    if d is None:
        return "Unknown"
    return f"{d.strftime('%B')} {d.day}, {d.year}"


def _build_schedule_context_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        raise ValueError("Empty schedule grid dataframe")

    df_played = df[df["played"]]
    if df_played.empty:
        raise ValueError("No played games in schedule grid")

    per_date = df.groupby("date")["played"].sum().reset_index()
    per_date["has_any_game"] = per_date["played"] > 0

    spring_start, opening_day = _detect_opening_day(per_date)

    df_teams = df[df["team_abbr"].notna()].copy()
    final_day = _compute_final_day(df_teams, per_date) if not df_teams.empty else df_played["date"].max()

    # league-wide days between opening and final day
    per_range = per_date[(per_date["date"] >= opening_day) & (per_date["date"] <= final_day)]
    league_game_days = int(per_range["has_any_game"].sum())
    league_off_days = int((~per_range["has_any_game"]).sum())

    derby_date, asg_date, extra_off_date = _detect_all_star(per_range)

    lines: list[str] = []
    lines.append("#### Schedule Overview")
    lines.append("")
    lines.append(f"- Spring Training Opens: {_fmt_date(spring_start)}")
    lines.append(f"- Opening Day: {_fmt_date(opening_day)}")
    lines.append(f"- Final Day: {_fmt_date(final_day)}")
    lines.append(f"- League-wide game days: {league_game_days}")
    lines.append(f"- League-wide full off-days (no games at all): {league_off_days}")
    lines.append("")

    lines.append("#### All-Star Break")
    lines.append("")
    if derby_date and asg_date and extra_off_date:
        lines.append(f"- HR Derby: {_fmt_date(derby_date)}")
        lines.append(f"- All-Star Game: {_fmt_date(asg_date)}")
        lines.append(f"- Extra off-day: {_fmt_date(extra_off_date)}")
    else:
        lines.append("(All-Star break could not be detected from the schedule grid.)")
    lines.append("")

    # Playoffs spans
    if df_teams.empty:
        lines.append("#### Playoffs")
        lines.append("")
        lines.append("(Playoff spans unavailable: no team metadata matched the schedule grid.)")
        lines.append("")
    else:
        spans = _detect_playoff_spans(df_teams, final_day)
        lines.append("#### Playoffs")
        lines.append("")
        if spans:
            for label in ["DCS", "CCS", "GS"]:
                if label in spans:
                    start, end = spans[label]
                    lines.append(f"- {label}: {_fmt_date(start)} - {_fmt_date(end)}")
            lines.append("")
        else:
            lines.append("(No playoff spans detected after the regular season.)")
            lines.append("")

    lines.append("#### Rest Distribution")
    lines.append("")

    if df_teams.empty:
        lines.append("(Team-level rest/brutal stretches unavailable: no team metadata matched the schedule grid.)")
        lines.append("")
    else:
        team_key = "team_abbr"
        teams = sorted(df_teams[team_key].unique())
        all_dates = pd.date_range(opening_day, final_day, freq="D").date

        records: list[dict[str, Any]] = []
        for team in teams:
            tdf = df_teams[(df_teams[team_key] == team) & (df_teams["played"])]
            dates_played = sorted({d for d in tdf["date"] if opening_day <= d <= final_day})
            off_days = [d for d in all_dates if d not in dates_played]
            records.append({"team": team, "games": len(dates_played), "off_days": len(off_days)})

        team_df = pd.DataFrame(records)
        most_rest = team_df.sort_values("off_days", ascending=False).head(3)
        least_rest = team_df.sort_values("off_days", ascending=True).head(3)

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

    lines.append("#### Brutal Stretches")
    lines.append("")
    if df_teams.empty:
        lines.append("(Team-level brutal stretches unavailable: no team metadata matched the schedule grid.)")
        return "\n".join(lines)

    WINDOW_DAYS = 15
    MIN_GAMES = 14
    brutal_rows: list[dict[str, Any]] = []

    team_key = "team_abbr"
    all_dates_seq = list(pd.date_range(opening_day, final_day, freq="D").date)
    n_dates = len(all_dates_seq)

    for team in sorted(df_teams[team_key].unique()):
        tdf = df_teams[df_teams[team_key] == team]
        played_map: dict[Any, bool] = {}
        for d, played in zip(tdf["date"], tdf["played"]):
            if opening_day <= d <= final_day:
                played_map[d] = bool(played)

        vals = [1 if played_map.get(d, False) else 0 for d in all_dates_seq]
        if n_dates < WINDOW_DAYS:
            continue

        window_sum = sum(vals[:WINDOW_DAYS])
        best_games = window_sum
        best_start_idx = 0

        for start in range(1, n_dates - WINDOW_DAYS + 1):
            window_sum += vals[start + WINDOW_DAYS - 1] - vals[start - 1]
            if window_sum > best_games:
                best_games = window_sum
                best_start_idx = start

        if best_games >= MIN_GAMES:
            start_date = all_dates_seq[best_start_idx]
            end_date = all_dates_seq[best_start_idx + WINDOW_DAYS - 1]
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

    if brutal_df is not None and not brutal_df.empty:
        lines.append(
            f"Defined as any {WINDOW_DAYS}-day window with at least {MIN_GAMES} games "
            f"(no more than {WINDOW_DAYS - MIN_GAMES} off-day)."
        )
        lines.append("")
        for _, row in brutal_df.iterrows():
            lines.append(
                f"- {row['team']}: {int(row['games'])} games in {int(row['window_days'])} days "
                f"({_fmt_date(row['start_date'])} - {_fmt_date(row['end_date'])}, "
                f"{int(row['off_days'])} off-day{'s' if row['off_days'] != 1 else ''})"
            )
    else:
        lines.append(
            f"(No brutal stretches found under the rule: {WINDOW_DAYS} days, at least {MIN_GAMES} games.)"
        )

    return "\n".join(lines)


def build_schedule_context_fragment(season: int, league_id: int) -> Path:
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
