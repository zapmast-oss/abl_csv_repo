import argparse
import datetime as dt
import re
from pathlib import Path
from io import StringIO

import pandas as pd


def parse_scores_html(html_text: str, season: int, league_id: int, filename: str):
    """
    Parse one daily scoreboard HTML into:
        - games_by_team: one row per team per game
        - games: one row per game with home/away & runs
    """
    # filename pattern: .../league_{league_id}_scores_{season}_MM_DD.html
    m = re.search(
        rf"league_{league_id}_scores_{season}_(\d{{2}})_(\d{{2}})\.html", filename
    )
    if not m:
        return pd.DataFrame(), pd.DataFrame()

    month = int(m.group(1))
    day = int(m.group(2))
    game_date = dt.date(season, month, day)

    tables = pd.read_html(StringIO(html_text))

    team_rows = []
    game_rows = []

    for idx, df in enumerate(tables):
        # Linescore tables: 2 rows (away/home), have R/H/E & "Unnamed: 0" (team name)
        if (
            df.shape[0] == 2
            and "R" in df.columns
            and "H" in df.columns
            and "E" in df.columns
            and "Unnamed: 0" in df.columns
        ):
            away = df.iloc[0]
            home = df.iloc[1]

            game_id = f"{season}_{month:02d}{day:02d}_{idx}"

            # Team-per-game rows
            team_rows.append(
                {
                    "season": season,
                    "league_id": league_id,
                    "game_date": game_date,
                    "game_id": game_id,
                    "order": 0,
                    "team_name": str(away["Unnamed: 0"]),
                    "is_home": 0,
                    "runs": int(away["R"]),
                    "hits": int(away["H"]),
                    "errors": int(away["E"]),
                }
            )
            team_rows.append(
                {
                    "season": season,
                    "league_id": league_id,
                    "game_date": game_date,
                    "game_id": game_id,
                    "order": 1,
                    "team_name": str(home["Unnamed: 0"]),
                    "is_home": 1,
                    "runs": int(home["R"]),
                    "hits": int(home["H"]),
                    "errors": int(home["E"]),
                }
            )

            # Game-level row
            game_rows.append(
                {
                    "season": season,
                    "league_id": league_id,
                    "game_date": game_date,
                    "game_id": game_id,
                    "away_team_name": str(away["Unnamed: 0"]),
                    "home_team_name": str(home["Unnamed: 0"]),
                    "away_runs": int(away["R"]),
                    "home_runs": int(home["R"]),
                }
            )

    return pd.DataFrame(team_rows), pd.DataFrame(game_rows)


def build_team_game_flags(team_games: pd.DataFrame) -> pd.DataFrame:
    """
    Add opponent runs, win/loss flags, month and week_index to team_games.
    """
    if team_games.empty:
        return team_games

    df = team_games.copy()

    # Opponent runs via self-merge
    opp = df[["game_id", "order", "runs"]].copy()
    opp["opp_order"] = 1 - opp["order"]
    opp = opp.rename(columns={"runs": "opp_runs"})

    df = df.merge(
        opp[["game_id", "opp_order", "opp_runs"]],
        left_on=["game_id", "order"],
        right_on=["game_id", "opp_order"],
        how="left",
    ).drop(columns=["opp_order"])

    df["win"] = (df["runs"] > df["opp_runs"]).astype(int)
    df["loss"] = (df["runs"] < df["opp_runs"]).astype(int)

    # Date features
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["month"] = df["game_date"].dt.to_period("M").astype(str)

    min_date = df["game_date"].min()
    df["week_index"] = ((df["game_date"] - min_date).dt.days // 7 + 1).astype(int)

    return df


def compute_team_monthly_weekly(team_games: pd.DataFrame):
    """
    Compute team-level monthly and weekly summaries from team_games
    (with win/loss and opp_runs columns).
    """
    if team_games.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = team_games.copy()

    # Monthly
    monthly = (
        df.groupby(
            ["season", "league_id", "team_name", "month"], as_index=False
        )
        .agg(
            games=("game_id", "nunique"),
            wins=("win", "sum"),
            losses=("loss", "sum"),
            runs_for=("runs", "sum"),
            runs_against=("opp_runs", "sum"),
        )
    )
    monthly["run_diff"] = monthly["runs_for"] - monthly["runs_against"]

    # Weekly
    weekly = (
        df.groupby(
            ["season", "league_id", "team_name", "week_index"], as_index=False
        )
        .agg(
            games=("game_id", "nunique"),
            wins=("win", "sum"),
            losses=("loss", "sum"),
            runs_for=("runs", "sum"),
            runs_against=("opp_runs", "sum"),
        )
    )
    weekly["run_diff"] = weekly["runs_for"] - weekly["runs_against"]

    return monthly, weekly


def compute_series_summary(games: pd.DataFrame) -> pd.DataFrame:
    """
    Compute series-level summary by team pair and contiguous blocks of games.
    A new series starts when:
      - date gap > 1 day for that pair, or
      - home team changes.
    """
    if games.empty:
        return pd.DataFrame()

    df = games.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["home_win"] = (df["home_runs"] > df["away_runs"]).astype(int)
    df["away_win"] = (df["away_runs"] > df["home_runs"]).astype(int)

    # Pair key (order-independent)
    df["pair_key"] = df.apply(
        lambda r: " / ".join(
            sorted([r["home_team_name"], r["away_team_name"]])
        ),
        axis=1,
    )

    series_chunks = []
    for pair, grp in df.groupby("pair_key", sort=False):
        g = grp.sort_values("game_date").copy()
        g["game_date"] = pd.to_datetime(g["game_date"])
        g["date_diff"] = g["game_date"].diff().dt.days.fillna(99)
        g["home_change"] = g["home_team_name"].ne(
            g["home_team_name"].shift(1)
        ).fillna(True)
        g["new_series"] = (g["date_diff"] > 1) | g["home_change"]
        g["series_index"] = g["new_series"].cumsum()
        series_chunks.append(g)

    df2 = pd.concat(series_chunks, ignore_index=True)

    # Aggregate series
    series_summary = (
        df2.groupby(
            ["season", "league_id", "pair_key", "series_index"], as_index=False
        )
        .agg(
            start_date=("game_date", "min"),
            end_date=("game_date", "max"),
            games=("game_id", "nunique"),
            home_team=(
                "home_team_name",
                lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0],
            ),
            home_team_wins=("home_win", "sum"),
            away_team_wins=("away_win", "sum"),
            total_home_runs=("home_runs", "sum"),
            total_away_runs=("away_runs", "sum"),
        )
    )

    def result(row):
        return (
            f"{row['home_team']} {row['home_team_wins']}-"
            f"{row['away_team_wins']} (runs {row['total_home_runs']}-"
            f"{row['total_away_runs']})"
        )

    series_summary["series_result"] = series_summary.apply(result, axis=1)
    return series_summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Parse OOTP almanac daily scoreboard HTML into game logs, "
            "then monthly, weekly, and series summaries."
        )
    )
    parser.add_argument(
        "--almanac-zip",
        required=True,
        help="Path to almanac_YYYY.zip",
    )
    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help="Season year (e.g. 1972).",
    )
    parser.add_argument(
        "--league-id",
        type=int,
        default=200,
        help="League ID to parse (ABL = 200). Default: 200.",
    )
    parser.add_argument(
        "--out-root",
        default="csv/out/almanac",
        help="Root output folder for parsed data (default: csv/out/almanac).",
    )

    args = parser.parse_args(argv)

    zip_path = Path(args.almanac_zip)
    out_root = Path(args.out_root) / str(args.season)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[DEBUG] almanac_zip={zip_path}")
    print(f"[DEBUG] season={args.season}, league_id={args.league_id}")
    print(f"[DEBUG] out_root={out_root}")

    if not zip_path.exists():
        raise FileNotFoundError(f"almanac zip not found: {zip_path}")

    # Collect all daily scores files from the zip
    import zipfile

    scores_pattern = f"almanac_{args.season}/leagues/league_{args.league_id}_scores_{args.season}_"
    team_rows = []
    game_rows = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [n for n in zf.namelist() if scores_pattern in n]
        print(f"[INFO] Found {len(names)} daily scores files in zip")

        for name in sorted(names):
            html = zf.read(name).decode("utf-8", errors="ignore")
            tg, gg = parse_scores_html(html, args.season, args.league_id, name)
            if not tg.empty:
                team_rows.append(tg)
            if not gg.empty:
                game_rows.append(gg)

    if team_rows:
        team_games = pd.concat(team_rows, ignore_index=True)
    else:
        team_games = pd.DataFrame()

    if game_rows:
        games = pd.concat(game_rows, ignore_index=True)
    else:
        games = pd.DataFrame()

    print(f"[INFO] Team-games rows: {len(team_games)}")
    print(f"[INFO] Games rows: {len(games)}")

    # Save raw
    by_team_path = out_root / f"games_{args.season}_league{args.league_id}_by_team.csv"
    games_path = out_root / f"games_{args.season}_league{args.league_id}.csv"

    if not team_games.empty:
        team_games.to_csv(by_team_path, index=False)
        print(f"[OK] Wrote team-by-game logs to {by_team_path}")
    if not games.empty:
        games.to_csv(games_path, index=False)
        print(f"[OK] Wrote game logs to {games_path}")

    # Enrich with win/loss + month/week_index
    team_games_flags = build_team_game_flags(team_games)
    monthly, weekly = compute_team_monthly_weekly(team_games_flags)
    series_summary = compute_series_summary(games)

    if not monthly.empty:
        monthly_path = (
            out_root / f"team_monthly_summary_{args.season}_league{args.league_id}.csv"
        )
        monthly.to_csv(monthly_path, index=False)
        print(f"[OK] Wrote team monthly summary to {monthly_path}")

    if not weekly.empty:
        weekly_path = (
            out_root / f"team_weekly_summary_{args.season}_league{args.league_id}.csv"
        )
        weekly.to_csv(weekly_path, index=False)
        print(f"[OK] Wrote team weekly summary to {weekly_path}")

    if not series_summary.empty:
        series_path = (
            out_root / f"series_summary_{args.season}_league{args.league_id}.csv"
        )
        series_summary.to_csv(series_path, index=False)
        print(f"[OK] Wrote series summary to {series_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
