from __future__ import annotations

import pandas as pd
from pathlib import Path
import io


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[2]
ABL_STATS = ROOT / "csv" / "abl_statistics"
OUT_DIR = ROOT / "csv" / "out" / "star_schema"
TEAM_PARK_PATH = (
    ABL_STATS
    / "abl_statistics_team_statistics___info_-_sortable_stats_team_pers_park.csv"
)
TEAM_STAFF_PATH = (
    ABL_STATS
    / "abl_statistics_team_statistics___info_-_sortable_stats_abl_staff.csv"
)
COACHES_PATH = ROOT / "csv" / "ootp_csv" / "coaches.csv"

STAFF_ROLE_COLUMNS = [
    "GM",
    "MA",
    "BN",
    "PC",
    "HC",
    "SC",
    "TT",
    "OWN",
    "1BC",
    "3BC",
]

STAFF_COMMENT_BLOCK = [
    "# Column Documentation: csv\\abl_statistics\\abl_statistics_team_statistics___info_-_sortable_stats_abl_staff.csv",
    "# ID: Team ID",
    "# Team Name: Franchise name",
    "# Abbr: Team abbreviation",
    "# GM: General manager",
    "# MA: Manager (dugout skipper)",
    "# BN: Bench coach",
    "# PC: Pitching coach",
    "# HC: Hitting coach",
    "# SC: Scouting director",
    "# TT: Team trainer",
    "# OWN: Owner",
    "# 1BC: First-base coach",
    "# 3BC: Third-base coach",
]

COACH_LOOKUP: pd.DataFrame | None = None
_TEAM_LOOKUP: pd.DataFrame | None = None


def read_csv(path: Path) -> pd.DataFrame:
    skip_rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.lstrip().startswith("#"):
                skip_rows += 1
                continue
            break
    df = pd.read_csv(path, skiprows=skip_rows)
    if "ID" in df.columns:
        df["ID"] = pd.to_numeric(df["ID"], errors="coerce")
    return df


def merge_on_id(
    left: pd.DataFrame,
    right: pd.DataFrame,
    suffixes: tuple[str, str],
) -> pd.DataFrame:
    return left.merge(right, on="ID", how="outer", suffixes=suffixes)


def get_team_lookup() -> pd.DataFrame:
    global _TEAM_LOOKUP
    if _TEAM_LOOKUP is None:
        lookup = read_csv(TEAM_PARK_PATH)
        cols = [col for col in lookup.columns if col in {"ID", "Team Name", "Abbr"}]
        _TEAM_LOOKUP = lookup[cols].copy()
    return _TEAM_LOOKUP


def ensure_team_id(df: pd.DataFrame, name_col: str = "Team Name") -> pd.DataFrame:
    if "ID" not in df.columns:
        lookup = get_team_lookup()
        if name_col not in df.columns:
            raise KeyError(f"{name_col} column not found for team lookup.")
        mapping = lookup.set_index(name_col)["ID"]
        df["ID"] = df[name_col].map(mapping)
        df["ID"] = pd.to_numeric(df["ID"], errors="coerce")
    return df


def get_coach_lookup() -> pd.Series:
    global COACH_LOOKUP
    if COACH_LOOKUP is None:
        coaches = pd.read_csv(COACHES_PATH)
        coaches["full_name"] = (
            coaches["first_name"].astype(str).str.strip()
            + " "
            + coaches["last_name"].astype(str).str.strip()
        ).str.lower().str.strip()
        COACH_LOOKUP = coaches[["coach_id", "full_name"]].drop_duplicates(
            subset="full_name", keep="first"
        )
    return COACH_LOOKUP.set_index("full_name")["coach_id"]


def attach_coach_ids(df: pd.DataFrame) -> pd.DataFrame:
    lookup = get_coach_lookup()
    for column in STAFF_ROLE_COLUMNS:
        id_col = f"{column}_ID"
        df[id_col] = (
            df[column]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(lookup)
        )
    return df


def write_output(df: pd.DataFrame, filename: str, summary: list[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUT_DIR / filename
    df.to_csv(output_path, index=False)
    summary.append(f"Created {filename} with {len(df)} rows")


def build_dim_player_profile(summary: list[str]) -> None:
    indicative1 = read_csv(
        ABL_STATS / "abl_statistics_player_statistics_-_sortable_stats_player_indicative_1.csv"
    )
    indicative2 = read_csv(
        ABL_STATS / "abl_statistics_player_statistics_-_sortable_stats_player_indicative_2.csv"
    )
    misc_info = read_csv(
        ABL_STATS / "abl_statistics_player_statistics_-_sortable_stats_player_misc_info.csv"
    )

    profile = merge_on_id(indicative1, indicative2, suffixes=("", "_indic2"))
    profile = merge_on_id(profile, misc_info, suffixes=("", "_misc"))
    write_output(profile, "dim_player_profile.csv", summary)


def build_dim_player_ratings(summary: list[str]) -> None:
    bat = read_csv(
        ABL_STATS / "abl_statistics_player_statistics_-_sortable_stats_player_bat_ratings.csv"
    )
    write_output(bat, "dim_player_batting_ratings.csv", summary)

    pitch = read_csv(
        ABL_STATS / "abl_statistics_player_statistics_-_sortable_stats_player_pitch_ratings.csv"
    )
    write_output(pitch, "dim_player_pitching_ratings.csv", summary)

    field = read_csv(
        ABL_STATS / "abl_statistics_player_statistics_-_sortable_stats_player_field_ratings.csv"
    )
    write_output(field, "dim_player_fielding_ratings.csv", summary)


def build_dim_team_park(summary: list[str]) -> None:
    parks = read_csv(
        ABL_STATS
        / "abl_statistics_team_statistics___info_-_sortable_stats_team_pers_park.csv"
    )
    write_output(parks, "dim_team_park.csv", summary)


def build_dim_team_staff(summary: list[str]) -> None:
    staff = read_csv(TEAM_STAFF_PATH)
    staff_with_ids = attach_coach_ids(staff.copy())
    write_output(staff_with_ids, "dim_team_staff.csv", summary)

    buffer = io.StringIO()
    staff_with_ids.to_csv(buffer, index=False)
    comment_text = "\n".join(STAFF_COMMENT_BLOCK) + "\n"
    TEAM_STAFF_PATH.write_text(comment_text + buffer.getvalue(), encoding="utf-8")


def build_fact_team_batting(summary: list[str]) -> None:
    stats = read_csv(
        ABL_STATS
        / "abl_statistics_team_statistics___info_-_sortable_stats_batting_stats.csv"
    )
    stats = ensure_team_id(stats, name_col="Team Name")
    xtra = read_csv(
        ABL_STATS
        / "abl_statistics_team_statistics___info_-_sortable_stats_batting_xtra.csv"
    )
    merged = merge_on_id(stats, xtra, suffixes=("_stats", "_xtra"))
    write_output(merged, "fact_team_batting.csv", summary)


def build_fact_team_pitching(summary: list[str]) -> None:
    pitch1 = read_csv(
        ABL_STATS
        / "abl_statistics_team_statistics___info_-_sortable_stats_pitching_1.csv"
    )
    pitch2 = read_csv(
        ABL_STATS
        / "abl_statistics_team_statistics___info_-_sortable_stats_pitching_2.csv"
    )
    merged = merge_on_id(pitch1, pitch2, suffixes=("_p1", "_p2"))
    write_output(merged, "fact_team_pitching.csv", summary)


def build_fact_team_catcher_fielding(summary: list[str]) -> None:
    field1 = read_csv(
        ABL_STATS
        / "abl_statistics_team_statistics___info_-_sortable_stats_c_fielding_1.csv"
    )
    field2 = read_csv(
        ABL_STATS
        / "abl_statistics_team_statistics___info_-_sortable_stats_c_fielding_2.csv"
    )
    merged = merge_on_id(field1, field2, suffixes=("_cf1", "_cf2"))
    write_output(merged, "fact_team_catcher_fielding.csv", summary)


def build_fact_team_standings(summary: list[str]) -> None:
    standings = read_csv(
        ABL_STATS
        / "abl_statistics_team_statistics___info_-_sortable_stats_team_cur_rec_hist.csv"
    )
    write_output(standings, "fact_team_standings.csv", summary)


def build_fact_team_financials(summary: list[str]) -> None:
    financials = read_csv(
        ABL_STATS
        / "abl_statistics_team_statistics___info_-_sortable_stats_team_finan.csv"
    )
    write_output(financials, "fact_team_financials.csv", summary)


def build_fact_player_batting(summary: list[str]) -> None:
    bat_stats = read_csv(
        ABL_STATS
        / "abl_statistics_player_statistics_-_sortable_stats_player_bat_stats.csv"
    )
    write_output(bat_stats, "fact_player_batting.csv", summary)


def build_fact_player_pitching(summary: list[str]) -> None:
    pitch1 = read_csv(
        ABL_STATS
        / "abl_statistics_player_statistics_-_sortable_stats_player_pitch_stats_1.csv"
    )
    pitch2 = read_csv(
        ABL_STATS
        / "abl_statistics_player_statistics_-_sortable_stats_player_pitch_stats_2.csv"
    )
    merged = merge_on_id(pitch1, pitch2, suffixes=("_p1", "_p2"))
    write_output(merged, "fact_player_pitching.csv", summary)


def build_fact_player_fielding(summary: list[str]) -> None:
    field_stats = read_csv(
        ABL_STATS
        / "abl_statistics_player_statistics_-_sortable_stats_player_field_stats.csv"
    )
    write_output(field_stats, "fact_player_fielding.csv", summary)


def main() -> None:
    summary: list[str] = []

    build_dim_player_profile(summary)
    build_dim_player_ratings(summary)
    build_dim_team_park(summary)
    build_dim_team_staff(summary)

    build_fact_team_batting(summary)
    build_fact_team_pitching(summary)
    build_fact_team_catcher_fielding(summary)
    build_fact_team_standings(summary)
    build_fact_team_financials(summary)
    build_fact_player_batting(summary)
    build_fact_player_pitching(summary)
    build_fact_player_fielding(summary)

    for line in summary:
        print(line)


if __name__ == "__main__":
    main()
