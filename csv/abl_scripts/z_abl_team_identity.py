import pandas as pd
from pathlib import Path

CSV_OUT_DIR = Path("csv/out/csv_out")
STAR_DIR = Path("csv/out/star_schema")


def load_team_lookup() -> pd.DataFrame:
    """Return dim_team_park with team_id + abbr + name."""
    dim = pd.read_csv(STAR_DIR / "dim_team_park.csv")
    # Normalize a minimal set of columns
    col_map = {}
    for c in dim.columns:
        lc = c.lower()
        if lc in ("id", "team_id") and "team_id" not in col_map:
            col_map[c] = "team_id"
        elif "abbr" in lc and "team_abbr" not in col_map:
            col_map[c] = "team_abbr"
        elif ("team name" in lc or lc == "name") and "team_name" not in col_map:
            col_map[c] = "team_name"

    dim = dim[list(col_map.keys())].rename(columns=col_map)
    dim["team_abbr"] = dim["team_abbr"].str.strip()
    return dim


def _pick_row(df: pd.DataFrame, team_id: int, label_cols: list[str]) -> dict:
    """Helper: grab one row for team_id and keep only selected columns."""
    if "team_id" not in df.columns:
        raise SystemExit("Expected 'team_id' column in identity CSV.")
    row = df[df["team_id"] == team_id]
    if row.empty:
        return {}
    rec = row.iloc[0]
    out = {}
    for col in label_cols:
        if col in rec.index:
            out[col] = rec[col]
    return out


def build_team_identity_card(team_abbr: str) -> dict:
    """
    Build a compact identity card for a single team.

    Inputs:
      - team_abbr: e.g. "CHI"

    Output shape:
      {
        "team_abbr": "CHI",
        "team_name": "Chicago Fire",
        "team_id": 12,
        "identity": {
          "run_creation": {...},
          "one_run": {...},
          "heat_check": {...},
          "babip_luck": {...},
          "system_crash": {...},
        },
      }
    """
    team_abbr = team_abbr.strip().upper()

    # Look up team_id + display name
    teams = load_team_lookup()
    match = teams[teams["team_abbr"].str.upper() == team_abbr]
    if match.empty:
        raise SystemExit(f"No team found for abbr={team_abbr!r}")
    team_row = match.iloc[0]
    team_id = int(team_row["team_id"])
    team_name = str(team_row["team_name"])

    # 1) Run Creation Profile
    rc_df = pd.read_csv(CSV_OUT_DIR / "z_ABL_Run_Creation_Profile.csv")
    run_creation = _pick_row(
        rc_df,
        team_id,
        [
            "team_display",
            "hr_pct",
            "2out_rbi",
            "2out_rbi_pct",
            "steal_attempts_per_game",
            "hr_rbi_share",
            "power_flag",
            "pressure_flag",
            "clutch_flag",
            "rating",
        ],
    )

    # 2) One-Run Identity
    one_df = pd.read_csv(CSV_OUT_DIR / "z_ABL_One_Run_Record.csv")
    one_run = _pick_row(
        one_df,
        team_id,
        [
            "overall_winpct",
            "one_run_g",
            "one_run_w",
            "one_run_l",
            "one_run_winpct",
            "one_run_diff_winpct",
            "one_run_share",
        ],
    )

    # 3) Heat Check
    heat_df = pd.read_csv(CSV_OUT_DIR / "z_ABL_Heat_Check.csv")
    heat_check = _pick_row(
        heat_df,
        team_id,
        [
            "current_win_streak",
            "longest_win_streak",
            "rolling_pct_10",
            "rolling_pct_20",
            "rolling_pct_30",
            "heat_flag",
            "rating",
        ],
    )

    # 4) BABIP Luck
    babip_df = pd.read_csv(CSV_OUT_DIR / "z_ABL_Team_BABIP_Luck.csv")
    babip_luck = _pick_row(
        babip_df,
        team_id,
        [
            "team_babip",
            "league_babip",
            "off_babip_diff",
            "def_babip_diff",
            "bat_flag",
            "pitch_flag",
            "rating",
        ],
    )

    # 5) System Crash (Slumps)
    crash_df = pd.read_csv(CSV_OUT_DIR / "z_ABL_System_Crash_Slumps_Current.csv")
    system_crash = _pick_row(
        crash_df,
        team_id,
        [
            "current_losing_streak",
            "longest_losing_streak",
            "l10_record",
            "crash_flag",
            "rating",
        ],
    )

    card = {
        "team_abbr": team_abbr,
        "team_name": team_name,
        "team_id": team_id,
        "identity": {
            "run_creation": run_creation,
            "one_run": one_run,
            "heat_check": heat_check,
            "babip_luck": babip_luck,
            "system_crash": system_crash,
        },
    }
    return card
