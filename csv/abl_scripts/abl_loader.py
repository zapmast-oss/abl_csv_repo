"""Responsible for reading OOTP CSV exports for the ABL; never writes or deletes files."""

from pathlib import Path

import pandas as pd

from abl_config import CSV_ROOT, LEAGUE_ID, csv_path


def read_csv(name: str) -> pd.DataFrame:
    """Load a CSV from the ABL export folder using csv_path(name). This must never modify files."""
    path = csv_path(name)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path.resolve()}")
    return pd.read_csv(path)


def filter_league(df: pd.DataFrame, league_id: int = LEAGUE_ID) -> pd.DataFrame:
    """If df has a league indicator column, filter it down to the requested league_id."""
    possible_columns = {"league_id", "leagueid", "league", "lg_id"}
    columns = {col.lower(): col for col in df.columns}
    for candidate in possible_columns:
        if candidate in columns:
            column_name = columns[candidate]
            return df[df[column_name] == league_id]
    return df


def read_league_csv(name: str, league_id: int = LEAGUE_ID) -> pd.DataFrame:
    """Load a CSV and return only rows that belong to the requested league."""
    data = read_csv(name)
    return filter_league(data, league_id)
