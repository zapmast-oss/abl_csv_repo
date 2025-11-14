"""Centralized configuration for ABL scripts; read-only helpers for OOTP CSV data."""

import os
from datetime import datetime
from pathlib import Path

LEAGUE_ID = 200
TEAM_IDS = list(range(1, 25))

# Project layout helpers -----------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

# Raw OOTP exports live under ootp_csv/. Allow overrides via env vars.
DEFAULT_OOTP_ROOT = ROOT_DIR / "ootp_csv"
RAW_CSV_ROOT = Path(os.environ.get("ABL_OOTP_ROOT", DEFAULT_OOTP_ROOT)).resolve()

# Locally curated CSVs (abl_*.csv) get parked in abl_csv/.
DEFAULT_ANALYTICS_ROOT = ROOT_DIR / "abl_csv"
ANALYTICS_CSV_ROOT = Path(os.environ.get("ABL_ANALYTICS_ROOT", DEFAULT_ANALYTICS_ROOT)).resolve()

# Output directories for generated CSV/TXT artifacts.
CSV_OUT_ROOT = Path(os.environ.get("ABL_CSV_OUT", ROOT_DIR / "out" / "csv_out")).resolve()
TXT_OUT_ROOT = Path(os.environ.get("ABL_TXT_OUT", ROOT_DIR / "out" / "txt_out")).resolve()

# Backwards-compatible alias used by older scripts.
CSV_ROOT = RAW_CSV_ROOT


def csv_path(name: str) -> Path:
    """Return the full path to a raw OOTP CSV."""
    return RAW_CSV_ROOT / name


def abl_csv_path(name: str) -> Path:
    """Return the path to an analytics CSV (abl_*.csv, standings snapshots, etc)."""
    return ANALYTICS_CSV_ROOT / name


def stamp_text_block(text: str) -> str:
    """Insert a 'Generated on' line beneath the title of any text report."""
    payload = (text or "").rstrip("\n")
    lines = payload.splitlines()
    timestamp = f"Generated on: {datetime.now():%Y-%m-%d %H:%M:%S}"
    if not lines:
        return f"{timestamp}\n"

    insert_at = 1
    if len(lines) > 1:
        underline = lines[1].strip()
        if underline and len(set(underline)) == 1 and underline[0] in {"=", "-", "_"}:
            insert_at = 2
    stamped = lines[:insert_at] + [timestamp, ""] + lines[insert_at:]
    return "\n".join(stamped).rstrip() + "\n"
