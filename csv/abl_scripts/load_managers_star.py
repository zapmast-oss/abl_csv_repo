from __future__ import annotations

import argparse
import csv
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

CSV_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CSV_ROOT.parent
DEFAULT_DB = REPO_ROOT / "data_work" / "abl.db"
DEFAULT_SRC = CSV_ROOT / "out" / "csv_out" / "abl_managers_summary.csv"


def parse_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def parse_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def ensure_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if "TEXT" in existing and "last_refreshed_at" not in existing:
        conn.execute(f"ALTER TABLE {table} RENAME COLUMN TEXT TO last_refreshed_at")
        existing.add("last_refreshed_at")
    for column, col_type in columns.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dim_manager (
            manager_id INTEGER PRIMARY KEY,
            name TEXT UNIQUE,
            first_seen_season INT,
            last_seen_season INT,
            total_titles INT,
            career_wins INT,
            career_losses INT,
            career_win_pct REAL,
            current_team TEXT,
            last_refreshed_at TEXT
        )
        """
    )
    ensure_columns(
        conn,
        "dim_manager",
        {
            "first_seen_season": "INT",
            "last_seen_season": "INT",
            "last_refreshed_at": "TEXT",
        },
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fact_manager_season (
            manager_id INT,
            season INT,
            team TEXT,
            wins INT,
            losses INT,
            win_pct REAL,
            titles_this_season INT,
            PRIMARY KEY (manager_id, season),
            FOREIGN KEY(manager_id) REFERENCES dim_manager(manager_id)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_dim_mgr_name ON dim_manager(name)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_fact_mgr_season ON fact_manager_season(manager_id, season)"
    )
    conn.execute("DROP VIEW IF EXISTS v_manager_career")
    conn.execute(
        """
        CREATE VIEW v_manager_career AS
        SELECT
            manager_id,
            name,
            total_titles,
            career_wins,
            career_losses,
            career_win_pct,
            current_team
        FROM dim_manager
        ORDER BY total_titles DESC, career_win_pct DESC
        """
    )


def upsert_manager(conn: sqlite3.Connection, row: dict) -> int:
    name = row["name"].strip()
    titles = parse_int(row.get("titles"))
    championships = parse_int(row.get("championships"))
    total_titles = titles if titles is not None else championships
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    data = {
        "name": name,
        "first_seen": None,
        "last_seen": None,
        "total_titles": total_titles,
        "career_wins": parse_int(row.get("career_wins")),
        "career_losses": parse_int(row.get("career_losses")),
        "career_win_pct": parse_float(row.get("career_win_pct")),
        "current_team": (row.get("current_team") or "").strip() or None,
        "last_refreshed_at": now,
    }
    conn.execute(
        """
        INSERT INTO dim_manager (name, first_seen_season, last_seen_season, total_titles,
                                 career_wins, career_losses, career_win_pct, current_team, last_refreshed_at)
        VALUES (:name, :first_seen, :last_seen, :total_titles,
                :career_wins, :career_losses, :career_win_pct, :current_team, :last_refreshed_at)
        ON CONFLICT(name) DO UPDATE SET
            total_titles=excluded.total_titles,
            career_wins=excluded.career_wins,
            career_losses=excluded.career_losses,
            career_win_pct=excluded.career_win_pct,
            current_team=excluded.current_team,
            last_refreshed_at=excluded.last_refreshed_at
        """,
        data,
    )
    cur = conn.execute("SELECT manager_id FROM dim_manager WHERE name = ?", (name,))
    manager_id = cur.fetchone()[0]
    return manager_id


def upsert_fact_row(conn: sqlite3.Connection, manager_id: int, row: dict) -> None:
    wins = parse_int(row.get("wins"))
    losses = parse_int(row.get("losses"))
    win_pct = parse_float(row.get("win_pct"))
    titles_this_season = parse_int(row.get("championships"))
    team = (row.get("team") or "").strip() or None
    conn.execute(
        """
        INSERT INTO fact_manager_season (manager_id, season, team, wins, losses, win_pct, titles_this_season)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(manager_id, season) DO UPDATE SET
            team=excluded.team,
            wins=excluded.wins,
            losses=excluded.losses,
            win_pct=excluded.win_pct,
            titles_this_season=excluded.titles_this_season
        """,
        (manager_id, 0, team, wins, losses, win_pct, titles_this_season),
    )


def load_rows(source: Path) -> Iterable[dict]:
    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def main() -> None:
    parser = argparse.ArgumentParser(description="Load manager summary CSV into SQLite star schema tables.")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="Path to SQLite database (default data_work/abl.db)")
    parser.add_argument(
        "--src",
        default=str(DEFAULT_SRC),
        help="Source CSV path (default out/csv_out/abl_managers_summary.csv)",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.parent.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)

    src_path = Path(args.src)
    if not src_path.exists():
        raise FileNotFoundError(f"Source CSV missing: {src_path}")

    with sqlite3.connect(db_path) as conn:
        ensure_schema(conn)
        for row in load_rows(src_path):
            if not row.get("name"):
                continue
            manager_id = upsert_manager(conn, row)
            upsert_fact_row(conn, manager_id, row)
        conn.commit()
    print(f"Loaded managers into {db_path}")


if __name__ == "__main__":
    main()
