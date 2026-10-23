"""SQLite storage for contracts, progress and reports."""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from . import config

_write_lock = threading.RLock()

COLUMNS: dict[str, str] = {
    "id": "TEXT PRIMARY KEY",
    "filename": "TEXT NOT NULL",
    "file_path": "TEXT NOT NULL",
    "text_path": "TEXT",
    "passages_path": "TEXT",
    "pages": "INTEGER DEFAULT 0",
    "size_bytes": "INTEGER DEFAULT 0",
    "model": "TEXT",
    "perspective": "TEXT",
    "status": "TEXT NOT NULL DEFAULT 'queued'",
    "progress": "INTEGER NOT NULL DEFAULT 0",
    "stage": "TEXT NOT NULL DEFAULT 'Queued'",
    "error": "TEXT",
    "cancel_requested": "INTEGER NOT NULL DEFAULT 0",
    "report_json": "TEXT",
    "risk_indicator": "INTEGER",
    "findings_count": "INTEGER",
    "index_state": "TEXT NOT NULL DEFAULT 'none'",
    "index_note": "TEXT",
    "created_at": "TEXT NOT NULL",
    "updated_at": "TEXT NOT NULL",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(config.DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    """Create the table and add any missing columns (so databases from older versions keep working)."""
    with _write_lock, connect() as conn:
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.DatabaseError:
            pass
        ddl = ", ".join(f"{name} {spec}" for name, spec in COLUMNS.items())
        conn.execute(f"CREATE TABLE IF NOT EXISTS contracts ({ddl})")
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(contracts)")}
        for name, spec in COLUMNS.items():
            if name not in existing:
                # ALTER TABLE cannot add PRIMARY KEY / NOT NULL-without-default columns; strip those.
                safe_spec = spec.replace("PRIMARY KEY", "").replace("NOT NULL", "").strip()
                conn.execute(f"ALTER TABLE contracts ADD COLUMN {name} {safe_spec}")
        conn.commit()


def insert_contract(contract_id: str, filename: str, file_path: str, size_bytes: int, model: str, perspective: str = "neutral") -> None:
    created = now_iso()
    with _write_lock, connect() as conn:
        conn.execute(
            "INSERT INTO contracts (id, filename, file_path, size_bytes, model, perspective, status, progress, stage, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 'queued', 0, 'Waiting in queue', ?, ?)",
            (contract_id, filename, file_path, size_bytes, model, perspective, created, created),
        )
        conn.commit()


def get(contract_id: str) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM contracts WHERE id = ?", (contract_id,)).fetchone()
        return dict(row) if row else None


def update(contract_id: str, **fields: Any) -> None:
    if not fields:
        return
    unknown = set(fields) - set(COLUMNS)
    if unknown:
        raise ValueError(f"Unknown column(s): {', '.join(sorted(unknown))}")
    fields["updated_at"] = now_iso()
    assignments = ", ".join(f"{key} = ?" for key in fields)
    values = list(fields.values()) + [contract_id]
    with _write_lock, connect() as conn:
        conn.execute(f"UPDATE contracts SET {assignments} WHERE id = ?", values)
        conn.commit()


def delete(contract_id: str) -> None:
    with _write_lock, connect() as conn:
        conn.execute("DELETE FROM contracts WHERE id = ?", (contract_id,))
        conn.commit()


def list_recent(limit: int = 30) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, filename, pages, status, progress, stage, risk_indicator, findings_count, index_state, created_at "
            "FROM contracts ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def ids_with_status(*statuses: str) -> list[str]:
    marks = ",".join("?" for _ in statuses)
    with connect() as conn:
        rows = conn.execute(f"SELECT id FROM contracts WHERE status IN ({marks}) ORDER BY created_at", statuses).fetchall()
    return [row["id"] for row in rows]
