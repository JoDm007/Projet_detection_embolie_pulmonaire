# backend/db/database.py
# Connexion SQLite. Une connexion courte par opération : simple et sûr avec le worker
# d'analyse qui tourne dans un autre thread (mode WAL = lectures pendant les écritures).
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
_SCHEMA_FILE = Path(__file__).with_name("schema.sql")
_db_path: Path | None = None


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def init_db(db_path: Path) -> None:
    """Crée le fichier et le schéma s'ils n'existent pas (idempotent)."""
    global _db_path
    _db_path = db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA_FILE.read_text(encoding="utf-8"))
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        if row is None:
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))


@contextmanager
def connect():
    if _db_path is None:
        raise RuntimeError("init_db() doit être appelé avant tout accès à la base.")
    conn = sqlite3.connect(_db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
