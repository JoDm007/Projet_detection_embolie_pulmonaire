# backend/db/repositories.py
# Fonctions d'accès aux tables. Les autres blocs passent par ici, jamais par du SQL direct.
import json
from datetime import datetime, timedelta, timezone

from .database import connect, utcnow


# ─────────────────────────────── Utilisateurs ───────────────────────────────
class UserRepo:
    @staticmethod
    def create(username: str, password_hash: str, role: str) -> int:
        with connect() as conn:
            cur = conn.execute(
                "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
                (username, password_hash, role, utcnow()))
            return cur.lastrowid

    @staticmethod
    def get_by_username(username: str):
        with connect() as conn:
            return conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()

    @staticmethod
    def get(user_id: int):
        with connect() as conn:
            return conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()

    @staticmethod
    def list():
        with connect() as conn:
            return conn.execute(
                "SELECT id, username, role, active, locked_until, created_at FROM users ORDER BY id"
            ).fetchall()

    @staticmethod
    def set_password(user_id: int, password_hash: str) -> None:
        with connect() as conn:
            conn.execute("UPDATE users SET password_hash = ?, failed_logins = 0, locked_until = NULL "
                         "WHERE id = ?", (password_hash, user_id))
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))

    @staticmethod
    def set_active(user_id: int, active: bool) -> None:
        with connect() as conn:
            conn.execute("UPDATE users SET active = ? WHERE id = ?", (int(active), user_id))
            if not active:
                conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))

    @staticmethod
    def register_failure(user_id: int, max_failures: int, lockout_minutes: int) -> None:
        """Incrémente le compteur d'échecs ; bloque le compte au-delà du seuil."""
        with connect() as conn:
            row = conn.execute("SELECT failed_logins FROM users WHERE id = ?", (user_id,)).fetchone()
            failures = (row["failed_logins"] if row else 0) + 1
            locked_until = None
            if failures >= max_failures:
                locked_until = (datetime.now(timezone.utc)
                                + timedelta(minutes=lockout_minutes)).isoformat(timespec="seconds")
                failures = 0
            conn.execute("UPDATE users SET failed_logins = ?, locked_until = COALESCE(?, locked_until) "
                         "WHERE id = ?", (failures, locked_until, user_id))

    @staticmethod
    def reset_failures(user_id: int) -> None:
        with connect() as conn:
            conn.execute("UPDATE users SET failed_logins = 0, locked_until = NULL WHERE id = ?",
                         (user_id,))


# ─────────────────────────────── Sessions ───────────────────────────────
class SessionRepo:
    @staticmethod
    def create(token_hash: str, user_id: int, hours: int) -> None:
        expires = (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat(timespec="seconds")
        with connect() as conn:
            conn.execute("INSERT INTO sessions (token_hash, user_id, created_at, expires_at) "
                         "VALUES (?, ?, ?, ?)", (token_hash, user_id, utcnow(), expires))

    @staticmethod
    def get_user(token_hash: str):
        """Utilisateur actif lié à une session non expirée, sinon None."""
        with connect() as conn:
            return conn.execute(
                "SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id "
                "WHERE s.token_hash = ? AND s.expires_at > ? AND u.active = 1",
                (token_hash, utcnow())).fetchone()

    @staticmethod
    def delete(token_hash: str) -> None:
        with connect() as conn:
            conn.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))

    @staticmethod
    def purge_expired() -> int:
        with connect() as conn:
            return conn.execute("DELETE FROM sessions WHERE expires_at <= ?", (utcnow(),)).rowcount


# ─────────────────────────────── Analyses ───────────────────────────────
class AnalysisRepo:
    @staticmethod
    def create(analysis_id: str, user_id: int, source: str, engine: str, case_id: str | None) -> None:
        with connect() as conn:
            conn.execute(
                "INSERT INTO analyses (id, user_id, source, case_id, engine, status, step, created_at) "
                "VALUES (?, ?, ?, ?, ?, 'queued', 'En attente', ?)",
                (analysis_id, user_id, source, case_id, engine, utcnow()))

    @staticmethod
    def get(analysis_id: str):
        with connect() as conn:
            return conn.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone()

    @staticmethod
    def list_for_user(user_id: int, include_all: bool = False):
        with connect() as conn:
            if include_all:
                return conn.execute("SELECT a.*, u.username FROM analyses a JOIN users u "
                                    "ON u.id = a.user_id ORDER BY a.created_at DESC").fetchall()
            return conn.execute("SELECT a.*, u.username FROM analyses a JOIN users u "
                                "ON u.id = a.user_id WHERE a.user_id = ? "
                                "ORDER BY a.created_at DESC", (user_id,)).fetchall()

    @staticmethod
    def set_progress(analysis_id: str, step: str, progress: float) -> None:
        with connect() as conn:
            conn.execute("UPDATE analyses SET status = 'running', step = ?, progress = ? WHERE id = ?",
                         (step, progress, analysis_id))

    @staticmethod
    def set_done(analysis_id: str, summary: dict) -> None:
        with connect() as conn:
            conn.execute("UPDATE analyses SET status = 'done', step = 'Terminé', progress = 1, "
                         "summary_json = ?, finished_at = ? WHERE id = ?",
                         (json.dumps(summary, ensure_ascii=False), utcnow(), analysis_id))

    @staticmethod
    def set_failed(analysis_id: str, message: str) -> None:
        with connect() as conn:
            conn.execute("UPDATE analyses SET status = 'failed', step = 'Échec', error = ?, "
                         "finished_at = ? WHERE id = ?", (message, utcnow(), analysis_id))

    @staticmethod
    def fail_interrupted() -> int:
        """Au démarrage : les analyses laissées en cours par un arrêt du serveur sont closes."""
        with connect() as conn:
            return conn.execute(
                "UPDATE analyses SET status = 'failed', step = 'Échec', "
                "error = 'Analyse interrompue par un redémarrage du serveur.', finished_at = ? "
                "WHERE status IN ('queued', 'running')", (utcnow(),)).rowcount

    @staticmethod
    def older_than(iso_ts: str):
        with connect() as conn:
            return conn.execute("SELECT id FROM analyses WHERE created_at < ? "
                                "AND status IN ('done', 'failed')", (iso_ts,)).fetchall()

    @staticmethod
    def delete(analysis_id: str) -> None:
        with connect() as conn:
            conn.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))


# ─────────────────────────────── Audit ───────────────────────────────
class AuditRepo:
    @staticmethod
    def log(action: str, username: str | None = None, target: str | None = None,
            detail: str | None = None) -> None:
        with connect() as conn:
            conn.execute("INSERT INTO audit_log (ts, username, action, target, detail) "
                         "VALUES (?, ?, ?, ?, ?)", (utcnow(), username, action, target, detail))

    @staticmethod
    def last(limit: int = 50):
        with connect() as conn:
            return conn.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?",
                                (limit,)).fetchall()
