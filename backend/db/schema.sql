-- backend/db/schema.sql — schéma v1 de app_data/app.db
-- Aucune donnée patient identifiante : les analyses ne connaissent qu'un UUID,
-- et pour la démo un identifiant de cas FUMPE (jeu de données public).

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    username        TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash   TEXT NOT NULL,
    role            TEXT NOT NULL CHECK (role IN ('radiologue', 'admin')),
    active          INTEGER NOT NULL DEFAULT 1,
    failed_logins   INTEGER NOT NULL DEFAULT 0,
    locked_until    TEXT,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    token_hash  TEXT PRIMARY KEY,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at  TEXT NOT NULL,
    expires_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS analyses (
    id            TEXT PRIMARY KEY,
    user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    source        TEXT NOT NULL CHECK (source IN ('upload', 'demo')),
    case_id       TEXT,
    engine        TEXT NOT NULL,
    status        TEXT NOT NULL CHECK (status IN ('queued', 'running', 'done', 'failed')),
    step          TEXT,
    progress      REAL NOT NULL DEFAULT 0,
    error         TEXT,
    summary_json  TEXT,
    created_at    TEXT NOT NULL,
    finished_at   TEXT
);

CREATE INDEX IF NOT EXISTS idx_analyses_user ON analyses(user_id, created_at);

CREATE TABLE IF NOT EXISTS audit_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TEXT NOT NULL,
    username  TEXT,
    action    TEXT NOT NULL,
    target    TEXT,
    detail    TEXT
);
