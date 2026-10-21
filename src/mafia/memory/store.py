"""Per-agent SQLite memory store (DRD 7.1). One store per agent; all access is serialized by a lock
because the engine calls agents from worker threads."""
from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
  id            INTEGER PRIMARY KEY,
  type          TEXT NOT NULL CHECK (type IN ('episodic','belief','lesson','reflection')),
  text          TEXT NOT NULL,
  about         TEXT NOT NULL DEFAULT '[]',
  role_context  TEXT,
  game_created  INTEGER NOT NULL,
  game_updated  INTEGER NOT NULL,
  importance    INTEGER NOT NULL CHECK (importance BETWEEN 1 AND 10),
  confidence    TEXT,
  status        TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active','superseded','faded','merged')),
  superseded_by INTEGER REFERENCES memories(id),
  cites         TEXT NOT NULL DEFAULT '[]',
  last_retrieved_game INTEGER,
  times_retrieved INTEGER NOT NULL DEFAULT 0,
  games_applied INTEGER NOT NULL DEFAULT 0,
  wins_when_applied INTEGER NOT NULL DEFAULT 0,
  embedding     BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS memory_versions (
  memory_id INTEGER NOT NULL REFERENCES memories(id),
  game      INTEGER NOT NULL,
  old_text  TEXT NOT NULL,
  old_confidence TEXT
);
CREATE TABLE IF NOT EXISTS summaries (
  subject   TEXT NOT NULL,
  game      INTEGER NOT NULL,
  text      TEXT NOT NULL,
  PRIMARY KEY (subject, game)
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE INDEX IF NOT EXISTS idx_mem_status ON memories(status, type);
"""


@dataclass
class Memory:
    id: int
    type: str
    text: str
    about: list[str]
    role_context: str | None
    game_created: int
    game_updated: int
    importance: int
    confidence: str | None
    status: str
    superseded_by: int | None
    cites: list[int]
    last_retrieved_game: int | None
    times_retrieved: int
    games_applied: int
    wins_when_applied: int
    embedding: np.ndarray


def _row(r: sqlite3.Row) -> Memory:
    d = dict(r)
    d["about"] = json.loads(d["about"])
    d["cites"] = json.loads(d["cites"])
    d["embedding"] = np.frombuffer(d["embedding"], dtype=np.float32)
    return Memory(**d)


class MemoryStore:
    def __init__(self, path, embedder):
        self.embedder = embedder
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.executescript(SCHEMA)
            self._db.commit()

    def _emb(self, text: str) -> bytes:
        return np.asarray(self.embedder.embed(text), dtype=np.float32).tobytes()

    def _insert(self, type, text, about, role_context, game, importance, confidence, cites, created=None) -> int:
        """Insert without committing; caller holds the lock."""
        cur = self._db.execute(
            "INSERT INTO memories (type,text,about,role_context,game_created,game_updated,importance,confidence,cites,embedding)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (type, text, json.dumps(list(about)), role_context, created or game, game, importance, confidence,
             json.dumps(list(cites)), self._emb(text)),
        )
        return cur.lastrowid

    def add(self, type, text, *, about=(), role_context=None, game, importance, confidence=None, cites=(), created=None) -> int:
        with self._lock:
            try:
                i = self._insert(type, text, about, role_context, game, importance, confidence, cites, created)
                self._db.commit()
                return i
            except Exception:
                self._db.rollback()
                raise

    def backup(self, path) -> None:
        """Consistent copy of the whole DB (crash checkpoint)."""
        import sqlite3
        with self._lock:
            dst = sqlite3.connect(str(path))
            try:
                self._db.backup(dst)
            finally:
                dst.close()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def get(self, id: int) -> Memory:
        with self._lock:
            r = self._db.execute("SELECT * FROM memories WHERE id=?", (id,)).fetchone()
        if r is None:
            raise KeyError(id)
        return _row(r)

    def active(self, types=None, status="active") -> list[Memory]:
        q, args = "SELECT * FROM memories WHERE status=?", [status]
        if types:
            q += f" AND type IN ({','.join('?' * len(types))})"
            args += list(types)
        with self._lock:
            return [_row(r) for r in self._db.execute(q + " ORDER BY id", args)]

    def update(self, id, text, confidence, importance, game) -> None:
        """Rewrite in place; the previous text/confidence go to memory_versions."""
        emb = self._emb(text)
        with self._lock:
            old = self._db.execute("SELECT text, confidence FROM memories WHERE id=?", (id,)).fetchone()
            if old is None:
                raise KeyError(id)
            self._db.execute("INSERT INTO memory_versions VALUES (?,?,?,?)", (id, game, old["text"], old["confidence"]))
            self._db.execute(
                "UPDATE memories SET text=?, confidence=?, importance=?, game_updated=?, embedding=? WHERE id=?",
                (text, confidence, importance, game, emb, id),
            )
            self._db.commit()

    def supersede(self, old_id, text, *, game, importance=None, confidence=None, cites=()) -> int:
        """Add a replacement (inheriting type/about/role/importance from the old one) and hide the old, atomically."""
        with self._lock:
            old = self.get(old_id)
            try:
                new_id = self._insert(old.type, text, old.about, old.role_context, game,
                                      importance or old.importance, confidence or old.confidence, cites)
                self._db.execute("UPDATE memories SET status='superseded', superseded_by=? WHERE id=?", (new_id, old_id))
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
        return new_id

    def reinforce(self, id, importance, game, cites=()) -> None:
        """A near-duplicate was proposed: keep the text, raise importance, refresh game_updated, union cites."""
        with self._lock:
            m = self.get(id)
            merged = m.cites + [c for c in dict.fromkeys(cites) if c not in m.cites and c != id]
            self._db.execute("UPDATE memories SET importance=?, game_updated=?, cites=? WHERE id=?",
                             (min(10, max(m.importance, importance)), game, json.dumps(merged), id))
            self._db.commit()

    def near_duplicate(self, type, text, threshold, about=None, role=None, exclude=None) -> Memory | None:
        """Most similar active memory of `type` (beliefs: sharing a subject; lessons: same role_context)
        at cosine >= threshold, else None. `exclude` is an id to skip (a supersede target)."""
        e = np.asarray(self.embedder.embed(text), dtype=np.float32)
        best, top = None, threshold
        for m in self.active((type,)):
            if m.id == exclude or (type == "lesson" and m.role_context != role):
                continue
            if type == "belief" and not (set(m.about) & set(about or ()) or not (m.about or about)):
                continue  # different subjects never merge
            s = float(m.embedding @ e)
            if s >= top:
                best, top = m, s
        return best

    def retire(self, old_id, new_id) -> None:
        """Hide `old_id` as superseded by an existing memory."""
        with self._lock:
            self._db.execute("UPDATE memories SET status='superseded', superseded_by=? WHERE id=?", (new_id, old_id))
            self._db.commit()

    def set_status(self, ids, status) -> None:
        ids = list(ids)
        with self._lock:
            self._db.executemany("UPDATE memories SET status=? WHERE id=?", [(status, i) for i in ids])
            self._db.commit()

    def touch(self, ids, game) -> None:
        """Retrieval bookkeeping."""
        with self._lock:
            self._db.executemany(
                "UPDATE memories SET last_retrieved_game=?, times_retrieved=times_retrieved+1 WHERE id=?",
                [(game, i) for i in ids])
            self._db.commit()

    def bump_lessons(self, ids, won: bool) -> None:
        with self._lock:
            self._db.executemany(
                "UPDATE memories SET games_applied=games_applied+1, wins_when_applied=wins_when_applied+? WHERE id=?",
                [(int(won), i) for i in ids])
            self._db.commit()

    def versions(self, id) -> list[tuple[int, str, str | None]]:
        with self._lock:
            return [(r["game"], r["old_text"], r["old_confidence"]) for r in self._db.execute(
                "SELECT game, old_text, old_confidence FROM memory_versions WHERE memory_id=? ORDER BY rowid", (id,))]

    def get_summaries(self) -> dict[str, str]:
        with self._lock:
            rows = self._db.execute(
                "SELECT subject, text FROM summaries s WHERE game=(SELECT MAX(game) FROM summaries WHERE subject=s.subject)"
            ).fetchall()
        return {r["subject"]: r["text"] for r in rows}

    def set_summaries(self, game, summaries: dict[str, str]) -> None:
        with self._lock:
            self._db.executemany("INSERT OR REPLACE INTO summaries VALUES (?,?,?)",
                                 [(s, game, t) for s, t in summaries.items()])
            self._db.commit()

    def meta_get(self, key, default=None):
        with self._lock:
            r = self._db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return r["value"] if r else default

    def meta_set(self, key, value) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, value))
            self._db.commit()

    def seed(self, friend_summaries: dict[str, str], self_summary: str) -> None:
        """Game 0 starting impressions (DRD 7.1)."""
        self.set_summaries(0, {**friend_summaries, "self": self_summary})
