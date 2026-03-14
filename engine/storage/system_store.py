from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any, Iterable, Optional


def _utcnow() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def _dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _loads(payload: str | None) -> Any:
    if not payload:
        return {}
    return json.loads(payload)


class SystemStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._ensure_schema()

    @contextmanager
    def connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        statements = [
            """
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                mode TEXT,
                status TEXT NOT NULL,
                progress REAL NOT NULL,
                message TEXT NOT NULL,
                result_json TEXT NOT NULL,
                error TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS mode_state (
                mode TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
        ]
        with self._lock, self.connection() as conn:
            for statement in statements:
                conn.execute(statement)
            conn.commit()

    def execute(self, sql: str, params: Iterable[Any] = ()) -> None:
        with self._lock, self.connection() as conn:
            conn.execute(sql, tuple(params))
            conn.commit()

    def fetch_all(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._lock, self.connection() as conn:
            cursor = conn.execute(sql, tuple(params))
            return cursor.fetchall()

    def fetch_one(self, sql: str, params: Iterable[Any] = ()) -> Optional[sqlite3.Row]:
        rows = self.fetch_all(sql, params)
        return rows[0] if rows else None

    def create_task(self, task_id: str, task_type: str, mode: str | None) -> None:
        now = _utcnow()
        self.execute(
            """
            INSERT INTO tasks(task_id, task_type, mode, status, progress, message, result_json, error, created_at, updated_at)
            VALUES (?, ?, ?, 'queued', 0.0, '', '{}', '', ?, ?)
            """,
            (task_id, task_type, mode, now, now),
        )

    def update_task(
        self,
        task_id: str,
        *,
        status: str,
        progress: float | None = None,
        message: str | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        row = self.fetch_one("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
        if row is None:
            return
        payload = _loads(row["result_json"])
        if result is not None:
            payload = result
        self.execute(
            """
            UPDATE tasks
            SET status = ?, progress = ?, message = ?, result_json = ?, error = ?, updated_at = ?
            WHERE task_id = ?
            """,
            (
                status,
                progress if progress is not None else row["progress"],
                message if message is not None else row["message"],
                _dumps(payload),
                error if error is not None else row["error"],
                _utcnow(),
                task_id,
            ),
        )

    def list_tasks(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.fetch_all(
            """
            SELECT *
            FROM tasks
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [
            {
                "task_id": row["task_id"],
                "task_type": row["task_type"],
                "mode": row["mode"],
                "status": row["status"],
                "progress": row["progress"],
                "message": row["message"],
                "result": _loads(row["result_json"]),
                "error": row["error"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def get_task(self, task_id: str) -> Optional[dict[str, Any]]:
        row = self.fetch_one("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
        if row is None:
            return None
        return {
            "task_id": row["task_id"],
            "task_type": row["task_type"],
            "mode": row["mode"],
            "status": row["status"],
            "progress": row["progress"],
            "message": row["message"],
            "result": _loads(row["result_json"]),
            "error": row["error"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def save_mode_state(self, mode: str, payload: dict[str, Any]) -> None:
        self.execute(
            """
            INSERT INTO mode_state(mode, payload_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(mode) DO UPDATE SET payload_json=excluded.payload_json, updated_at=excluded.updated_at
            """,
            (mode, _dumps(payload), _utcnow()),
        )

    def load_mode_state(self, mode: str) -> dict[str, Any]:
        row = self.fetch_one("SELECT payload_json FROM mode_state WHERE mode = ?", (mode,))
        return _loads(row["payload_json"]) if row else {}

    def list_mode_state(self) -> list[dict[str, Any]]:
        rows = self.fetch_all("SELECT mode, payload_json FROM mode_state ORDER BY mode")
        return [{"mode": row["mode"], **_loads(row["payload_json"])} for row in rows]

