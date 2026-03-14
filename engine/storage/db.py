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


class StateStore:
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
            CREATE TABLE IF NOT EXISTS symbols (
                symbol TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                sector TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS daily_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                as_of_date TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS text_events (
                event_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                published_at TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_name TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                url TEXT NOT NULL,
                importance_hint REAL NOT NULL,
                sentiment_hint REAL NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS research_cards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                as_of_date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                provider_name TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS portfolio_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                as_of_date TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS risk_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                as_of_date TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                status TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                limit_price REAL NOT NULL,
                target_weight REAL NOT NULL,
                notes_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS fills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id INTEGER,
                run_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                price REAL NOT NULL,
                fees REAL NOT NULL,
                status TEXT NOT NULL,
                notes_json TEXT NOT NULL,
                filled_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS positions (
                symbol TEXT PRIMARY KEY,
                quantity INTEGER NOT NULL,
                avg_cost REAL NOT NULL,
                last_price REAL NOT NULL,
                market_value REAL NOT NULL,
                industry TEXT NOT NULL,
                acquired_at TEXT NOT NULL,
                peak_price REAL NOT NULL,
                stop_loss_pct REAL NOT NULL,
                take_profit_pct REAL NOT NULL,
                time_stop_days INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS daily_runs (
                run_id TEXT PRIMARY KEY,
                as_of_date TEXT NOT NULL,
                status TEXT NOT NULL,
                degrade_mode INTEGER NOT NULL,
                stage TEXT NOT NULL,
                summary_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS model_memory (
                key TEXT PRIMARY KEY,
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

    def executemany(self, sql: str, rows: Iterable[Iterable[Any]]) -> None:
        with self._lock, self.connection() as conn:
            conn.executemany(sql, rows)
            conn.commit()

    def fetch_all(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._lock, self.connection() as conn:
            cursor = conn.execute(sql, tuple(params))
            return cursor.fetchall()

    def fetch_one(self, sql: str, params: Iterable[Any] = ()) -> Optional[sqlite3.Row]:
        rows = self.fetch_all(sql, params)
        return rows[0] if rows else None

    def upsert_symbol(self, symbol: str, name: str, sector: str) -> None:
        self.execute(
            """
            INSERT INTO symbols(symbol, name, sector, is_active, updated_at)
            VALUES (?, ?, ?, 1, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                name=excluded.name,
                sector=excluded.sector,
                is_active=1,
                updated_at=excluded.updated_at
            """,
            (symbol, name, sector, _utcnow()),
        )

    def replace_text_events(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        payload_rows = [
            (
                row["event_id"],
                row["symbol"],
                row["published_at"],
                row["source_type"],
                row["source_name"],
                row["title"],
                row["content"],
                row.get("url", ""),
                row.get("importance_hint", 0.0),
                row.get("sentiment_hint", 0.0),
                _utcnow(),
            )
            for row in rows
        ]
        self.executemany(
            """
            INSERT INTO text_events(
                event_id, symbol, published_at, source_type, source_name, title, content, url,
                importance_hint, sentiment_hint, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(event_id) DO UPDATE SET
                title=excluded.title,
                content=excluded.content,
                url=excluded.url,
                importance_hint=excluded.importance_hint,
                sentiment_hint=excluded.sentiment_hint
            """,
            payload_rows,
        )

    def save_research_cards(self, run_id: str, as_of_date: str, cards: list[dict[str, Any]]) -> None:
        self.execute("DELETE FROM research_cards WHERE run_id = ?", (run_id,))
        self.executemany(
            """
            INSERT INTO research_cards(run_id, as_of_date, symbol, provider_name, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (run_id, as_of_date, card["symbol"], card.get("provider_name", "system"), json.dumps(card), _utcnow())
                for card in cards
            ],
        )

    def save_portfolio_decision(self, run_id: str, as_of_date: str, payload: dict[str, Any]) -> None:
        self.execute("DELETE FROM portfolio_decisions WHERE run_id = ?", (run_id,))
        self.execute(
            """
            INSERT INTO portfolio_decisions(run_id, as_of_date, payload_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (run_id, as_of_date, json.dumps(payload), _utcnow()),
        )

    def save_risk_review(self, run_id: str, as_of_date: str, payload: dict[str, Any]) -> None:
        self.execute("DELETE FROM risk_reviews WHERE run_id = ?", (run_id,))
        self.execute(
            """
            INSERT INTO risk_reviews(run_id, as_of_date, payload_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (run_id, as_of_date, json.dumps(payload), _utcnow()),
        )

    def save_daily_snapshot(self, run_id: str, as_of_date: str, payload: dict[str, Any]) -> None:
        self.execute(
            """
            INSERT INTO daily_snapshots(run_id, as_of_date, payload_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (run_id, as_of_date, json.dumps(payload), _utcnow()),
        )

    def save_run(self, run_id: str, as_of_date: str, status: str, stage: str, degrade_mode: bool, summary: dict[str, Any]) -> None:
        now = _utcnow()
        self.execute(
            """
            INSERT INTO daily_runs(run_id, as_of_date, status, degrade_mode, stage, summary_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                status=excluded.status,
                degrade_mode=excluded.degrade_mode,
                stage=excluded.stage,
                summary_json=excluded.summary_json,
                updated_at=excluded.updated_at
            """,
            (run_id, as_of_date, status, int(degrade_mode), stage, json.dumps(summary), now, now),
        )

    def save_orders(self, run_id: str, rows: list[dict[str, Any]]) -> None:
        self.execute("DELETE FROM orders WHERE run_id = ?", (run_id,))
        if not rows:
            return
        self.executemany(
            """
            INSERT INTO orders(run_id, symbol, action, status, quantity, limit_price, target_weight, notes_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    row["symbol"],
                    row["action"],
                    row["status"],
                    row["quantity"],
                    row["limit_price"],
                    row["target_weight"],
                    json.dumps(row.get("notes", [])),
                    _utcnow(),
                )
                for row in rows
            ],
        )

    def save_fills(self, run_id: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        self.executemany(
            """
            INSERT INTO fills(order_id, run_id, symbol, action, quantity, price, fees, status, notes_json, filled_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row.get("order_id"),
                    run_id,
                    row["symbol"],
                    row["action"],
                    row["quantity"],
                    row["price"],
                    row["fees"],
                    row["status"],
                    json.dumps(row.get("notes", [])),
                    row["filled_at"],
                )
                for row in rows
            ],
        )

    def replace_positions(self, rows: list[dict[str, Any]]) -> None:
        self.execute("DELETE FROM positions", ())
        if not rows:
            return
        self.executemany(
            """
            INSERT INTO positions(
                symbol, quantity, avg_cost, last_price, market_value, industry, acquired_at,
                peak_price, stop_loss_pct, take_profit_pct, time_stop_days, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row["symbol"],
                    row["quantity"],
                    row["avg_cost"],
                    row["last_price"],
                    row["market_value"],
                    row["industry"],
                    row["acquired_at"],
                    row["peak_price"],
                    row["stop_loss_pct"],
                    row["take_profit_pct"],
                    row["time_stop_days"],
                    _utcnow(),
                )
                for row in rows
            ],
        )

    def save_memory(self, key: str, payload: dict[str, Any]) -> None:
        self.execute(
            """
            INSERT INTO model_memory(key, payload_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET payload_json=excluded.payload_json, updated_at=excluded.updated_at
            """,
            (key, json.dumps(payload), _utcnow()),
        )

    def load_memory(self, key: str) -> dict[str, Any]:
        row = self.fetch_one("SELECT payload_json FROM model_memory WHERE key = ?", (key,))
        return json.loads(row["payload_json"]) if row else {}

    def latest_run(self) -> Optional[dict[str, Any]]:
        row = self.fetch_one(
            "SELECT run_id, as_of_date, status, degrade_mode, stage, summary_json FROM daily_runs ORDER BY created_at DESC LIMIT 1"
        )
        if row is None:
            return None
        return {
            "run_id": row["run_id"],
            "as_of_date": row["as_of_date"],
            "status": row["status"],
            "degrade_mode": bool(row["degrade_mode"]),
            "stage": row["stage"],
            "summary": json.loads(row["summary_json"]),
        }

    def list_positions(self) -> list[dict[str, Any]]:
        rows = self.fetch_all("SELECT * FROM positions ORDER BY market_value DESC")
        return [dict(row) for row in rows]

    def list_events(self, symbol: str, as_of_date: str, limit: int = 8) -> list[dict[str, Any]]:
        rows = self.fetch_all(
            """
            SELECT * FROM text_events
            WHERE symbol = ? AND published_at <= ?
            ORDER BY published_at DESC
            LIMIT ?
            """,
            (symbol, as_of_date, limit),
        )
        return [dict(row) for row in rows]

    def list_cards_for_run(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.fetch_all(
            "SELECT payload_json FROM research_cards WHERE run_id = ? ORDER BY symbol",
            (run_id,),
        )
        return [json.loads(row["payload_json"]) for row in rows]

    def load_decision_for_run(self, run_id: str) -> Optional[dict[str, Any]]:
        row = self.fetch_one("SELECT payload_json FROM portfolio_decisions WHERE run_id = ?", (run_id,))
        return json.loads(row["payload_json"]) if row else None

    def load_risk_for_run(self, run_id: str) -> Optional[dict[str, Any]]:
        row = self.fetch_one("SELECT payload_json FROM risk_reviews WHERE run_id = ?", (run_id,))
        return json.loads(row["payload_json"]) if row else None

    def load_backtest_runs(self) -> list[dict[str, Any]]:
        rows = self.fetch_all("SELECT as_of_date, summary_json FROM daily_runs ORDER BY as_of_date")
        return [{"as_of_date": row["as_of_date"], "summary": json.loads(row["summary_json"])} for row in rows]
