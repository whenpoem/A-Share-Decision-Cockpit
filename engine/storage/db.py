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
                order_ref TEXT,
                run_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                status TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                limit_price REAL NOT NULL,
                target_weight REAL NOT NULL,
                notes_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS fills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_ref TEXT,
                run_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                price REAL NOT NULL,
                fees REAL NOT NULL,
                status TEXT NOT NULL,
                notes_json TEXT NOT NULL,
                realized_pnl REAL NOT NULL DEFAULT 0.0,
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
            """
            CREATE TABLE IF NOT EXISTS approval_tickets (
                ticket_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS position_memory (
                symbol TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS decision_journal (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                as_of_date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                stage TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS portfolio_memory_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                as_of_date TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS backtest_runs (
                run_id TEXT PRIMARY KEY,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                status TEXT NOT NULL,
                summary_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
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

    def reset_runtime_state(self) -> None:
        tables = [
            "positions",
            "orders",
            "fills",
            "approval_tickets",
            "decision_journal",
            "daily_runs",
            "daily_snapshots",
            "research_cards",
            "portfolio_decisions",
            "risk_reviews",
            "position_memory",
            "portfolio_memory_history",
            "model_memory",
        ]
        with self._lock, self.connection() as conn:
            for table in tables:
                conn.execute(f"DELETE FROM {table}")
            conn.commit()

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

    def list_symbols(self, limit: int | None = None) -> list[dict[str, Any]]:
        if limit is None:
            rows = self.fetch_all(
                """
                SELECT symbol, name, sector
                FROM symbols
                WHERE is_active = 1
                ORDER BY updated_at DESC, symbol
                """
            )
        else:
            rows = self.fetch_all(
                """
                SELECT symbol, name, sector
                FROM symbols
                WHERE is_active = 1
                ORDER BY updated_at DESC, symbol
                LIMIT ?
                """,
                (limit,),
            )
        return [dict(row) for row in rows]

    def load_symbols(self, symbols: list[str]) -> list[dict[str, Any]]:
        if not symbols:
            return []
        placeholders = ",".join("?" for _ in symbols)
        rows = self.fetch_all(
            f"""
            SELECT symbol, name, sector
            FROM symbols
            WHERE is_active = 1 AND symbol IN ({placeholders})
            """,
            symbols,
        )
        by_symbol = {row["symbol"]: dict(row) for row in rows}
        return [by_symbol[symbol] for symbol in symbols if symbol in by_symbol]

    def replace_text_events(self, rows: list[dict[str, Any]]) -> None:
        with self._lock, self.connection() as conn:
            conn.execute("DELETE FROM text_events")
            if rows:
                conn.executemany(
                    """
                    INSERT INTO text_events(
                        event_id, symbol, published_at, source_type, source_name, title, content, url,
                        importance_hint, sentiment_hint, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
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
                    ],
                )
            conn.commit()

    def save_research_cards(self, run_id: str, as_of_date: str, cards: list[dict[str, Any]]) -> None:
        self.execute("DELETE FROM research_cards WHERE run_id = ?", (run_id,))
        if not cards:
            return
        self.executemany(
            """
            INSERT INTO research_cards(run_id, as_of_date, symbol, provider_name, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    as_of_date,
                    card["symbol"],
                    card.get("provider_name", "system"),
                    _dumps(card),
                    _utcnow(),
                )
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
            (run_id, as_of_date, _dumps(payload), _utcnow()),
        )

    def save_risk_review(self, run_id: str, as_of_date: str, payload: dict[str, Any]) -> None:
        self.execute("DELETE FROM risk_reviews WHERE run_id = ?", (run_id,))
        self.execute(
            """
            INSERT INTO risk_reviews(run_id, as_of_date, payload_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (run_id, as_of_date, _dumps(payload), _utcnow()),
        )

    def save_daily_snapshot(self, run_id: str, as_of_date: str, payload: dict[str, Any]) -> None:
        self.execute(
            """
            INSERT INTO daily_snapshots(run_id, as_of_date, payload_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (run_id, as_of_date, _dumps(payload), _utcnow()),
        )

    def load_daily_snapshot(self, run_id: str) -> dict[str, Any]:
        row = self.fetch_one(
            """
            SELECT payload_json
            FROM daily_snapshots
            WHERE run_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (run_id,),
        )
        return _loads(row["payload_json"]) if row else {}

    def save_run(
        self,
        run_id: str,
        as_of_date: str,
        status: str,
        stage: str,
        degrade_mode: bool,
        summary: dict[str, Any],
    ) -> None:
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
            (run_id, as_of_date, status, int(degrade_mode), stage, _dumps(summary), now, now),
        )

    def save_orders(self, run_id: str, rows: list[dict[str, Any]]) -> None:
        self.execute("DELETE FROM orders WHERE run_id = ?", (run_id,))
        if not rows:
            return
        now = _utcnow()
        self.executemany(
            """
            INSERT INTO orders(order_ref, run_id, symbol, action, status, quantity, limit_price, target_weight, notes_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row.get("order_ref", ""),
                    run_id,
                    row["symbol"],
                    row["action"],
                    row["status"],
                    row["quantity"],
                    row["limit_price"],
                    row["target_weight"],
                    _dumps(row.get("notes", [])),
                    now,
                    now,
                )
                for row in rows
            ],
        )

    def update_order_status(self, order_ref: str, status: str, notes: list[str] | None = None) -> None:
        row = self.fetch_one("SELECT notes_json FROM orders WHERE order_ref = ?", (order_ref,))
        if row is None:
            return
        existing = _loads(row["notes_json"])
        merged = list(dict.fromkeys([*(existing or []), *((notes or []))]))
        self.execute(
            "UPDATE orders SET status = ?, notes_json = ?, updated_at = ? WHERE order_ref = ?",
            (status, _dumps(merged), _utcnow(), order_ref),
        )

    def save_fills(self, run_id: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        self.executemany(
            """
            INSERT INTO fills(order_ref, run_id, symbol, action, quantity, price, fees, status, notes_json, realized_pnl, filled_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row.get("order_ref", ""),
                    run_id,
                    row["symbol"],
                    row["action"],
                    row["quantity"],
                    row["price"],
                    row["fees"],
                    row["status"],
                    _dumps(row.get("notes", [])),
                    float(row.get("realized_pnl", 0.0)),
                    row["filled_at"],
                )
                for row in rows
            ],
        )

    def replace_positions(self, rows: list[dict[str, Any]]) -> None:
        with self._lock, self.connection() as conn:
            conn.execute("DELETE FROM positions")
            if rows:
                conn.executemany(
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
            conn.commit()

    def save_memory(self, key: str, payload: dict[str, Any]) -> None:
        self.execute(
            """
            INSERT INTO model_memory(key, payload_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET payload_json=excluded.payload_json, updated_at=excluded.updated_at
            """,
            (key, _dumps(payload), _utcnow()),
        )

    def load_memory(self, key: str) -> dict[str, Any]:
        row = self.fetch_one("SELECT payload_json FROM model_memory WHERE key = ?", (key,))
        return _loads(row["payload_json"]) if row else {}

    def upsert_position_memory(self, symbol: str, payload: dict[str, Any]) -> None:
        self.execute(
            """
            INSERT INTO position_memory(symbol, payload_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET payload_json=excluded.payload_json, updated_at=excluded.updated_at
            """,
            (symbol, _dumps(payload), _utcnow()),
        )

    def load_position_memory(self, symbol: str) -> dict[str, Any]:
        row = self.fetch_one("SELECT payload_json FROM position_memory WHERE symbol = ?", (symbol,))
        return _loads(row["payload_json"]) if row else {}

    def list_position_memories(self) -> list[dict[str, Any]]:
        rows = self.fetch_all("SELECT payload_json FROM position_memory ORDER BY symbol")
        return [_loads(row["payload_json"]) for row in rows]

    def append_decision_journal(
        self,
        entry_id: str,
        run_id: str,
        as_of_date: str,
        symbol: str,
        stage: str,
        payload: dict[str, Any],
    ) -> None:
        self.execute(
            """
            INSERT OR REPLACE INTO decision_journal(id, run_id, as_of_date, symbol, stage, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (entry_id, run_id, as_of_date, symbol, stage, _dumps(payload), _utcnow()),
        )

    def list_decision_journal(self, limit: int = 100, symbol: str | None = None) -> list[dict[str, Any]]:
        if symbol:
            rows = self.fetch_all(
                """
                SELECT id, run_id, as_of_date, symbol, stage, payload_json
                FROM decision_journal
                WHERE symbol = ?
                ORDER BY as_of_date DESC, created_at DESC
                LIMIT ?
                """,
                (symbol, limit),
            )
        else:
            rows = self.fetch_all(
                """
                SELECT id, run_id, as_of_date, symbol, stage, payload_json
                FROM decision_journal
                ORDER BY as_of_date DESC, created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
        return [
            {
                "id": row["id"],
                "run_id": row["run_id"],
                "as_of_date": row["as_of_date"],
                "symbol": row["symbol"],
                "stage": row["stage"],
                "payload": _loads(row["payload_json"]),
            }
            for row in rows
        ]

    def append_portfolio_memory(self, as_of_date: str, payload: dict[str, Any]) -> None:
        self.execute(
            """
            INSERT INTO portfolio_memory_history(as_of_date, payload_json, created_at)
            VALUES (?, ?, ?)
            """,
            (as_of_date, _dumps(payload), _utcnow()),
        )

    def latest_portfolio_memory(self) -> dict[str, Any]:
        row = self.fetch_one(
            "SELECT payload_json FROM portfolio_memory_history ORDER BY as_of_date DESC, id DESC LIMIT 1"
        )
        return _loads(row["payload_json"]) if row else {}

    def list_portfolio_memory(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.fetch_all(
            """
            SELECT as_of_date, payload_json
            FROM portfolio_memory_history
            ORDER BY as_of_date DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [{"as_of_date": row["as_of_date"], "payload": _loads(row["payload_json"])} for row in rows]

    def save_approval_tickets(self, tickets: list[dict[str, Any]]) -> None:
        if not tickets:
            return
        self.executemany(
            """
            INSERT OR REPLACE INTO approval_tickets(ticket_id, run_id, symbol, side, status, payload_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    ticket["ticket_id"],
                    ticket["run_id"],
                    ticket["symbol"],
                    ticket["side"],
                    ticket["status"],
                    _dumps(ticket),
                    ticket["created_at"],
                    ticket["updated_at"],
                )
                for ticket in tickets
            ],
        )

    def get_approval_ticket(self, ticket_id: str) -> Optional[dict[str, Any]]:
        row = self.fetch_one("SELECT payload_json FROM approval_tickets WHERE ticket_id = ?", (ticket_id,))
        return _loads(row["payload_json"]) if row else None

    def list_approval_tickets(self, status: str | None = None) -> list[dict[str, Any]]:
        if status:
            rows = self.fetch_all(
                """
                SELECT payload_json
                FROM approval_tickets
                WHERE status = ?
                ORDER BY updated_at DESC
                """,
                (status,),
            )
        else:
            rows = self.fetch_all(
                "SELECT payload_json FROM approval_tickets ORDER BY updated_at DESC"
            )
        return [_loads(row["payload_json"]) for row in rows]

    def update_approval_ticket(
        self,
        ticket_id: str,
        *,
        status: str,
        broker_order_id: str = "",
    ) -> Optional[dict[str, Any]]:
        ticket = self.get_approval_ticket(ticket_id)
        if ticket is None:
            return None
        ticket["status"] = status
        ticket["updated_at"] = _utcnow()
        if broker_order_id:
            ticket["broker_order_id"] = broker_order_id
        self.execute(
            """
            UPDATE approval_tickets
            SET status = ?, payload_json = ?, updated_at = ?
            WHERE ticket_id = ?
            """,
            (status, _dumps(ticket), ticket["updated_at"], ticket_id),
        )
        return ticket

    def save_backtest_run(
        self,
        run_id: str,
        start_date: str,
        end_date: str,
        status: str,
        summary: dict[str, Any],
    ) -> None:
        now = _utcnow()
        self.execute(
            """
            INSERT INTO backtest_runs(run_id, start_date, end_date, status, summary_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                status=excluded.status,
                summary_json=excluded.summary_json,
                updated_at=excluded.updated_at
            """,
            (run_id, start_date, end_date, status, _dumps(summary), now, now),
        )

    def list_backtest_runs(self) -> list[dict[str, Any]]:
        rows = self.fetch_all(
            """
            SELECT run_id, start_date, end_date, status, summary_json
            FROM backtest_runs
            ORDER BY updated_at DESC
            """
        )
        return [
            {
                "run_id": row["run_id"],
                "start_date": row["start_date"],
                "end_date": row["end_date"],
                "status": row["status"],
                "summary": _loads(row["summary_json"]),
            }
            for row in rows
        ]

    def load_backtest_run(self, run_id: str) -> Optional[dict[str, Any]]:
        row = self.fetch_one(
            """
            SELECT run_id, start_date, end_date, status, summary_json
            FROM backtest_runs
            WHERE run_id = ?
            """,
            (run_id,),
        )
        if row is None:
            return None
        return {
            "run_id": row["run_id"],
            "start_date": row["start_date"],
            "end_date": row["end_date"],
            "status": row["status"],
            "summary": _loads(row["summary_json"]),
        }

    def latest_run(self) -> Optional[dict[str, Any]]:
        row = self.fetch_one(
            """
            SELECT run_id, as_of_date, status, degrade_mode, stage, summary_json
            FROM daily_runs
            ORDER BY updated_at DESC
            LIMIT 1
            """
        )
        if row is None:
            return None
        return {
            "run_id": row["run_id"],
            "as_of_date": row["as_of_date"],
            "status": row["status"],
            "degrade_mode": bool(row["degrade_mode"]),
            "stage": row["stage"],
            "summary": _loads(row["summary_json"]),
        }

    def list_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.fetch_all(
            """
            SELECT run_id, as_of_date, status, degrade_mode, stage, summary_json
            FROM daily_runs
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [
            {
                "run_id": row["run_id"],
                "as_of_date": row["as_of_date"],
                "status": row["status"],
                "degrade_mode": bool(row["degrade_mode"]),
                "stage": row["stage"],
                "summary": _loads(row["summary_json"]),
            }
            for row in rows
        ]

    def list_positions(self) -> list[dict[str, Any]]:
        rows = self.fetch_all("SELECT * FROM positions ORDER BY market_value DESC")
        return [dict(row) for row in rows]

    def list_events(self, symbol: str, as_of_date: str, limit: int = 8) -> list[dict[str, Any]]:
        rows = self.fetch_all(
            """
            SELECT *
            FROM text_events
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
        return [_loads(row["payload_json"]) for row in rows]

    def load_decision_for_run(self, run_id: str) -> Optional[dict[str, Any]]:
        row = self.fetch_one("SELECT payload_json FROM portfolio_decisions WHERE run_id = ?", (run_id,))
        return _loads(row["payload_json"]) if row else None

    def load_risk_for_run(self, run_id: str) -> Optional[dict[str, Any]]:
        row = self.fetch_one("SELECT payload_json FROM risk_reviews WHERE run_id = ?", (run_id,))
        return _loads(row["payload_json"]) if row else None

    def list_orders(self, run_id: str | None = None) -> list[dict[str, Any]]:
        if run_id:
            rows = self.fetch_all(
                "SELECT * FROM orders WHERE run_id = ? ORDER BY created_at DESC",
                (run_id,),
            )
        else:
            rows = self.fetch_all("SELECT * FROM orders ORDER BY created_at DESC")
        return [
            {
                **dict(row),
                "notes": _loads(row["notes_json"]),
            }
            for row in rows
        ]

    def list_fills(self, run_id: str | None = None) -> list[dict[str, Any]]:
        if run_id:
            rows = self.fetch_all(
                "SELECT * FROM fills WHERE run_id = ? ORDER BY filled_at DESC",
                (run_id,),
            )
        else:
            rows = self.fetch_all("SELECT * FROM fills ORDER BY filled_at DESC")
        return [
            {
                **dict(row),
                "notes": _loads(row["notes_json"]),
            }
            for row in rows
        ]
