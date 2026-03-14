from __future__ import annotations

import json
from datetime import datetime
from uuid import uuid4

from engine.agents.providers import DeepSeekProvider, ProviderChain, QwenProvider
from engine.agents.service import DecisionAgent, ResearchAgent
from engine.config import Settings
from engine.market.service import MarketService
from engine.risk.service import RiskService
from engine.sim.service import SimulationService
from engine.storage.db import StateStore
from engine.text.service import TextService
from typing import Optional

from engine.types import DailyRunSummary, PositionState


class DailyRunner:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings.load()
        self.store = StateStore(self.settings.db_path)
        self.market_service = MarketService(self.settings, self.store)
        self.text_service = TextService(self.settings, self.store)
        self.provider_chain = ProviderChain(
            DeepSeekProvider(self.settings.primary_provider),
            QwenProvider(self.settings.fallback_provider),
        )
        self.research_agent = ResearchAgent(self.settings, self.provider_chain)
        self.decision_agent = DecisionAgent(self.settings, self.provider_chain)
        self.risk_service = RiskService(self.settings)
        self.sim_service = SimulationService(self.settings)

    def refresh_market(self, start_date: str = "2023-01-01", end_date: Optional[str] = None):
        return self.market_service.refresh_market(start_date=start_date, end_date=end_date)

    def refresh_text(self, snapshot, positions: Optional[dict[str, PositionState]] = None):
        return self.text_service.refresh_text(
            snapshot.priors,
            snapshot.financials,
            snapshot.bars,
            positions,
        )

    def run_daily(self, as_of_date: Optional[str] = None) -> DailyRunSummary:
        market_snapshot = self.refresh_market()
        positions = self._load_positions()
        self.refresh_text(market_snapshot, positions)
        cash_balance = self.store.load_memory("portfolio_cash").get("value", self.settings.default_capital)
        run_at = market_snapshot.as_of_date if as_of_date is None else datetime.fromisoformat(as_of_date)
        event_packs = self.text_service.build_event_packs(
            as_of_date=run_at,
            priors=market_snapshot.priors,
            financials=market_snapshot.financials,
            positions=positions,
        )
        run_id = uuid4().hex[:12]
        research_cards = []
        degrade_mode = False
        call_records = []
        for pack in event_packs:
            result = self.research_agent.run(pack)
            degrade_mode = degrade_mode or result.degrade_mode
            research_cards.append(result.payload)
            call_records.extend(result.call_records)
        positions_payload, latest_nav = self._position_payload(positions, cash_balance)
        decision_result = self.decision_agent.run(
            as_of_date=run_at,
            market_view=market_snapshot.priors[0].market_regime if market_snapshot.priors else "neutral",
            cards=research_cards,
            positions_payload=positions_payload,
            priors_payload=[pack.prior.model_dump(mode="json") for pack in event_packs],
        )
        degrade_mode = degrade_mode or decision_result.degrade_mode
        call_records.extend(decision_result.call_records)
        prior_map = {item.symbol: item for item in market_snapshot.priors}
        peak_nav = self.store.load_memory("portfolio_peak_nav").get("value", latest_nav or self.settings.default_capital)
        risk_review = self.risk_service.review(run_at, decision_result.payload, prior_map, positions, latest_nav, peak_nav)
        orders = self.sim_service.prepare_orders(risk_review, prior_map, positions, max(latest_nav, self.settings.default_capital))
        fills, updated_positions, ending_cash = self.sim_service.submit_orders(
            run_at,
            orders,
            market_snapshot.bars,
            positions,
            prior_map,
            cash_balance,
        )
        updated_nav = self._portfolio_nav(updated_positions) + ending_cash
        self.store.save_memory("portfolio_peak_nav", {"value": max(float(peak_nav), updated_nav)})
        self.store.save_memory("portfolio_cash", {"value": ending_cash})
        summary = DailyRunSummary(
            run_id=run_id,
            as_of_date=run_at,
            status="degraded" if degrade_mode else "ok",
            stage="completed",
            degrade_mode=degrade_mode,
            candidate_symbols=[pack.symbol for pack in event_packs],
            approved_symbols=[row.symbol for row in risk_review.approved],
            clipped_symbols=[row.symbol for row in risk_review.clipped],
            rejected_symbols=[row.symbol for row in risk_review.rejected],
            fills=fills,
            risk_flags=risk_review.risk_flags,
            notes=[record["detail"] for record in call_records if not record.get("success", False)],
            cash_balance=ending_cash,
            ending_nav=updated_nav,
        )
        self._persist_run(run_id, run_at, market_snapshot, research_cards, decision_result.payload, risk_review, orders, fills, updated_positions, summary)
        return summary

    def dashboard_summary(self) -> dict:
        latest = self.store.latest_run()
        if latest is None:
            return {
                "run": None,
                "portfolio": {"positions": [], "gross_exposure": 0.0, "nav": self.settings.default_capital, "cash": self.settings.default_capital},
                "signals": [],
                "risk": {"traffic_light": "amber", "flags": ["no_run_yet"]},
            }
        positions = self.store.list_positions()
        cash_balance = self.store.load_memory("portfolio_cash").get("value", self.settings.default_capital)
        nav = (sum(position["market_value"] for position in positions) + cash_balance) or self.settings.default_capital
        gross = sum(position["market_value"] for position in positions) / nav if nav else 0.0
        traffic_light = "red" if latest["degrade_mode"] or gross > self.settings.max_gross_exposure else "green"
        return {
            "run": latest,
            "portfolio": {"positions": positions, "gross_exposure": gross, "nav": nav, "cash": cash_balance},
            "signals": self.store.list_cards_for_run(latest["run_id"]),
            "risk": {"traffic_light": traffic_light, "flags": latest["summary"].get("risk_flags", [])},
        }

    def today_signals(self) -> dict:
        latest = self.store.latest_run()
        if latest is None:
            return {"run": None, "cards": [], "decision": None, "risk": None}
        return {
            "run": latest,
            "cards": self.store.list_cards_for_run(latest["run_id"]),
            "decision": self.store.load_decision_for_run(latest["run_id"]),
            "risk": self.store.load_risk_for_run(latest["run_id"]),
        }

    def current_portfolio(self) -> dict:
        positions = self.store.list_positions()
        cash_balance = self.store.load_memory("portfolio_cash").get("value", self.settings.default_capital)
        nav = sum(position["market_value"] for position in positions) + cash_balance
        return {
            "positions": positions,
            "nav": nav,
            "cash": cash_balance,
            "gross_exposure": sum(position["market_value"] for position in positions) / nav if nav else 0.0,
        }

    def run_details(self, run_id: str) -> dict:
        latest = self.store.fetch_one("SELECT summary_json FROM daily_runs WHERE run_id = ?", (run_id,))
        if latest is None:
            return {"run_id": run_id, "found": False}
        return {
            "run_id": run_id,
            "summary": json.loads(latest["summary_json"]),
            "cards": self.store.list_cards_for_run(run_id),
            "decision": self.store.load_decision_for_run(run_id),
            "risk": self.store.load_risk_for_run(run_id),
        }

    def backtest_summary(self) -> dict:
        runs = self.store.load_backtest_runs()
        equity_curve = []
        nav = self.settings.default_capital
        peak = nav
        max_drawdown = 0.0
        for item in runs:
            nav = float(item["summary"].get("ending_nav", nav))
            peak = max(peak, nav)
            max_drawdown = max(max_drawdown, 1.0 - nav / peak if peak else 0.0)
            equity_curve.append({"as_of_date": item["as_of_date"], "nav": nav})
        total_return = (nav / self.settings.default_capital) - 1.0 if self.settings.default_capital else 0.0
        return {
            "runs": len(runs),
            "equity_curve": equity_curve,
            "total_return": total_return,
            "max_drawdown": max_drawdown,
        }

    def _persist_run(self, run_id, as_of, market_snapshot, research_cards, decision, risk_review, orders, fills, updated_positions, summary) -> None:
        as_of_str = as_of.isoformat()
        self.store.save_research_cards(run_id, as_of_str, [card.model_dump(mode="json") for card in research_cards])
        self.store.save_portfolio_decision(run_id, as_of_str, decision.model_dump(mode="json"))
        self.store.save_risk_review(run_id, as_of_str, risk_review.model_dump(mode="json"))
        self.store.save_orders(
            run_id,
            [
                dict(order.model_dump(mode="json"), status="prepared", notes=[])
                for order in orders
            ],
        )
        self.store.save_fills(run_id, [fill.model_dump(mode="json") for fill in fills])
        self.store.replace_positions([position.model_dump(mode="json") for position in updated_positions.values()])
        self.store.save_daily_snapshot(
            run_id,
            as_of_str,
            {
                "priors": [item.model_dump(mode="json") for item in market_snapshot.priors],
                "summary": summary.model_dump(mode="json"),
            },
        )
        self.store.save_run(run_id, as_of_str, summary.status, summary.stage, summary.degrade_mode, summary.model_dump(mode="json"))
        artifact_path = self.settings.artifact_storage / f"{run_id}.json"
        artifact_path.write_text(
            json.dumps(
                {
                    "summary": summary.model_dump(mode="json"),
                    "cards": [card.model_dump(mode="json") for card in research_cards],
                    "decision": decision.model_dump(mode="json"),
                    "risk_review": risk_review.model_dump(mode="json"),
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    def _load_positions(self) -> dict[str, PositionState]:
        return {row["symbol"]: PositionState.model_validate(row) for row in self.store.list_positions()}

    @staticmethod
    def _portfolio_nav(positions: dict[str, PositionState]) -> float:
        return sum(position.market_value for position in positions.values()) or 0.0

    def _position_payload(self, positions: dict[str, PositionState], cash_balance: float) -> tuple[list[dict], float]:
        nav = self._portfolio_nav(positions) + cash_balance
        if nav <= 0:
            return [], self.settings.default_capital
        payload = []
        for position in positions.values():
            payload.append(
                {
                    "symbol": position.symbol,
                    "weight": position.market_value / nav,
                    "avg_cost": position.avg_cost,
                    "last_price": position.last_price,
                    "market_value": position.market_value,
                    "industry": position.industry,
                }
            )
        return payload, nav
