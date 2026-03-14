from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from engine.config import Settings
from engine.types import PositionState, PriorSignal, RiskDecision, RiskReview, TradeIntentSet


class RiskService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def review(
        self,
        as_of_date: datetime,
        decision: TradeIntentSet,
        priors: dict[str, PriorSignal],
        positions: dict[str, PositionState],
        latest_nav: float,
        peak_nav: float,
    ) -> RiskReview:
        sector_usage = defaultdict(float)
        current_usage = 0.0
        for symbol, position in positions.items():
            prior = priors.get(symbol)
            if prior is None:
                continue
            weight = 0.0 if latest_nav <= 0 else position.market_value / latest_nav
            sector_usage[prior.sector] += weight
            current_usage += weight
        approved: list[RiskDecision] = []
        clipped: list[RiskDecision] = []
        delayed: list[RiskDecision] = []
        rejected: list[RiskDecision] = []
        risk_flags: list[str] = list(decision.portfolio_risks)
        turnover_budget = self.settings.max_daily_turnover
        drawdown = 0.0 if peak_nav <= 0 else max(0.0, 1.0 - latest_nav / peak_nav)
        no_new_entries = drawdown >= self.settings.portfolio_drawdown_limit or "LLM unavailable" in decision.rationale
        if drawdown >= self.settings.portfolio_drawdown_limit:
            risk_flags.append("portfolio_drawdown_circuit_breaker")
        for intent in decision.trade_intents:
            prior = priors.get(intent.symbol)
            if prior is None:
                rejected.append(
                    RiskDecision(
                        symbol=intent.symbol,
                        requested_action=intent.action,
                        requested_weight=intent.target_weight,
                        approved_weight=0.0,
                        final_status="rejected",
                        risk_flags=["missing_prior"],
                        rationale="No prior signal available.",
                    )
                )
                continue
            flags: list[str] = []
            current_weight = self._position_weight(intent.symbol, positions, latest_nav)
            approved_weight = min(intent.target_weight, intent.max_weight, self.settings.max_position_weight)
            status = "approved"
            if intent.symbol in self.settings.blacklist_symbols:
                status = "rejected"
                approved_weight = 0.0
                flags.append("blacklisted")
            if intent.action == "buy" and intent.evidence_count < 2:
                status = "rejected"
                approved_weight = 0.0
                flags.append("insufficient_evidence")
            if intent.action == "buy" and prior.liquidity_score < 0.35:
                status = "rejected"
                approved_weight = 0.0
                flags.append("insufficient_liquidity")
            if intent.action == "buy" and prior.prior_avoid_score >= 0.75:
                approved_weight = min(approved_weight, self.settings.max_position_weight * 0.4)
                status = "clipped" if approved_weight > 0 else "rejected"
                flags.append("high_prior_avoid")
            if intent.action == "buy" and prior.downside_risk_score >= 0.72:
                approved_weight = min(approved_weight, self.settings.max_position_weight * 0.5)
                status = "clipped" if approved_weight > 0 else "rejected"
                flags.append("high_downside_risk")
            if no_new_entries and intent.action == "buy":
                status = "rejected"
                approved_weight = 0.0
                flags.append("degraded_no_new_entries")
            sector_after = sector_usage[prior.sector] + approved_weight
            if intent.action == "buy" and sector_after > self.settings.max_sector_weight:
                approved_weight = max(0.0, self.settings.max_sector_weight - sector_usage[prior.sector])
                status = "clipped" if approved_weight > 0 else "rejected"
                flags.append("sector_cap")
            projected_gross = current_usage + max(0.0, approved_weight - current_weight)
            if intent.action == "buy" and projected_gross > self.settings.max_gross_exposure:
                approved_weight = max(0.0, self.settings.max_gross_exposure - current_usage)
                status = "clipped" if approved_weight > 0 else "rejected"
                flags.append("gross_exposure_cap")
            turnover_need = abs(approved_weight - current_weight)
            if turnover_need > turnover_budget:
                approved_weight = max(0.0, current_weight + turnover_budget)
                status = "delayed" if approved_weight <= current_weight else "clipped"
                flags.append("daily_turnover_cap")
                turnover_need = turnover_budget
            turnover_budget = max(0.0, turnover_budget - turnover_need)
            decision_row = RiskDecision(
                symbol=intent.symbol,
                requested_action=intent.action,
                requested_weight=float(intent.target_weight),
                approved_weight=float(max(0.0, approved_weight)),
                final_status=status,  # type: ignore[arg-type]
                risk_flags=flags,
                rationale=intent.thesis,
            )
            if status == "approved":
                approved.append(decision_row)
                sector_usage[prior.sector] += approved_weight
                current_usage += max(0.0, approved_weight - current_weight)
            elif status == "clipped":
                clipped.append(decision_row)
                sector_usage[prior.sector] += approved_weight
                current_usage += max(0.0, approved_weight - current_weight)
            elif status == "delayed":
                delayed.append(decision_row)
            else:
                rejected.append(decision_row)
        for symbol, position in positions.items():
            if position.last_price <= position.avg_cost * (1.0 - position.stop_loss_pct):
                risk_flags.append(f"fixed_stop:{symbol}")
            if position.last_price >= position.avg_cost * (1.0 + position.take_profit_pct):
                risk_flags.append(f"take_profit_watch:{symbol}")
        return RiskReview(
            as_of_date=as_of_date,
            approved=approved,
            clipped=clipped,
            delayed=delayed,
            rejected=rejected,
            risk_flags=sorted(set(risk_flags)),
        )

    @staticmethod
    def _position_weight(symbol: str, positions: dict[str, PositionState], nav: float) -> float:
        position = positions.get(symbol)
        if position is None or nav <= 0:
            return 0.0
        return position.market_value / nav

