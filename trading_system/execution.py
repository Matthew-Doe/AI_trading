from __future__ import annotations

import math
from datetime import datetime
from typing import Iterable
from zoneinfo import ZoneInfo

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    OrderClass,
    StopLossRequest,
    TakeProfitRequest,
)

from trading_system.config import TradingConfig
from trading_system.models import (
    HeldPositionSignal,
    OrderPlan,
    PendingOrderReview,
    SymbolMarketData,
    TradeDecision,
)
from trading_system.telegram import TelegramNotifier


class ExecutionError(RuntimeError):
    pass


class AlpacaExecutionEngine:
    def __init__(self, config: TradingConfig, logger, run_id: str | None = None):
        self.config = config
        self.logger = logger
        self.run_id = run_id or "manual-run"
        self.client = TradingClient(
            api_key=config.alpaca_api_key,
            secret_key=config.alpaca_secret_key,
            paper=True,
            url_override=config.alpaca_paper_base_url,
        )
        self.telegram = TelegramNotifier(config, logger)

    def _standard_trade_cap(
        self,
        *,
        side: str,
        equity: float,
        buying_power: float,
        cash: float,
    ) -> tuple[float, float, str]:
        capital_base = min(equity, buying_power)
        if (
            side == "long"
            and equity > 0
            and cash / equity >= self.config.cash_rich_available_cash_threshold
        ):
            return (
                cash * self.config.cash_rich_trade_pct,
                self.config.cash_rich_trade_pct,
                "available cash",
            )
        return (
            capital_base * self.config.max_single_trade_pct,
            self.config.max_single_trade_pct,
            "capital",
        )

    def build_order_plans(
        self,
        decisions: Iterable[TradeDecision],
        selected_symbols: list[SymbolMarketData],
    ) -> list[OrderPlan]:
        account = self.client.get_account()
        equity = float(account.equity)
        buying_power = float(account.buying_power)
        cash = float(getattr(account, "cash", buying_power))
        if self._daily_loss_limit_reached(account):
            self.logger.warning("Skipping new orders because the daily loss limit has been reached.")
            return []
        open_positions = {position.symbol for position in self.client.get_all_positions()}
        capital_base = min(equity, buying_power)

        symbol_map = {item.symbol: item for item in selected_symbols}
        actionables = [item for item in decisions if item.action in {"long", "short"} and item.allocation > 0]
        total_allocation = sum(item.allocation for item in actionables)
        if total_allocation > self.config.max_total_exposure:
            scale = self.config.max_total_exposure / total_allocation
            for item in actionables:
                item.allocation *= scale

        plans: list[OrderPlan] = []
        for decision in actionables:
            if decision.symbol in open_positions:
                self.logger.info("Skipping %s because a position already exists.", decision.symbol)
                continue
            if decision.action == "short" and not self.config.allow_shorting:
                self.logger.info("Skipping %s short because shorting is disabled.", decision.symbol)
                continue

            market_data = symbol_map.get(decision.symbol)
            if market_data is None:
                self.logger.warning("No market data found for %s during execution.", decision.symbol)
                continue

            stop_distance = max(market_data.indicators.atr14, market_data.close * 0.01)
            planned_stop_distance = max(
                market_data.indicators.atr14 * self.config.stop_atr_multiple,
                market_data.close * 0.01,
            )
            risk_budget = capital_base * self.config.max_risk_per_trade
            risk_qty = int(risk_budget / planned_stop_distance) if planned_stop_distance > 0 else 0
            alloc_notional = capital_base * decision.allocation
            alloc_qty = int(alloc_notional / market_data.close) if market_data.close > 0 else 0
            max_single_trade_notional, max_trade_pct_used, max_trade_cap_basis = self._standard_trade_cap(
                side=decision.action,
                equity=equity,
                buying_power=buying_power,
                cash=cash,
            )
            standard_max_trade_qty = (
                int(max_single_trade_notional / market_data.close) if market_data.close > 0 else 0
            )
            approval_required = False
            approval_granted = False
            max_trade_qty = standard_max_trade_qty
            if (
                decision.action == "long"
                and decision.confidence >= self.config.high_confidence_threshold
                and market_data.close > 0
            ):
                override_notional = cash * self.config.high_confidence_trade_pct
                override_qty = int(override_notional / market_data.close)
                if override_qty > standard_max_trade_qty:
                    approval_required = True
                    approval = self.telegram.request_trade_approval(
                        run_id=self.run_id,
                        order_plan=OrderPlan(
                            symbol=decision.symbol,
                            side=decision.action,
                            qty=override_qty,
                            notional=override_qty * market_data.close,
                            confidence=decision.confidence,
                            allocation=decision.allocation,
                            reason="telegram approval request",
                            max_trade_pct=self.config.high_confidence_trade_pct,
                            telegram_approval_required=True,
                        ),
                        standard_notional=max_single_trade_notional,
                        requested_notional=override_notional,
                        cash_available=cash,
                    )
                    approval_granted = approval.approved
                    if approval_granted:
                        max_trade_qty = override_qty
                        max_trade_pct_used = self.config.high_confidence_trade_pct
                        max_trade_cap_basis = "available cash"
                        self.logger.info(
                            "Telegram approval granted for %s. Raising max trade cap to %.2f%% of cash.",
                            decision.symbol,
                            self.config.high_confidence_trade_pct * 100,
                        )
                    else:
                        self.logger.info(
                            "Telegram approval not granted for %s. Keeping standard max trade cap.",
                            decision.symbol,
                        )
            qty = min(risk_qty, alloc_qty, max_trade_qty)
            if qty <= 0:
                self.logger.info("Skipping %s because computed quantity is zero.", decision.symbol)
                continue
            if qty * market_data.close > equity * self.config.max_position_weight:
                qty = int((equity * self.config.max_position_weight) / market_data.close)
            if qty <= 0:
                self.logger.info("Skipping %s because position-weight cap reduced quantity to zero.", decision.symbol)
                continue

            entry_limit_price, stop_price, take_profit_price = self._planned_prices(
                side=decision.action,
                close=market_data.close,
                stop_distance=planned_stop_distance,
            )

            plans.append(
                OrderPlan(
                    symbol=decision.symbol,
                    side=decision.action,
                    qty=qty,
                    notional=qty * market_data.close,
                    confidence=decision.confidence,
                    allocation=decision.allocation,
                    reason=(
                        f"allocation={decision.allocation:.2f}, risk_qty={risk_qty}, "
                        f"alloc_qty={alloc_qty}, max_trade_qty={max_trade_qty}, "
                        f"cap_basis={max_trade_cap_basis}"
                    ),
                    max_trade_pct=max_trade_pct_used,
                    telegram_approval_required=approval_required,
                    telegram_approval_granted=approval_granted,
                    entry_limit_price=entry_limit_price,
                    stop_price=stop_price,
                    take_profit_price=take_profit_price,
                    risk_notional=qty * planned_stop_distance,
                    order_style=self.config.order_style,
                )
            )

        return plans

    def _daily_loss_limit_reached(self, account) -> bool:
        last_equity = float(getattr(account, "last_equity", 0.0) or 0.0)
        equity = float(getattr(account, "equity", 0.0) or 0.0)
        if last_equity <= 0:
            return False
        daily_return = (equity - last_equity) / last_equity
        return daily_return <= -self.config.max_daily_loss_pct

    def _planned_prices(
        self,
        *,
        side: str,
        close: float,
        stop_distance: float,
    ) -> tuple[float | None, float | None, float | None]:
        if close <= 0:
            return None, None, None
        buffer = self.config.entry_limit_buffer_pct
        reward_distance = stop_distance * self.config.take_profit_r_multiple
        if side == "long":
            entry = close * (1 + buffer)
            stop = max(0.01, close - stop_distance)
            take_profit = close + reward_distance
        else:
            entry = close * (1 - buffer)
            stop = close + stop_distance
            take_profit = max(0.01, close - reward_distance)
        return round(entry, 2), round(stop, 2), round(take_profit, 2)

    def review_pending_orders(
        self,
        market_data: list[SymbolMarketData],
        now: datetime | None = None,
    ) -> list[PendingOrderReview]:
        if not self._is_before_market_open(now):
            self.logger.info("Skipping pending-order review because the market is already open.")
            return []

        market_map = {item.symbol: item for item in market_data}
        existing_orders = self.client.get_orders(
            filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500)
        )
        reviews: list[PendingOrderReview] = []

        for order in existing_orders:
            symbol = getattr(order, "symbol", "")
            item = market_map.get(symbol)
            side = str(getattr(order, "side", "")).lower()
            status = str(getattr(order, "status", "open"))
            order_id = str(getattr(order, "id", "")) or None
            submitted_at = str(getattr(order, "submitted_at", "")) or None
            latest_price = item.premarket.latest_price if item else None
            reference_close = item.close if item else None
            price_change_pct = None
            if latest_price is not None and reference_close:
                price_change_pct = (latest_price - reference_close) / reference_close

            action = "keep"
            reason = "No extended-hours review trigger."
            if item is None:
                action = "keep"
                reason = "No market data available for pending-order review."
            elif latest_price is None:
                action = "keep"
                reason = "No extended-hours price available."
            elif (
                price_change_pct is not None
                and abs(price_change_pct) >= self.config.pending_order_review_max_gap_pct
            ):
                action = "cancel"
                reason = (
                    f"Extended-hours move of {price_change_pct:.2%} exceeded "
                    f"{self.config.pending_order_review_max_gap_pct:.2%} review threshold."
                )
                if order_id:
                    self.client.cancel_order_by_id(order_id)
                    self.logger.info(
                        "Cancelled pending order %s for %s after extended-hours move of %.2f%%",
                        order_id,
                        symbol,
                        price_change_pct * 100,
                    )
                    self.telegram.send_message(
                        f"Pending order review for {self.run_id}\n"
                        f"{symbol} {side} order cancelled before the open.\n"
                        f"Reference close: ${reference_close:.2f}\n"
                        f"Extended-hours price: ${latest_price:.2f}\n"
                        f"Move: {price_change_pct:.2%}\n"
                        f"Reason: {reason}"
                    )

            reviews.append(
                PendingOrderReview(
                    symbol=symbol,
                    order_id=order_id,
                    side=side,
                    status=status,
                    submitted_at=submitted_at,
                    reference_close=reference_close,
                    extended_hours_price=latest_price,
                    price_change_pct=price_change_pct,
                    action=action,
                    reason=reason,
                )
            )

        return reviews

    def evaluate_held_positions(
        self,
        decisions: Iterable[TradeDecision],
        selected_symbols: list[SymbolMarketData],
    ) -> list[HeldPositionSignal]:
        account = self.client.get_account()
        equity = float(account.equity)
        buying_power = float(account.buying_power)
        cash = float(getattr(account, "cash", buying_power))
        symbol_map = {item.symbol: item for item in selected_symbols}
        decision_map = {item.symbol: item for item in decisions}
        signals: list[HeldPositionSignal] = []

        for position in self.client.get_all_positions():
            symbol = position.symbol
            current_qty = int(abs(float(position.qty)))
            market_value = abs(float(position.market_value))
            current_weight = market_value / equity if equity > 0 else 0.0
            side = "short" if float(position.qty) < 0 else "long"
            decision = decision_map.get(symbol)
            market_data = symbol_map.get(symbol)

            signal = "hold"
            confidence = 0.5
            target_qty = current_qty
            reason = "No stronger action identified."
            max_trade_pct = self.config.max_single_trade_pct

            if market_data is None or decision is None or decision.action == "skip":
                signal = "sell"
                confidence = max(self.config.min_confidence, 0.7)
                target_qty = 0
                reason = "Held position is no longer in the active conviction set."
            elif decision.action != side:
                signal = "sell"
                confidence = max(decision.confidence, self.config.min_confidence)
                target_qty = 0
                reason = f"Model now prefers {decision.action} over existing {side} exposure."
            elif decision.action == side and (
                decision.allocation - current_weight >= self.config.buy_more_threshold
            ):
                max_single_trade_notional, max_trade_pct, max_trade_cap_basis = self._standard_trade_cap(
                    side=decision.action,
                    equity=equity,
                    buying_power=buying_power,
                    cash=cash,
                )
                alloc_notional = equity * min(decision.allocation, self.config.max_total_exposure)
                capped_notional = min(alloc_notional, max_single_trade_notional)
                target_qty = int(capped_notional / market_data.close) if market_data.close > 0 else current_qty
                if target_qty > current_qty:
                    signal = "buy_more"
                    confidence = decision.confidence
                    reason = (
                        f"Target allocation {decision.allocation:.2f} exceeds current weight "
                        f"{current_weight:.2f} by at least {self.config.buy_more_threshold:.2f}, "
                        f"capped at {max_trade_pct:.2%} of {max_trade_cap_basis}."
                    )
                else:
                    target_qty = current_qty
                    confidence = decision.confidence
                    reason = "Target allocation does not require additional size."
            else:
                confidence = decision.confidence if decision else 0.5
                reason = "Existing position remains aligned with current model conviction."

            signals.append(
                HeldPositionSignal(
                    symbol=symbol,
                    current_side=side,
                    signal=signal,
                    confidence=confidence,
                    current_qty=current_qty,
                    target_qty=target_qty,
                    delta_qty=target_qty - current_qty,
                    reason=reason,
                    max_trade_pct=max_trade_pct,
                )
            )

        return signals

    @staticmethod
    def _would_sell_long_at_loss(position) -> bool:
        try:
            qty = abs(float(position.qty))
            market_value = abs(float(position.market_value))
            avg_entry_price = float(getattr(position, "avg_entry_price", 0.0) or 0.0)
            if qty <= 0 or avg_entry_price <= 0:
                unrealized_plpc = getattr(position, "unrealized_plpc", None)
                if unrealized_plpc is not None:
                    return float(unrealized_plpc) < 0
                return False
            current_price = market_value / qty
            return current_price < avg_entry_price
        except (TypeError, ValueError, ZeroDivisionError):
            unrealized_plpc = getattr(position, "unrealized_plpc", None)
            if unrealized_plpc is None:
                return False
            try:
                return float(unrealized_plpc) < 0
            except (TypeError, ValueError):
                return False

    def build_held_position_order_plans(
        self,
        held_signals: list[HeldPositionSignal],
        selected_symbols: list[SymbolMarketData],
    ) -> list[OrderPlan]:
        symbol_map = {item.symbol: item for item in selected_symbols}
        plans: list[OrderPlan] = []
        for signal in held_signals:
            market_data = symbol_map.get(signal.symbol)
            if market_data is None:
                continue
            if signal.signal == "hold":
                continue
            qty = abs(signal.delta_qty) if signal.signal == "buy_more" else signal.current_qty
            if qty <= 0:
                continue
            if signal.signal == "buy_more":
                side = "buy_more" if signal.current_side == "long" else "sell"
            else:
                side = "sell" if signal.current_side == "long" else "buy_more"
            plans.append(
                OrderPlan(
                    symbol=signal.symbol,
                    side=side,
                    qty=qty,
                    notional=qty * market_data.close,
                    confidence=signal.confidence,
                    allocation=0.0,
                    reason=signal.reason,
                    max_trade_pct=signal.max_trade_pct,
                    order_style=self.config.order_style,
                )
            )
        return plans

    def submit_orders(self, order_plans: list[OrderPlan]) -> list[dict]:
        results: list[dict] = []
        existing_orders = self.client.get_orders(
            filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500)
        )
        open_symbols = {order.symbol for order in existing_orders}

        for plan in order_plans:
            if plan.symbol in open_symbols:
                self.logger.info("Skipping %s because an open order already exists.", plan.symbol)
                results.append(
                    {
                        "symbol": plan.symbol,
                        "status": "skipped_open_order",
                        "qty": plan.qty,
                        "side": plan.side,
                        "notional": round(plan.notional, 2),
                        "confidence": round(plan.confidence, 4),
                        "max_trade_pct": plan.max_trade_pct,
                        "telegram_approval_required": plan.telegram_approval_required,
                        "telegram_approval_granted": plan.telegram_approval_granted,
                        "reason": "Open order already exists for symbol.",
                    }
                )
                continue

            side = OrderSide.BUY if plan.side in {"long", "buy_more"} else OrderSide.SELL
            request = self._build_order_request(plan, side)
            if self.config.execute_orders:
                order = self.client.submit_order(order_data=request)
                payload = {
                    "symbol": plan.symbol,
                    "status": getattr(order, "status", "submitted"),
                    "qty": plan.qty,
                    "side": plan.side,
                    "order_id": str(getattr(order, "id", "")) or None,
                    "notional": round(plan.notional, 2),
                    "confidence": round(plan.confidence, 4),
                    "max_trade_pct": plan.max_trade_pct,
                    "telegram_approval_required": plan.telegram_approval_required,
                    "telegram_approval_granted": plan.telegram_approval_granted,
                }
                self.logger.info("Submitted order for %s qty=%s side=%s", plan.symbol, plan.qty, plan.side)
            else:
                payload = {
                    "symbol": plan.symbol,
                    "status": "dry_run",
                    "qty": plan.qty,
                    "side": plan.side,
                    "notional": round(plan.notional, 2),
                    "confidence": round(plan.confidence, 4),
                    "max_trade_pct": plan.max_trade_pct,
                    "telegram_approval_required": plan.telegram_approval_required,
                    "telegram_approval_granted": plan.telegram_approval_granted,
                }
                self.logger.info("Dry-run order for %s qty=%s side=%s", plan.symbol, plan.qty, plan.side)
            results.append(payload)
            self.telegram.send_trade_summary(run_id=self.run_id, order_plan=plan, payload=payload)

        return results

    def _build_order_request(self, plan: OrderPlan, side: OrderSide):
        if (
            plan.order_style in {"limit", "bracket_limit"}
            and plan.entry_limit_price is not None
        ):
            kwargs = {
                "symbol": plan.symbol,
                "qty": plan.qty,
                "side": side,
                "time_in_force": TimeInForce.DAY,
                "limit_price": plan.entry_limit_price,
            }
            if (
                plan.order_style == "bracket_limit"
                and plan.stop_price is not None
                and plan.take_profit_price is not None
            ):
                kwargs["order_class"] = OrderClass.BRACKET
                kwargs["take_profit"] = TakeProfitRequest(limit_price=plan.take_profit_price)
                kwargs["stop_loss"] = StopLossRequest(stop_price=plan.stop_price)
            return LimitOrderRequest(**kwargs)
        return MarketOrderRequest(
            symbol=plan.symbol,
            qty=plan.qty,
            side=side,
            time_in_force=TimeInForce.DAY,
        )

    def _is_before_market_open(self, now: datetime | None = None) -> bool:
        market_now = now or datetime.now(ZoneInfo(self.config.market_timezone))
        market_time = market_now.timetz().replace(tzinfo=None)
        return market_time < datetime.strptime("09:30", "%H:%M").time()
