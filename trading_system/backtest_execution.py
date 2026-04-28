from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from trading_system.models import OrderPlan, TradeDecision, SymbolMarketData

MarketSnapshot = float | SymbolMarketData

@dataclass
class BacktestPosition:
    symbol: str
    qty: int
    entry_price: float
    entry_time: datetime
    side: str  # "long" or "short"
    stop_price: float | None = None
    take_profit_price: float | None = None
    highest_price: float | None = None
    lowest_price: float | None = None
    sizing_reason: str = ""
    risk_notional: float = 0.0
    stop_distance: float = 0.0

@dataclass
class BacktestTradeRecord:
    symbol: str
    side: str
    entry_time: datetime
    entry_price: float
    exit_time: datetime | None = None
    exit_price: float | None = None
    exit_reason: str | None = None # "stop_loss", "take_profit", "time_expiry", "manual"
    qty: int = 0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    costs: float = 0.0
    holding_period_days: int = 0
    sizing_reason: str = ""
    risk_notional: float = 0.0
    mfe_pct: float = 0.0
    mae_pct: float = 0.0

class BacktestExecutionEngine:
    def __init__(
        self, 
        initial_cash: float = 100000.0, 
        slippage_pct: float = 0.001,  # 0.1% per leg
        commission_fixed: float = 0.0,
        max_position_size_pct: float = 0.15,
        wash_sale_cooldown_days: int = 31,
        config: Any = None,
        market_data_service: Any = None
    ):
        self.cash = initial_cash
        self.equity = initial_cash
        self.positions: dict[str, BacktestPosition] = {}
        self.trades: list[BacktestTradeRecord] = []
        self.slippage_pct = slippage_pct
        self.commission_fixed = commission_fixed
        self.max_position_size_pct = max_position_size_pct
        self.wash_sale_cooldown_days = max(0, wash_sale_cooldown_days)
        self.config = config
        self.loss_cooldowns: dict[str, datetime] = {}
        self.tax_blocked_decisions: list[dict[str, Any]] = []
        self.sizing_logs: list[dict[str, Any]] = []
        self.exit_adjustment_logs: list[dict[str, Any]] = []
        self.market_data_service = market_data_service

    def update_equity(self, current_prices: dict[str, MarketSnapshot]):
        """Calculates current portfolio value based on latest prices."""
        pos_value = 0.0
        for symbol, pos in self.positions.items():
            price = self._mark_price(current_prices.get(symbol), fallback=pos.entry_price)
            if pos.side == "long":
                pos_value += pos.qty * price
            else:
                # Short: Profit = (Entry - Current) * Qty
                # Short Equity = (Entry * Qty) + ((Entry - Current) * Qty)
                # But more simply: We owe 'Current * Qty' to the market.
                # Equity = Cash + (Entry*Qty) - (Current*Qty)
                pos_value += (pos.entry_price - price) * pos.qty
        self.equity = self.cash + pos_value

    def process_decisions(
        self, 
        decisions: list[TradeDecision], 
        all_symbol_prices: dict[str, MarketSnapshot],
        current_time: datetime
    ):
        """Executes new decisions and checks existing positions for stops/targets."""
        
        # 1. Manage Existing Positions (Check Stops/Targets)
        active_symbols = {dec.symbol for dec in decisions if dec.action != "skip"}
        symbols_to_remove = []
        for symbol, pos in self.positions.items():
            snapshot = all_symbol_prices.get(symbol)
            if snapshot is None:
                continue
            price = self._mark_price(snapshot, fallback=pos.entry_price)
            self._update_excursions(pos, price)
            
            exit_triggered = False
            reason = ""
            
            if pos.side == "long":
                if pos.stop_price and price <= pos.stop_price:
                    exit_triggered, reason = True, "breakeven_stop" if pos.stop_price >= pos.entry_price else "stop_loss"
                elif pos.take_profit_price and price >= pos.take_profit_price:
                    exit_triggered, reason = True, "target_hit"
            else: # short
                if pos.stop_price and price >= pos.stop_price:
                    exit_triggered, reason = True, "breakeven_stop" if pos.stop_price <= pos.entry_price else "stop_loss"
                elif pos.take_profit_price and price <= pos.take_profit_price:
                    exit_triggered, reason = True, "target_hit"

            days_held = (current_time - pos.entry_time).days
            if not exit_triggered:
                exit_triggered, reason = self._staged_exit_decision(
                    pos=pos,
                    price=price,
                    current_time=current_time,
                    active_symbols=active_symbols,
                    days_held=days_held,
                )

            if exit_triggered:
                self._close_position(pos, price, current_time, reason)
                symbols_to_remove.append(symbol)
        
        for s in symbols_to_remove:
            del self.positions[s]
            self.update_equity(all_symbol_prices) # Immediate equity update

        # 2. Execute New Decisions
        for dec in decisions:
            if dec.action == "skip": continue
            if dec.symbol in self.positions: continue
            if self._is_tax_blocked(dec.symbol, current_time):
                self.tax_blocked_decisions.append(
                    {
                        "time": current_time.isoformat(),
                        "symbol": dec.symbol,
                        "action": dec.action,
                        "confidence": dec.confidence,
                        "allocation": dec.allocation,
                        "blocked_until": self.loss_cooldowns[dec.symbol].isoformat(),
                        "reason": "wash_sale_loss_cooldown",
                    }
                )
                continue
            
            snapshot = all_symbol_prices.get(dec.symbol)
            if snapshot is None: continue
            price = self._entry_price(snapshot)

            # Entry Price explicitly includes slippage
            fill_price = price * (1 + self.slippage_pct) if dec.action == "long" else price * (1 - self.slippage_pct)
            
            # Position Sizing
            qty, sizing_reason, risk_notional, stop_distance = self._size_position(
                decision=dec,
                snapshot=snapshot,
                fill_price=fill_price,
                mark_price=price,
            )
            if qty <= 0: continue
            stop_price, take_profit_price = self._planned_exit_prices(
                decision=dec,
                snapshot=snapshot,
                side=dec.action,
                mark_price=price,
                stop_distance=stop_distance,
            )

            # Open Position
            cost = (qty * fill_price)
            self.cash -= cost
            self.positions[dec.symbol] = BacktestPosition(
                symbol=dec.symbol,
                qty=qty,
                entry_price=fill_price,
                entry_time=current_time,
                side=dec.action,
                stop_price=stop_price,
                take_profit_price=take_profit_price,
                highest_price=price,
                lowest_price=price,
                sizing_reason=sizing_reason,
                risk_notional=risk_notional,
                stop_distance=stop_distance,
            )
            self.sizing_logs.append(
                {
                    "time": current_time.isoformat(),
                    "symbol": dec.symbol,
                    "side": dec.action,
                    "qty": qty,
                    "fill_price": round(fill_price, 4),
                    "notional": round(qty * fill_price, 2),
                    "risk_notional": round(risk_notional, 2),
                    "stop_distance": round(stop_distance, 4),
                    "stop_price": stop_price,
                    "take_profit_price": take_profit_price,
                    "reason": sizing_reason,
                }
            )
            self.update_equity(all_symbol_prices) # Immediate equity update

    @staticmethod
    def _mark_price(snapshot: MarketSnapshot | None, *, fallback: float) -> float:
        if isinstance(snapshot, SymbolMarketData):
            return snapshot.close
        if snapshot is None:
            return fallback
        return float(snapshot)

    @staticmethod
    def _entry_price(snapshot: MarketSnapshot) -> float:
        if isinstance(snapshot, SymbolMarketData):
            return float(snapshot.premarket.latest_price or snapshot.close)
        return float(snapshot)

    def _size_position(
        self,
        *,
        decision: TradeDecision,
        snapshot: MarketSnapshot,
        fill_price: float,
        mark_price: float,
    ) -> tuple[int, str, float, float]:
        config = self.config
        if config is None or not isinstance(snapshot, SymbolMarketData):
            alloc = min(decision.allocation or 0.05, self.max_position_size_pct)
            target_notional = min(self.equity * alloc, self.cash)
            qty = int(target_notional / fill_price)
            return qty, f"legacy_allocation alloc={alloc:.4f}", 0.0, 0.0

        stop_distance = max(
            snapshot.indicators.atr14 * config.stop_atr_multiple,
            mark_price * 0.01,
        )
        risk_budget = self.equity * config.max_risk_per_trade
        risk_qty = int(risk_budget / stop_distance) if stop_distance > 0 else 0
        allocation = decision.allocation or config.max_single_trade_pct
        alloc_notional = self.equity * allocation
        alloc_qty = int(alloc_notional / fill_price) if fill_price > 0 else 0
        max_trade_pct = config.max_single_trade_pct
        max_trade_notional = self.equity * config.max_single_trade_pct
        cap_basis = "equity"

        if (
            decision.action == "long"
            and self.equity > 0
            and self.cash / self.equity >= config.cash_rich_available_cash_threshold
        ):
            max_trade_pct = config.cash_rich_trade_pct
            max_trade_notional = self.cash * config.cash_rich_trade_pct
            cap_basis = "available_cash"
        if (
            decision.action == "long"
            and decision.confidence >= config.high_confidence_threshold
            and self.cash > 0
        ):
            max_trade_pct = max(max_trade_pct, config.high_confidence_trade_pct)
            max_trade_notional = max(max_trade_notional, self.cash * config.high_confidence_trade_pct)
            cap_basis = "high_confidence_cash"

        max_trade_qty = int(max_trade_notional / fill_price) if fill_price > 0 else 0
        cash_qty = int(self.cash / fill_price) if fill_price > 0 else 0
        max_position_qty = (
            int((self.equity * config.max_position_weight) / fill_price)
            if fill_price > 0
            else 0
        )
        candidates = {
            "risk": risk_qty,
            "allocation": alloc_qty,
            "trade_cap": max_trade_qty,
            "cash": cash_qty,
            "position_weight": max_position_qty,
        }
        qty = max(0, min(candidates.values()))
        binding = [name for name, value in candidates.items() if value == qty]
        sizing_reason = (
            f"binding={','.join(binding)} allocation={allocation:.4f} "
            f"risk_qty={risk_qty} alloc_qty={alloc_qty} max_trade_qty={max_trade_qty} "
            f"cash_qty={cash_qty} max_position_qty={max_position_qty} "
            f"cap_pct={max_trade_pct:.4f} cap_basis={cap_basis}"
        )
        return qty, sizing_reason, qty * stop_distance, stop_distance

    def _planned_exit_prices(
        self,
        *,
        decision: TradeDecision,
        snapshot: MarketSnapshot,
        side: str,
        mark_price: float,
        stop_distance: float,
    ) -> tuple[float | None, float | None]:
        config = self.config
        if config is None or stop_distance <= 0 or not isinstance(snapshot, SymbolMarketData):
            return decision.invalidation_price, decision.target_price
        reward_distance = stop_distance * config.take_profit_r_multiple
        if side == "long":
            return round(max(0.01, mark_price - stop_distance), 2), round(mark_price + reward_distance, 2)
        return round(mark_price + stop_distance, 2), round(max(0.01, mark_price - reward_distance), 2)

    @staticmethod
    def _update_excursions(pos: BacktestPosition, price: float) -> None:
        pos.highest_price = price if pos.highest_price is None else max(pos.highest_price, price)
        pos.lowest_price = price if pos.lowest_price is None else min(pos.lowest_price, price)

    def _staged_exit_decision(
        self,
        *,
        pos: BacktestPosition,
        price: float,
        current_time: datetime,
        active_symbols: set[str],
        days_held: int,
    ) -> tuple[bool, str]:
        config = self.config
        min_days = getattr(config, "backtest_min_thesis_days", 2)
        max_days = getattr(config, "backtest_max_hold_days", 7)
        breakeven_days = getattr(config, "backtest_breakeven_after_days", 2)
        if pos.side == "long":
            unrealized_pct = (price - pos.entry_price) / pos.entry_price if pos.entry_price else 0.0
            if days_held >= breakeven_days and price > pos.entry_price:
                old_stop = pos.stop_price
                pos.stop_price = max(pos.stop_price or 0.0, pos.entry_price)
                if old_stop != pos.stop_price:
                    self.exit_adjustment_logs.append(
                        {
                            "time": current_time.isoformat(),
                            "symbol": pos.symbol,
                            "action": "ratchet_stop_to_breakeven",
                            "old_stop": old_stop,
                            "new_stop": pos.stop_price,
                            "unrealized_pct": round(unrealized_pct, 4),
                        }
                    )
            if days_held >= min_days and unrealized_pct < 0 and pos.symbol not in active_symbols:
                return True, "thesis_failed"
            if days_held >= max_days:
                return True, "extended_hold"
        else:
            unrealized_pct = (pos.entry_price - price) / pos.entry_price if pos.entry_price else 0.0
            if days_held >= breakeven_days and price < pos.entry_price:
                old_stop = pos.stop_price
                pos.stop_price = min(pos.stop_price or float("inf"), pos.entry_price)
                if old_stop != pos.stop_price:
                    self.exit_adjustment_logs.append(
                        {
                            "time": current_time.isoformat(),
                            "symbol": pos.symbol,
                            "action": "ratchet_stop_to_breakeven",
                            "old_stop": old_stop,
                            "new_stop": pos.stop_price,
                            "unrealized_pct": round(unrealized_pct, 4),
                        }
                    )
            if days_held >= min_days and unrealized_pct < 0 and pos.symbol not in active_symbols:
                return True, "thesis_failed"
            if days_held >= max_days:
                return True, "extended_hold"
        return False, ""

    def _close_position(self, pos: BacktestPosition, price: float, exit_time: datetime, reason: str):
        # Exit Price with Slippage
        fill_price = price * (1 - self.slippage_pct) if pos.side == "long" else price * (1 + self.slippage_pct)
        
        if pos.side == "long":
            proceeds = pos.qty * fill_price
            gross_pnl = (fill_price - pos.entry_price) * pos.qty
        else:
            # For shorts, we effectively get back the cash we used to secure it (pos.qty * pos.entry_price)
            # plus/minus the price movement.
            gross_pnl = (pos.entry_price - fill_price) * pos.qty
            proceeds = (pos.qty * pos.entry_price) + gross_pnl

        self.cash += proceeds
        
        total_costs = (abs(pos.entry_price - (pos.entry_price / (1+self.slippage_pct))) * pos.qty) + \
                      (abs(fill_price - (fill_price / (1-self.slippage_pct))) * pos.qty)
        
        record = BacktestTradeRecord(
            symbol=pos.symbol,
            side=pos.side,
            entry_time=pos.entry_time,
            entry_price=pos.entry_price,
            exit_time=exit_time,
            exit_price=fill_price,
            exit_reason=reason,
            qty=pos.qty,
            gross_pnl=gross_pnl,
            net_pnl=gross_pnl - total_costs,
            costs=total_costs,
            holding_period_days=(exit_time - pos.entry_time).days,
            sizing_reason=pos.sizing_reason,
            risk_notional=pos.risk_notional,
            mfe_pct=self._mfe_pct(pos),
            mae_pct=self._mae_pct(pos),
        )
        self.trades.append(record)
        if record.net_pnl < 0 and self.wash_sale_cooldown_days:
            self.loss_cooldowns[pos.symbol] = exit_time + timedelta(
                days=self.wash_sale_cooldown_days
            )

    def _is_tax_blocked(self, symbol: str, current_time: datetime) -> bool:
        blocked_until = self.loss_cooldowns.get(symbol)
        if blocked_until is None:
            return False
        if current_time < blocked_until:
            return True
        del self.loss_cooldowns[symbol]
        return False

    def get_summary(self) -> dict[str, Any]:
        net_pnl = sum(t.net_pnl for t in self.trades)
        wins = [t for t in self.trades if t.net_pnl > 0]
        return {
            "final_equity": round(self.equity, 2),
            "total_net_pnl": round(net_pnl, 2),
            "trade_count": len(self.trades),
            "win_rate": round(len(wins) / len(self.trades), 4) if self.trades else 0,
            "total_costs": round(sum(t.costs for t in self.trades), 2)
        }

    @staticmethod
    def _mfe_pct(pos: BacktestPosition) -> float:
        if not pos.entry_price:
            return 0.0
        if pos.side == "long":
            return round(((pos.highest_price or pos.entry_price) - pos.entry_price) / pos.entry_price, 4)
        return round((pos.entry_price - (pos.lowest_price or pos.entry_price)) / pos.entry_price, 4)

    @staticmethod
    def _mae_pct(pos: BacktestPosition) -> float:
        if not pos.entry_price:
            return 0.0
        if pos.side == "long":
            return round(((pos.lowest_price or pos.entry_price) - pos.entry_price) / pos.entry_price, 4)
        return round((pos.entry_price - (pos.highest_price or pos.entry_price)) / pos.entry_price, 4)

    def get_tax_summary(self) -> dict[str, Any]:
        loss_sales = [
            t for t in self.trades if t.exit_time is not None and t.net_pnl < 0
        ]
        future_buys = self._future_replacement_buy_matches(loss_sales)
        disallowed_loss = sum(item["disallowed_loss_estimate"] for item in future_buys)
        raw_realized_pnl = sum(t.net_pnl for t in self.trades)
        gross_gains = sum(t.net_pnl for t in self.trades if t.net_pnl > 0)
        gross_losses = -sum(t.net_pnl for t in loss_sales)

        by_symbol: dict[str, dict[str, Any]] = {}
        for item in future_buys:
            symbol = item["symbol"]
            summary = by_symbol.setdefault(
                symbol,
                {
                    "loss_sales": 0,
                    "matched_rebuy_shares": 0,
                    "disallowed_loss_estimate": 0.0,
                },
            )
            summary["loss_sales"] += 1
            summary["matched_rebuy_shares"] += item["matched_rebuy_shares"]
            summary["disallowed_loss_estimate"] += item["disallowed_loss_estimate"]

        ranked_symbols = [
            {
                "symbol": symbol,
                "loss_sales": values["loss_sales"],
                "matched_rebuy_shares": values["matched_rebuy_shares"],
                "disallowed_loss_estimate": round(values["disallowed_loss_estimate"], 2),
            }
            for symbol, values in sorted(
                by_symbol.items(),
                key=lambda kv: kv[1]["disallowed_loss_estimate"],
                reverse=True,
            )
        ]

        return {
            "wash_sale_cooldown_days": self.wash_sale_cooldown_days,
            "tax_blocked_entry_count": len(self.tax_blocked_decisions),
            "tax_blocked_entries": self.tax_blocked_decisions,
            "loss_sale_count": len(loss_sales),
            "wash_sale_candidate_count": len(future_buys),
            "gross_short_term_gains": round(gross_gains, 2),
            "gross_short_term_losses": round(gross_losses, 2),
            "raw_realized_pnl": round(raw_realized_pnl, 2),
            "wash_sale_disallowed_loss_estimate": round(disallowed_loss, 2),
            "taxable_realized_pnl_estimate": round(raw_realized_pnl + disallowed_loss, 2),
            "ranked_wash_sale_symbols": ranked_symbols,
            "wash_sale_candidates": future_buys,
            "notes": [
                "Wash-sale estimates use same-symbol replacement buys inside 30 calendar days after a loss sale.",
                "This is a backtest estimate, not tax advice; real reporting needs broker tax lots and account-wide activity.",
            ],
        }

    def _future_replacement_buy_matches(
        self, loss_sales: list[BacktestTradeRecord]
    ) -> list[dict[str, Any]]:
        buy_events: list[dict[str, Any]] = []
        for idx, trade in enumerate(self.trades):
            if trade.side == "long":
                buy_events.append(
                    {
                        "symbol": trade.symbol,
                        "time": trade.entry_time,
                        "qty_remaining": trade.qty,
                        "trade_index": idx,
                    }
                )
        for symbol, pos in self.positions.items():
            if pos.side == "long":
                buy_events.append(
                    {
                        "symbol": symbol,
                        "time": pos.entry_time,
                        "qty_remaining": pos.qty,
                        "trade_index": None,
                    }
                )

        matches: list[dict[str, Any]] = []
        sorted_losses = sorted(loss_sales, key=lambda trade: trade.exit_time or trade.entry_time)
        for loss_trade in sorted_losses:
            if loss_trade.exit_time is None or loss_trade.qty <= 0:
                continue
            window_end = loss_trade.exit_time + timedelta(days=30)
            shares_needed = loss_trade.qty
            matched_shares = 0
            replacement_buys: list[dict[str, Any]] = []

            for buy in sorted(buy_events, key=lambda item: item["time"]):
                if buy["symbol"] != loss_trade.symbol:
                    continue
                if buy["trade_index"] is not None and self.trades[buy["trade_index"]] is loss_trade:
                    continue
                if not (loss_trade.exit_time <= buy["time"] <= window_end):
                    continue
                if buy["qty_remaining"] <= 0:
                    continue

                shares = min(shares_needed, buy["qty_remaining"])
                buy["qty_remaining"] -= shares
                shares_needed -= shares
                matched_shares += shares
                replacement_buys.append(
                    {
                        "time": buy["time"].isoformat(),
                        "shares": shares,
                    }
                )
                if shares_needed == 0:
                    break

            if matched_shares:
                loss_per_share = -loss_trade.net_pnl / loss_trade.qty
                matches.append(
                    {
                        "symbol": loss_trade.symbol,
                        "loss_exit_time": loss_trade.exit_time.isoformat(),
                        "loss_amount": round(-loss_trade.net_pnl, 2),
                        "loss_qty": loss_trade.qty,
                        "matched_rebuy_shares": matched_shares,
                        "disallowed_loss_estimate": round(loss_per_share * matched_shares, 2),
                        "replacement_buys": replacement_buys,
                    }
                )

        return matches
