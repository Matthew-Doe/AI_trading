from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
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
    confidence: float = 0.0
    allocation: float = 0.0
    take_profit_armed: bool = False
    best_price: float | None = None

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
    confidence: float = 0.0
    allocation: float = 0.0

class BacktestExecutionEngine:
    def __init__(
        self, 
        initial_cash: float = 100000.0, 
        slippage_pct: float = 0.001,  # 0.1% per leg
        commission_fixed: float = 0.0,
        max_position_size_pct: float = 0.15,
        market_data_service: Any = None,
        min_cash_reserve_pct: float = 0.0,
        base_hold_days: int = 3,
        winner_hold_days: int = 10,
        trailing_take_profit_pct: float = 0.05,
        take_profit_r_multiple: float = 2.0,
        loser_penalty_factor: float = 0.5,
        loser_skip_after: int = 2,
    ):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.equity = initial_cash
        self.positions: dict[str, BacktestPosition] = {}
        self.trades: list[BacktestTradeRecord] = []
        self.slippage_pct = slippage_pct
        self.commission_fixed = commission_fixed
        self.max_position_size_pct = max_position_size_pct
        self.market_data_service = market_data_service
        self.min_cash_reserve_pct = min_cash_reserve_pct
        self.base_hold_days = base_hold_days
        self.winner_hold_days = winner_hold_days
        self.trailing_take_profit_pct = trailing_take_profit_pct
        self.take_profit_r_multiple = take_profit_r_multiple
        self.loser_penalty_factor = loser_penalty_factor
        self.loser_skip_after = loser_skip_after

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
        symbols_to_remove = []
        for symbol, pos in self.positions.items():
            snapshot = all_symbol_prices.get(symbol)
            if snapshot is None:
                continue
            price = self._mark_price(snapshot, fallback=pos.entry_price)
            
            exit_triggered = False
            reason = ""
            
            if pos.side == "long":
                if pos.take_profit_armed:
                    pos.best_price = max(pos.best_price or pos.entry_price, price)
                    trailing_stop = pos.best_price * (1.0 - self.trailing_take_profit_pct)
                    if price <= trailing_stop:
                        exit_triggered, reason = True, "trailing_take_profit"
                if not exit_triggered and pos.stop_price and price <= pos.stop_price:
                    exit_triggered, reason = True, "stop_loss"
                elif not exit_triggered and pos.take_profit_price and price >= pos.take_profit_price:
                    pos.take_profit_armed = True
                    pos.best_price = max(pos.best_price or pos.entry_price, price)
            else: # short
                if pos.take_profit_armed:
                    pos.best_price = min(pos.best_price or pos.entry_price, price)
                    trailing_stop = pos.best_price * (1.0 + self.trailing_take_profit_pct)
                    if price >= trailing_stop:
                        exit_triggered, reason = True, "trailing_take_profit"
                if not exit_triggered and pos.stop_price and price >= pos.stop_price:
                    exit_triggered, reason = True, "stop_loss"
                elif not exit_triggered and pos.take_profit_price and price <= pos.take_profit_price:
                    pos.take_profit_armed = True
                    pos.best_price = min(pos.best_price or pos.entry_price, price)

            days_held = (current_time - pos.entry_time).days
            is_winner = price > pos.entry_price if pos.side == "long" else price < pos.entry_price
            hold_limit = self.winner_hold_days if is_winner else self.base_hold_days
            if not exit_triggered and days_held >= hold_limit:
                exit_triggered, reason = True, "time_expiry"

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
            recent_losses = self._recent_symbol_losses(dec.symbol)
            if recent_losses >= self.loser_skip_after:
                continue
            
            snapshot = all_symbol_prices.get(dec.symbol)
            if snapshot is None: continue
            price = self._entry_price(snapshot)

            # Entry Price explicitly includes slippage
            fill_price = price * (1 + self.slippage_pct) if dec.action == "long" else price * (1 - self.slippage_pct)
            
            # Position Sizing
            loser_penalty = self.loser_penalty_factor ** recent_losses
            alloc = min((dec.allocation or 0.05) * loser_penalty, self.max_position_size_pct)
            target_notional = self.equity * alloc
            available_cash = max(0.0, self.cash - (self.equity * self.min_cash_reserve_pct))
            
            if target_notional > available_cash:
                target_notional = available_cash
            
            qty = int(target_notional / fill_price)
            if qty <= 0: continue

            take_profit_price = self._take_profit_price(dec, fill_price)

            # Open Position
            cost = (qty * fill_price)
            self.cash -= cost
            self.positions[dec.symbol] = BacktestPosition(
                symbol=dec.symbol,
                qty=qty,
                entry_price=fill_price,
                entry_time=current_time,
                side=dec.action,
                stop_price=dec.invalidation_price,
                take_profit_price=take_profit_price,
                confidence=dec.confidence,
                allocation=alloc,
                best_price=fill_price,
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

    def _take_profit_price(self, decision: TradeDecision, fill_price: float) -> float | None:
        if decision.target_price is not None:
            return decision.target_price
        if decision.invalidation_price is None:
            return None
        if decision.action == "long":
            risk = fill_price - decision.invalidation_price
            return fill_price + (risk * self.take_profit_r_multiple) if risk > 0 else None
        if decision.action == "short":
            risk = decision.invalidation_price - fill_price
            return fill_price - (risk * self.take_profit_r_multiple) if risk > 0 else None
        return None

    def _recent_symbol_losses(self, symbol: str) -> int:
        losses = 0
        for trade in reversed(self.trades):
            if trade.symbol != symbol:
                continue
            if trade.net_pnl > 0:
                break
            losses += 1
            if losses >= self.loser_skip_after:
                break
        return losses

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
            confidence=pos.confidence,
            allocation=pos.allocation,
        )
        self.trades.append(record)

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
