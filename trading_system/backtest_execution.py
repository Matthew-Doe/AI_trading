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

class BacktestExecutionEngine:
    def __init__(
        self, 
        initial_cash: float = 100000.0, 
        slippage_pct: float = 0.001,  # 0.1% per leg
        commission_fixed: float = 0.0,
        max_position_size_pct: float = 0.15,
        market_data_service: Any = None
    ):
        self.cash = initial_cash
        self.equity = initial_cash
        self.positions: dict[str, BacktestPosition] = {}
        self.trades: list[BacktestTradeRecord] = []
        self.slippage_pct = slippage_pct
        self.commission_fixed = commission_fixed
        self.max_position_size_pct = max_position_size_pct
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
        symbols_to_remove = []
        for symbol, pos in self.positions.items():
            snapshot = all_symbol_prices.get(symbol)
            if snapshot is None:
                continue
            price = self._mark_price(snapshot, fallback=pos.entry_price)
            
            exit_triggered = False
            reason = ""
            
            if pos.side == "long":
                if pos.stop_price and price <= pos.stop_price:
                    exit_triggered, reason = True, "stop_loss"
                elif pos.take_profit_price and price >= pos.take_profit_price:
                    exit_triggered, reason = True, "take_profit"
            else: # short
                if pos.stop_price and price >= pos.stop_price:
                    exit_triggered, reason = True, "stop_loss"
                elif pos.take_profit_price and price <= pos.take_profit_price:
                    exit_triggered, reason = True, "take_profit"

            days_held = (current_time - pos.entry_time).days
            if not exit_triggered and days_held >= 3:
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
            
            snapshot = all_symbol_prices.get(dec.symbol)
            if snapshot is None: continue
            price = self._entry_price(snapshot)

            # Entry Price explicitly includes slippage
            fill_price = price * (1 + self.slippage_pct) if dec.action == "long" else price * (1 - self.slippage_pct)
            
            # Position Sizing
            alloc = min(dec.allocation or 0.05, self.max_position_size_pct)
            target_notional = self.equity * alloc
            
            if target_notional > self.cash:
                target_notional = self.cash
            
            qty = int(target_notional / fill_price)
            if qty <= 0: continue

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
                take_profit_price=dec.target_price
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
            holding_period_days=(exit_time - pos.entry_time).days
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
