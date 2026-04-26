from datetime import datetime
from types import SimpleNamespace

from trading_system.config import TradingConfig
from trading_system.execution import AlpacaExecutionEngine
from trading_system.main import load_mock_universe
from trading_system.models import TradeDecision


class DummyLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


class FakeTradingClient:
    def __init__(self, equity="10000", buying_power="10000", cash="10000"):
        self.positions = [
            SimpleNamespace(symbol="NVDA", qty="10", market_value="1000", avg_entry_price="95"),
            SimpleNamespace(symbol="TSLA", qty="-5", market_value="-600", avg_entry_price="110"),
        ]
        self.orders = []
        self.cancelled_order_ids = []
        self.account = SimpleNamespace(equity=equity, buying_power=buying_power, cash=cash)

    def get_account(self):
        return self.account

    def get_all_positions(self):
        return self.positions

    def get_orders(self, filter=None):
        return self.orders

    def cancel_order_by_id(self, order_id):
        self.cancelled_order_ids.append(order_id)


def test_evaluate_held_positions_returns_three_signal_types():
    engine = AlpacaExecutionEngine.__new__(AlpacaExecutionEngine)
    engine.config = TradingConfig(min_confidence=0.6, buy_more_threshold=0.05)
    engine.logger = DummyLogger()
    engine.client = FakeTradingClient()

    selected = load_mock_universe()
    decisions = [
        TradeDecision(symbol="NVDA", action="long", confidence=0.9, allocation=0.20),
        TradeDecision(symbol="TSLA", action="skip", confidence=0.3, allocation=0.0),
    ]
    signals = engine.evaluate_held_positions(decisions, selected)
    signal_map = {item.symbol: item for item in signals}

    assert signal_map["NVDA"].signal == "hold"
    assert signal_map["TSLA"].signal == "sell"


def test_evaluate_held_positions_avoids_selling_long_at_loss():
    engine = AlpacaExecutionEngine.__new__(AlpacaExecutionEngine)
    engine.config = TradingConfig(min_confidence=0.6, buy_more_threshold=0.05)
    engine.logger = DummyLogger()
    engine.client = FakeTradingClient()

    engine.client.positions = [
        SimpleNamespace(symbol="AAPL", qty="10", market_value="1000", avg_entry_price="120"),
    ]

    selected = load_mock_universe()
    decisions = [TradeDecision(symbol="AAPL", action="skip", confidence=0.3, allocation=0.0)]
    signals = engine.evaluate_held_positions(decisions, selected)

    assert len(signals) == 1
    assert signals[0].signal == "sell"
    assert "active conviction" in signals[0].reason


class FakeTelegramNotifier:
    def __init__(self, approved: bool):
        self.approved = approved
        self.messages: list[dict] = []

    def request_trade_approval(self, **kwargs):
        self.messages.append(kwargs)
        return SimpleNamespace(approved=self.approved)

    def send_trade_summary(self, **kwargs):
        self.messages.append(kwargs)

    def send_message(self, message):
        self.messages.append({"message": message})


def test_high_confidence_long_can_use_telegram_override():
    engine = AlpacaExecutionEngine.__new__(AlpacaExecutionEngine)
    engine.config = TradingConfig(
        min_confidence=0.6,
        max_single_trade_pct=0.02,
        cash_rich_trade_pct=0.05,
        cash_rich_available_cash_threshold=0.20,
        high_confidence_trade_pct=0.10,
        high_confidence_threshold=0.95,
    )
    engine.logger = DummyLogger()
    engine.client = FakeTradingClient()
    engine.telegram = FakeTelegramNotifier(approved=True)
    engine.run_id = "test-run"

    selected = load_mock_universe()
    decisions = [TradeDecision(symbol="AAPL", action="long", confidence=0.97, allocation=0.50)]
    plans = engine.build_order_plans(decisions, selected)

    assert len(plans) == 1
    assert plans[0].telegram_approval_required is True
    assert plans[0].telegram_approval_granted is True
    assert plans[0].max_trade_pct == 0.10


def test_long_uses_cash_rich_cap_when_available_cash_is_high():
    engine = AlpacaExecutionEngine.__new__(AlpacaExecutionEngine)
    engine.config = TradingConfig(
        min_confidence=0.6,
        max_single_trade_pct=0.01,
        cash_rich_trade_pct=0.05,
        cash_rich_available_cash_threshold=0.20,
    )
    engine.logger = DummyLogger()
    engine.client = FakeTradingClient(equity="10000", buying_power="10000", cash="10000")
    engine.telegram = FakeTelegramNotifier(approved=False)
    engine.run_id = "test-run"

    selected = load_mock_universe()
    decisions = [TradeDecision(symbol="AAPL", action="long", confidence=0.85, allocation=0.50)]
    plans = engine.build_order_plans(decisions, selected)

    assert len(plans) == 1
    assert plans[0].qty == 3
    assert plans[0].max_trade_pct == 0.05
    assert "cap_basis=available cash" in plans[0].reason


def test_long_keeps_standard_cap_when_available_cash_is_not_high():
    engine = AlpacaExecutionEngine.__new__(AlpacaExecutionEngine)
    engine.config = TradingConfig(
        min_confidence=0.6,
        max_single_trade_pct=0.01,
        cash_rich_trade_pct=0.05,
        cash_rich_available_cash_threshold=0.20,
    )
    engine.logger = DummyLogger()
    engine.client = FakeTradingClient(equity="10000", buying_power="10000", cash="1000")
    engine.telegram = FakeTelegramNotifier(approved=False)
    engine.run_id = "test-run"

    selected = load_mock_universe()
    decisions = [TradeDecision(symbol="AAPL", action="long", confidence=0.85, allocation=0.50)]
    plans = engine.build_order_plans(decisions, selected)

    assert plans == []


def test_high_confidence_long_stays_at_standard_cap_without_telegram_approval():
    engine = AlpacaExecutionEngine.__new__(AlpacaExecutionEngine)
    engine.config = TradingConfig(
        min_confidence=0.6,
        max_single_trade_pct=0.02,
        cash_rich_trade_pct=0.05,
        cash_rich_available_cash_threshold=0.20,
        high_confidence_trade_pct=0.10,
        high_confidence_threshold=0.95,
    )
    engine.logger = DummyLogger()
    engine.client = FakeTradingClient()
    engine.telegram = FakeTelegramNotifier(approved=False)
    engine.run_id = "test-run"

    selected = load_mock_universe()
    decisions = [TradeDecision(symbol="AAPL", action="long", confidence=0.97, allocation=0.50)]
    plans = engine.build_order_plans(decisions, selected)

    assert len(plans) == 1
    assert plans[0].telegram_approval_required is True
    assert plans[0].telegram_approval_granted is False
    assert plans[0].max_trade_pct == 0.05


def test_buy_more_signal_uses_cash_rich_cap_when_available_cash_is_high():
    engine = AlpacaExecutionEngine.__new__(AlpacaExecutionEngine)
    engine.config = TradingConfig(
        min_confidence=0.6,
        buy_more_threshold=0.05,
        max_single_trade_pct=0.01,
        cash_rich_trade_pct=0.05,
        cash_rich_available_cash_threshold=0.20,
    )
    engine.logger = DummyLogger()
    engine.client = FakeTradingClient(equity="10000", buying_power="10000", cash="10000")
    engine.client.positions = [
        SimpleNamespace(symbol="AAPL", qty="1", market_value="130", avg_entry_price="120"),
    ]

    selected = load_mock_universe()
    decisions = [TradeDecision(symbol="AAPL", action="long", confidence=0.9, allocation=0.50)]
    signals = engine.evaluate_held_positions(decisions, selected)

    assert len(signals) == 1
    assert signals[0].signal == "buy_more"
    assert signals[0].target_qty == 3
    assert signals[0].max_trade_pct == 0.05
    assert "5.00% of available cash" in signals[0].reason


def test_review_pending_orders_cancels_large_extended_hours_move_before_open():
    engine = AlpacaExecutionEngine.__new__(AlpacaExecutionEngine)
    engine.config = TradingConfig(pending_order_review_max_gap_pct=0.03)
    engine.logger = DummyLogger()
    engine.client = FakeTradingClient()
    engine.telegram = FakeTelegramNotifier(approved=False)
    engine.run_id = "test-run"
    engine.client.orders = [
        SimpleNamespace(
            id="ord-1",
            symbol="AAPL",
            side="buy",
            status="new",
            submitted_at="2026-04-16T12:00:00Z",
        )
    ]

    selected = load_mock_universe()
    for item in selected:
        if item.symbol == "AAPL":
            item.close = 130.0
            item.premarket.latest_price = 140.0
            break

    reviews = engine.review_pending_orders(
        selected,
        now=datetime(2026, 4, 16, 8, 30),
    )

    assert len(reviews) == 1
    assert reviews[0].action == "cancel"
    assert engine.client.cancelled_order_ids == ["ord-1"]


def test_review_pending_orders_skips_after_open():
    engine = AlpacaExecutionEngine.__new__(AlpacaExecutionEngine)
    engine.config = TradingConfig(pending_order_review_max_gap_pct=0.03)
    engine.logger = DummyLogger()
    engine.client = FakeTradingClient()
    engine.telegram = FakeTelegramNotifier(approved=False)
    engine.run_id = "test-run"
    engine.client.orders = [
        SimpleNamespace(id="ord-1", symbol="AAPL", side="buy", status="new", submitted_at="2026-04-16T12:00:00Z")
    ]


def test_order_plan_includes_limit_stop_and_take_profit_prices():
    engine = AlpacaExecutionEngine.__new__(AlpacaExecutionEngine)
    engine.config = TradingConfig(
        min_confidence=0.6,
        max_single_trade_pct=0.20,
        cash_rich_trade_pct=0.20,
        cash_rich_available_cash_threshold=0.20,
        order_style="bracket_limit",
        stop_atr_multiple=1.5,
        take_profit_r_multiple=2.0,
        entry_limit_buffer_pct=0.002,
    )
    engine.logger = DummyLogger()
    engine.client = FakeTradingClient(equity="100000", buying_power="100000", cash="100000")
    engine.telegram = FakeTelegramNotifier(approved=False)
    engine.run_id = "test-run"

    selected = load_mock_universe()
    decisions = [TradeDecision(symbol="AAPL", action="long", confidence=0.85, allocation=0.10)]
    plans = engine.build_order_plans(decisions, selected)

    assert len(plans) == 1
    assert plans[0].order_style == "bracket_limit"
    assert plans[0].entry_limit_price > selected[2].close
    assert plans[0].stop_price < selected[2].close
    assert plans[0].take_profit_price > selected[2].close
    assert plans[0].risk_notional > 0


def test_daily_loss_kill_switch_blocks_new_orders():
    engine = AlpacaExecutionEngine.__new__(AlpacaExecutionEngine)
    engine.config = TradingConfig(
        min_confidence=0.6,
        max_daily_loss_pct=0.02,
        max_single_trade_pct=0.20,
    )
    engine.logger = DummyLogger()
    engine.client = FakeTradingClient(equity="97000", buying_power="97000", cash="97000")
    engine.client.account.last_equity = "100000"
    engine.telegram = FakeTelegramNotifier(approved=False)
    engine.run_id = "test-run"

    selected = load_mock_universe()
    decisions = [TradeDecision(symbol="AAPL", action="long", confidence=0.85, allocation=0.10)]

    assert engine.build_order_plans(decisions, selected) == []

    reviews = engine.review_pending_orders(
        load_mock_universe(),
        now=datetime(2026, 4, 16, 10, 0),
    )

    assert reviews == []
    assert engine.client.cancelled_order_ids == []


def test_submit_orders_records_skip_when_open_order_exists():
    engine = AlpacaExecutionEngine.__new__(AlpacaExecutionEngine)
    engine.config = TradingConfig(execute_orders=False)
    engine.logger = DummyLogger()
    engine.client = FakeTradingClient()
    engine.telegram = FakeTelegramNotifier(approved=False)
    engine.run_id = "test-run"
    engine.client.orders = [SimpleNamespace(symbol="AAPL")]

    results = engine.submit_orders(
        [
            SimpleNamespace(
                symbol="AAPL",
                side="long",
                qty=3,
                notional=390.0,
                confidence=0.91,
                max_trade_pct=0.01,
                telegram_approval_required=False,
                telegram_approval_granted=False,
            )
        ]
    )

    assert len(results) == 1
    assert results[0]["status"] == "skipped_open_order"
    assert "Open order already exists" in results[0]["reason"]
