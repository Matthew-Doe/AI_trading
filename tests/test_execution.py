from datetime import datetime
from types import SimpleNamespace

from trading_system.config import TradingConfig
from trading_system.execution import AlpacaExecutionEngine, ExecutionError
from trading_system.live_strategy_state import LivePositionState, LiveStrategyStateStore
from trading_system.main import load_mock_universe
from trading_system.models import OrderPlan, TradeDecision
from trading_system.tax_state import TaxLossCooldownState
from trading_system.utils import write_json


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

    def submit_order(self, order_data):
        return SimpleNamespace(id="submitted-1", status="submitted")


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
    assert signals[0].tax_loss_exit is True
    assert signals[0].estimated_tax_loss == 200.0
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


def test_live_paper_backtest_style_bypasses_telegram_approval():
    engine = AlpacaExecutionEngine.__new__(AlpacaExecutionEngine)
    engine.config = TradingConfig(
        enable_live_paper_backtest_style=True,
        allow_live_large_trade_approval=False,
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
    assert engine.telegram.messages == []
    assert plans[0].telegram_approval_required is False
    assert plans[0].max_trade_pct == 0.10


def test_live_backtest_style_defers_thesis_failure_and_persists_count(tmp_path):
    engine = AlpacaExecutionEngine.__new__(AlpacaExecutionEngine)
    engine.config = TradingConfig(
        enable_live_paper_backtest_style=True,
        enable_conditional_hold_extension=True,
        conditional_hold_extension_observations=2,
        conditional_hold_max_adverse_pct=0.05,
    )
    engine.state_store = LiveStrategyStateStore(tmp_path / "state.json")
    engine.state_store.record_entry(
        LivePositionState(
            symbol="AAPL",
            side="long",
            quantity=10,
            entry_price=100.0,
            stop_price=95.0,
            take_profit_price=110.0,
            entry_timestamp="2026-01-02T14:30:00+00:00",
        )
    )

    plan = engine.review_live_backtest_style_position("AAPL", current_price=98.0, thesis_failed=True)

    assert plan is None
    assert engine.state_store.positions["AAPL"].thesis_failure_deferrals == 1


def test_live_backtest_style_partial_exit_only_once(tmp_path):
    engine = AlpacaExecutionEngine.__new__(AlpacaExecutionEngine)
    engine.config = TradingConfig(
        enable_live_paper_backtest_style=True,
        enable_partial_profit_taking=True,
        partial_profit_take_fraction=0.5,
        partial_profit_trailing_stop_pct=0.05,
    )
    engine.state_store = LiveStrategyStateStore(tmp_path / "state.json")
    engine.state_store.record_entry(
        LivePositionState(
            symbol="AAPL",
            side="long",
            quantity=10,
            entry_price=100.0,
            stop_price=95.0,
            take_profit_price=110.0,
            entry_timestamp="2026-01-02T14:30:00+00:00",
        )
    )

    first = engine.review_live_backtest_style_position("AAPL", current_price=112.0)
    reloaded = LiveStrategyStateStore(tmp_path / "state.json")
    engine.state_store = reloaded
    second = engine.review_live_backtest_style_position("AAPL", current_price=113.0)

    assert first is not None
    assert first.side == "sell"
    assert first.qty == 5
    assert "partial_exit=true" in first.reason
    assert reloaded.positions["AAPL"].partial_profit_taken is True
    assert second is None


def test_live_backtest_style_held_position_plans_use_state_review(tmp_path):
    engine = AlpacaExecutionEngine.__new__(AlpacaExecutionEngine)
    engine.config = TradingConfig(
        enable_live_paper_backtest_style=True,
        enable_partial_profit_taking=True,
        partial_profit_take_fraction=0.5,
    )
    engine.state_store = LiveStrategyStateStore(tmp_path / "state.json")
    engine.state_store.record_entry(
        LivePositionState(
            symbol="AAPL",
            side="long",
            quantity=10,
            entry_price=100.0,
            stop_price=95.0,
            take_profit_price=110.0,
            entry_timestamp="2026-01-02T14:30:00+00:00",
        )
    )
    selected = load_mock_universe()
    selected_by_symbol = {item.symbol: item for item in selected}
    selected_by_symbol["AAPL"].close = 112.0

    plans = engine.build_held_position_order_plans(
        [
            SimpleNamespace(
                symbol="AAPL",
                signal="hold",
                reason="aligned",
                current_side="long",
                current_qty=10,
                delta_qty=0,
                confidence=0.8,
                max_trade_pct=0.0,
            )
        ],
        selected,
    )

    assert len(plans) == 1
    assert plans[0].qty == 5
    assert "partial_exit=true" in plans[0].reason


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


def test_submit_orders_dry_run_includes_broker_lifecycle_fields():
    engine = AlpacaExecutionEngine.__new__(AlpacaExecutionEngine)
    engine.config = TradingConfig(execute_orders=False)
    engine.logger = DummyLogger()
    engine.client = FakeTradingClient()
    engine.telegram = FakeTelegramNotifier(approved=False)
    engine.run_id = "test-run"
    engine.tax_state = TaxLossCooldownState.__new__(TaxLossCooldownState)
    engine.tax_blocked_orders = []

    results = engine.submit_orders(
        [
            OrderPlan(
                symbol="AAPL",
                side="long",
                qty=3,
                notional=390.0,
                confidence=0.91,
                allocation=0.1,
                reason="test",
                entry_limit_price=131.0,
            )
        ]
    )

    assert results[0]["broker_order_id"] is None
    assert results[0]["client_order_id"] is None
    assert results[0]["submitted_at"] is None
    assert results[0]["filled_qty"] == 0
    assert results[0]["average_fill_price"] is None
    assert results[0]["limit_price"] == 131.0


def test_live_paper_backtest_style_refuses_non_paper_url(tmp_path):
    engine = AlpacaExecutionEngine.__new__(AlpacaExecutionEngine)
    engine.config = TradingConfig(
        enable_live_paper_backtest_style=True,
        alpaca_paper_base_url="https://api.alpaca.markets",
        live_paper_readiness_path=str(tmp_path / "readiness.json"),
    )
    engine.client = FakeTradingClient()

    try:
        engine.validate_live_paper_backtest_style_readiness()
    except ExecutionError as exc:
        assert "paper Alpaca endpoint" in str(exc)
    else:
        raise AssertionError("non-paper URL should be refused")


def test_live_paper_backtest_style_requires_empty_account_and_bias_readiness(tmp_path):
    readiness = tmp_path / "readiness.json"
    write_json(readiness, {"acceptance_grade": True, "strategy": "hold_tax_partial"})
    engine = AlpacaExecutionEngine.__new__(AlpacaExecutionEngine)
    engine.config = TradingConfig(
        enable_live_paper_backtest_style=True,
        alpaca_paper_base_url="https://paper-api.alpaca.markets",
        live_paper_readiness_path=str(readiness),
        require_empty_paper_account=True,
    )
    engine.client = FakeTradingClient()

    try:
        engine.validate_live_paper_backtest_style_readiness()
    except ExecutionError as exc:
        assert "empty paper account" in str(exc)
    else:
        raise AssertionError("non-empty paper account should be refused")

    engine.client.positions = []
    engine.client.orders = []
    assert engine.validate_live_paper_backtest_style_readiness() is None


def test_live_tax_loss_cooldown_blocks_new_long_entries(tmp_path):
    engine = AlpacaExecutionEngine.__new__(AlpacaExecutionEngine)
    engine.config = TradingConfig(
        run_dir=tmp_path,
        min_confidence=0.6,
        max_single_trade_pct=0.20,
        tax_loss_cooldown_days=31,
    )
    engine.logger = DummyLogger()
    engine.client = FakeTradingClient(equity="100000", buying_power="100000", cash="100000")
    engine.telegram = FakeTelegramNotifier(approved=False)
    engine.run_id = "test-run"
    engine.tax_state = TaxLossCooldownState(tmp_path / "tax_loss_cooldowns.json")
    engine.tax_blocked_orders = []
    engine.tax_state.record_loss_sale(symbol="AAPL", cooldown_days=31, estimated_loss=123.45)

    selected = load_mock_universe()
    decisions = [TradeDecision(symbol="AAPL", action="long", confidence=0.85, allocation=0.10)]

    plans = engine.build_order_plans(decisions, selected)

    assert plans == []
    assert engine.tax_blocked_orders == [
        {
            "symbol": "AAPL",
            "blocked_until": engine.tax_state.cooldowns["AAPL"]["blocked_until"],
            "reason": "tax_loss_cooldown",
        }
    ]


def test_live_tax_adjusted_reentry_allows_reduced_size(tmp_path):
    engine = AlpacaExecutionEngine.__new__(AlpacaExecutionEngine)
    engine.config = TradingConfig(
        run_dir=tmp_path,
        min_confidence=0.6,
        max_single_trade_pct=0.20,
        tax_loss_cooldown_days=31,
        enable_tax_adjusted_ev_reentry=True,
        tax_reentry_min_confidence=0.90,
        tax_reentry_min_expected_value_pct=2.0,
        tax_reentry_size_multiplier=0.50,
    )
    engine.logger = DummyLogger()
    engine.client = FakeTradingClient(equity="100000", buying_power="100000", cash="100000")
    engine.telegram = FakeTelegramNotifier(approved=False)
    engine.run_id = "test-run"
    engine.tax_state = TaxLossCooldownState(tmp_path / "tax_loss_cooldowns.json")
    engine.tax_blocked_orders = []
    engine.tax_state.record_loss_sale(symbol="AAPL", cooldown_days=31, estimated_loss=123.45)

    selected = load_mock_universe()
    decisions = [
        TradeDecision(
            symbol="AAPL",
            action="long",
            confidence=0.95,
            allocation=0.10,
            expected_value_pct=3.0,
        )
    ]

    plans = engine.build_order_plans(decisions, selected)

    assert len(plans) == 1
    assert plans[0].allocation == 0.05
    assert "tax_adjusted_reentry=true" in plans[0].reason


def test_submit_orders_records_live_tax_loss_cooldown(tmp_path):
    engine = AlpacaExecutionEngine.__new__(AlpacaExecutionEngine)
    engine.config = TradingConfig(
        run_dir=tmp_path,
        execute_orders=True,
        tax_loss_cooldown_days=31,
    )
    engine.logger = DummyLogger()
    engine.client = FakeTradingClient()
    engine.telegram = FakeTelegramNotifier(approved=False)
    engine.run_id = "test-run"
    engine.tax_state = TaxLossCooldownState(tmp_path / "tax_loss_cooldowns.json")
    engine.tax_blocked_orders = []

    results = engine.submit_orders(
        [
            OrderPlan(
                symbol="AAPL",
                side="sell",
                qty=10,
                notional=1000.0,
                confidence=0.7,
                allocation=0.0,
                reason="test loss sale",
                tax_loss_exit=True,
                estimated_tax_loss=200.0,
            )
        ]
    )

    assert results[0]["status"] == "submitted"
    assert results[0]["tax_loss_cooldown"]["symbol"] == "AAPL"
    assert engine.tax_state.is_blocked("AAPL")


def test_live_backtest_style_submit_records_entry_state(tmp_path):
    engine = AlpacaExecutionEngine.__new__(AlpacaExecutionEngine)
    engine.config = TradingConfig(
        run_dir=tmp_path,
        execute_orders=True,
        enable_live_paper_backtest_style=True,
    )
    engine.logger = DummyLogger()
    engine.client = FakeTradingClient()
    engine.telegram = FakeTelegramNotifier(approved=False)
    engine.run_id = "test-run"
    engine.tax_state = TaxLossCooldownState(tmp_path / "tax_loss_cooldowns.json")
    engine.tax_blocked_orders = []
    engine.state_store = LiveStrategyStateStore(tmp_path / "live_strategy_state.json")

    engine.submit_orders(
        [
            OrderPlan(
                symbol="AAPL",
                side="long",
                qty=10,
                notional=1000.0,
                confidence=0.9,
                allocation=0.1,
                reason="entry",
                entry_limit_price=101.0,
                stop_price=95.0,
                take_profit_price=110.0,
            )
        ]
    )

    assert engine.state_store.positions["AAPL"].quantity == 10
    assert engine.state_store.positions["AAPL"].stop_price == 95.0
