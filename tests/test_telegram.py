from trading_system.models import OrderPlan
from trading_system.telegram import TelegramNotifier
from trading_system.config import TradingConfig


class DummyLogger:
    def warning(self, *args, **kwargs):
        pass


def test_humanize_trade_reason_for_sizing_trace():
    plan = OrderPlan(
        symbol="LRCX",
        side="long",
        qty=3,
        notional=783.11,
        confidence=0.75,
        allocation=0.14,
        reason="allocation=0.14, risk_qty=88, alloc_qty=55, max_trade_qty=3",
        max_trade_pct=0.01,
    )

    message = TelegramNotifier._humanize_trade_reason(plan)

    assert "target allocation 14%" in message
    assert "max trade cap" in message
    assert "allocation=0.14" not in message


def test_humanize_trade_reason_preserves_plain_language_reason():
    plan = OrderPlan(
        symbol="AAPL",
        side="sell",
        qty=10,
        notional=1000.0,
        confidence=0.8,
        allocation=0.0,
        reason="Held position is no longer in the active conviction set.",
    )

    assert TelegramNotifier._humanize_trade_reason(plan) == plan.reason


def test_send_message_is_best_effort_when_request_fails():
    notifier = TelegramNotifier(
        TradingConfig(telegram_bot_token="token", telegram_chat_id="chat"),
        DummyLogger(),
    )

    class BrokenSession:
        def post(self, *args, **kwargs):
            raise RuntimeError("network unavailable")

    notifier.session = BrokenSession()

    assert notifier.send_message("hello") is None
