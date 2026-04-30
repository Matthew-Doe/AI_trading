from trading_system.config import TradingConfig
from trading_system.scheduler import live_paper_phase_schedule


def test_live_paper_phase_schedule_uses_entry_and_exit_times():
    config = TradingConfig(
        enable_live_paper_backtest_style=True,
        live_entry_review_time_et="09:45",
        live_exit_review_time_et="15:45",
    )

    assert live_paper_phase_schedule(config) == {
        "entry_review": (9, 45),
        "exit_review": (15, 45),
    }
