from __future__ import annotations

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from trading_system.config import TradingConfig
from trading_system.main import parse_args, run_pipeline
from trading_system.portfolio_summary import send_market_close_summary
from trading_system.utils import get_logger


def main() -> int:
    config = TradingConfig()
    args = parse_args()
    scheduler = BlockingScheduler(timezone=config.market_timezone)
    for hour, minute in config.scheduled_times:
        scheduler.add_job(
            lambda: run_pipeline(config, args),
            trigger=CronTrigger(
                day_of_week="mon-fri",
                hour=hour,
                minute=minute,
            ),
            id=f"trading_run_{hour:02d}{minute:02d}",
            replace_existing=True,
        )
    scheduler.add_job(
        lambda: send_market_close_summary(
            config, get_logger(config.log_dir, "market_close_summary")
        ),
        trigger=CronTrigger(
            day_of_week="mon-fri",
            hour=config.market_close_summary_hour,
            minute=config.market_close_summary_minute,
        ),
        id="market_close_summary",
        replace_existing=True,
    )
    schedule_text = ", ".join(
        f"{hour:02d}:{minute:02d}" for hour, minute in config.scheduled_times
    )
    print(
        f"Scheduler active for {schedule_text} plus close summary at "
        f"{config.market_close_summary_hour:02d}:{config.market_close_summary_minute:02d} "
        f"{config.market_timezone} on weekdays"
    )
    scheduler.start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
