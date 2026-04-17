from trading_system.config import TradingConfig, _parse_schedule_times


def test_parse_schedule_times_multiple_entries():
    assert _parse_schedule_times("08:30,12:30,15:30", 8, 30) == (
        (8, 30),
        (12, 30),
        (15, 30),
    )


def test_trading_config_accepts_explicit_schedule_times(tmp_path):
    config = TradingConfig(
        log_dir=tmp_path / "logs",
        run_dir=tmp_path / "runs",
        cache_dir=tmp_path / ".cache",
        scheduled_times=((8, 30), (12, 30), (15, 30)),
    )

    assert config.scheduled_times == ((8, 30), (12, 30), (15, 30))
