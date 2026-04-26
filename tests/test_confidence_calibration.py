from datetime import UTC, datetime

from trading_system.data import DataIngestionError
from trading_system.confidence_calibration import (
    ConfidenceBucket,
    build_historical_decision_outcomes,
    calibrate_confidence,
    label_decision_correctness,
)


def test_label_decision_correctness_for_long_short_and_skip():
    assert label_decision_correctness("long", 0.03, actionable_move_pct=0.02) is True
    assert label_decision_correctness("short", -0.04, actionable_move_pct=0.02) is True
    assert label_decision_correctness("skip", 0.01, actionable_move_pct=0.02) is True
    assert label_decision_correctness("skip", -0.03, actionable_move_pct=0.02) is False


def test_calibrate_confidence_uses_bucket_hit_rate():
    buckets = [
        ConfidenceBucket(
            lower=0.8,
            upper=0.9,
            sample_count=5,
            correct_count=4,
            hit_rate=0.8,
        )
    ]

    calibrated = calibrate_confidence(
        raw_confidence=0.84,
        buckets=buckets,
        global_hit_rate=0.55,
        minimum_samples=3,
    )

    assert calibrated == 0.82


def test_calibrate_confidence_excludes_non_terminal_upper_bound():
    buckets = [
        ConfidenceBucket(
            lower=0.8,
            upper=0.9,
            sample_count=10,
            correct_count=8,
            hit_rate=0.8,
        )
    ]

    calibrated = calibrate_confidence(
        raw_confidence=0.9,
        buckets=buckets,
        global_hit_rate=0.55,
        minimum_samples=3,
    )

    assert calibrated == 0.8


def test_calibrate_confidence_includes_terminal_upper_bound():
    buckets = [
        ConfidenceBucket(
            lower=0.9,
            upper=1.0,
            sample_count=10,
            correct_count=9,
            hit_rate=0.9,
        )
    ]

    calibrated = calibrate_confidence(
        raw_confidence=1.0,
        buckets=buckets,
        global_hit_rate=0.55,
        minimum_samples=3,
    )

    assert calibrated == 0.95


def test_calibrate_confidence_returns_raw_when_unmatched():
    calibrated = calibrate_confidence(
        raw_confidence=0.42,
        buckets=[],
        global_hit_rate=0.55,
        minimum_samples=3,
    )

    assert calibrated == 0.46


def test_calibrate_confidence_uses_bucket_hit_rate_at_exact_minimum_samples():
    buckets = [
        ConfidenceBucket(
            lower=0.8,
            upper=0.9,
            sample_count=3,
            correct_count=2,
            hit_rate=0.67,
        )
    ]

    calibrated = calibrate_confidence(
        raw_confidence=0.84,
        buckets=buckets,
        global_hit_rate=0.55,
        minimum_samples=3,
    )

    assert calibrated == 0.76


def test_calibrate_confidence_falls_back_when_bucket_is_sparse():
    buckets = [
        ConfidenceBucket(
            lower=0.8,
            upper=0.9,
            sample_count=1,
            correct_count=1,
            hit_rate=1.0,
        )
    ]

    calibrated = calibrate_confidence(
        raw_confidence=0.84,
        buckets=buckets,
        global_hit_rate=0.55,
        minimum_samples=3,
    )

    assert round(calibrated, 2) == 0.75


def test_label_decision_correctness_rejects_invalid_action():
    try:
        label_decision_correctness("hold", 0.03, actionable_move_pct=0.02)
    except ValueError as exc:
        assert "Unsupported action" in str(exc)
    else:
        raise AssertionError("ValueError was not raised")


def test_build_historical_decision_outcomes_skips_runs_without_forward_history(tmp_path):
    run_dir = tmp_path / "runs" / "20260415T123002Z"
    run_dir.mkdir(parents=True)
    (run_dir / "decisions.json").write_text(
        '[{"symbol":"AAPL","action":"long","confidence":0.82,"allocation":0.4}]',
        encoding="utf-8",
    )

    class StubMarketDataService:
        def fetch_forward_close_window(self, symbol: str, *, as_of: datetime, trading_days_ahead: int):
            del symbol
            del as_of
            del trading_days_ahead
            raise DataIngestionError("Insufficient forward close history for AAPL at 2026-04-15")

    outcomes = build_historical_decision_outcomes(
        run_root=tmp_path / "runs",
        market_data_service=StubMarketDataService(),
        actionable_move_pct=0.02,
        now=datetime(2026, 4, 22, tzinfo=UTC),
    )

    assert outcomes == []


def test_build_historical_decision_outcomes_returns_labeled_outcome(tmp_path):
    run_dir = tmp_path / "runs" / "20260415T123002Z"
    run_dir.mkdir(parents=True)
    (run_dir / "decisions.json").write_text(
        '[{"symbol":"AAPL","action":"long","confidence":0.82,"allocation":0.4}]',
        encoding="utf-8",
    )

    class StubMarketDataService:
        def fetch_forward_close_window(self, symbol: str, *, as_of: datetime, trading_days_ahead: int):
            assert symbol == "AAPL"
            assert as_of == datetime(2026, 4, 15, 12, 30, 2, tzinfo=UTC)
            assert trading_days_ahead == 3
            return 100.0, 103.0, "2026-04-21T16:00:00+00:00"

    outcomes = build_historical_decision_outcomes(
        run_root=tmp_path / "runs",
        market_data_service=StubMarketDataService(),
        actionable_move_pct=0.02,
        now=datetime(2026, 4, 22, tzinfo=UTC),
    )

    assert len(outcomes) == 1
    assert outcomes[0].run_id == "20260415T123002Z"
    assert outcomes[0].symbol == "AAPL"
    assert outcomes[0].action == "long"
    assert outcomes[0].raw_confidence == 0.82
    assert outcomes[0].forward_return == 0.03
    assert outcomes[0].is_correct is True
    assert outcomes[0].forward_as_of == "2026-04-21T16:00:00+00:00"


def test_build_historical_decision_outcomes_skips_future_run_ids_without_lookup(tmp_path):
    run_dir = tmp_path / "runs" / "20260425T123002Z"
    run_dir.mkdir(parents=True)
    (run_dir / "decisions.json").write_text(
        '[{"symbol":"AAPL","action":"long","confidence":0.82,"allocation":0.4}]',
        encoding="utf-8",
    )

    class StubMarketDataService:
        def __init__(self):
            self.calls = 0

        def fetch_forward_close_window(self, symbol: str, *, as_of: datetime, trading_days_ahead: int):
            del symbol
            del as_of
            del trading_days_ahead
            self.calls += 1
            raise AssertionError("future runs should be skipped before lookup")

    market_data_service = StubMarketDataService()

    outcomes = build_historical_decision_outcomes(
        run_root=tmp_path / "runs",
        market_data_service=market_data_service,
        actionable_move_pct=0.02,
        now=datetime(2026, 4, 22, tzinfo=UTC),
    )

    assert outcomes == []
    assert market_data_service.calls == 0


def test_build_historical_decision_outcomes_reads_backtest_day_folders(tmp_path):
    run_dir = tmp_path / "backtest" / "2026-03-02"
    run_dir.mkdir(parents=True)
    (run_dir / "decisions.json").write_text(
        '[{"symbol":"AAPL","action":"long","confidence":0.82,"allocation":0.4}]',
        encoding="utf-8",
    )

    class StubMarketDataService:
        def fetch_forward_close_window(self, symbol: str, *, as_of: datetime, trading_days_ahead: int):
            assert symbol == "AAPL"
            assert as_of == datetime(2026, 3, 2, tzinfo=UTC)
            assert trading_days_ahead == 3
            return 100.0, 103.0, "2026-03-05T16:00:00+00:00"

    outcomes = build_historical_decision_outcomes(
        run_root=tmp_path / "backtest",
        market_data_service=StubMarketDataService(),
        actionable_move_pct=0.02,
        now=datetime(2026, 3, 6, tzinfo=UTC),
    )

    assert len(outcomes) == 1
    assert outcomes[0].run_id == "2026-03-02"
    assert outcomes[0].is_correct is True


def test_build_historical_decision_outcomes_skips_current_backtest_day(tmp_path):
    run_dir = tmp_path / "backtest" / "2026-03-02"
    run_dir.mkdir(parents=True)
    (run_dir / "decisions.json").write_text(
        '[{"symbol":"AAPL","action":"long","confidence":0.82,"allocation":0.4}]',
        encoding="utf-8",
    )

    class StubMarketDataService:
        def fetch_forward_close_window(self, symbol: str, *, as_of: datetime, trading_days_ahead: int):
            raise AssertionError("current backtest day should not be used for calibration")

    outcomes = build_historical_decision_outcomes(
        run_root=tmp_path / "backtest",
        market_data_service=StubMarketDataService(),
        actionable_move_pct=0.02,
        now=datetime(2026, 3, 2, tzinfo=UTC),
    )

    assert outcomes == []
