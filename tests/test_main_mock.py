import argparse

from trading_system.config import TradingConfig
from trading_system.main import run_pipeline


def test_mock_pipeline_runs_successfully(tmp_path):
    config = TradingConfig(
        log_dir=tmp_path / "logs",
        run_dir=tmp_path / "runs",
        telegram_bot_token="",
        telegram_chat_id="",
    )
    args = argparse.Namespace(
        mock=True,
        llm_provider=None,
        openai_api_key=None,
        anthropic_api_key=None,
        llm_debate_model=None,
        llm_decision_model=None,
        include_news=False,
        replay_run_id=None,
        phase="full",
    )
    result = run_pipeline(config, args)
    assert result == 0
    run_dirs = list((tmp_path / "runs").iterdir())
    assert run_dirs
    assert (run_dirs[0] / "decisions.json").exists()
    assert (run_dirs[0] / "llm_usage.json").exists()
    assert (run_dirs[0] / "run_metrics.json").exists()
    assert (run_dirs[0] / "order_plans.json").exists()
    assert (run_dirs[0] / "report.json").exists()
    assert (run_dirs[0] / "report.html").exists()


def test_generate_report_for_latest_run(tmp_path):
    config = TradingConfig(
        log_dir=tmp_path / "logs",
        run_dir=tmp_path / "runs",
        telegram_bot_token="",
        telegram_chat_id="",
    )
    args = argparse.Namespace(
        mock=True,
        llm_provider=None,
        openai_api_key=None,
        anthropic_api_key=None,
        llm_debate_model=None,
        llm_decision_model=None,
        include_news=False,
        replay_run_id=None,
        phase="full",
    )
    assert run_pipeline(config, args) == 0

    from trading_system.main import generate_report_for_latest_run

    report_path = generate_report_for_latest_run(config)

    assert report_path.name == "report.html"
    assert report_path.exists()


def test_mock_exit_review_phase_skips_new_entry_plans(tmp_path):
    config = TradingConfig(
        log_dir=tmp_path / "logs",
        run_dir=tmp_path / "runs",
        telegram_bot_token="",
        telegram_chat_id="",
    )
    args = argparse.Namespace(
        mock=True,
        llm_provider=None,
        openai_api_key=None,
        anthropic_api_key=None,
        llm_debate_model=None,
        llm_decision_model=None,
        include_news=False,
        replay_run_id=None,
        phase="exit_review",
    )

    assert run_pipeline(config, args) == 0
    run_dir = next((tmp_path / "runs").iterdir())
    import json

    assert json.loads((run_dir / "order_plans.json").read_text(encoding="utf-8")) == []
