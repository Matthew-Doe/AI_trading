from __future__ import annotations

import sys
from datetime import date

import pytest

from scripts import run_live_paper_daily


def test_should_run_is_false_before_start_date(monkeypatch):
    monkeypatch.setenv("LIVE_PAPER_START_DATE", "2026-05-04")

    assert run_live_paper_daily.should_run(date(2026, 5, 1)) is False


def test_should_run_is_true_on_start_date(monkeypatch):
    monkeypatch.setenv("LIVE_PAPER_START_DATE", "2026-05-04")

    assert run_live_paper_daily.should_run(date(2026, 5, 4)) is True


def test_main_skips_before_start_date(monkeypatch, capsys):
    calls = []

    monkeypatch.setenv("LIVE_PAPER_START_DATE", "2026-05-04")
    monkeypatch.setattr(run_live_paper_daily, "should_run", lambda: False)
    monkeypatch.setattr(run_live_paper_daily.trading_main, "main", lambda: calls.append("called"))

    assert run_live_paper_daily.main() == 0
    assert calls == []
    assert "Skipping live paper run before 2026-05-04." in capsys.readouterr().out


def test_main_forwards_full_phase_by_default(monkeypatch):
    captured_argv = []

    def fake_main() -> int:
        captured_argv.extend(sys.argv)
        return 17

    monkeypatch.setattr(run_live_paper_daily, "should_run", lambda: True)
    monkeypatch.setattr(run_live_paper_daily.trading_main, "main", fake_main)
    monkeypatch.setattr(sys, "argv", ["scripts/run_live_paper_daily.py"])

    assert run_live_paper_daily.main() == 17
    assert captured_argv == ["scripts/run_live_paper_daily.py", "--phase", "full"]
    assert sys.argv == ["scripts/run_live_paper_daily.py"]


def test_main_preserves_explicit_phase_and_extra_args(monkeypatch):
    captured_argv = []

    def fake_main() -> int:
        captured_argv.extend(sys.argv)
        return 0

    monkeypatch.setattr(run_live_paper_daily, "should_run", lambda: True)
    monkeypatch.setattr(run_live_paper_daily.trading_main, "main", fake_main)
    monkeypatch.setattr(
        sys,
        "argv",
        ["scripts/run_live_paper_daily.py", "--mock", "--phase", "entry_review"],
    )

    assert run_live_paper_daily.main() == 0
    assert captured_argv == [
        "scripts/run_live_paper_daily.py",
        "--mock",
        "--phase",
        "entry_review",
    ]


def test_invalid_start_date_exits(monkeypatch):
    monkeypatch.setenv("LIVE_PAPER_START_DATE", "05/04/2026")

    with pytest.raises(SystemExit, match="Invalid LIVE_PAPER_START_DATE"):
        run_live_paper_daily.should_run(date(2026, 5, 4))
