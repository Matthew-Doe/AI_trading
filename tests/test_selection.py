from trading_system.main import load_mock_universe
from trading_system.selection import CandidateSelector


class DummyLogger:
    def info(self, *args, **kwargs):
        pass


def test_candidate_selection_returns_ranked_subset():
    selector = CandidateSelector(DummyLogger())
    universe = load_mock_universe()
    selected = selector.select(universe, 5)
    assert len(selected) == 5
    assert all("total" in item.score_breakdown for item in selected)
    scores = [item.score_breakdown["total"] for item in selected]
    assert scores == sorted(scores, reverse=True)


def test_candidate_selection_excludes_untradeable_symbols():
    selector = CandidateSelector(DummyLogger())
    universe = load_mock_universe()
    universe[0].is_tradeable = False
    universe[0].data_quality_flags = ["extreme_20d_return"]

    selected = selector.select(universe, 5)

    assert universe[0].symbol not in {item.symbol for item in selected}


def test_candidate_selection_penalizes_overextended_symbols():
    selector = CandidateSelector(DummyLogger())
    clean = load_mock_universe()[0]
    overextended = load_mock_universe()[1]
    overextended.raw_metrics["price_change_20d"] = 0.90
    overextended.indicators.rsi14 = 96.0

    clean_score = selector.score_symbol(clean).score_breakdown["total"]
    overextended_score = selector.score_symbol(overextended).score_breakdown["total"]

    assert overextended_score < clean_score
