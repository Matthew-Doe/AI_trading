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

