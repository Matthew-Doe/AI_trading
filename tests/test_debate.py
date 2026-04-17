from trading_system.debate import OllamaDebateEngine


class DummyLogger:
    def warning(self, *args, **kwargs):
        pass


def test_extract_json_finds_embedded_object():
    raw = 'preface {"symbol":"AAPL","position":"bull","confidence":0.8,"arguments":["a"],"risks":["r"],"key_levels":{"support":1,"resistance":2}} suffix'
    payload = OllamaDebateEngine._extract_json(raw)

    assert payload["symbol"] == "AAPL"


def test_extract_json_raises_when_no_object_present():
    try:
        OllamaDebateEngine._extract_json("plain text without braces")
    except Exception as exc:  # noqa: BLE001
        assert "No JSON object found" in str(exc)
    else:
        raise AssertionError("Expected JSON extraction failure")
