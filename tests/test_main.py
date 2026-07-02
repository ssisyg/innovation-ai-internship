"""
Tests for backend/main.py: the signal-parsing helper, API-key auth,
and the SSE-streaming /analyze endpoint.

The /analyze test never calls the real LangGraph agent or OpenAI --
it monkeypatches `agent_app.stream` with a small fake generator that
mimics the shape LangGraph produces, and `memory.save` so we can
assert on what would have been persisted.
"""

import json

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, ToolMessage

from backend.main import app, parse_signal

client = TestClient(app)
VALID_API_KEY = "test-api-key"  # matches tests/conftest.py env override


class TestParseSignal:
    def test_extracts_labelled_signal(self):
        assert parse_signal("Final Signal: BUY\nConfidence: HIGH") == "BUY"

    def test_case_insensitive_and_tolerant_of_spacing(self):
        assert parse_signal("final   signal:sell") == "SELL"

    def test_falls_back_to_keyword_scan_when_unlabelled(self):
        assert parse_signal("The recommendation is to hold for now.") == "HOLD"

    def test_empty_or_missing_text_is_unknown(self):
        assert parse_signal("") == "UNKNOWN"
        assert parse_signal(None) == "UNKNOWN"


class TestPublicEndpoints:
    def test_root(self):
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_health(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "healthy"}


class TestAuth:
    def test_history_without_api_key_is_rejected(self):
        resp = client.get("/history")
        assert resp.status_code == 401

    def test_history_with_wrong_api_key_is_rejected(self):
        resp = client.get("/history", headers={"x-api-key": "wrong-key"})
        assert resp.status_code == 401

    def test_history_with_correct_api_key_succeeds(self, monkeypatch):
        monkeypatch.setattr(
            "backend.main.memory.get_history",
            lambda ticker=None, limit=10: [{"ticker": "AAPL", "signal": "BUY"}],
        )
        resp = client.get("/history", headers={"x-api-key": VALID_API_KEY})
        assert resp.status_code == 200
        assert resp.json()["count"] == 1

    def test_analyze_without_api_key_is_rejected(self):
        resp = client.post("/analyze", json={"ticker": "AAPL"})
        assert resp.status_code == 401


class TestAnalyzeStream:
    def test_stream_emits_tool_and_final_events_then_saves_to_memory(self, monkeypatch):
        def fake_stream(state):
            # 1) agent decides to call a tool
            yield {
                "agent": {
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[{"name": "PriceTool", "args": {"ticker": "AAPL"}, "id": "1"}],
                        )
                    ]
                }
            }
            # 2) tool result comes back
            yield {
                "tools": {
                    "messages": [
                        ToolMessage(
                            content=json.dumps({"success": True, "data": {"current_price": 100}}),
                            name="PriceTool",
                            tool_call_id="1",
                        )
                    ]
                }
            }
            # 3) agent produces the final recommendation
            yield {
                "agent": {
                    "messages": [
                        AIMessage(
                            content="Final Signal: BUY\nConfidence: HIGH\nReasoning:\n- strong momentum\n"
                        )
                    ]
                }
            }

        saved = {}

        def fake_save(ticker, signal, reasoning, tool_data=None):
            saved["ticker"] = ticker
            saved["signal"] = signal
            saved["tool_data"] = tool_data

        monkeypatch.setattr("backend.main.agent_app.stream", fake_stream)
        monkeypatch.setattr("backend.main.memory.save", fake_save)

        with client.stream(
            "POST",
            "/analyze",
            json={"ticker": "AAPL", "query": "Should I buy?"},
            headers={"x-api-key": VALID_API_KEY},
        ) as resp:
            assert resp.status_code == 200
            events = [
                json.loads(line[len("data: "):])
                for line in resp.iter_lines()
                if line.startswith("data: ")
            ]

        event_types = [e["type"] for e in events]
        assert "tool_call" in event_types
        assert "tool_result" in event_types
        assert "final" in event_types
        assert events[-1] == {"type": "done", "signal": "BUY"}

        assert saved["ticker"] == "AAPL"
        assert saved["signal"] == "BUY"
        assert len(saved["tool_data"]) == 1

    def test_stream_defaults_query_when_omitted(self, monkeypatch):
        captured = {}

        def fake_stream(state):
            captured["query"] = state["messages"][0].content
            yield {
                "agent": {
                    "messages": [AIMessage(content="Final Signal: HOLD\nConfidence: LOW\nReasoning:\n- n/a\n")]
                }
            }

        monkeypatch.setattr("backend.main.agent_app.stream", fake_stream)
        monkeypatch.setattr("backend.main.memory.save", lambda **kwargs: None)

        with client.stream(
            "POST",
            "/analyze",
            json={"ticker": "AAPL"},
            headers={"x-api-key": VALID_API_KEY},
        ) as resp:
            list(resp.iter_lines())

        assert "detailed investment recommendation" in captured["query"]

    def test_stream_reports_errors_as_error_event(self, monkeypatch):
        def broken_stream(state):
            raise RuntimeError("graph exploded")
            yield  # pragma: no cover - unreachable, keeps this a generator

        monkeypatch.setattr("backend.main.agent_app.stream", broken_stream)

        with client.stream(
            "POST",
            "/analyze",
            json={"ticker": "AAPL"},
            headers={"x-api-key": VALID_API_KEY},
        ) as resp:
            events = [
                json.loads(line[len("data: "):])
                for line in resp.iter_lines()
                if line.startswith("data: ")
            ]

        assert events[-1]["type"] == "error"
        assert "graph exploded" in events[-1]["message"]
