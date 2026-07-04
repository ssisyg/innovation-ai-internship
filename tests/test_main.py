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
        body = resp.json()
        assert body["status"] == "healthy"
        # redis is optional infra (caching/rate-limiting) -- health should
        # report its state but never depend on it to return 200.
        assert body["redis"] in ("connected", "unavailable")


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


class TestAnalyzeCache:
    """
    Proves the Redis cache added in main.py actually does something:
    an identical (ticker, query) request within the TTL must be served
    from cache without invoking the LangGraph agent a second time.

    Redis itself is faked (not a real connection) so this test doesn't
    depend on infra being up -- same reasoning as mocking agent_app.stream
    elsewhere in this file. See backend/main.py's try/except around every
    redis_client call: cache misses/errors always fall through to a real
    run, so this also implicitly documents that fallback contract.
    """

    def test_repeat_request_hits_cache_and_skips_agent(self, monkeypatch):
        call_count = {"n": 0}

        def fake_stream(state):
            call_count["n"] += 1
            yield {
                "agent": {
                    "messages": [
                        AIMessage(content="Final Signal: BUY\nConfidence: HIGH\nReasoning:\n- momentum\n")
                    ]
                }
            }

        store = {}

        async def fake_get(key):
            return store.get(key)

        async def fake_set(key, value, ex=None):
            store[key] = value

        monkeypatch.setattr("backend.main.agent_app.stream", fake_stream)
        monkeypatch.setattr("backend.main.memory.save", lambda **kwargs: None)
        monkeypatch.setattr("backend.main.redis_client.get", fake_get)
        monkeypatch.setattr("backend.main.redis_client.set", fake_set)

        payload = {"ticker": "AAPL", "query": "Should I buy?"}
        headers = {"x-api-key": VALID_API_KEY}

        # First call: cache is empty -> real run, and it populates the cache.
        with client.stream("POST", "/analyze", json=payload, headers=headers) as resp:
            first_events = [
                json.loads(line[len("data: "):])
                for line in resp.iter_lines()
                if line.startswith("data: ")
            ]
        assert call_count["n"] == 1
        assert first_events[-1] == {"type": "done", "signal": "BUY"}

        # Second, identical call: must be served from cache -- the agent
        # is never invoked again, and the done event is flagged cached.
        with client.stream("POST", "/analyze", json=payload, headers=headers) as resp:
            second_events = [
                json.loads(line[len("data: "):])
                for line in resp.iter_lines()
                if line.startswith("data: ")
            ]
        assert call_count["n"] == 1  # <-- the whole point: agent NOT called again
        assert second_events[-1] == {"type": "done", "signal": "BUY", "cached": True}

    def test_different_query_is_not_a_cache_hit(self, monkeypatch):
        call_count = {"n": 0}

        def fake_stream(state):
            call_count["n"] += 1
            yield {
                "agent": {
                    "messages": [AIMessage(content="Final Signal: HOLD\nConfidence: LOW\nReasoning:\n- n/a\n")]
                }
            }

        store = {}

        async def fake_get(key):
            return store.get(key)

        async def fake_set(key, value, ex=None):
            store[key] = value

        monkeypatch.setattr("backend.main.agent_app.stream", fake_stream)
        monkeypatch.setattr("backend.main.memory.save", lambda **kwargs: None)
        monkeypatch.setattr("backend.main.redis_client.get", fake_get)
        monkeypatch.setattr("backend.main.redis_client.set", fake_set)

        headers = {"x-api-key": VALID_API_KEY}
        with client.stream("POST", "/analyze", json={"ticker": "AAPL", "query": "query one"}, headers=headers) as resp:
            list(resp.iter_lines())
        with client.stream("POST", "/analyze", json={"ticker": "AAPL", "query": "query two"}, headers=headers) as resp:
            list(resp.iter_lines())

        assert call_count["n"] == 2  # different cache keys -> agent runs both times
