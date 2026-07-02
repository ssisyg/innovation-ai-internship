"""
Unit tests for the LangGraph orchestration logic in backend/agent.py:
should_continue (routing), ensure_final (format guardrail), and
agent_node (the LLM call + memory-context wiring).

The LLM itself is always mocked here (`backend.agent.llm_with_tools.invoke`)
so these tests are fast, deterministic, and never call OpenAI.
"""

from langchain_core.messages import AIMessage, HumanMessage

from backend.agent import should_continue, ensure_final, agent_node, SIGNAL_RE


def make_state(messages, iterations=0, ticker="AAPL"):
    return {"messages": messages, "ticker": ticker, "iterations": iterations}


class TestShouldContinue:
    def test_routes_to_tools_when_tool_calls_present(self):
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "PriceTool", "args": {"ticker": "AAPL"}, "id": "1"}],
        )
        state = make_state([msg], iterations=1)

        assert should_continue(state) == "tools"

    def test_routes_to_end_when_no_tool_calls(self):
        msg = AIMessage(content="Final Signal: HOLD")
        state = make_state([msg], iterations=1)

        assert should_continue(state) == "end"

    def test_stops_at_max_iterations_even_with_pending_tool_calls(self):
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "PriceTool", "args": {}, "id": "1"}],
        )
        state = make_state([msg], iterations=5)

        assert should_continue(state) == "end"


class TestEnsureFinal:
    def test_appends_hold_fallback_when_signal_missing(self):
        msg = AIMessage(content="The stock looks interesting but risky.")
        state = make_state([msg])

        result = ensure_final(state)
        content = result["messages"][-1].content

        assert SIGNAL_RE.search(content)
        assert "Final Signal: HOLD" in content

    def test_leaves_content_unchanged_when_signal_already_present(self):
        original = "Final Signal: BUY\nConfidence: HIGH\nReasoning:\n- strong RSI\n"
        msg = AIMessage(content=original)
        state = make_state([msg])

        result = ensure_final(state)

        assert result["messages"][-1].content == original


class _FakeLLM:
    """
    Stand-in for `llm_with_tools`.

    In recent langchain-core versions, `llm.bind_tools(...)` returns a
    pydantic-model-backed RunnableBinding that rejects
    `monkeypatch.setattr(obj, "invoke", fn)` on the *instance* (pydantic
    only allows setting declared fields). So instead of patching the
    `.invoke` attribute on the real object, we swap out the whole
    `backend.agent.llm_with_tools` module-level name for one of these.
    """

    def __init__(self, invoke_fn):
        self._invoke_fn = invoke_fn

    def invoke(self, messages):
        return self._invoke_fn(messages)


class TestAgentNode:
    def test_first_iteration_injects_memory_context_into_system_prompt(self, monkeypatch):
        captured = {}

        def fake_invoke(messages):
            captured["messages"] = messages
            return AIMessage(content="Final Signal: HOLD\nConfidence: LOW\nReasoning:\n- n/a\n")

        monkeypatch.setattr("backend.agent.llm_with_tools", _FakeLLM(fake_invoke))
        monkeypatch.setattr(
            "backend.agent.memory.format_for_prompt",
            lambda ticker=None, limit=5: "Past recommendations for AAPL:\n  - [2026-01-01] AAPL -> BUY: ...",
        )

        state = make_state([HumanMessage(content="Ticker: AAPL. Analyze it.")], iterations=0)
        result = agent_node(state)

        system_msg = captured["messages"][0]
        assert "Past recommendations for AAPL" in system_msg.content
        assert result["iterations"] == 1

    def test_later_iterations_do_not_re_fetch_memory(self, monkeypatch):
        calls = {"count": 0}

        def fake_format_for_prompt(ticker=None, limit=5):
            calls["count"] += 1
            return "should not be called"

        def fake_invoke(messages):
            return AIMessage(content="Final Signal: HOLD\nConfidence: LOW\nReasoning:\n- n/a\n")

        monkeypatch.setattr("backend.agent.llm_with_tools", _FakeLLM(fake_invoke))
        monkeypatch.setattr("backend.agent.memory.format_for_prompt", fake_format_for_prompt)

        state = make_state([HumanMessage(content="Ticker: AAPL. Analyze it.")], iterations=1)
        agent_node(state)

        assert calls["count"] == 0

    def test_llm_failure_produces_hold_fallback_response(self, monkeypatch):
        def broken_invoke(messages):
            raise RuntimeError("connection refused")

        monkeypatch.setattr("backend.agent.llm_with_tools", _FakeLLM(broken_invoke))
        monkeypatch.setattr(
            "backend.agent.memory.format_for_prompt",
            lambda ticker=None, limit=5: "No previous recommendations on record.",
        )

        state = make_state([HumanMessage(content="Ticker: AAPL. Analyze it.")], iterations=0)
        result = agent_node(state)

        content = result["messages"][-1].content
        assert "Final Signal: HOLD" in content
        assert "connection refused" in content
