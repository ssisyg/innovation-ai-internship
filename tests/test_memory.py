"""
Unit tests for MemoryStore in backend/agent.py.

Every test builds its own MemoryStore against a throwaway SQLite file
under pytest's `tmp_path`, so these never touch the real
data/memory/agent_memory.db used by the running app.
"""

from backend.agent import MemoryStore


def test_save_and_get_history_orders_most_recent_first(tmp_path):
    store = MemoryStore(db_path=str(tmp_path / "memory.db"))

    store.save(ticker="AAPL", signal="BUY", reasoning="Strong momentum")
    store.save(ticker="AAPL", signal="HOLD", reasoning="Mixed signals")
    store.save(ticker="TSLA", signal="SELL", reasoning="Weak sentiment")

    all_records = store.get_history(limit=10)
    assert len(all_records) == 3
    assert all_records[0]["ticker"] == "TSLA"  # most recently saved


def test_get_history_filters_by_ticker(tmp_path):
    store = MemoryStore(db_path=str(tmp_path / "memory.db"))

    store.save(ticker="AAPL", signal="BUY", reasoning="r1")
    store.save(ticker="AAPL", signal="HOLD", reasoning="r2")
    store.save(ticker="TSLA", signal="SELL", reasoning="r3")

    aapl_only = store.get_history(ticker="AAPL", limit=10)
    assert len(aapl_only) == 2
    assert all(r["ticker"] == "AAPL" for r in aapl_only)


def test_get_history_respects_limit(tmp_path):
    store = MemoryStore(db_path=str(tmp_path / "memory.db"))

    for i in range(5):
        store.save(ticker="AAPL", signal="HOLD", reasoning=f"round {i}")

    records = store.get_history(ticker="AAPL", limit=3)
    assert len(records) == 3


def test_save_persists_tool_data_as_json(tmp_path):
    store = MemoryStore(db_path=str(tmp_path / "memory.db"))

    store.save(
        ticker="NVDA",
        signal="BUY",
        reasoning="Solid quant signal",
        tool_data=[{"tool": "PriceTool", "result": {"success": True}}],
    )

    record = store.get_history(ticker="NVDA", limit=1)[0]
    assert "PriceTool" in record["tool_data"]


def test_format_for_prompt_with_no_history(tmp_path):
    store = MemoryStore(db_path=str(tmp_path / "memory.db"))

    text = store.format_for_prompt(ticker="AAPL")

    assert text == "No previous recommendations on record."


def test_format_for_prompt_includes_ticker_and_signal(tmp_path):
    store = MemoryStore(db_path=str(tmp_path / "memory.db"))
    store.save(ticker="NVDA", signal="BUY", reasoning="x" * 200)

    text = store.format_for_prompt(ticker="NVDA")

    assert "NVDA" in text
    assert "BUY" in text
    # reasoning is truncated to 120 chars in the summary line
    assert "x" * 130 not in text
