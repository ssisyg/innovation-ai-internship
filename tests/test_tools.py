"""
Unit tests for the four LangChain tools in backend/agent.py.

These call each tool's `.invoke({...})` interface directly (the same
interface the LangGraph ToolNode uses), so we're exercising the real
tool wiring, not just the underlying plain functions.
"""

from backend.agent import PriceTool, SentimentTool, QuantTool, RAGTool


class TestPriceTool:
    def test_known_ticker_returns_indicators(self, known_ticker):
        result = PriceTool.invoke({"ticker": known_ticker})

        assert result["success"] is True
        data = result["data"]
        assert data["ticker"] == known_ticker
        assert isinstance(data["current_price"], float)
        assert isinstance(data["rsi"], float)
        assert isinstance(data["macd"], float)
        assert data["rsi_signal"] in {"OVERBOUGHT", "OVERSOLD", "NEUTRAL"}
        assert data["data_source"] in {"local_csv", "yfinance"}
        assert "data_as_of" in data

    def test_unknown_ticker_fails_without_hitting_network(self, monkeypatch, unknown_ticker):
        # No local CSV exists for this ticker; stub the yfinance fallback so
        # the test never makes a real network call.
        monkeypatch.setattr("backend.agent._fetch_yfinance", lambda *a, **k: None)

        result = PriceTool.invoke({"ticker": unknown_ticker})

        assert result["success"] is False
        assert "error" in result

    def test_unexpected_exception_is_caught_and_reported(self, monkeypatch, known_ticker):
        def boom(*args, **kwargs):
            raise RuntimeError("disk on fire")

        monkeypatch.setattr("backend.agent._load_local_ohlcv", boom)
        monkeypatch.setattr("backend.agent._fetch_yfinance", lambda *a, **k: None)

        result = PriceTool.invoke({"ticker": known_ticker})

        assert result["success"] is False
        assert "disk on fire" in result["error"]


class TestSentimentTool:
    def test_known_ticker_returns_sentiment_summary(self, known_ticker):
        result = SentimentTool.invoke({"ticker": known_ticker})

        assert result["success"] is True
        data = result["data"]
        assert data["ticker"] == known_ticker
        assert data["signal"] in {"BULLISH", "BEARISH", "NEUTRAL"}
        assert isinstance(data["avg_sentiment"], float)
        assert isinstance(data["sample_headlines"], list)
        assert len(data["sample_headlines"]) <= 3

    def test_unknown_ticker_reports_no_data(self, unknown_ticker):
        result = SentimentTool.invoke({"ticker": unknown_ticker})

        assert result["success"] is False
        assert result["error"] == "No sentiment data"


class TestQuantTool:
    def test_known_ticker_returns_prediction_and_shap(self, known_ticker):
        result = QuantTool.invoke({"ticker": known_ticker})

        assert result["success"] is True
        data = result["data"]
        assert data["prediction"] in {"UP", "DOWN"}
        assert data["signal"] in {"BUY", "SELL", "HOLD"}
        assert 0.0 <= data["up_probability"] <= 1.0
        assert len(data["top_shap_features"]) == 5
        for feat in data["top_shap_features"]:
            assert "feature" in feat and "mean_abs_shap" in feat

        lo, hi = data["confidence_interval"]
        assert 0.0 <= lo <= data["up_probability"] <= hi <= 1.0

    def test_unknown_ticker_reports_missing_precomputed_data(self, unknown_ticker):
        result = QuantTool.invoke({"ticker": unknown_ticker})

        assert result["success"] is False
        assert "No data available" in result["error"]

    def test_signal_matches_probability_thresholds(self, known_ticker):
        result = QuantTool.invoke({"ticker": known_ticker})
        data = result["data"]
        prob = data["up_probability"]

        if prob > 0.55:
            assert data["signal"] == "BUY"
        elif prob < 0.45:
            assert data["signal"] == "SELL"
        else:
            assert data["signal"] == "HOLD"


class TestRAGTool:
    def test_semantic_search_filters_by_ticker(self, known_ticker):
        result = RAGTool.invoke({"query": "earnings beat expectations", "ticker": known_ticker})

        assert result["success"] is True
        results = result["data"]["results"]
        assert isinstance(results, list)
        for item in results:
            assert item["ticker"] == known_ticker
            # relevance = 1 - cosine_distance; cosine distance can range
            # 0 (identical) to 2 (opposite), so relevance can legitimately
            # go negative for weakly/un-related results.
            assert -1.0 <= item["relevance"] <= 1.0
            assert "headline" in item and "sentiment" in item

    def test_semantic_search_without_ticker_filter(self):
        result = RAGTool.invoke({"query": "stock market volatility"})

        assert result["success"] is True
        assert isinstance(result["data"]["results"], list)
