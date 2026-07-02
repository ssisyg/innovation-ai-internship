"""
Week 12 Agent Evaluation Framework.

Runs the compiled LangGraph agent (backend.agent.agent_app) end-to-end
against a set of known tickers -- the same graph backend/main.py serves
over SSE, just invoked directly here instead of through the API -- and
scores four dimensions called out in the project statement:

  1. Tool Calling Accuracy
     Did the agent successfully call the three mandatory tools every
     run needs (Price / Sentiment / Quant)? RAGTool is conditional --
     see SYSTEM_PROMPT's RAGTOOL RULE -- so it's scored separately as
     "did the agent follow its own policy for when to call RAG",
     re-derived from SentimentTool/QuantTool's actual results rather
     than assumed to always be required.

  2. Response Latency
     Wall-clock time from request to final answer, per ticker.

  3. Groundedness
     Does the agent's final BUY/SELL/HOLD signal agree with QuantTool's
     own signal? QuantTool is the one place in the pipeline that emits
     a hard, checkable number (up_probability -> signal), so we use it
     as the faithfulness anchor. Disagreement isn't automatically
     "wrong" (the LLM may be weighing sentiment/RAG evidence too) --
     it's a rate to eyeball, not a hard pass/fail.

  4. Hallucination Check
     Flags final reasoning that references the *kind* of evidence a
     tool produces (SHAP values, sentiment scores, RSI, headlines...)
     when that tool never actually returned successfully -- i.e.
     numbers the model could only have invented.

IMPORTANT: this calls the real LLM once per ticker, so it costs money
and takes real wall-clock time. Use --n or --tickers to keep runs
small while iterating; run the full ticker set before a submission or
demo.

Usage:
    python scripts/evaluate_agent.py                     # sample of 5
    python scripts/evaluate_agent.py --n 10
    python scripts/evaluate_agent.py --tickers AAPL,NVDA,TSLA
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd
from langchain_core.messages import HumanMessage

# Make sure the project root (parent of this scripts/ dir) is on
# sys.path, so `from backend.agent import ...` below resolves no
# matter how this script is invoked (`python scripts/evaluate_agent.py`
# from the project root, from inside scripts/, via `python -m`, etc.).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agent import agent_app, AgentState, META_CACHE
from backend.main import parse_signal


CORE_TOOLS = {"PriceTool", "SentimentTool", "QuantTool"}  # mandatory every run
OPTIONAL_TOOLS = {"RAGTool"}  # conditional -- see RAGTOOL RULE in SYSTEM_PROMPT

# Keywords that should only show up in the final reasoning if the
# corresponding tool actually ran successfully. Used by the
# hallucination check -- deliberately conservative (misses subtler
# hallucinations) rather than noisy with false positives.
TOOL_KEYWORDS = {
    "PriceTool": ["rsi", "macd", "bollinger", "moving average"],
    "SentimentTool": ["sentiment score", "bullish", "bearish"],
    "QuantTool": ["shap", "predicted return", "up_probability",
                  "model prediction", "confidence interval"],
    # NOTE: deliberately NOT using "headline" here. SentimentTool also
    # legitimately returns headlines (see its `sample_headlines` field),
    # so "headline" alone can't tell RAGTool's evidence apart from
    # SentimentTool's. Use RAGTool-specific citation phrasing instead --
    # this is narrower (may miss some real hallucinations) but avoids
    # flagging legitimate SentimentTool-grounded headline mentions.
    "RAGTool": ["semantic search", "retrieved article", "relevance score"],
}


def run_agent_once(ticker: str) -> dict:
    """Runs the graph for one ticker; returns raw data for scoring."""
    query = f"Analyze {ticker} and provide a detailed investment recommendation."
    state: AgentState = {
        "messages": [HumanMessage(content=f"Ticker: {ticker}. {query}")],
        "ticker": ticker,
        "iterations": 0,
    }

    tools_called = []
    tool_results = {}  # tool_name -> list of parsed result dicts
    final_response = ""

    start = time.perf_counter()
    for step in agent_app.stream(state):
        node = list(step.keys())[0]
        data = step[node]

        if node == "agent":
            msg = data["messages"][-1]
            if getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    tools_called.append(tc["name"])
            elif msg.content:
                final_response = msg.content

        elif node == "tools":
            for m in data["messages"]:
                try:
                    result = json.loads(m.content)
                except Exception:
                    result = {"raw": str(m.content)}
                tool_results.setdefault(getattr(m, "name", "unknown"), []).append(result)

        elif node == "final_fix":
            msg = data["messages"][-1]
            if getattr(msg, "content", None):
                final_response = msg.content

    latency = time.perf_counter() - start

    return {
        "ticker": ticker,
        "latency_sec": round(latency, 2),
        "tools_called": tools_called,
        "tool_results": tool_results,
        "final_response": final_response,
    }


def _successful_tools(run: dict) -> set:
    return {
        name for name, results in run["tool_results"].items()
        if any(r.get("success") for r in results)
    }


def score_tool_accuracy(run: dict) -> dict:
    successful = _successful_tools(run)
    core_coverage = len(CORE_TOOLS & successful) / len(CORE_TOOLS)
    return {
        "core_tool_accuracy": round(core_coverage, 2),
        "core_tools_missing_or_failed": sorted(CORE_TOOLS - successful),
    }


def score_rag_policy(run: dict) -> dict:
    """
    Independently re-derives the RAGTool rule from SYSTEM_PROMPT
    (call RAG if SentimentTool is NEUTRAL, or SentimentTool and
    QuantTool disagree on direction) using the tool results actually
    returned, then checks whether the agent's own behavior matched it.
    This is the automatable half of "did the agent follow its policy";
    the "not confident without it" clause in the prompt is judgment-based
    and isn't checked here.
    """
    successful = _successful_tools(run)
    rag_called = "RAGTool" in run["tools_called"]

    sentiment_signal = quant_signal = None
    for r in run["tool_results"].get("SentimentTool", []):
        if r.get("success"):
            sentiment_signal = r["data"].get("signal")
            break
    for r in run["tool_results"].get("QuantTool", []):
        if r.get("success"):
            quant_signal = r["data"].get("signal")
            break

    if sentiment_signal is None or quant_signal is None:
        return {"rag_expected": None, "rag_called": rag_called, "rag_policy_followed": None}

    bullish = {"BULLISH", "BUY"}
    bearish = {"BEARISH", "SELL"}
    disagreement = (
        (sentiment_signal in bullish and quant_signal in bearish)
        or (sentiment_signal in bearish and quant_signal in bullish)
    )
    rag_expected = sentiment_signal == "NEUTRAL" or disagreement

    # Not calling RAG when expected is a miss; calling it when not
    # strictly required is fine (the prompt also allows it for
    # low-confidence cases we can't check automatically).
    policy_followed = rag_called if rag_expected else True

    return {
        "rag_expected": rag_expected,
        "rag_called": rag_called,
        "rag_policy_followed": policy_followed,
    }


def score_groundedness(run: dict, signal: str) -> dict:
    """Does the final signal agree with QuantTool's own signal?"""
    quant_signal = None
    for r in run["tool_results"].get("QuantTool", []):
        if r.get("success"):
            quant_signal = r["data"].get("signal")
            break

    if quant_signal is None:
        return {"groundedness": None, "quant_signal": None}

    return {"groundedness": int(signal == quant_signal), "quant_signal": quant_signal}


def score_hallucination(run: dict) -> dict:
    """Flags reasoning that references a tool's evidence when that
    tool never actually returned successfully."""
    text = run["final_response"].lower()
    successful = _successful_tools(run)

    flags = []
    for tool_name, keywords in TOOL_KEYWORDS.items():
        if tool_name in successful:
            continue
        for kw in keywords:
            if kw in text:
                flags.append(f"mentions '{kw}' but {tool_name} never succeeded")
                break

    return {"hallucination_flag": len(flags) > 0, "hallucination_details": flags}


def score_run(run: dict) -> dict:
    """Turns one raw agent run into a scored record (no LLM calls here)."""
    signal = parse_signal(run["final_response"])

    record = {
        "ticker": run["ticker"],
        "signal": signal,
        "latency_sec": run["latency_sec"],
        "response_len": len(run["final_response"]),
        "tools_called": run["tools_called"],
    }
    record.update(score_tool_accuracy(run))
    record.update(score_rag_policy(run))
    record.update(score_groundedness(run, signal))
    record.update(score_hallucination(run))
    return record


def plot_summary(df: pd.DataFrame, out_dir: str, timestamp: str) -> str:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(df["ticker"], df["latency_sec"])
    axes[0].set_title("Response Latency by Ticker")
    axes[0].set_ylabel("seconds")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(df["ticker"], df["core_tool_accuracy"])
    axes[1].set_title("Core Tool Accuracy by Ticker\n(Price/Sentiment/Quant)")
    axes[1].set_ylabel("coverage (0-1)")
    axes[1].set_ylim(0, 1.1)
    axes[1].tick_params(axis="x", rotation=45)

    fig.tight_layout()
    chart_path = os.path.join(out_dir, f"evaluation_chart_{timestamp}.png")
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    return chart_path


def main():
    parser = argparse.ArgumentParser(description="Agent Evaluation Framework")
    parser.add_argument("--tickers", type=str, default=None,
                         help="Comma-separated tickers, e.g. AAPL,NVDA,TSLA")
    parser.add_argument("--n", type=int, default=5,
                         help="If --tickers not given, sample this many known tickers")
    parser.add_argument("--out-dir", type=str, default="data/models/eval")
    args = parser.parse_args()

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
    else:
        available = sorted(META_CACHE["ticker"].unique())
        tickers = available[:args.n]

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Evaluating agent on {len(tickers)} tickers: {tickers}")
    records = []
    raw_runs = []
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] running {ticker}...")
        try:
            run = run_agent_once(ticker)
            records.append(score_run(run))
            raw_runs.append(run)
        except Exception as e:
            print(f"  FAILED: {e}")
            records.append({"ticker": ticker, "error": str(e)})

    df = pd.DataFrame(records)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    csv_path = os.path.join(args.out_dir, f"evaluation_report_{timestamp}.csv")
    df.to_csv(csv_path, index=False)

    # Full raw runs (final_response text + every tool's raw output) --
    # not meant for the spreadsheet, meant for manually auditing *why*
    # a groundedness/hallucination score came out the way it did.
    raw_path = os.path.join(args.out_dir, f"evaluation_raw_{timestamp}.json")
    with open(raw_path, "w") as f:
        json.dump(raw_runs, f, indent=2)

    summary = {
        "n_tickers": len(df),
        "avg_latency_sec": round(df["latency_sec"].dropna().mean(), 2) if "latency_sec" in df else None,
        "avg_core_tool_accuracy": round(df["core_tool_accuracy"].dropna().mean(), 2) if "core_tool_accuracy" in df else None,
        "rag_called_rate": round(df["rag_called"].dropna().mean(), 2) if "rag_called" in df else None,
        "rag_policy_followed_rate": round(df["rag_policy_followed"].dropna().mean(), 2) if "rag_policy_followed" in df else None,
        "groundedness_rate": round(df["groundedness"].dropna().mean(), 2) if "groundedness" in df else None,
        "hallucination_rate": round(df["hallucination_flag"].dropna().mean(), 2) if "hallucination_flag" in df else None,
    }
    summary_path = os.path.join(args.out_dir, f"evaluation_summary_{timestamp}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\nFull report: {csv_path}")
    print(f"Raw runs:    {raw_path}")
    print(f"Summary:     {summary_path}")

    try:
        chart_path = plot_summary(df, args.out_dir, timestamp)
        print(f"Chart:       {chart_path}")
    except Exception as e:
        print(f"(skipped chart: {e})")


if __name__ == "__main__":
    main()
