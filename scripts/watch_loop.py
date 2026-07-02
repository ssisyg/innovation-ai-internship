"""
Loop Engineering layer on top of the ReAct agent harness.

backend/agent.py's LangGraph StateGraph is the *inner* loop: observe a
tool result, decide the next action, repeat -- capped at 5 iterations,
scoped to one ticker, one request. This script is the *outer* loop: it
decides WHEN and HOW OFTEN to run that inner loop at all, without a
person clicking "analyze" for each ticker each time.

Follows the pattern most 2026 loop-engineering write-ups converge on
(Osmani/Steinberger): a loop is "cron plus a decision-maker" -- an
external scheduler ticks it, and each tick the loop decides what's
worth doing, not a hardcoded branch. Concretely, this script has:

  - Budget / iteration caps (--budget-calls, --max-cycles): a bug or a
    bad watchlist can't rack up unbounded LLM spend unattended.
  - No-progress detection: a ticker whose signal hasn't changed since
    last cycle is still re-checked, but not re-surfaced as an alert --
    otherwise every cycle "agrees with itself" and alerts become noise
    nobody reads.
  - A verification gate on what counts as alert-worthy: only a signal
    *change*, or a HIGH-confidence actionable BUY/SELL, gets surfaced.
    Everything else is logged to state but stays quiet.

Real-world usage: run ONE cycle per invocation, triggered by an OS
scheduler (cron / Windows Task Scheduler), not an infinite Python
process:

    python scripts/watch_loop.py --once
    # cron: 0 9 * * 1-5 cd /path/to/repo && venv/bin/python scripts/watch_loop.py --once

For a live demo without setting up an external scheduler, --daemon
loops in-process with a sleep between cycles instead:

    python scripts/watch_loop.py --daemon --max-cycles 3 --interval-min 60

State (last-seen signal per ticker) persists to
data/models/eval/watch_state.json between invocations, so "did this
change" detection works across separate --once runs, not just within
one --daemon process. Every alert is also appended to
data/models/eval/watch_alerts.jsonl as an append-only audit log.

Like scripts/evaluate_agent.py, this calls the real LLM once per
ticker per cycle -- --budget-calls exists specifically to bound that.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

# Make sure the project root is on sys.path regardless of how/where
# this script is invoked from (see the same comment in evaluate_agent.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.main import parse_signal
from scripts.evaluate_agent import run_agent_once


DEFAULT_WATCHLIST = ["AAPL", "NVDA", "TSLA", "AMD", "MSFT"]
STATE_PATH = "data/models/eval/watch_state.json"
ALERT_LOG_PATH = "data/models/eval/watch_alerts.jsonl"


def load_state() -> dict:
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {}


def save_state(state: dict) -> None:
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def append_alert(alert: dict) -> None:
    os.makedirs(os.path.dirname(ALERT_LOG_PATH), exist_ok=True)
    with open(ALERT_LOG_PATH, "a") as f:
        f.write(json.dumps(alert) + "\n")


def extract_confidence(text: str) -> str:
    upper = text.upper()
    for level in ("HIGH", "MEDIUM", "LOW"):
        if f"CONFIDENCE: {level}" in upper:
            return level
    return "UNKNOWN"


def run_cycle(watchlist: list, state: dict, budget_calls: int) -> tuple:
    """
    Runs the agent once per ticker (bounded by budget_calls), decides
    what's alert-worthy against the previous cycle's state, and
    returns (updated_state, alerts_raised, calls_made).
    """
    alerts = []
    calls_made = 0

    for ticker in watchlist:
        if calls_made >= budget_calls:
            print(f"  budget of {budget_calls} calls hit this cycle, "
                  f"skipping remaining {len(watchlist) - calls_made} ticker(s)")
            break

        print(f"  checking {ticker}...")
        run = run_agent_once(ticker)
        calls_made += 1

        signal = parse_signal(run["final_response"])
        confidence = extract_confidence(run["final_response"])
        prev_signal = state.get(ticker, {}).get("signal")

        changed = prev_signal is not None and prev_signal != signal
        actionable = signal in ("BUY", "SELL") and confidence == "HIGH"
        noteworthy = changed or actionable

        record = {
            "ticker": ticker,
            "signal": signal,
            "confidence": confidence,
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }
        state[ticker] = record

        if noteworthy:
            alert = {
                **record,
                "reason": "signal_changed" if changed else "high_confidence_actionable",
                "previous_signal": prev_signal,
            }
            alerts.append(alert)
            append_alert(alert)
            print(f"    ALERT: {ticker} -> {signal} ({confidence}) [{alert['reason']}]")
        else:
            print(f"    {ticker} -> {signal} ({confidence}), nothing new")

    return state, alerts, calls_made


def main():
    parser = argparse.ArgumentParser(description="Loop Engineering: autonomous watchlist scanner")
    parser.add_argument("--watchlist", type=str, default=None,
                         help="Comma-separated tickers; defaults to a small built-in list")
    parser.add_argument("--budget-calls", type=int, default=20,
                         help="Max agent invocations allowed in one cycle")
    parser.add_argument("--once", action="store_true",
                         help="Run exactly one cycle and exit (default; use this from "
                              "cron/Task Scheduler)")
    parser.add_argument("--daemon", action="store_true",
                         help="Loop in-process with a sleep between cycles, for demos. "
                              "Prefer --once + an OS scheduler for real deployments.")
    parser.add_argument("--interval-min", type=int, default=60,
                         help="Minutes between cycles when --daemon is set")
    parser.add_argument("--max-cycles", type=int, default=3,
                         help="How many cycles to run when --daemon is set")
    args = parser.parse_args()

    watchlist = (
        [t.strip().upper() for t in args.watchlist.split(",")]
        if args.watchlist else DEFAULT_WATCHLIST
    )

    state = load_state()
    cycles = args.max_cycles if args.daemon else 1
    no_progress_streak = 0

    for cycle_num in range(1, cycles + 1):
        print(f"\n=== Cycle {cycle_num}/{cycles} — {datetime.now(timezone.utc).isoformat()} ===")
        state, alerts, calls_made = run_cycle(watchlist, state, args.budget_calls)
        save_state(state)

        no_progress_streak = 0 if alerts else no_progress_streak + 1
        print(f"  cycle used {calls_made}/{args.budget_calls} of the call budget, "
              f"{len(alerts)} alert(s) raised"
              + (f", no-progress streak: {no_progress_streak}" if not alerts else ""))

        if args.daemon and cycle_num < cycles:
            print(f"  sleeping {args.interval_min} min until next cycle...")
            time.sleep(args.interval_min * 60)

    print(f"\nDone. State: {STATE_PATH}  Alert log: {ALERT_LOG_PATH}")


if __name__ == "__main__":
    main()
