# Harness Engineering & Loop Engineering

*(This section can be pasted into the final report or README as-is,
or trimmed down. Written to be dropped under an "Architecture" or
"Engineering Practices" heading.)*

Beyond the model-and-prompt layer, this project applies two practices
from the current (2026) agent-engineering literature: **harness
engineering** — the scaffolding that turns a raw LLM into a reliable
agent — and **loop engineering** — the control layer that decides when
and how often that agent runs at all.

## Harness Engineering

The harness is everything that wraps the model: the tool interface it
can act through, the context it's given, how failures are caught, and
how the whole thing is observed and evaluated. This project's harness
has six concrete pieces:

| Harness component | Implementation |
|---|---|
| Tool surface | `PriceTool`, `SentimentTool`, `QuantTool`, `RAGTool` — each a `@tool`-decorated function with a typed input/output schema |
| Context management | `SYSTEM_PROMPT` + `MemoryStore.format_for_prompt()`, which injects prior recommendations for the same ticker into the first LLM call of each run |
| Execution environment | FastAPI + SSE streaming (`backend/main.py`), containerized via a multi-stage Dockerfile |
| Feedback / correction | `ensure_final()` — a graph node that guarantees a parseable `Final Signal:` line even if the model drifts from the required format; a conditional error-fallback edge if a tool call errors |
| Observability | LangSmith tracing across every node transition (Thought → Action → Observation) |
| Evaluation criteria | `scripts/evaluate_agent.py` — Tool Calling Accuracy, Response Latency, Groundedness, and a Hallucination Check, run against the live agent rather than assumed from unit tests alone |

The evaluation framework specifically exists because a harness that
isn't measured is just architecture diagrams — it's what turns
"the agent has tools" into a number you can track across changes to
the prompt or the model.

## Loop Engineering

The LangGraph `StateGraph` (`agent_node` → `should_continue` →
`tools` → back to `agent_node`, capped at 5 iterations) is the
**inner** loop: it drives one ticker analysis to completion within a
single request.

`scripts/watch_loop.py` adds an **outer** loop on top of it: a
scheduler-driven cycle that decides *whether a ticker is even worth
re-analyzing right now*, rather than waiting for a person to click
"analyze." It follows the pattern most 2026 loop-engineering write-ups
converge on — a loop is "cron plus a decision-maker," not a fixed
schedule that blindly re-runs everything:

- **Budget caps** (`--budget-calls`, `--max-cycles`) bound how many
  LLM calls one invocation can make, so a bad watchlist or a bug can't
  run away unattended.
- **No-progress detection**: a ticker whose signal hasn't changed
  since the last cycle is still re-checked (state is updated) but
  isn't re-surfaced as an alert — otherwise every cycle "agrees with
  itself" and the alert log becomes noise nobody reads.
- **A verification gate on what's alert-worthy**: only a signal
  *change*, or a HIGH-confidence actionable BUY/SELL, gets written to
  `watch_alerts.jsonl`. Everything else is logged to state quietly.

In production this is meant to run as `watch_loop.py --once`,
triggered by an OS scheduler (cron / Windows Task Scheduler) on a
fixed cadence — an in-process `--daemon` mode with a sleep interval is
included only for live demos where setting up an external scheduler
isn't practical.

## Why this framing, not just "we built an agent"

Prompt engineering is about the words sent to the model; context
engineering is about what it can see; harness engineering is about
the environment it runs in; loop engineering is about the cycle that
decides when it runs at all. Each layer wraps the previous one. This
project touches all four, but the harness and loop layers are the
ones most internship-style agent projects skip — most stop at
"the agent has tools and a prompt." Naming them explicitly is what
separates "I called an LLM in a while loop" from "I designed the
scaffolding and control policy around it," which is the actual
differentiator for AIE/MLE roles.
