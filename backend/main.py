import os
import json
import re
import asyncio
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from backend.agent import agent_app, memory, AgentState
from langchain_core.messages import HumanMessage


app = FastAPI(title="AI Investment Agent API", version="1.0")


# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# ✅ FIX: API KEY 从 .env 读取，不硬编码在源码里
# 在 .env 里加一行: API_KEY=innovation-ai-2024
# =========================
API_KEY = os.getenv("API_KEY", "innovation-ai-2024")

def verify_api_key(x_api_key: str = Header(default=None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key


# =========================
# Request Models
# =========================
class AnalyzeRequest(BaseModel):
    ticker: str
    query: str = None


# =========================
# Signal 解析
# =========================
SIGNAL_RE = re.compile(r"Final\s*Signal\s*:\s*(BUY|SELL|HOLD)", re.IGNORECASE)

def parse_signal(text: str) -> str:
    if not text:
        return "UNKNOWN"
    match = SIGNAL_RE.search(text)
    if match:
        return match.group(1).upper()
    # 宽松兜底
    upper = text.upper()
    for word in ("BUY", "SELL", "HOLD"):
        if word in upper:
            return word
    return "UNKNOWN"


# =========================
# SSE STREAM
# =========================
async def stream_agent(ticker: str, query: str) -> AsyncGenerator[str, None]:

    if not query:
        query = f"Analyze {ticker} and provide a detailed investment recommendation."

    state: AgentState = {
        "messages": [HumanMessage(content=f"Ticker: {ticker}. {query}")],
        "ticker": ticker,
        "iterations": 0,
    }

    final_response = ""
    signal = "UNKNOWN"
    collected_tool_data = []

    try:
        for step in agent_app.stream(state):
            node = list(step.keys())[0]
            data = step[node]

            # ---------- agent ----------
            if node == "agent":
                msg = data["messages"][-1]

                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        yield f"data: {json.dumps({'type': 'tool_call', 'tool': tc['name'], 'args': tc['args']})}\n\n"

                elif msg.content:
                    final_response = msg.content
                    yield f"data: {json.dumps({'type': 'final', 'content': final_response})}\n\n"

            # ---------- tools ----------
            elif node == "tools":
                for m in data["messages"]:
                    try:
                        result = json.loads(m.content)
                    except Exception:
                        result = {"raw": str(m.content)}

                    tool_name = getattr(m, "name", "unknown")
                    collected_tool_data.append({"tool": tool_name, "result": result})

                    yield f"data: {json.dumps({'type': 'tool_result', 'tool': tool_name, 'result': result})}\n\n"

            # ---------- final_fix ----------
            elif node == "final_fix":
                # ensure_final 可能修改了内容，重新读一次
                msg = data["messages"][-1]
                if hasattr(msg, "content") and msg.content:
                    final_response = msg.content

            await asyncio.sleep(0.01)

        # ---------- 解析信号 & 落库 ----------
        if final_response:
            signal = parse_signal(final_response)
            try:
                memory.save(
                    ticker=ticker,
                    signal=signal,
                    reasoning=final_response,
                    tool_data=collected_tool_data,
                )
            except Exception as save_err:
                print(f"[memory.save error] {save_err}")

        yield f"data: {json.dumps({'type': 'done', 'signal': signal})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


# =========================
# Routes
# =========================
@app.get("/")
def root():
    return {"status": "ok", "service": "AI Investment Agent"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/analyze")
async def analyze(req: AnalyzeRequest, api_key: str = Depends(verify_api_key)):
    return StreamingResponse(
        stream_agent(req.ticker, req.query),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.get("/history")
def get_history(
    ticker: str = None,
    limit: int = 10,
    api_key: str = Depends(verify_api_key),
):
    records = memory.get_history(ticker=ticker, limit=limit)
    return {"records": records, "count": len(records)}


@app.get("/performance")
def get_performance(api_key: str = Depends(verify_api_key)):
    import pandas as pd

    out = {}
    if os.path.exists("data/models/model_comparison.csv"):
        out["models"] = pd.read_csv("data/models/model_comparison.csv").to_dict("records")
    if os.path.exists("data/backtest/backtest_results.csv"):
        out["backtest"] = pd.read_csv("data/backtest/backtest_results.csv").to_dict("records")
    return out