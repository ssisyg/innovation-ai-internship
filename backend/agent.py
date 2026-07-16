import os
import re
import json
import sqlite3
import time
import joblib
import operator

import pandas as pd
import numpy as np
import yfinance as yf
import ta

from datetime import datetime, timezone
from typing import TypedDict, Annotated, Sequence, Optional
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

import chromadb
from sentence_transformers import SentenceTransformer


# =========================
# 全局加载
# =========================
BEST_MODEL   = joblib.load("data/models/best_model.pkl")
BEST_NAME    = joblib.load("data/models/best_model_name.pkl")
SHAP_IMP     = pd.read_csv("data/models/shap/shap_importance.csv")
SCHEMA       = joblib.load("data/ml/schema.pkl")
SCALER       = joblib.load("data/ml/scaler.pkl")
SENTIMENT_DF = pd.read_csv("data/sentiment/news_with_sentiment.csv")
SIMILAR_CASE_INDEX = joblib.load("data/models/similar_case_index.pkl")
SIMILAR_CASE_INFO  = pd.read_csv("data/models/similar_case_info.csv", parse_dates=["date"])


# Week 8 QuantTool 用的预计算特征缓存（已经过 scaler，直接送模型）
X_TEST_CACHE = pd.read_csv("data/ml/X_test.csv",   index_col=0, parse_dates=True)
META_CACHE   = pd.read_csv("data/ml/test_meta.csv", index_col=0, parse_dates=True)
print(f"X_TEST_CACHE: {X_TEST_CACHE.shape}, tickers: {META_CACHE['ticker'].nunique()}")

print("Loading Sentence Transformer...")
EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")

# ticker -> sector 映射，供 RAGTool 检索行业/宏观新闻使用
SECTOR_MAP = {
    "NVDA": "semiconductor", "AMD": "semiconductor", "INTC": "semiconductor",
    "QCOM": "semiconductor", "AVGO": "semiconductor", "MU": "semiconductor",
    "ARM": "semiconductor", "AMAT": "semiconductor",
    "AAPL": "big_tech", "MSFT": "big_tech", "GOOGL": "big_tech",
    "META": "big_tech", "AMZN": "big_tech", "NFLX": "big_tech",
    "V": "fintech", "MA": "fintech", "PYPL": "fintech", "COIN": "fintech", "HOOD": "fintech",
    "JPM": "banking", "GS": "banking", "BAC": "banking", "WFC": "banking",
    "MS": "banking", "BLK": "banking",
    "CRM": "cloud_saas", "NOW": "cloud_saas", "SNOW": "cloud_saas",
    "PLTR": "cloud_saas", "NET": "cloud_saas", "DDOG": "cloud_saas", "MDB": "cloud_saas",
    "WMT": "retail", "TGT": "retail", "COST": "retail", "EBAY": "retail", "SHOP": "retail",
    "UNH": "healthcare", "ISRG": "healthcare", "DXCM": "healthcare",
    "ENPH": "clean_energy", "FSLR": "clean_energy", "RIVN": "clean_energy",
    "TSLA": "clean_energy", "ABNB": "consumer_travel",
}

os.makedirs("data/rag", exist_ok=True)
CHROMA_CLIENT = chromadb.PersistentClient(path="data/rag/chroma_db")

try:
    COLLECTION = CHROMA_CLIENT.get_collection("news_headlines")
    print(f"ChromaDB loaded ({COLLECTION.count()} docs)")
except Exception:
    COLLECTION = CHROMA_CLIENT.create_collection("news_headlines")
    df_news    = SENTIMENT_DF.dropna(subset=["headline"])
    headlines  = df_news["headline"].tolist()

    embeddings = []
    for i in range(0, len(headlines), 64):
        emb = EMBEDDER.encode(headlines[i:i+64], show_progress_bar=False)
        embeddings.extend(emb.tolist())

    COLLECTION.add(
        ids=[f"doc_{i}" for i in range(len(df_news))],
        documents=headlines,
        embeddings=embeddings,
        metadatas=[{"ticker": t, "sentiment": s}
                   for t, s in zip(df_news["ticker"], df_news["sentiment_label"])]
    )
    print(f"ChromaDB built ({len(df_news)} docs)")


# =========================
# 宏观/行业新闻 Collection（独立于按ticker的新闻，用于RAGTool补充信息增量）
# =========================
try:
    MACRO_COLLECTION = CHROMA_CLIENT.get_collection("macro_sector_news")
    print(f"MacroChromaDB loaded ({MACRO_COLLECTION.count()} docs)")
except Exception:
    macro_csv_path = "data/news/macro_sector_news.csv"
    if os.path.exists(macro_csv_path):
        MACRO_COLLECTION = CHROMA_CLIENT.create_collection("macro_sector_news")
        df_macro = pd.read_csv(macro_csv_path).dropna(subset=["headline"])
        macro_headlines = df_macro["headline"].tolist()

        macro_embeddings = []
        for i in range(0, len(macro_headlines), 64):
            emb = EMBEDDER.encode(macro_headlines[i:i+64], show_progress_bar=False)
            macro_embeddings.extend(emb.tolist())

        MACRO_COLLECTION.add(
            ids=[f"macro_{i}" for i in range(len(df_macro))],
            documents=macro_headlines,
            embeddings=macro_embeddings,
            metadatas=[{"sector": s, "date": d, "source": src}
                       for s, d, src in zip(df_macro["sector"], df_macro["date"], df_macro["source"])]
        )
        print(f"MacroChromaDB built ({len(df_macro)} docs)")
    else:
        MACRO_COLLECTION = None
        print("警告: 未找到 data/news/macro_sector_news.csv，请先运行 scripts/build_macro_news.py")


# =========================
# LLM
# =========================
llm = ChatOpenAI(
    model="gpt-5.5",
    temperature=0,
    max_completion_tokens=1000,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)


# =========================
# Tools
# =========================
def _load_local_ohlcv(ticker: str, min_rows: int = 60) -> pd.DataFrame | None:
    """
    优先从本地 data/raw/{ticker}_price.csv 加载 OHLCV 数据。
    文件不存在或行数不足时返回 None。
    """
    path = f"data/raw/{ticker}_price.csv"
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, header=[0, 1], index_col=0)
        df.index = pd.to_datetime(df.index)
        df.columns = df.columns.get_level_values(0)   # 拍平多级列名
        df = df.dropna().sort_index()
        if len(df) < min_rows:
            return None
        return df
    except Exception:
        return None


def _fetch_yfinance(ticker: str, period: str = "6mo",
                    min_rows: int = 60, max_attempts: int = 3) -> pd.DataFrame | None:
    """带指数退避的 yfinance 拉取，作为本地数据的备用来源。"""
    for attempt in range(max_attempts):
        try:
            hist = yf.Ticker(ticker).history(period=period, auto_adjust=True)
            if len(hist) >= min_rows:
                return hist
        except Exception:
            pass
        time.sleep(2 ** (attempt + 1))   # 2 → 4 → 8 秒
    return None


@tool
def PriceTool(ticker: str) -> dict:
    """Get stock price, RSI, MACD and technical indicators for a ticker."""
    try:
        # ① 优先读本地 CSV（Week 1-2 已下载的历史数据）
        df = _load_local_ohlcv(ticker, min_rows=30)
        source = "local_csv"

        # ② 本地没有再尝试 yfinance
        if df is None:
            hist = _fetch_yfinance(ticker, period="6mo", min_rows=30)
            if hist is None or len(hist) < 30:
                return {"success": False, "error": "Price data unavailable (local & yfinance both failed)"}
            df = hist
            source = "yfinance"

        close = df["Close"]
        rsi   = float(ta.momentum.rsi(close, window=14).iloc[-1])
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd  = float((ema12 - ema26).iloc[-1])

        # 取最后一个交易日的日期，标注在结果里让 LLM 知道数据截止时间
        last_date = str(df.index[-1])[:10]

        return {
            "success": True,
            "data": {
                "ticker": ticker,
                "current_price": round(float(close.iloc[-1]), 2),
                "rsi": round(rsi, 2),
                "macd": round(macd, 4),
                "rsi_signal": ("OVERBOUGHT" if rsi > 70
                               else "OVERSOLD" if rsi < 30
                               else "NEUTRAL"),
                "data_source": source,
                "data_as_of": last_date,
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
def SentimentTool(ticker: str) -> dict:
    """Get news sentiment analysis for a stock ticker."""
    try:
        df_t = SENTIMENT_DF[SENTIMENT_DF["ticker"] == ticker]
        if len(df_t) == 0:
            return {"success": False, "error": "No sentiment data"}

        mean_score = float(df_t["sentiment_numeric"].mean())

        return {
            "success": True,
            "data": {
                "ticker": ticker,
                "avg_sentiment": round(mean_score, 3),
                "signal": "BULLISH" if mean_score > 0.1 else "BEARISH" if mean_score < -0.1 else "NEUTRAL",
                "sample_headlines": df_t["headline"].head(3).tolist()
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
def QuantTool(ticker: str) -> dict:
    """Run ML model prediction for next-day stock movement using pre-computed or live-updated features."""
    try:
        X_input = None
        last_date = None
        data_mode = None

        # ① 优先读每日增量更新的缓存（daily_update.py 产出）
        daily_path = "data/daily_cache/today_features.csv"
        print(f"[DEBUG] 检查daily_path是否存在: {os.path.exists(daily_path)}")
        print(f"[DEBUG] daily_path绝对路径: {os.path.abspath(daily_path)}")

        if os.path.exists(daily_path):
            df_daily = pd.read_csv(daily_path, index_col=0)
            print(f"[DEBUG] daily缓存文件共有 {len(df_daily)} 行，ticker列表: {sorted(df_daily['ticker'].unique())[:10]}...")

            row = df_daily[df_daily["ticker"] == ticker]
            print(f"[DEBUG] 筛选 {ticker} 后找到 {len(row)} 行")

            if len(row) > 0:
                X_input = row[X_TEST_CACHE.columns].tail(1)
                last_date = "today (live)"
                data_mode = "daily_cache"
                print(f"[DEBUG] 成功使用daily_cache数据, X_input shape: {X_input.shape}")

        # ② 没有今日缓存（或没找到这只ticker），退回原来的测试集兜底
        if X_input is None:
            print(f"[DEBUG] 走兜底逻辑：precomputed_test_set")
            mask = META_CACHE["ticker"].values == ticker
            if mask.sum() == 0:
                return {
                    "success": False,
                    "error": f"No data available for {ticker} (neither daily cache nor test set). "
                             f"Available: {sorted(META_CACHE['ticker'].unique())[:10]}..."
                }

            row_indices = mask.nonzero()[0]
            X_input = X_TEST_CACHE.iloc[row_indices].tail(1)
            last_date = str(X_input.index[-1])[:10]
            data_mode = "precomputed_test_set"

        pred = int(BEST_MODEL.predict(X_input)[0])
        prob = float(BEST_MODEL.predict_proba(X_input)[0][1])

        top_shap = SHAP_IMP.head(5)[["feature", "mean_abs_shap"]].to_dict("records")

        return {
            "success": True,
            "data": {
                "ticker": ticker,
                "model": BEST_NAME,
                "prediction": "UP" if pred == 1 else "DOWN",
                "up_probability": round(prob, 3),
                "confidence_interval": [
                    round(max(0.0, prob - 0.1), 3),
                    round(min(1.0, prob + 0.1), 3),
                ],
                "signal": "BUY" if prob > 0.55 else "SELL" if prob < 0.45 else "HOLD",
                "top_shap_features": top_shap,
                "data_as_of": last_date,
                "data_mode": data_mode,
            }
        }

    except Exception as e:
        print(f"[DEBUG] QuantTool异常: {e}")
        return {"success": False, "error": str(e)}




@tool
def RAGTool(query: str, ticker: str = None) -> dict:
    """Retrieve relevant financial news (company-specific) and sector/macro context using semantic search."""
    try:
        emb = EMBEDDER.encode([query]).tolist()

        # ① 按ticker检索的公司新闻（原有逻辑）
        company_results = COLLECTION.query(
            query_embeddings=emb,
            n_results=5,
            where={"ticker": ticker} if ticker else None
        )

        out = []
        for doc, meta, dist in zip(
            company_results["documents"][0],
            company_results["metadatas"][0],
            company_results["distances"][0]
        ):
            out.append({
                "headline": doc,
                "ticker": meta["ticker"],
                "sentiment": meta["sentiment"],
                "relevance": round(1 - dist, 3),
                "source_type": "company_news"
            })

        # ② 按行业检索的宏观新闻（新增，跟公司新闻数据源不重叠）
        sector = SECTOR_MAP.get(ticker)
        if sector and MACRO_COLLECTION is not None:
            macro_results = MACRO_COLLECTION.query(
                query_embeddings=emb,
                n_results=3,
                where={"sector": sector}
            )
            for doc, meta, dist in zip(
                macro_results["documents"][0],
                macro_results["metadatas"][0],
                macro_results["distances"][0]
            ):
                out.append({
                    "headline": doc,
                    "sector": meta["sector"],
                    "source": meta.get("source", ""),
                    "relevance": round(1 - dist, 3),
                    "source_type": "sector_macro"
                })

        return {"success": True, "data": {"query": query, "ticker_sector": sector, "results": out}}

    except Exception as e:
        return {"success": False, "error": str(e)}
    

@tool
def SimilarCaseTool(ticker: str) -> dict:
    """Find historically similar market situations (based on technical + sentiment feature similarity) and how they resolved next-day."""
    try:
        # 复用跟QuantTool一样的"取当前特征"逻辑：优先今日缓存，没有则退回测试集
        X_input = None
        daily_path = "data/daily_cache/today_features.csv"
        if os.path.exists(daily_path):
            df_daily = pd.read_csv(daily_path, index_col=0)
            row = df_daily[df_daily["ticker"] == ticker]
            if len(row) > 0:
                X_input = row[X_TEST_CACHE.columns].tail(1)

        if X_input is None:
            mask = META_CACHE["ticker"].values == ticker
            if mask.sum() == 0:
                return {"success": False, "error": f"No feature data available for {ticker}"}
            row_indices = mask.nonzero()[0]
            X_input = X_TEST_CACHE.iloc[row_indices].tail(1)

        # 在训练集里找20个最相近的历史样本
        distances, indices = SIMILAR_CASE_INDEX.kneighbors(X_input.values, n_neighbors=20)

        similar_cases = SIMILAR_CASE_INFO.iloc[indices[0]].copy()
        similar_cases["distance"] = distances[0]

        up_rate = float(similar_cases["target"].mean())
        # 挑几个具体案例作为可引用的证据
        examples = similar_cases.sort_values("distance").head(5)[["ticker", "date", "target"]].copy()
        examples["date"] = examples["date"].astype(str).str[:10]
        examples["outcome"] = examples["target"].map({1: "UP", 0: "DOWN"})

        return {
            "success": True,
            "data": {
                "ticker": ticker,
                "n_similar_cases": len(similar_cases),
                "historical_up_rate": round(up_rate, 3),
                "example_cases": examples[["ticker", "date", "outcome"]].to_dict("records"),
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}



# =========================
# Memory
# =========================
class MemoryStore:
    def __init__(self, db_path="data/memory/agent_memory.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            ticker TEXT,
            signal TEXT,
            reasoning TEXT,
            tool_data TEXT
        )""")
        conn.commit()
        conn.close()

    # ✅ FIX Bug3: format_for_prompt 真正读取历史记录
    def format_for_prompt(self, ticker: str = None, limit: int = 5) -> str:
        rows = self.get_history(ticker=ticker, limit=limit)
        if not rows:
            return "No previous recommendations on record."
        lines = []
        for r in rows:
            ts = r["timestamp"][:10]  # 只取日期部分 YYYY-MM-DD
            lines.append(f"  - [{ts}] {r['ticker']} → {r['signal']}: {r['reasoning'][:120]}...")
        header = f"Past recommendations for {ticker}:" if ticker else "Recent recommendations:"
        return header + "\n" + "\n".join(lines)

    def save(self, ticker: str, signal: str, reasoning: str, tool_data: dict = None):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """INSERT INTO recommendations (timestamp, ticker, signal, reasoning, tool_data)
               VALUES (?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                ticker,
                signal,
                reasoning,
                json.dumps(tool_data) if tool_data is not None else None,
            ),
        )
        conn.commit()
        conn.close()

    def get_history(self, ticker: str = None, limit: int = 10):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        if ticker:
            cur.execute(
                "SELECT * FROM recommendations WHERE ticker = ? ORDER BY id DESC LIMIT ?",
                (ticker, limit),
            )
        else:
            cur.execute(
                "SELECT * FROM recommendations ORDER BY id DESC LIMIT ?",
                (limit,),
            )
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows


memory = MemoryStore()


# =========================
# SYSTEM PROMPT
# =========================
SYSTEM_PROMPT = """
You are an AI investment research assistant.

TOOLS AVAILABLE:
- PriceTool    → current price, RSI, MACD, technical indicators
- SentimentTool → news sentiment scores (FinBERT)
- QuantTool    → ML model next-day prediction + SHAP explanation
- RAGTool      → semantic search over financial news (conditional, see below)
- SimilarCaseTool → finds historically similar market situations (by feature similarity) and their actual next-day outcome (conditional, see below)

WORKFLOW: Always call Price → Sentiment → Quant, in that order. Then
decide whether to call RAGTool and/or SimilarCaseTool using the rules
below before giving your final answer.

RAGTOOL RULE (evaluate this yourself from the tool results you already
have — do not guess):
- Call RAGTool if SentimentTool's `signal` is "NEUTRAL", OR
- Call RAGTool if SentimentTool's `signal` and QuantTool's `signal`
  point in different directions (e.g. one is bullish/BUY and the
  other is bearish/SELL), OR
- Call RAGTool if you are not confident enough to state HIGH or
  MEDIUM confidence in your final answer without it.
- Otherwise RAGTool is optional — skip it and go straight to your
  final answer to keep latency down.
When you skip RAGTool, do not claim to have reviewed specific news
articles in your reasoning.

SIMILARCASETOOL RULE:
- Call SimilarCaseTool if QuantTool's `up_probability` is between 0.45
  and 0.60 (i.e. the model's conviction is weak), to check whether
  historical precedent supports or contradicts the model's prediction.
- Otherwise it is optional — skip it if QuantTool's signal is already
  strongly one-directional.
- When you do call it, compare `historical_up_rate` against QuantTool's
  `up_probability`: if they agree, mention this as corroborating
  evidence; if they diverge meaningfully, flag this as a reason for
  lower confidence.

CRITICAL OUTPUT FORMAT (MANDATORY — NO EXCEPTIONS):
Your final answer MUST start with exactly one of these lines, with NOTHING before it:

Final Signal: BUY
Final Signal: SELL
Final Signal: HOLD

Then immediately follow with:

Confidence: HIGH / MEDIUM / LOW
Reasoning:
- key point 1
- key point 2
- key point 3
Risk Factors:
- risk 1
- risk 2

Rules:
- Do NOT use "Bullish"/"Bearish"/"Moderate" instead of BUY/SELL/HOLD.
- Do NOT put any text before "Final Signal:".
- If ALL tools fail, still output "Final Signal: HOLD" with low confidence.
- You MAY add additional analysis AFTER the required block.
"""


# =========================
# State
# =========================
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    ticker: str
    iterations: int


tools_list = [PriceTool, SentimentTool, QuantTool, RAGTool, SimilarCaseTool]
llm_with_tools = llm.bind_tools(tools_list)


# =========================
# Agent Node
# ✅ FIX Bug1: SystemMessage 每轮都拼入本地 messages，
#              返回值里不含 SystemMessage，避免重复累积到 state
# ✅ FIX Bug3: 第一轮把 memory history 也拼进去
# =========================
def agent_node(state: AgentState):
    iterations = state.get("iterations", 0)
    ticker     = state.get("ticker", "")

    # 构建本次 invoke 的消息列表（不存回 state，只用于这一次调用）
    if iterations == 0:
        # 第一轮：system + memory history + 用户消息
        history_context = memory.format_for_prompt(ticker=ticker or None)
        system_with_memory = SYSTEM_PROMPT.rstrip() + f"\n\n{history_context}"
        invoke_messages = (
            [SystemMessage(content=system_with_memory)]
            + list(state["messages"])
        )
    else:
        # 后续轮（工具返回后）：system + 完整对话历史（含工具结果）
        invoke_messages = (
            [SystemMessage(content=SYSTEM_PROMPT)]
            + list(state["messages"])
        )

    try:
        response = llm_with_tools.invoke(invoke_messages)
    except Exception as e:
        response = AIMessage(content=f"""Final Signal: HOLD
Confidence: LOW
Reasoning:
- LLM call failed: {str(e)}
Risk Factors:
- API error, treat output with caution
""")

    return {
        "messages": [response],
        "iterations": iterations + 1,
    }


def should_continue(state: AgentState):
    last = state["messages"][-1]

    if state["iterations"] >= 5:   # 给多一轮缓冲
        return "end"

    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"

    return "end"


# ✅ FIX Bug1 保险层：确保最终输出一定含 Final Signal 格式
SIGNAL_RE = re.compile(r"Final\s*Signal\s*:\s*(BUY|SELL|HOLD)", re.IGNORECASE)

def ensure_final(state: AgentState):
    last = state["messages"][-1]
    if isinstance(last, AIMessage):
        content = last.content or ""
        if not SIGNAL_RE.search(content):
            # 格式不对，追加一个兜底块
            last.content = content + (
                "\n\nFinal Signal: HOLD\n"
                "Confidence: LOW\n"
                "Reasoning:\n"
                "- Model output did not follow required format\n"
                "Risk Factors:\n"
                "- Format fallback triggered; verify data manually\n"
            )
    return state


# =========================
# Graph
# =========================
tool_node = ToolNode(tools_list)

graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_node("final_fix", ensure_final)

graph.set_entry_point("agent")

graph.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", "end": "final_fix"}
)

graph.add_edge("tools", "agent")
graph.add_edge("final_fix", END)

agent_app = graph.compile()

print("Agent ready.")
print("Memory methods:", [m for m in dir(memory) if not m.startswith("_")])