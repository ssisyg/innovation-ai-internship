"""
每日收盘后增量更新脚本
用途：拉取当天价格+新闻，计算当天特征，更新QuantTool/SentimentTool/RAGTool能读到的缓存
建议：美股收盘后（美东16:00，约北京时间次日04:00或05:00，视夏令时）用任务计划程序跑一次
"""

import pandas as pd
import numpy as np
import joblib
import os
import time
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import ta

load_dotenv()
TIINGO_API_KEY = os.getenv("TIINGO_API_KEY")

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "META", "AMZN", "NFLX", "TSLA", "ABNB",
    "NVDA", "AMD", "INTC", "QCOM", "AVGO", "MU", "ARM", "AMAT",
    "V", "MA", "PYPL", "COIN", "HOOD",
    "JPM", "GS", "BAC", "WFC", "MS", "BLK",
    "CRM", "NOW", "SNOW", "PLTR", "NET", "DDOG", "MDB",
    "WMT", "TGT", "COST", "EBAY", "SHOP",
    "UNH", "ISRG", "DXCM",
    "ENPH", "FSLR", "RIVN"
]

FEATURE_COLS = joblib.load("data/ml/feature_cols.pkl")
SCALER = joblib.load("data/ml/scaler.pkl")


# =========================
# 第一步：价格增量更新
# =========================
def update_price(ticker):
    """只拉最近几天的价格，追加到已有CSV，不重新拉全部历史"""
    path = f"data/raw/{ticker}_price.csv"
    df_old = pd.read_csv(path, index_col=0)
    df_old.index = pd.to_datetime(df_old.index)
    last_date = df_old.index.max()

    start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    end = datetime.today().strftime("%Y-%m-%d")

    if start > end:
        return df_old  # 今天已经更新过，不用再拉

    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
    params = {"startDate": start, "endDate": end, "token": TIINGO_API_KEY, "format": "json"}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    new = pd.DataFrame(resp.json())

    if len(new) == 0:
        return df_old  # 今天还没收盘/没有新数据

    out_new = pd.DataFrame({
        "Date": pd.to_datetime(new["date"]).dt.tz_localize(None),
        "Open": new["adjOpen"], "High": new["adjHigh"],
        "Low": new["adjLow"], "Close": new["adjClose"], "Volume": new["adjVolume"]
    }).set_index("Date")

    df_all = pd.concat([df_old, out_new])
    df_all = df_all[~df_all.index.duplicated(keep="last")].sort_index()
    df_all.to_csv(path)
    return df_all


# =========================
# 第二步：技术指标增量计算（只取最后一行，但要用最近70天算窗口指标）
# =========================
def compute_today_features(df_price, ticker):
    """跟Week3的add_features逻辑一致，但只需要最近的窗口数据，返回最后一行"""
    df = df_price.tail(70).copy()  # 70天足够覆盖SMA_50这类最长窗口的指标
    close = df["Close"]

    df["SMA_20"] = ta.trend.sma_indicator(close, window=20)
    df["SMA_50"] = ta.trend.sma_indicator(close, window=50)
    df["EMA_20"] = ta.trend.ema_indicator(close, window=20)
    df["RSI_14"] = ta.momentum.rsi(close, window=14)

    macd = ta.trend.MACD(close)
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_diff"] = macd.macd_diff()

    bb = ta.volatility.BollingerBands(close, window=20)
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_lower"] = bb.bollinger_lband()
    df["BB_width"] = bb.bollinger_wband()

    df["Volatility_20"] = close.pct_change().rolling(20).std()
    df["Return_1d"] = close.pct_change(1)
    df["Return_3d"] = close.pct_change(3)
    df["Return_5d"] = close.pct_change(5)

    today_row = df.iloc[[-1]].copy()  # 只要最后一行（今天）
    today_row["ticker"] = ticker
    return today_row


# =========================
# 第三步：今日新闻抓取 + FinBERT情感（复用Week5/6逻辑，改成只抓今天）
# =========================
def scrape_today_news(ticker):
    """跟Week5的scrape_finviz_news逻辑一致，这里真正过滤出'今天'的新闻"""
    import requests
    from bs4 import BeautifulSoup

    url = f"https://finviz.com/quote.ashx?t={ticker}"
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://finviz.com/"}
    resp = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(resp.text, "html.parser")
    news_table = soup.find("table", {"id": "news-table"})

    news = []
    if news_table:
        rows = news_table.find_all("tr")
        current_date = None
        today_str = datetime.today().strftime("%b-%d-%y")  # 例如 "Jul-14-26"

        for row in rows:  # 不再只看前15条，要扫描到超出今天为止
            cols = row.find_all("td")
            if len(cols) < 2:
                continue
            date_col = cols[0].text.strip()
            headline = cols[1].text.strip()

            if len(date_col) > 8:
                current_date = date_col.split(" ")[0]

            date_str = current_date
            if date_str == "Today":
                date_str = today_str

            # 关键修复：只保留日期真正等于今天的新闻
            if date_str == today_str:
                news.append({"ticker": ticker, "date": date_str, "headline": headline})
            elif current_date is not None and date_str != today_str and len(news) > 0:
                # 已经翻到比今天更早的日期、且之前已经收集过今天的新闻了，说明今天的部分扫完了，提前结束
                break

    return news



def analyze_today_sentiment(news_list, sentiment_pipeline):
    if not news_list:
        return {"sentiment_mean": 0.0, "sentiment_pos_ratio": 0.0,
                "sentiment_neg_ratio": 0.0, "news_count_log": 0.0}

    headlines = [n["headline"][:512] for n in news_list]
    preds = sentiment_pipeline(headlines, batch_size=16)
    numeric = [1 if p["label"] == "positive" else -1 if p["label"] == "negative" else 0 for p in preds]

    return {
        "sentiment_mean": float(np.mean(numeric)),
        "sentiment_pos_ratio": float(np.mean([p["label"] == "positive" for p in preds])),
        "sentiment_neg_ratio": float(np.mean([p["label"] == "negative" for p in preds])),
        "news_count_log": float(np.log1p(len(news_list)))
    }


# =========================
# 第四步：拼成完整的23维特征行，用已有scaler做标准化
# =========================
def build_today_row(price_features_row, sentiment_dict, sentiment_history):
    row = price_features_row.copy()
    for k, v in sentiment_dict.items():
        row[k] = v

    # sentiment_lag1/lag3/lag5 需要过去几天的情感历史（从缓存里读，见下方主流程）
    row["sentiment_lag1"] = sentiment_history.get("lag1", 0.0)
    row["sentiment_lag3"] = sentiment_history.get("lag3", 0.0)
    row["sentiment_lag5"] = sentiment_history.get("lag5", 0.0)

    X_today = row[FEATURE_COLS]
    X_today_scaled = pd.DataFrame(
        SCALER.transform(X_today.values.reshape(1, -1)),
        columns=FEATURE_COLS,
        index=row.index
    )
    return X_today_scaled


# =========================
# 主流程
# =========================
def run_daily_update(sentiment_pipeline):
    os.makedirs("data/daily_cache", exist_ok=True)
    all_today_rows = []

    for ticker in TICKERS:
        try:
            df_price = update_price(ticker)
            price_row = compute_today_features(df_price, ticker)

            news = scrape_today_news(ticker)
            sent = analyze_today_sentiment(news, sentiment_pipeline)

            # 简化：sentiment_lag先用今天的值填充，后续可以从昨天的缓存里读真实历史
            sentiment_history = {"lag1": sent["sentiment_mean"],
                                  "lag3": sent["sentiment_mean"],
                                  "lag5": sent["sentiment_mean"]}

            today_scaled = build_today_row(price_row, sent, sentiment_history)
            today_scaled["ticker"] = ticker
            all_today_rows.append(today_scaled)

            print(f"{ticker} 更新完成")
            time.sleep(2)
        except Exception as e:
            print(f"{ticker} 更新失败: {e}")

    df_cache = pd.concat(all_today_rows)
    df_cache.to_csv("data/daily_cache/today_features.csv")
    print(f"\n今日缓存已更新，覆盖 {len(df_cache)} 只股票")



if __name__ == "__main__":
    from transformers import pipeline
    sentiment_pipeline = pipeline("text-classification", model="ProsusAI/finbert", device=-1)
    run_daily_update(sentiment_pipeline)
