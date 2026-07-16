"""
一次性脚本：抓取行业/宏观新闻，存成CSV，供agent.py建MACRO_COLLECTION使用
"""
import requests
import pandas as pd
import time
import os
from dotenv import load_dotenv

load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

SECTOR_QUERIES = {
    "semiconductor": ["semiconductor industry chip export restrictions", "AI chip demand supply"],
    "big_tech": ["big tech antitrust regulation", "tech sector AI investment"],
    "fintech": ["digital payments fintech regulation", "cryptocurrency market trends"],
    "banking": ["Federal Reserve interest rate decision", "banking sector regulation"],
    "cloud_saas": ["enterprise software spending trends", "cloud computing market growth"],
    "retail": ["consumer spending retail sales trends", "inflation impact on retail"],
    "healthcare": ["healthcare medical device regulation", "FDA approval trends"],
    "clean_energy": ["clean energy policy incentives", "electric vehicle market trends"],
    "consumer_travel": ["travel industry consumer spending trends", "hospitality sector outlook"],
}


def fetch_sector_news(query):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query, "language": "en", "sortBy": "publishedAt",
        "pageSize": 20, "apiKey": NEWSAPI_KEY
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json().get("articles", [])


all_macro_news = []
for sector, queries in SECTOR_QUERIES.items():
    for q in queries:
        try:
            articles = fetch_sector_news(q)
            for a in articles:
                all_macro_news.append({
                    "sector": sector,
                    "query": q,
                    "headline": a["title"],
                    "date": a["publishedAt"][:10],
                    "source": a["source"]["name"]
                })
            print(f"{sector} / '{q}': {len(articles)}条")
        except Exception as e:
            print(f"{sector} / '{q}' 出错: {e}")
        time.sleep(1)

df_macro = pd.DataFrame(all_macro_news).drop_duplicates(subset=["headline"])
os.makedirs("data/news", exist_ok=True)
df_macro.to_csv("data/news/macro_sector_news.csv", index=False)
print(f"\n总计: {len(df_macro)}条宏观/行业新闻")
