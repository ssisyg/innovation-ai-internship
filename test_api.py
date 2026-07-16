import requests

resp = requests.post(
    "http://127.0.0.1:8000/analyze",
    headers={"X-API-Key": "innovation-ai-2024"},   # 换成你.env里实际设的API_KEY，如果没设就用这个默认值
    json={"ticker": "AAPL", "query": "Should I buy AAPL?"},
    stream=True
)

print(f"状态码: {resp.status_code}")
print("---")

for line in resp.iter_lines():
    if line:
        print(line.decode("utf-8"))

