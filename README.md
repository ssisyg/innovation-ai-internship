📊 Investment Agent System (Week 11–12)
🚀 Overview

This project is an end-to-end LLM-powered investment analysis agent system built with LangGraph + FastAPI + Docker, designed for real-world financial reasoning, tool orchestration, and automated evaluation.

It supports:

Multi-tool agent reasoning (Price / Sentiment / Quant / RAG)
Streaming inference via FastAPI (SSE)
Persistent memory (ChromaDB-based)
Full evaluation framework (latency / accuracy / groundedness / hallucination)
CI/CD pipeline with automated testing and Docker validation
🧠 System Architecture
User Request
    ↓
FastAPI (/analyze, /history)
    ↓
LangGraph Agent (ReAct loop)
    ↓
4 Tools
   ├── PriceTool
   ├── SentimentTool
   ├── QuantTool
   └── RAGTool
    ↓
Memory Layer (ChromaDB)
    ↓
Final Investment Decision (BUY / SELL / HOLD)
⚙️ Core Features
1. LangGraph Agent (ReAct-based)
Iterative reasoning loop (Think → Act → Observe)
Dynamically decides which tools to call
Supports multi-step financial analysis
2. Tool System (4 Core Tools)
Tool	Function
PriceTool	Market price + indicators
SentimentTool	News sentiment scoring
QuantTool	ML-based prediction (up_probability)
RAGTool	News retrieval + context grounding
3. Streaming API (FastAPI SSE)
Real-time token / tool streaming
Frontend-ready event stream
Supports error event handling

Endpoints:

POST /analyze
GET  /history
GET  /health
4. Memory System
Persistent conversation & analysis storage
Built on ChromaDB
Enables historical reasoning context injection
5. Evaluation Framework (Week 12)

Automated evaluation across multiple dimensions:

Metrics:
Tool Calling Accuracy
Response Latency
Groundedness (vs QuantTool signal)
Hallucination Detection
Output:
CSV report
JSON summary
Visualization charts

Example:

data/models/eval/
 ├── evaluation_report.csv
 ├── evaluation_summary.json
 └── evaluation_chart.png
🐳 Docker Deployment
Build image
docker build -t investment-agent-backend .
Run container
docker run -p 8000:8000 --env-file .env \
-v ${PWD}/data/memory:/app/data/memory \
investment-agent-backend
Health check
curl http://localhost:8000/health

Expected response:

{"status": "healthy"}
🔁 CI/CD Pipeline (GitHub Actions)

Located at:

.github/workflows/ci.yml
Pipeline stages:
1. Test Job
Install requirements-backend.txt
Install requirements-test.txt
Run pytest -v
2. Docker Build Job (depends on test)
Build Docker image
Start container
Validate /health endpoint
Print logs for debugging

⚠️ No secrets required (fully mock-based testing)

🧪 Testing

Run locally:

pytest -v

Test coverage includes:

Agent logic
API endpoints
Memory system
Tool execution
Streaming behavior
📈 Evaluation Example Output
{
  "n_tickers": 5,
  "avg_latency_sec": 3.21,
  "avg_tool_accuracy": 0.92,
  "groundedness_rate": 0.84,
  "hallucination_rate": 0.06
}
📦 Project Structure
innovation-ai-internship/
├── backend/
├── scripts/
│   ├── evaluate_agent.py
│   └── watch_loop.py
├── tests/
├── data/
├── docker-compose.yml
├── Dockerfile
├── requirements-backend.txt
├── requirements-test.txt
└── .github/workflows/ci.yml
🧩 Tech Stack
LangGraph
FastAPI
ChromaDB
PyTorch (CPU-only)
scikit-learn
Docker
GitHub Actions
pytest
🚀 Key Improvements (Week 11–12)
Added full CI/CD pipeline
Introduced evaluation framework
Dockerized full system (multi-stage build)
Optimized dependencies (runtime vs dev split)
Added streaming + error handling robustness
Enabled reproducible testing environment
📌 Notes
CI runs fully mock-based (no external API keys required)
Docker build is CPU-only optimized
Evaluation framework runs real LLM inference per ticker
📫 Future Improvements
Add Redis caching layer
Add GPU support for embedding models
Add frontend dashboard for evaluation metrics
Improve hallucination detection with semantic matching
