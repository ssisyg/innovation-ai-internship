# 🚀 AI Investment Agent System

A production-grade multi-agent AI system for investment analysis, featuring tool-augmented reasoning, real-time evaluation, and full CI/CD + Docker deployment.

---

## 📌 Project Overview

This project builds an **LLM-powered investment agent** that analyzes stock tickers using:

- 📊 Price data analysis
- 🧠 Sentiment analysis
- 📈 Quantitative prediction model (SHAP-based)
- 🌐 News RAG retrieval

The agent is built using **LangGraph** and exposes both:
- REST API (FastAPI)
- Streaming execution (SSE)

---

## 🧠 Key Features

### 🔹 Multi-tool Agent (LangGraph)
- Tool orchestration with structured reasoning loop
- Supports:
  - PriceTool
  - SentimentTool
  - QuantTool
  - RAGTool

---

### 🔹 Real-time Streaming API
- Server-Sent Events (SSE)
- Token-level + tool-level streaming
- Frontend-ready design

---

### 🔹 Memory System
- Persistent conversation memory
- Redis / local memory abstraction
- Historical context injection

---

### 🔹 Evaluation Framework (Week 12)
Automated agent benchmarking across:

- Tool calling accuracy
- Latency measurement
- Groundedness vs Quant signal
- Hallucination detection

Output:
- CSV report
- JSON summary
- Performance visualization

---

### 🔹 Observability (LangSmith)
- Full tracing of:
  - Tool calls
  - Agent decisions
  - Execution graph

---

### 🔹 Dockerized Deployment
- Multi-stage Docker build
- CPU-only optimized Torch
- Lightweight runtime image
- `/health` monitoring endpoint

---

### 🔹 CI/CD Pipeline (GitHub Actions)

Automatically runs:

#### 1. Test Stage
- pytest execution
- mock-based agent tests
- memory + tool validation

#### 2. Docker Build Stage
- Docker image build
- container health check
- ensures production readiness

---

## 🏗️ System Architecture

```mermaid
graph TD
    A[Client / Frontend] --> B[FastAPI Server]
    B --> C[LangGraph Agent]

    C --> D[Price Tool]
    C --> E[Sentiment Tool]
    C --> F[Quant Model + SHAP]
    C --> G[RAG Retriever]

    D --> H[Market Data]
    E --> I[News / Social Signals]
    F --> J[ML Model Inference]
    G --> K[Vector DB (Chroma)]

    C --> L[Memory Store]
    C --> M[LangSmith Tracing]

    B --> N[Streaming SSE Response]
