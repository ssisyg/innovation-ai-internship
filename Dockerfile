# =====================================================================
# Multi-stage Dockerfile for the AI Investment Agent backend
# (backend/agent.py + backend/main.py, served via FastAPI + Uvicorn).
#
# Build:
#   docker build -t investment-agent-backend .
#
# Run (env vars passed at runtime, never baked into the image):
#   docker run -p 8000:8000 --env-file .env \
#     -v $(pwd)/data/memory:/app/data/memory \
#     investment-agent-backend
#
# The volume mount keeps data/memory/agent_memory.db (the SQLite
# MemoryStore) on the host, so recommendation history survives
# container restarts/rebuilds instead of living only inside the
# container's writable layer.
# =====================================================================

# ---------- Stage 1: build ----------
# Compile/install everything here so the final image doesn't need a
# C/C++ toolchain (chromadb's hnswlib and some sklearn/lightgbm/xgboost
# extras need one to build from source on some platforms).
FROM python:3.12-slim AS builder

WORKDIR /app

# deb.debian.org can be slow/unreachable from some networks. Point apt
# at a mirror and retry a couple of times before giving up, rather than
# failing the whole build on one transient network blip.
# If deb.debian.org already works fine on your network, you can delete
# the two `sed` lines below and keep the plain apt-get call.
RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g; s|security.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources \
    && for i in 1 2 3; do \
        apt-get update && apt-get install -y --no-install-recommends build-essential git && break || sleep 5; \
       done \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-backend.txt .

# Install the CPU-only torch wheel FIRST and explicitly. PyPI's default
# `torch` wheel bundles CUDA libraries (multiple GB) that are useless in
# a CPU-only API container -- we only need torch here so
# sentence-transformers can run the MiniLM embedding model for RAGTool.
# Once torch is already satisfied, the subsequent install of
# requirements-backend.txt (which also lists torch as a transitive need
# via sentence-transformers) is a no-op for that package.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements-backend.txt

# ---------- Stage 2: runtime ----------
FROM python:3.12-slim

WORKDIR /app

# libgomp1 is a runtime (not build-time) requirement of xgboost/lightgbm
RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g; s|security.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources \
    && for i in 1 2 3; do \
        apt-get update && apt-get install -y --no-install-recommends libgomp1 && break || sleep 5; \
       done \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 appuser

# Bring over only the installed packages + entry-point scripts from the
# build stage -- not the compiler toolchain or pip caches.
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Application code
COPY backend/ ./backend/

# Pre-trained model, precomputed features/sentiment, and the pre-built
# ChromaDB index that agent.py loads at import time via paths relative
# to the working directory (e.g. "data/models/best_model.pkl"). Files
# not needed at runtime (raw training sets, EDA/backtest plots, news
# scraping output, notebooks) are filtered out by .dockerignore.
COPY data/models/ ./data/models/
COPY data/ml/ ./data/ml/
COPY data/sentiment/ ./data/sentiment/
# data/raw 和 data/rag 不再打包进镜像：
# - PriceTool 本地缺数据时会自动实时拉取（yfinance/Tiingo 兜底）
# - RAGTool 的 ChromaDB collection 缺失时会从 data/sentiment/ 自动重建

# Writable directory for the SQLite MemoryStore; mount a volume here
# at `docker run` time (see the comment at the top of this file) so
# recommendation history isn't lost when the container is recreated.
RUN mkdir -p /app/data/memory && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0) if urllib.request.urlopen('http://localhost:8000/health').status == 200 else sys.exit(1)"

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
