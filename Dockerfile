FROM python:3.10-slim-bullseye AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc build-essential git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ---------- Final image ----------
FROM python:3.10-slim-bullseye

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

WORKDIR /app

# copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# now copy just your app code (filtered by .dockerignore)
COPY . .

EXPOSE 8080

CMD streamlit run Sentiment-Analysis-App.py \
    --server.address=0.0.0.0 \
    --server.port=$PORT \
    --server.headless=true \
    --server.enableCORS=false \
    --browser.gatherUsageStats=false
