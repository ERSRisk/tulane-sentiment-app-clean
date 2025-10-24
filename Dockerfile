FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# You can set a default, but Cloud Run will inject PORT anyway
ENV PORT=8080

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc build-essential git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

# Shell form so $PORT expands; binds correctly for Cloud Run
CMD streamlit run Sentiment-Analysis-App.py \
    --server.address=0.0.0.0 \
    --server.port=$PORT \
    --server.headless=true \
    --server.enableCORS=false \
    --browser.gatherUsageStats=false
