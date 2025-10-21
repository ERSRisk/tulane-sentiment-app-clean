# Use a lightweight Python image
FROM python:3.11-slim

# Prevent Python from writing pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies your libraries might need
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libxml2-dev libxslt1-dev libjpeg62-turbo-dev zlib1g-dev \
    git curl && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python packages
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of your app
COPY . .

# Streamlit needs to bind to 0.0.0.0:$PORT for Cloud Run
EXPOSE 8080
CMD ["bash", "-lc", "streamlit run Sentiment-Analysis-App.py --server.port=${PORT:-8080} --server.address=0.0.0.0"]
