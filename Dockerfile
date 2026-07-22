# Guardião da Floresta - Docker Image
# Build with Gemma: Amazon Eco-Hack

FROM python:3.10-slim

LABEL maintainer="Your Name"
LABEL description="Guardião da Floresta - Edge AI for Amazon Conservation"
LABEL competition="Build with Gemma: Amazon Eco-Hack"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    libgomp1 \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY guardiao_core.py .
COPY app.py .
COPY data/ ./data/

# Create directories for models and uploads
RUN mkdir -p /app/models /app/uploads

# Expose Gradio port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Run the application
CMD ["python", "app.py"]
