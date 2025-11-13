FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY phase1_data_processing/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create necessary directories
RUN mkdir -p \
    phase1_data_processing/data/raw_audio \
    phase1_data_processing/data/processed_audio \
    phase1_data_processing/data/transcripts \
    phase1_data_processing/data/metadata \
    phase1_data_processing/outputs/llm_training_data \
    phase1_data_processing/outputs/analysis_reports \
    phase1_data_processing/logs

# Set the working directory to phase1
WORKDIR /app/phase1_data_processing

# Expose port for potential API (future use)
EXPOSE 8000

# Default command
CMD ["python", "run_pipeline.py"]