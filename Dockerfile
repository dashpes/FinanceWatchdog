FROM python:3.11-slim

WORKDIR /app

# Install cron and other system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    cron \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir ".[notifications,ai]"

# Copy application code
COPY src/ src/
COPY config/ config/

# Copy cron files and scripts
COPY cron/ cron/
COPY scripts/ scripts/

# Make scripts executable
RUN chmod +x scripts/*.sh

# Create data directories
RUN mkdir -p data logs

# Set environment
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "-m", "investment_monitor.cli", "--type", "regular"]
