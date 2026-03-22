# =========================
# Base Image
# =========================
FROM python:3.12-slim

# =========================
# Environment Variables
# =========================
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# =========================
# System Dependencies
# =========================
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# =========================
# Set Working Directory
# =========================
WORKDIR /app

# =========================
# Install Python Dependencies
# =========================
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# =========================
# Copy Project Files
# =========================
COPY . .

# =========================
# Create Required Directories
# =========================
RUN mkdir -p outputs/submissions

# =========================
# Default Command
# =========================
CMD ["python", "-m", "src.infer"]