FROM rocm/pytorch:rocm6.0_ubuntu22.04_py3.10_pytorch_2.1.1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/
COPY config.yaml .

# ROCm environment variables
ENV HSA_OVERRIDE_GFX_VERSION=11.0.0
ENV ROCR_VISIBLE_DEVICES=0

EXPOSE 8000

CMD ["python", "src/server.py", "--config", "config.yaml"]
