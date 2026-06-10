# 🚀 LLM Inference Server — AMD ROCm Optimized

A high-performance LLM inference server built specifically for **AMD GPU (ROCm)** using [vLLM](https://github.com/vllm-project/vllm) and [FastAPI](https://fastapi.tiangolo.com/). Designed for low-latency, high-throughput text generation at scale.

> **Why AMD ROCm?** This project targets AMD Instinct GPUs (MI210, MI250, MI300X) via the ROCm open software stack — offering a compelling alternative to CUDA for production AI inference.

---

## 📌 Features

- ⚡ **Fast inference** with continuous batching via vLLM
- 🔌 **OpenAI-compatible REST API** (`/v1/chat/completions`, `/v1/completions`)
- 📊 **Built-in benchmarking** — throughput, latency, TTFT (Time To First Token)
- 🐳 **Docker support** with ROCm base image
- 🔧 **Model-agnostic** — supports LLaMA, Mistral, Phi, Qwen, and more
- 📈 **Prometheus metrics** endpoint for monitoring

---

## 🖥️ Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | AMD RX 7900 XTX (24GB) | AMD Instinct MI210 / MI300X |
| RAM | 32 GB | 64 GB+ |
| Storage | 50 GB SSD | 200 GB NVMe |
| ROCm | 5.7+ | 6.0+ |

---

## 📁 Project Structure

```
llm-rocm-inference/
├── src/
│   ├── server.py          # FastAPI inference server
│   ├── engine.py          # vLLM engine wrapper
│   ├── routes/
│   │   ├── chat.py        # /v1/chat/completions
│   │   └── completions.py # /v1/completions
│   └── utils/
│       ├── metrics.py     # Prometheus metrics
│       └── logger.py      # Structured logging
├── benchmarks/
│   ├── bench_throughput.py  # Tokens/sec benchmark
│   ├── bench_latency.py     # TTFT & E2E latency
│   └── results/             # Benchmark output (JSON/CSV)
├── docker/
│   ├── Dockerfile           # ROCm-based image
│   └── docker-compose.yml
├── scripts/
│   ├── install_rocm.sh      # ROCm setup helper
│   └── download_model.py    # HuggingFace model downloader
├── tests/
│   └── test_api.py
├── requirements.txt
└── config.yaml
```

---

## ⚙️ Installation

### 1. Install ROCm

```bash
bash scripts/install_rocm.sh
```

Or follow the [official ROCm installation guide](https://rocm.docs.amd.com/en/latest/deploy/linux/index.html).

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Download a model

```bash
python scripts/download_model.py --model mistralai/Mistral-7B-Instruct-v0.2
```

---

## 🚀 Quick Start

### Run the server

```bash
python src/server.py \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.90
```

### Send a request

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Mistral-7B-Instruct-v0.2",
    "messages": [{"role": "user", "content": "Explain quantum computing in simple terms"}],
    "max_tokens": 256
  }'
```

---

## 🐳 Docker

```bash
cd docker
docker compose up --build
```

The `docker-compose.yml` automatically maps AMD GPU devices using `--device=/dev/kfd` and `--device=/dev/dri`.

---

## 📊 Benchmarks

Run throughput benchmark:

```bash
python benchmarks/bench_throughput.py \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --num-prompts 200 \
  --output-len 128
```

Run latency benchmark:

```bash
python benchmarks/bench_latency.py \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --batch-size 1 \
  --input-len 512 \
  --output-len 128
```

### Sample Results (AMD Instinct MI210)

| Model | Throughput (tok/s) | TTFT (ms) | Latency p99 (ms) |
|-------|--------------------|-----------|------------------|
| Mistral-7B | ~2,400 | ~45 | ~320 |
| LLaMA-3-8B | ~2,100 | ~52 | ~380 |
| Phi-3-mini | ~3,800 | ~28 | ~190 |

---

## 📡 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI-compatible chat |
| `/v1/completions` | POST | Text completion |
| `/v1/models` | GET | List loaded models |
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |

---

## 🛠️ Configuration (`config.yaml`)

```yaml
model:
  name: mistralai/Mistral-7B-Instruct-v0.2
  dtype: float16
  max_model_len: 4096

server:
  host: 0.0.0.0
  port: 8000
  workers: 1

rocm:
  gpu_memory_utilization: 0.90
  tensor_parallel_size: 1   # set to 2+ for multi-GPU
```

---

## 🤝 Contributing

Pull requests are welcome! Please open an issue first to discuss changes.

---

## 📄 License

MIT License — see [LICENSE](LICENSE)
