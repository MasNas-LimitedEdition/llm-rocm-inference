"""
LLM Inference Server — AMD ROCm Optimized
FastAPI server with OpenAI-compatible endpoints
"""

import argparse
import logging
import yaml
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from engine import InferenceEngine
from routes.chat import router as chat_router
from routes.completions import router as completions_router
from utils.logger import setup_logger
from utils.metrics import metrics_router

logger = setup_logger(__name__)
engine: InferenceEngine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    logger.info("🚀 Starting LLM Inference Server (AMD ROCm)")
    engine = InferenceEngine(
        model=app.state.config["model"]["name"],
        dtype=app.state.config["model"].get("dtype", "float16"),
        max_model_len=app.state.config["model"].get("max_model_len", 4096),
        gpu_memory_utilization=app.state.config["rocm"].get("gpu_memory_utilization", 0.9),
        tensor_parallel_size=app.state.config["rocm"].get("tensor_parallel_size", 1),
    )
    await engine.initialize()
    logger.info(f"✅ Model loaded: {app.state.config['model']['name']}")
    yield
    logger.info("🛑 Shutting down inference server...")
    await engine.shutdown()


def create_app(config: dict) -> FastAPI:
    app = FastAPI(
        title="LLM Inference Server (ROCm)",
        description="High-throughput LLM inference optimized for AMD GPUs",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.config = config
    app.state.get_engine = lambda: engine

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(chat_router, prefix="/v1")
    app.include_router(completions_router, prefix="/v1")
    app.include_router(metrics_router)

    Instrumentator().instrument(app).expose(app, endpoint="/metrics")

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": config["model"]["name"]}

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [{"id": config["model"]["name"], "object": "model"}],
        }

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="LLM Inference Server (ROCm)")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # CLI overrides
    if args.model:
        config["model"]["name"] = args.model
    if args.host:
        config["server"]["host"] = args.host
    if args.port:
        config["server"]["port"] = args.port
    if args.gpu_memory_utilization:
        config["rocm"]["gpu_memory_utilization"] = args.gpu_memory_utilization

    app = create_app(config)
    uvicorn.run(
        app,
        host=config["server"]["host"],
        port=config["server"]["port"],
        workers=config["server"].get("workers", 1),
        log_level="info",
    )
