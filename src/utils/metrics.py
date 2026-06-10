from fastapi import APIRouter
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

metrics_router = APIRouter()

REQUEST_COUNT = Counter("llm_requests_total", "Total inference requests", ["endpoint"])
REQUEST_LATENCY = Histogram("llm_request_latency_seconds", "Request latency", ["endpoint"])
TOKEN_COUNT = Counter("llm_tokens_total", "Total tokens generated")


@metrics_router.get("/metrics/custom")
async def custom_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
