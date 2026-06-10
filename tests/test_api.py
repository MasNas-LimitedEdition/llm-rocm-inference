"""
Basic API tests — run against a live server
"""

import pytest
import httpx

BASE_URL = "http://localhost:8000"


@pytest.fixture
def client():
    return httpx.Client(base_url=BASE_URL, timeout=60)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_list_models(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    assert "data" in resp.json()


def test_chat_completion(client):
    resp = client.post("/v1/chat/completions", json={
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 10,
        "temperature": 0.0,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert data["choices"][0]["message"]["role"] == "assistant"


def test_completion(client):
    resp = client.post("/v1/completions", json={
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "prompt": "The capital of France is",
        "max_tokens": 5,
        "temperature": 0.0,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "choices" in data


def test_usage_tokens(client):
    resp = client.post("/v1/chat/completions", json={
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 20,
    })
    data = resp.json()
    usage = data.get("usage", {})
    assert usage.get("prompt_tokens", 0) > 0
    assert usage.get("completion_tokens", 0) > 0
