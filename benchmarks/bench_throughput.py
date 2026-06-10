"""
Throughput Benchmark — Tokens per second on AMD ROCm GPU
Usage:
    python benchmarks/bench_throughput.py \
        --model mistralai/Mistral-7B-Instruct-v0.2 \
        --num-prompts 200 \
        --output-len 128
"""

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import List, Tuple

import aiohttp


SAMPLE_PROMPTS = [
    "Explain the theory of relativity in simple terms.",
    "Write a Python function to sort a list using quicksort.",
    "What are the main differences between AMD and NVIDIA GPU architectures?",
    "Summarize the history of machine learning in 3 paragraphs.",
    "How does attention mechanism work in transformer models?",
    "What is the difference between supervised and unsupervised learning?",
    "Explain how ROCm differs from CUDA for GPU computing.",
    "Write a haiku about artificial intelligence.",
    "What are the benefits of using vLLM for LLM inference?",
    "Describe the architecture of a typical large language model.",
]


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
    output_len: int,
) -> Tuple[float, int]:
    start = time.perf_counter()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": output_len,
        "temperature": 0.0,
    }
    async with session.post(url, json=payload) as resp:
        result = await resp.json()
    elapsed = time.perf_counter() - start
    tokens = result.get("usage", {}).get("total_tokens", output_len)
    return elapsed, tokens


async def run_benchmark(
    server_url: str,
    model: str,
    num_prompts: int,
    output_len: int,
    concurrency: int,
) -> dict:
    url = f"{server_url}/v1/chat/completions"
    prompts = [SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)] for i in range(num_prompts)]

    print(f"🔥 Running throughput benchmark")
    print(f"   Model      : {model}")
    print(f"   Prompts    : {num_prompts}")
    print(f"   Output len : {output_len} tokens")
    print(f"   Concurrency: {concurrency}")
    print()

    semaphore = asyncio.Semaphore(concurrency)
    results = []

    async def bounded_request(prompt):
        async with semaphore:
            return await send_request(session, url, model, prompt, output_len)

    total_start = time.perf_counter()
    async with aiohttp.ClientSession() as session:
        tasks = [bounded_request(p) for p in prompts]
        results = await asyncio.gather(*tasks)
    total_elapsed = time.perf_counter() - total_start

    latencies = [r[0] for r in results]
    total_tokens = sum(r[1] for r in results)
    throughput = total_tokens / total_elapsed

    stats = {
        "model": model,
        "num_prompts": num_prompts,
        "output_len": output_len,
        "concurrency": concurrency,
        "total_time_s": round(total_elapsed, 2),
        "throughput_tok_per_s": round(throughput, 1),
        "latency_mean_s": round(sum(latencies) / len(latencies), 3),
        "latency_p50_s": round(sorted(latencies)[len(latencies) // 2], 3),
        "latency_p99_s": round(sorted(latencies)[int(len(latencies) * 0.99)], 3),
        "total_tokens": total_tokens,
    }

    print("📊 Results:")
    print(f"   Total time     : {stats['total_time_s']}s")
    print(f"   Throughput     : {stats['throughput_tok_per_s']} tok/s")
    print(f"   Latency mean   : {stats['latency_mean_s']}s")
    print(f"   Latency p50    : {stats['latency_p50_s']}s")
    print(f"   Latency p99    : {stats['latency_p99_s']}s")

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--output-dir", default="benchmarks/results")
    args = parser.parse_args()

    stats = asyncio.run(
        run_benchmark(
            server_url=args.server_url,
            model=args.model,
            num_prompts=args.num_prompts,
            output_len=args.output_len,
            concurrency=args.concurrency,
        )
    )

    # Save results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    out_path = out_dir / f"throughput_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n💾 Results saved to {out_path}")


if __name__ == "__main__":
    main()
