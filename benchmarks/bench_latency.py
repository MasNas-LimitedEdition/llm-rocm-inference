"""
Latency Benchmark — TTFT & End-to-End latency on AMD ROCm GPU
Usage:
    python benchmarks/bench_latency.py \
        --model mistralai/Mistral-7B-Instruct-v0.2 \
        --batch-size 1 \
        --input-len 512 \
        --output-len 128
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp


async def measure_ttft(session, url, model, prompt, output_len):
    """Measure Time To First Token using streaming."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": output_len,
        "temperature": 0.0,
        "stream": True,
    }
    start = time.perf_counter()
    ttft = None
    e2e = None

    async with session.post(url, json=payload) as resp:
        async for line in resp.content:
            if line.startswith(b"data: ") and line != b"data: [DONE]\n":
                if ttft is None:
                    ttft = time.perf_counter() - start
        e2e = time.perf_counter() - start

    return ttft, e2e


async def run_latency_benchmark(
    server_url, model, batch_size, input_len, output_len, num_iters
):
    url = f"{server_url}/v1/chat/completions"
    prompt = "a " * input_len  # simple fixed-length prompt

    print(f"⏱️  Running latency benchmark")
    print(f"   Model      : {model}")
    print(f"   Batch size : {batch_size}")
    print(f"   Input len  : {input_len} tokens (approx)")
    print(f"   Output len : {output_len} tokens")
    print(f"   Iterations : {num_iters}")
    print()

    ttfts = []
    e2es = []

    async with aiohttp.ClientSession() as session:
        # Warmup
        print("🔥 Warming up...")
        for _ in range(3):
            await measure_ttft(session, url, model, prompt, output_len)

        print("📏 Measuring...")
        for i in range(num_iters):
            tasks = [
                measure_ttft(session, url, model, prompt, output_len)
                for _ in range(batch_size)
            ]
            results = await asyncio.gather(*tasks)
            for ttft, e2e in results:
                if ttft:
                    ttfts.append(ttft * 1000)  # ms
                if e2e:
                    e2es.append(e2e * 1000)  # ms
            if (i + 1) % 10 == 0:
                print(f"   Completed {i + 1}/{num_iters} iterations")

    def percentile(data, p):
        idx = int(len(data) * p / 100)
        return round(sorted(data)[idx], 1)

    stats = {
        "model": model,
        "batch_size": batch_size,
        "input_len": input_len,
        "output_len": output_len,
        "num_iters": num_iters,
        "ttft_mean_ms": round(sum(ttfts) / len(ttfts), 1),
        "ttft_p50_ms": percentile(ttfts, 50),
        "ttft_p95_ms": percentile(ttfts, 95),
        "ttft_p99_ms": percentile(ttfts, 99),
        "e2e_mean_ms": round(sum(e2es) / len(e2es), 1),
        "e2e_p50_ms": percentile(e2es, 50),
        "e2e_p95_ms": percentile(e2es, 95),
        "e2e_p99_ms": percentile(e2es, 99),
    }

    print("\n📊 Results:")
    print(f"   TTFT mean  : {stats['ttft_mean_ms']} ms")
    print(f"   TTFT p50   : {stats['ttft_p50_ms']} ms")
    print(f"   TTFT p99   : {stats['ttft_p99_ms']} ms")
    print(f"   E2E mean   : {stats['e2e_mean_ms']} ms")
    print(f"   E2E p99    : {stats['e2e_p99_ms']} ms")

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--num-iters", type=int, default=50)
    parser.add_argument("--output-dir", default="benchmarks/results")
    args = parser.parse_args()

    stats = asyncio.run(
        run_latency_benchmark(
            server_url=args.server_url,
            model=args.model,
            batch_size=args.batch_size,
            input_len=args.input_len,
            output_len=args.output_len,
            num_iters=args.num_iters,
        )
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    out_path = out_dir / f"latency_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n💾 Results saved to {out_path}")


if __name__ == "__main__":
    main()
