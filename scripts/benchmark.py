import asyncio
import time
import statistics
import httpx

BASE_URL = "http://localhost:8000/v1"

PROMPT_PAIRS = [
    (
        "Explain what a Redis vector database is in two sentences.",
        "What is a Redis vector database? Keep it to two sentences.",
    ),
    (
        "How do I sort a list in Python?",
        "Show me how to sort a Python list.",
    ),
    (
        "What is the capital of Japan?",
        "Name the capital city of Japan.",
    ),
]


async def run_benchmark():
    print("=" * 65)
    print("       SEMANTIC CACHE LATENCY BENCHMARK RUNNER")
    print("=" * 65)

    cold_latencies = []
    warm_latencies = []

    async with httpx.AsyncClient(timeout=120.0) as client:
        # Check health
        try:
            health = await client.get("http://localhost:8000/health")
            if health.status_code != 200:
                print("[!] Error: Proxy service not healthy.")
                return
        except Exception:
            print("[!] Error: Cannot connect to http://localhost:8000. Start uvicorn first.")
            return

        for idx, (p_original, p_paraphrase) in enumerate(PROMPT_PAIRS, start=1):
            print(f"\n[Test Case {idx}]")
            
            # 1. Cold Request (Cache MISS)
            payload_cold = {
                "model": "llama3.2:1b",
                "messages": [{"role": "user", "content": p_original}],
                "temperature": 0.0,
            }
            t0 = time.perf_counter()
            resp_cold = await client.post(f"{BASE_URL}/chat/completions", json=payload_cold)
            lat_cold = (time.perf_counter() - t0) * 1000.0
            cold_latencies.append(lat_cold)
            lookup_header = resp_cold.headers.get("X-Cache-Lookup", "N/A")
            print(f"  Cold Run  -> Status: {resp_cold.status_code} | Cache: {lookup_header} | Latency: {lat_cold:.2f} ms")

            # 2. Warm Request (Cache HIT via semantic similarity)
            payload_warm = {
                "model": "llama3.2:1b",
                "messages": [{"role": "user", "content": p_paraphrase}],
                "temperature": 0.0,
            }
            t1 = time.perf_counter()
            resp_warm = await client.post(
                f"{BASE_URL}/chat/completions",
                json=payload_warm,
                headers={"X-Similarity-Threshold": "0.85"},
            )
            lat_warm = (time.perf_counter() - t1) * 1000.0
            warm_latencies.append(lat_warm)
            lookup_warm = resp_warm.headers.get("X-Cache-Lookup", "N/A")
            sim_score = resp_warm.headers.get("X-Cache-Similarity", "N/A")
            print(f"  Warm Run  -> Status: {resp_warm.status_code} | Cache: {lookup_warm} (Sim: {sim_score}) | Latency: {lat_warm:.2f} ms")

    avg_cold = statistics.mean(cold_latencies)
    avg_warm = statistics.mean(warm_latencies)
    speedup = (avg_cold / avg_warm) if avg_warm > 0 else 0.0

    print("\n" + "=" * 65)
    print("                     BENCHMARK SUMMARY")
    print("=" * 65)
    print(f"  Avg Cold Request Latency (Model Gen) : {avg_cold:.2f} ms")
    print(f"  Avg Warm Request Latency (Cache Hit) : {avg_warm:.2f} ms")
    print(f"  Net Latency Reduction                : {(avg_cold - avg_warm):.2f} ms")
    print(f"  Effective Speedup Factor             : {speedup:.1f}x faster")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(run_benchmark())