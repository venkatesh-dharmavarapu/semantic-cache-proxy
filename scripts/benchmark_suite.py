import asyncio
import json
import statistics
import time
from typing import Any, Dict, List
import httpx


class SemanticCacheBenchmark:
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        concurrency: int = 20,
        requests_per_worker: int = 25,
    ):
        self.base_url = base_url
        self.concurrency = concurrency
        self.requests_per_worker = requests_per_worker
        self.latencies: List[float] = []
        self.tier_counts: Dict[str, int] = {
            "L1-EXACT": 0,
            "L2-SEMANTIC": 0,
            "MISS": 0,
            "ERROR": 0,
        }

    async def _worker(self, worker_id: int, client: httpx.AsyncClient):
        prompts = [
            "How do I sort a dictionary by value in Python?",
            "Sort a python dictionary by its values",
            "What is an abstract syntax tree?",
            "Explain AST in compiler design",
            "How does Docker multi-stage build optimize image size?",
            "Explain Docker multi-stage builds",
        ]

        for i in range(self.requests_per_worker):
            prompt = prompts[(worker_id + i) % len(prompts)]
            payload = {
                "model": "llama3.2:1b",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
            }
            start_t = time.perf_counter()
            try:
                res = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    headers={"X-Tenant-ID": "benchmark_tenant"},
                    timeout=15.0,
                )
                elapsed_ms = (time.perf_counter() - start_t) * 1000.0
                self.latencies.append(elapsed_ms)

                if res.status_code == 200:
                    tier = res.headers.get("X-Cache-Tier", "NONE")
                    lookup = res.headers.get("X-Cache-Lookup", "MISS")
                    if lookup == "HIT":
                        if "L1" in tier:
                            self.tier_counts["L1-EXACT"] += 1
                        else:
                            self.tier_counts["L2-SEMANTIC"] += 1
                    else:
                        self.tier_counts["MISS"] += 1
                else:
                    self.tier_counts["ERROR"] += 1
            except Exception:
                self.tier_counts["ERROR"] += 1

    async def run(self):
        print(f"[*] Starting benchmark: {self.concurrency} workers x {self.requests_per_worker} reqs...")
        start_time = time.perf_counter()
        async with httpx.AsyncClient() as client:
            tasks = [self._worker(w, client) for w in range(self.concurrency)]
            await asyncio.gather(*tasks)

        total_time = time.perf_counter() - start_time
        total_reqs = len(self.latencies)
        rps = total_reqs / total_time if total_time > 0 else 0

        self.latencies.sort()
        p50 = statistics.median(self.latencies) if self.latencies else 0.0
        p90 = self.latencies[int(len(self.latencies) * 0.90)] if self.latencies else 0.0
        p99 = self.latencies[int(len(self.latencies) * 0.99)] if self.latencies else 0.0

        hits = self.tier_counts["L1-EXACT"] + self.tier_counts["L2-SEMANTIC"]
        hit_rate = (hits / total_reqs * 100) if total_reqs > 0 else 0.0

        print("\n================ BENCHMARK REPORT ================")
        print(f"Total Requests Processed: {total_reqs}")
        print(f"Total Duration:           {total_time:.2f} s")
        print(f"Throughput:               {rps:.2f} req/s")
        print(f"Overall Hit Rate:         {hit_rate:.1f}%")
        print(f"  - L1 Exact Hits:        {self.tier_counts['L1-EXACT']}")
        print(f"  - L2 Semantic Hits:     {self.tier_counts['L2-SEMANTIC']}")
        print(f"  - Cache Misses:         {self.tier_counts['MISS']}")
        print(f"  - Errors:               {self.tier_counts['ERROR']}")
        print(f"P50 Latency:              {p50:.2f} ms")
        print(f"P90 Latency:              {p90:.2f} ms")
        print(f"P99 Latency:              {p99:.2f} ms")
        print("==================================================\n")


if __name__ == "__main__":
    bench = SemanticCacheBenchmark()
    asyncio.run(bench.run())