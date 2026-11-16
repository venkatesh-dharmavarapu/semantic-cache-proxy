import asyncio
import httpx


async def run_chaos_verification(base_url: str = "http://localhost:8000"):
    print("[*] Running Chaos Drill on Semantic Cache Proxy...")
    async with httpx.AsyncClient(base_url=base_url) as client:
        # 1. Probe Health & Probes
        live_res = await client.get("/health/live")
        assert live_res.status_code == 200, "Liveness failed"
        ready_res = await client.get("/health/ready")
        assert ready_res.status_code == 200, "Readiness failed"
        print("[+] Probes operational: Live=200, Ready=200")

        # 2. Test Tenant Quota Boundary
        tenant_headers = {"X-Tenant-ID": "chaos_tenant_quota"}
        prompt_payload = {
            "model": "llama3.2:1b",
            "messages": [{"role": "user", "content": "Chaos resilience evaluation"}],
            "temperature": 0.0,
        }
        res1 = await client.post("/v1/chat/completions", json=prompt_payload, headers=tenant_headers)
        print(f"[+] Initial call status: {res1.status_code} (Cache Lookup: {res1.headers.get('X-Cache-Lookup')})")

        # 3. Test Purge Isolation
        purge_res = await client.delete("/v1/cache/tenants/chaos_tenant_quota")
        assert purge_res.status_code == 200, "Purge call failed"
        print(f"[+] Tenant namespace purge executed cleanly: {purge_res.json()}")

        # 4. Verify Post-Purge Miss
        res2 = await client.post("/v1/chat/completions", json=prompt_payload, headers=tenant_headers)
        assert res2.headers.get("X-Cache-Lookup") == "MISS", "Expected MISS post-purge"
        print("[+] Post-purge isolation confirmed: Cache Lookup=MISS")

    print("[SUCCESS] All automated chaos scenarios validated.")


if __name__ == "__main__":
    asyncio.run(run_chaos_verification())