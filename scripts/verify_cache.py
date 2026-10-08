import asyncio
import os
import sys

# Ensure root directory is on the import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import settings
from app.core.semantic_cache import SemanticCache


async def main():
    print("=" * 60)
    print("  SEMANTIC CACHE VERIFICATION RUNNER")
    print("=" * 60)

    if not settings.OPENAI_API_KEY:
        print("[!] OPENAI_API_KEY is not set in .env.")
        print("[!] Unit tests with mocks can be run via: pytest -v")
        return

    cache = SemanticCache()

    prompt_a = "What is the capital of France?"
    prompt_b = "Tell me the capital city of France"

    messages_a = [{"role": "user", "content": prompt_a}]
    messages_b = [{"role": "user", "content": prompt_b}]
    model = "gpt-4o"

    print(f"\n1. Querying Prompt A: '{prompt_a}'...")
    res_a, vec_a, sys_h, p_h = await cache.lookup(messages_a, model=model)
    print(
        f"   Result: {'HIT' if res_a.is_hit else 'MISS'} (Latency: {res_a.lookup_latency_ms:.2f}ms)"
    )

    if not res_a.is_hit:
        print("   -> Simulating LLM response storage...")
        mock_response = {
            "id": "chatcmpl-mock",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "The capital of France is Paris.",
                    }
                }
            ],
        }
        cache.store(prompt_a, vec_a, mock_response, model, sys_h, p_h)
        print("   -> Entry indexed in Redis vector store.")

    print(f"\n2. Querying Prompt B (Semantically equivalent): '{prompt_b}'...")
    res_b, _, _, _ = await cache.lookup(
        messages_b, model=model, similarity_threshold=0.88
    )
    print(f"   Result: {'HIT' if res_b.is_hit else 'MISS'}")
    print(f"   Cosine Similarity Score: {res_b.similarity_score:.4f}")
    print(f"   Lookup Latency: {res_b.lookup_latency_ms:.2f}ms")
    if res_b.is_hit:
        print(
            f"   Returned Answer: {res_b.response_data['choices'][0]['message']['content']}"
        )

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())