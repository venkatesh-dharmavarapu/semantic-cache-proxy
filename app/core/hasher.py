import hashlib
import json
from typing import Any, Dict, List, Optional


class CacheKeyHasher:
    @staticmethod
    def hash_system_prompt(system_prompt: str) -> str:
        """Computes deterministic SHA-256 hash for system prompt."""
        cleaned = system_prompt.strip()
        return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()

    @staticmethod
    def hash_generation_params(
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """Deterministic hashing of inference hyperparameters."""
        normalized_params: Dict[str, Any] = {
            "model": model.strip().lower(),
            "temperature": round(temperature, 2) if temperature is not None else 1.0,
            "max_tokens": max_tokens if max_tokens is not None else 0,
            "top_p": round(top_p, 2) if top_p is not None else 1.0,
        }
        serialized = json.dumps(normalized_params, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def hash_conversation_context(
        messages: List[Dict[str, str]], window_size: int = 4
    ) -> str:
        """
        Hashes conversation context within a sliding window of recent turns.
        Excludes the final message (the active prompt to be vectorized).
        """
        if len(messages) <= 1:
            return "stateless"

        # Take up to `window_size` messages leading up to the final message
        context_slice = messages[-(window_size + 1) : -1]
        normalized_history = [
            f"{m.get('role', '')}:{m.get('content', '').strip()}"
            for m in context_slice
        ]
        serialized = "||".join(normalized_history)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()