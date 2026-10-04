import hashlib
import json
from typing import Any, Dict, Optional

class CacheKeyHasher:
    @staticmethod
    def hash_system_prompt(system_prompt: Optional[str]) -> str:
        """Generates a fixed SHA-256 hash for system instructions."""
        normalized = (system_prompt or "").strip().lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def hash_generation_params(
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
    ) -> str:
        """
        Creates a deterministic hash of inference hyperparameters.
        Identical inputs with different temperatures must not share cache hits.
        """
        params: Dict[str, Any] = {
            "model": model.strip().lower(),
            "temperature": round(temperature, 2) if temperature is not None else 1.0,
            "max_tokens": max_tokens,
            "top_p": round(top_p, 2) if top_p is not None else 1.0,
            "presence_penalty": round(presence_penalty, 2) if presence_penalty is not None else 0.0,
            "frequency_penalty": round(frequency_penalty, 2) if frequency_penalty is not None else 0.0,
        }
        
        serialized = json.dumps(params, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()