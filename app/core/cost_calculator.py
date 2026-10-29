from typing import Any, Dict


class TokenCostCalculator:
    # Baseline benchmark rates per 1,000 tokens (USD)
    MODEL_RATES: Dict[str, Dict[str, float]] = {
        "llama3.2:1b": {"prompt": 0.00005, "completion": 0.00010},
        "llama3.2:3b": {"prompt": 0.00010, "completion": 0.00020},
        "mistral": {"prompt": 0.00020, "completion": 0.00060},
        "default": {"prompt": 0.00015, "completion": 0.00060},
    }

    @classmethod
    def calculate_cost(cls, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        rates = cls.MODEL_RATES.get(model.lower(), cls.MODEL_RATES["default"])
        cost = (prompt_tokens / 1000.0 * rates["prompt"]) + (
            completion_tokens / 1000.0 * rates["completion"]
        )
        return round(cost, 6)

    @classmethod
    def extract_tokens(cls, response_data: Dict[str, Any]) -> tuple[int, int, int]:
        usage = response_data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
        return prompt_tokens, completion_tokens, total_tokens