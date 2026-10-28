from typing import Any, Dict


class NegativeCachePolicy:
    DEFAULT_NEGATIVE_TTL_SECONDS = 30
    REFUSAL_NEGATIVE_TTL_SECONDS = 60

    @staticmethod
    def is_negative_or_refusal(response_data: Dict[str, Any]) -> bool:
        """Detects whether response represents a model refusal or transient error."""
        choices = response_data.get("choices", [])
        if not choices:
            return True

        message = choices[0].get("message", {})
        content = message.get("content", "").lower()

        refusal_markers = [
            "i cannot fulfill this request",
            "as an ai, i am unable",
            "i am sorry, but i cannot",
            "model is overloaded",
            "service unavailable",
        ]

        return any(marker in content for marker in refusal_markers)

    @classmethod
    def get_adaptive_ttl(cls, response_data: Dict[str, Any], default_ttl: int) -> int:
        """Returns short TTL for negative responses or normal TTL for clean completions."""
        if cls.is_negative_or_refusal(response_data):
            return cls.REFUSAL_NEGATIVE_TTL_SECONDS
        return default_ttl