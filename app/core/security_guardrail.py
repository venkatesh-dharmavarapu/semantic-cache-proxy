import re
from typing import Any, Dict, List


class CacheSecurityGuardrail:
    # Common prompt injection, system override, and exfiltration signatures
    INJECTION_PATTERNS = [
        re.compile(r"ignore\s+(all\s+)?(previous|above)\s+instructions", re.IGNORECASE),
        re.compile(r"disregard\s+(all\s+)?prior\s+prompts", re.IGNORECASE),
        re.compile(r"system\s*override", re.IGNORECASE),
        re.compile(r"<\|\s*im_start\s*\|>", re.IGNORECASE),
        re.compile(r"reveal\s+(your\s+)?system\s+prompt", re.IGNORECASE),
    ]

    @classmethod
    def is_cacheable(cls, prompt: str, response_data: Dict[str, Any]) -> tuple[bool, str]:
        """
        Evaluates whether prompt and completion are safe to store in the semantic cache.
        Returns (is_allowed: bool, reason: str).
        """
        # 1. Inspect prompt for adversarial injection triggers
        for pattern in cls.INJECTION_PATTERNS:
            if pattern.search(prompt):
                return False, "adversarial_injection_detected"

        # 2. Inspect output content
        choices = response_data.get("choices", [])
        if not choices:
            return False, "empty_choices"

        content = choices[0].get("message", {}).get("content", "")
        if not content or len(content.strip()) < 2:
            return False, "empty_or_trivial_output"

        # Reject caching if model leaked system prompt markers
        if "you are a system prompt" in content.lower():
            return False, "system_leak_detected"

        return True, "valid"