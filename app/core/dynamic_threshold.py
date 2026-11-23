import math
import re
from typing import Set, Tuple


class DynamicThresholdCalibrator:
    """
    Dynamically adjusts cosine similarity thresholds based on prompt entropy,
    structural specificity, and action safety margins.
    """

    DESTRUCTIVE_ACTIONS: Set[str] = {
        "delete", "drop", "truncate", "remove", "kill", "purge",
        "destroy", "terminate", "wipe", "shutdown", "erase"
    }

    TECHNICAL_INDICATORS: Set[str] = {
        "kubernetes", "docker", "postgres", "redis", "asyncio", "pydantic",
        "fastapi", "pytest", "hnsw", "bm25", "parquet", "grpc", "protobuf",
        "algorithm", "complexity", "ast", "compiler", "mutex", "deadlock"
    }

    @classmethod
    def calculate_entropy(cls, text: str) -> float:
        """
        Calculates Shannon entropy across character unigrams to estimate information density.
        """
        cleaned = text.strip().lower()
        if not cleaned:
            return 0.0

        length = len(cleaned)
        char_counts = {}
        for char in cleaned:
            char_counts[char] = char_counts.get(char, 0) + 1

        entropy = 0.0
        for count in char_counts.values():
            prob = count / length
            entropy -= prob * math.log2(prob)

        return round(entropy, 4)

    @classmethod
    def calibrate_threshold(
        cls,
        prompt: str,
        base_threshold: float = 0.88,
    ) -> Tuple[float, float, str]:
        """
        Computes calibrated similarity threshold based on entropy and intent safety.

        Returns:
            (calibrated_threshold, entropy_score, adjustment_reason)
        """
        raw_tokens = re.findall(r"\w+", prompt.lower())
        token_set = set(raw_tokens)
        token_count = len(raw_tokens)

        entropy = cls.calculate_entropy(prompt)
        adjustment = 0.0
        reasons = []

        # 1. Action Safety Margin Check
        destructive_hits = token_set.intersection(cls.DESTRUCTIVE_ACTIONS)
        if destructive_hits:
            # Stricter margin for destructive actions
            adjustment += 0.06
            reasons.append(f"destructive_action({','.join(destructive_hits)})")

        # 2. Length & Vague Query Penalization
        if token_count <= 4:
            # Low length / high ambiguity -> tighten threshold
            adjustment += 0.05
            reasons.append("low_token_count")
        elif token_count >= 15:
            # High-length technical specificity -> relax threshold slightly
            adjustment -= 0.03
            reasons.append("high_token_count")

        # 3. Technical Indicator Bonus
        tech_hits = token_set.intersection(cls.TECHNICAL_INDICATORS)
        if len(tech_hits) >= 2:
            adjustment -= 0.02
            reasons.append(f"technical_density({len(tech_hits)})")

        # Bound calibrated threshold between [0.78, 0.96]
        calibrated = round(max(0.78, min(0.96, base_threshold + adjustment)), 4)
        reason_str = ";".join(reasons) if reasons else "baseline"

        return calibrated, entropy, reason_str