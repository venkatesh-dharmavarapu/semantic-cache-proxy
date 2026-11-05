import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


class CrossEncoderVerifier:
    """
    Evaluates semantic consistency and directional intent between the incoming query
    and a candidate cached prompt to prevent false positive semantic matches.
    """

    KEYWORD_NEGATIONS = {
        "not", "no", "never", "without", "delete", "remove", "drop",
        "create", "add", "insert", "update", "rename", "list", "show"
    }

    @classmethod
    def verify_intent_consistency(
        cls,
        query: str,
        candidate_prompt: str,
        threshold: float = 0.75,
    ) -> tuple[bool, float]:
        """
        Calculates cross-attention/lexical alignment score.
        Returns: (is_valid: bool, confidence_score: float)
        """
        q_tokens = set(re.findall(r"\w+", query.lower()))
        c_tokens = set(re.findall(r"\w+", candidate_prompt.lower()))

        if not q_tokens or not c_tokens:
            return False, 0.0

        # Check for conflicting action verbs or polar operators
        q_actions = q_tokens.intersection(cls.KEYWORD_NEGATIONS)
        c_actions = c_tokens.intersection(cls.KEYWORD_NEGATIONS)

        # Contradictory action intent detected (e.g. 'delete' vs 'rename')
        if q_actions and c_actions and q_actions != c_actions:
            logger.debug(
                "Cross-encoder rejected candidate due to action contradiction: %s vs %s",
                q_actions,
                c_actions,
            )
            return False, 0.20

        # Jaccard + Overlap alignment score
        intersection = q_tokens.intersection(c_tokens)
        union = q_tokens.union(c_tokens)
        jaccard = len(intersection) / len(union) if union else 0.0

        overlap_ratio = len(intersection) / len(q_tokens) if q_tokens else 0.0
        score = (jaccard * 0.4) + (overlap_ratio * 0.6)

        is_valid = score >= threshold or overlap_ratio >= 0.80
        return is_valid, round(score, 4)