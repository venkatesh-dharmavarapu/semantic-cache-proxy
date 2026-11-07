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

    STOPWORDS = {
        "a", "an", "the", "in", "on", "at", "to", "for", "of", "with",
        "by", "is", "are", "was", "were", "be", "been", "how", "what",
        "why", "can", "could", "do", "does", "did", "i", "you", "it",
        "me", "my", "please"
    }

    @classmethod
    def _stem(cls, word: str) -> str:
        """Lightweight heuristic stemmer for plurals and common suffixes."""
        w = word.lower()
        for suffix in ("ation", "tions", "tion", "ing", "ies", "es", "ed", "s"):
            if len(w) > len(suffix) + 3 and w.endswith(suffix):
                return w[:-len(suffix)]
        return w

    @classmethod
    def verify_intent_consistency(
        cls,
        query: str,
        candidate_prompt: str,
        threshold: float = 0.50,
    ) -> tuple[bool, float]:
        """
        Calculates cross-attention/lexical alignment score on content terms.
        Returns: (is_valid: bool, confidence_score: float)
        """
        raw_q = set(re.findall(r"\w+", query.lower()))
        raw_c = set(re.findall(r"\w+", candidate_prompt.lower()))

        if not raw_q or not raw_c:
            return False, 0.0

        # Check for conflicting action verbs or polar operators
        q_actions = raw_q.intersection(cls.KEYWORD_NEGATIONS)
        c_actions = raw_c.intersection(cls.KEYWORD_NEGATIONS)

        if q_actions and c_actions and q_actions != c_actions:
            logger.debug(
                "Cross-encoder rejected candidate due to action contradiction: %s vs %s",
                q_actions,
                c_actions,
            )
            return False, 0.20

        # Filter stopwords
        q_content = {w for w in raw_q if w not in cls.STOPWORDS} or raw_q
        c_content = {w for w in raw_c if w not in cls.STOPWORDS} or raw_c

        # Stemmed content sets for inflection resilience
        q_stemmed = {cls._stem(w) for w in q_content}
        c_stemmed = {cls._stem(w) for w in c_content}

        intersection = q_stemmed.intersection(c_stemmed)
        union = q_stemmed.union(c_stemmed)
        jaccard = len(intersection) / len(union) if union else 0.0

        overlap_q = len(intersection) / len(q_stemmed) if q_stemmed else 0.0
        overlap_c = len(intersection) / len(c_stemmed) if c_stemmed else 0.0
        max_overlap = max(overlap_q, overlap_c)

        score = (jaccard * 0.4) + (max_overlap * 0.6)

        is_valid = score >= threshold or max_overlap >= 0.60
        return is_valid, round(score, 4)