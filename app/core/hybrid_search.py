from typing import Any, Dict, List, Optional, Tuple


class HybridSearchFusion:
    RRF_K = 60

    @classmethod
    def compute_rrf_scores(
        cls,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Combines ranked lists from dense vector search and sparse full-text search
        using weighted Reciprocal Rank Fusion (RRF).
        """
        scores: Dict[str, float] = {}
        records: Dict[str, Dict[str, Any]] = {}

        # 1. Score dense candidates by rank position
        for rank, item in enumerate(dense_results, start=1):
            key = item.get("id") or item.get("key")
            if not key:
                continue
            records[key] = item
            rrf_val = dense_weight * (1.0 / (cls.RRF_K + rank))
            scores[key] = scores.get(key, 0.0) + rrf_val

        # 2. Score sparse BM25 candidates by rank position
        for rank, item in enumerate(sparse_results, start=1):
            key = item.get("id") or item.get("key")
            if not key:
                continue
            records[key] = item
            rrf_val = sparse_weight * (1.0 / (cls.RRF_K + rank))
            scores[key] = scores.get(key, 0.0) + rrf_val

        # Sort aggregated results descending by RRF score
        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        return [(records[k], scores[k]) for k in sorted_keys]

    @staticmethod
    def sanitize_text_query(query: str) -> str:
        """Sanitizes raw prompt text for safe RediSearch full-text syntax."""
        import re
        tokens = re.findall(r"\w+", query)
        if not tokens:
            return "*"
        # Join words with pipe (|) for OR matching or space for AND
        return " | ".join(tokens[:10])