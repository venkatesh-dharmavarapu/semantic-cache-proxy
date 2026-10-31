import json
import logging
from typing import Any, Dict, List
from app.core.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class ClusterTopologyInspector:
    def __init__(self, vector_store: VectorStoreManager):
        self.store = vector_store
        self.store.connect()

    def get_topology_map(self, max_clusters: int = 50) -> Dict[str, Any]:
        """Inspects all indexed centroids and computes cluster density and distribution."""
        client = self.store.client
        if not client:
            return {"status": "error", "message": "Database disconnected"}

        clusters: List[Dict[str, Any]] = []
        total_aliases = 0
        total_hits = 0

        for key in client.scan_iter("cache:llm:*"):
            if len(clusters) >= max_clusters:
                break

            fields = client.hmget(key, ["prompt", "model", "hit_count", "alias_count", "created_at"])
            if not fields or not fields[0]:
                continue

            prompt_text = fields[0].decode("utf-8") if fields[0] else ""
            model_tag = fields[1].decode("utf-8") if fields[1] else "unknown"
            hit_cnt = int(fields[2]) if fields[2] else 0
            alias_cnt = int(fields[3]) if fields[3] else 0
            created = float(fields[4]) if fields[4] else 0.0

            total_hits += hit_cnt
            total_aliases += alias_cnt

            clusters.append({
                "id": key.decode("utf-8") if isinstance(key, bytes) else key,
                "centroid_prompt": (prompt_text[:60] + "...") if len(prompt_text) > 60 else prompt_text,
                "model": model_tag,
                "hits": hit_cnt,
                "aliases": alias_cnt,
                "created_at": created,
                # Density score: weighted combination of re-use frequency and alias consolidation
                "density_score": round((hit_cnt * 1.0) + (alias_cnt * 1.5), 2),
            })

        # Sort clusters by highest density score
        clusters.sort(key=lambda c: c["density_score"], reverse=True)

        return {
            "status": "success",
            "cluster_count": len(clusters),
            "aggregate_hits": total_hits,
            "aggregate_aliases": total_aliases,
            "top_clusters": clusters,
        }