import json
from typing import Dict, Any, Optional, Tuple
import numpy as np
from redisvl.query import VectorQuery
from redisvl.query.filter import Tag
from app.core.vector_store import VectorStoreManager

class CacheReader:
    def __init__(self, vector_store: VectorStoreManager):
        self.store = vector_store

    def query_similarity(
        self,
        vector: list[float],
        model: str,
        system_prompt_hash: str,
        params_hash: str,
        similarity_threshold: float = 0.95,
    ) -> Optional[Tuple[Dict[str, Any], float, str]]:
        """
        Searches nearest neighbor vector with cosine distance.
        Redis cosine distance: similarity = 1 - vector_distance.
        
        Returns:
            (cached_response_dict, similarity_score, key) if hit, else None.
        """
        # Strict partition matching on model, system prompt, and parameters
        filter_expr = (
            (Tag("model") == model) &
            (Tag("system_prompt_hash") == system_prompt_hash) &
            (Tag("params_hash") == params_hash)
        )
        
        vector_bytes = np.array(vector, dtype=np.float32).tobytes()
        
        v_query = VectorQuery(
            vector=vector_bytes,
            vector_field_name="prompt_vector",
            return_fields=["prompt", "response_json", "vector_distance", "hit_count"],
            filter_expression=filter_expr,
            num_results=1,
            return_score=True
        )
        
        results = self.store.index.query(v_query)
        
        if not results:
            return None
        
        match = results[0]
        # RedisVL returns vector_distance. For cosine distance:
        distance = float(match.get("vector_distance", 1.0))
        similarity = 1.0 - distance
        
        if similarity >= similarity_threshold:
            # Increment hit counter for telemetry
            key = match["id"]
            self.store.client.hincrby(key, "hit_count", 1)
            response_json = json.loads(match["response_json"])
            return response_json, similarity, key
            
        return None