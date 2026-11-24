import hashlib
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SchemaHasher:
    @staticmethod
    def hash_response_format(
        response_format: Optional[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Computes a deterministic hash for structured JSON schemas or function calling definitions.
        Returns 'unstructured' if no schema contract is enforced.
        """
        if not response_format and not tools:
            return "unstructured"

        contract = {
            "response_format": response_format or {},
            "tools": tools or [],
        }
        canonical = json.dumps(contract, sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class DifferentialEngine:
    """
    Computes and applies JSON deltas for structured responses to minimize
    storage bloat while preserving exact JSON contracts.
    """

    @staticmethod
    def compute_patch(base_obj: Dict[str, Any], target_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Calculates RFC 6902-style JSON patch operations (replace, add, remove).
        """
        patch: List[Dict[str, Any]] = []

        # Key-level differential computation
        all_keys = set(base_obj.keys()).union(target_obj.keys())
        for k in all_keys:
            if k not in base_obj:
                patch.append({"op": "add", "path": f"/{k}", "value": target_obj[k]})
            elif k not in target_obj:
                patch.append({"op": "remove", "path": f"/{k}"})
            elif base_obj[k] != target_obj[k]:
                patch.append({"op": "replace", "path": f"/{k}", "value": target_obj[k]})

        return patch

    @staticmethod
    def apply_patch(base_obj: Dict[str, Any], patch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Reconstructs the target object by applying JSON patch operations to base.
        """
        result = dict(base_obj)
        for op in patch:
            path = op.get("path", "").lstrip("/")
            action = op.get("op")
            if action in ("add", "replace"):
                result[path] = op.get("value")
            elif action == "remove":
                result.pop(path, None)
        return result