import hashlib
import json
from typing import Any, Dict, List, Tuple


class ContextNormalizer:
    DEFAULT_WINDOW_TURNS = 3

    @classmethod
    def partition_context(
        cls,
        messages: List[Dict[str, Any]],
        window_size: int = DEFAULT_WINDOW_TURNS,
    ) -> Tuple[List[Dict[str, Any]], str, str]:
        """
        Partitions message history into:
        1. local_window: The last N prior messages.
        2. history_digest: SHA-256 digest of messages preceding the local window, or 'stateless'/'root'.
        3. current_prompt: The final user message content.
        """
        if not messages:
            return [], "stateless", ""

        chat_turns = [m for m in messages if m.get("role") != "system"]
        if not chat_turns:
            return [], "stateless", ""

        current_prompt = chat_turns[-1].get("content", "").strip()
        prior_turns = chat_turns[:-1]

        if not prior_turns:
            return [], "stateless", current_prompt

        if len(prior_turns) <= window_size:
            local_window = prior_turns
            history_digest = "root"
        else:
            split_idx = len(prior_turns) - window_size
            older_history = prior_turns[:split_idx]
            local_window = prior_turns[split_idx:]
            raw_history_str = json.dumps(older_history, sort_keys=True)
            history_digest = hashlib.sha256(raw_history_str.encode("utf-8")).hexdigest()[:16]

        return local_window, history_digest, current_prompt

    @classmethod
    def compute_turn_context_hash(
        cls,
        local_window: List[Dict[str, Any]],
        history_digest: str,
    ) -> str:
        """
        Computes a deterministic hash for multi-turn local dialogue state.
        For standalone single-turn messages (empty window and stateless history),
        it returns 'stateless' to remain compatible with pre-warmed & migrated entries.
        """
        if not local_window and history_digest in ("stateless", "empty"):
            return "stateless"

        normalized_window = [
            {"role": m.get("role"), "content": m.get("content", "").strip()}
            for m in local_window
        ]
        composite = f"{history_digest}::{json.dumps(normalized_window, sort_keys=True)}"
        return hashlib.sha256(composite.encode("utf-8")).hexdigest()