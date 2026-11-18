import hashlib
import json
from typing import Any, Dict, List, Tuple
from app.core.context_compactor import ContextCompactor


class ContextNormalizer:
    DEFAULT_WINDOW_TURNS = 3

    @classmethod
    def partition_context(
        cls,
        messages: List[Dict[str, Any]],
        window_size: int = DEFAULT_WINDOW_TURNS,
        compact_threshold_tokens: int = 2048,
    ) -> Tuple[List[Dict[str, Any]], str, str, bool, int]:
        """
        Partitions message history into:
        1. local_window: The last N prior messages.
        2. history_digest: SHA-256 digest of messages preceding the local window, or 'stateless'/'root'.
        3. current_prompt: The final user message content.
        4. was_compacted: Whether context compaction was triggered.
        5. tokens_saved: Estimated token reduction count.
        """
        if not messages:
            return [], "stateless", "", False, 0

        # 1. Apply compaction if context footprint is large
        compacted_messages, was_compacted, saved_tokens = ContextCompactor.compact_messages(
            messages, max_context_tokens=compact_threshold_tokens
        )

        chat_turns = [m for m in compacted_messages if m.get("role") != "system"]
        if not chat_turns:
            return [], "stateless", "", was_compacted, saved_tokens

        current_prompt = chat_turns[-1].get("content", "").strip()
        prior_turns = chat_turns[:-1]

        if not prior_turns:
            return [], "stateless", current_prompt, was_compacted, saved_tokens

        if len(prior_turns) <= window_size:
            local_window = prior_turns
            history_digest = "root"
        else:
            split_idx = len(prior_turns) - window_size
            older_history = prior_turns[:split_idx]
            local_window = prior_turns[split_idx:]
            raw_history_str = json.dumps(older_history, sort_keys=True)
            history_digest = hashlib.sha256(raw_history_str.encode("utf-8")).hexdigest()[:16]

        return local_window, history_digest, current_prompt, was_compacted, saved_tokens

    @classmethod
    def compute_turn_context_hash(
        cls,
        local_window: List[Dict[str, Any]],
        history_digest: str,
    ) -> str:
        if not local_window and history_digest in ("stateless", "empty"):
            return "stateless"

        normalized_window = [
            {"role": m.get("role"), "content": m.get("content", "").strip()}
            for m in local_window
        ]
        composite = f"{history_digest}::{json.dumps(normalized_window, sort_keys=True)}"
        return hashlib.sha256(composite.encode("utf-8")).hexdigest()