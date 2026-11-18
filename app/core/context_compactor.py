import logging
import re
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class ContextCompactor:
    """
    Compresses and normalizes long conversational dialogue trees to stay
    within optimal token budgets while preserving prompt intent and recent context.
    """

    CONVERSATIONAL_FILLER = {
        r"^(hello|hi|hey|good morning|good afternoon|good evening)[,.]?\s*": "",
        r"^(thank you|thanks|thanks a lot)[,.]?\s*": "",
        r"^(please|could you please|can you please)\s+": "",
    }

    @classmethod
    def estimate_tokens(cls, text: str) -> int:
        """Lightweight heuristic: ~4 characters per token."""
        if not text:
            return 0
        return max(1, len(text.strip()) // 4)

    @classmethod
    def clean_text(cls, text: str) -> str:
        """Removes conversational filler and normalizes whitespace."""
        cleaned = text.strip()
        for pattern, replacement in cls.CONVERSATIONAL_FILLER.items():
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE).strip()
        # Collapse multiple spaces and excessive newlines
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned

    @classmethod
    def compact_messages(
        cls,
        messages: List[Dict[str, Any]],
        max_context_tokens: int = 2048,
        recent_anchor_turns: int = 4,
    ) -> Tuple[List[Dict[str, Any]], bool, int]:
        """
        Compacts dialogue history if total token footprint exceeds max_context_tokens.
        
        Returns:
            (compacted_messages, was_compacted, saved_tokens)
        """
        if not messages:
            return [], False, 0

        # Separate system messages and chat turns
        system_msgs = [m for m in messages if m.get("role") == "system"]
        chat_turns = [m for m in messages if m.get("role") != "system"]

        if not chat_turns:
            return messages, False, 0

        initial_tokens = sum(cls.estimate_tokens(m.get("content", "")) for m in messages)
        if initial_tokens <= max_context_tokens:
            return messages, False, 0

        # Preserve recent anchor turns
        if len(chat_turns) <= recent_anchor_turns:
            # Clean recent turns without dropping
            compacted_chat = [
                {"role": m.get("role"), "content": cls.clean_text(m.get("content", ""))}
                for m in chat_turns
            ]
            final_msgs = system_msgs + compacted_chat
            final_tokens = sum(cls.estimate_tokens(m.get("content", "")) for m in final_msgs)
            return final_msgs, True, max(0, initial_tokens - final_tokens)

        # Partition into older history vs recent anchor
        split_idx = len(chat_turns) - recent_anchor_turns
        older_turns = chat_turns[:split_idx]
        recent_turns = chat_turns[split_idx:]

        # Compact older turns: extract key sentences or filter low-information messages
        compacted_older = []
        for turn in older_turns:
            content = cls.clean_text(turn.get("content", ""))
            # If message is short or boilerplate, retain trimmed content; if long, truncate to first 250 chars
            if len(content) > 300:
                content = content[:300].rsplit(" ", 1)[0] + "... [context truncated]"
            compacted_older.append({"role": turn.get("role"), "content": content})

        compacted_recent = [
            {"role": m.get("role"), "content": cls.clean_text(m.get("content", ""))}
            for m in recent_turns
        ]

        compacted_msgs = system_msgs + compacted_older + compacted_recent
        final_tokens = sum(cls.estimate_tokens(m.get("content", "")) for m in compacted_msgs)
        tokens_saved = max(0, initial_tokens - final_tokens)

        logger.info(
            "Context compacted: %d -> %d estimated tokens (saved ~%d)",
            initial_tokens,
            final_tokens,
            tokens_saved,
        )
        return compacted_msgs, True, tokens_saved