import os
import re
import uuid
from typing import Dict, Optional


class W3CTraceContext:
    """
    Implements W3C Trace Context specification (RFC compliant)
    Format: version-trace_id-parent_id-trace_flags
    Example: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01
    """
    TRACEPARENT_REGEX = re.compile(r"^([0-9a-f]{2})-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$")

    def __init__(
        self,
        trace_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        sampled: bool = True,
    ):
        self.version = "00"
        self.trace_id = trace_id or uuid.uuid4().hex
        self.span_id = uuid.uuid4().hex[:16]
        self.parent_id = parent_id
        self.flags = "01" if sampled else "00"

    @classmethod
    def from_header(cls, traceparent_header: Optional[str]) -> "W3CTraceContext":
        if traceparent_header:
            match = cls.TRACEPARENT_REGEX.match(traceparent_header.strip())
            if match:
                version, trace_id, parent_id, flags = match.groups()
                ctx = cls(trace_id=trace_id, parent_id=parent_id, sampled=(flags == "01"))
                ctx.version = version
                return ctx
        return cls()

    def to_header(self) -> str:
        return f"{self.version}-{self.trace_id}-{self.span_id}-{self.flags}"

    def child_context(self) -> "W3CTraceContext":
        """Spawns downstream child span sharing same trace_id."""
        child = W3CTraceContext(
            trace_id=self.trace_id,
            parent_id=self.span_id,
            sampled=(self.flags == "01"),
        )
        return child