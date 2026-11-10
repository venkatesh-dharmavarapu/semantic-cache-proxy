import os
import time
import uuid
from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class InvalidationScope(str, Enum):
    KEY_EVICTED = "KEY_EVICTED"
    MODEL_PURGED = "MODEL_PURGED"
    TENANT_PURGED = "TENANT_PURGED"
    THRESHOLD_UPDATED = "THRESHOLD_UPDATED"
    FLUSH_ALL = "FLUSH_ALL"


# Unique instance ID generated once per application process life cycle
NODE_INSTANCE_ID = f"node-{os.getpid()}-{uuid.uuid4().hex[:8]}"


class CacheInvalidationEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    origin_node: str = Field(default=NODE_INSTANCE_ID)
    scope: InvalidationScope
    target: str = Field(..., description="Target key, model name, tenant ID, or scope identifier")
    timestamp: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = Field(default_factory=dict)