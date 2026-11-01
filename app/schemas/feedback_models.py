from typing import Literal, Optional
from pydantic import BaseModel, Field


class CacheFeedbackRequest(BaseModel):
    cache_key: str = Field(..., description="Target Redis cache key (e.g., cache:llm:...)")
    rating: Literal["positive", "negative", "flag"]
    reason: Optional[str] = Field(default=None, description="Optional note or flag reason")
    threshold: Optional[int] = Field(default=2, ge=1, le=10, description="Negative threshold before eviction")


class CacheFeedbackResponse(BaseModel):
    status: str
    cache_key: str
    feedback_type: str
    current_negative_count: int
    evicted: bool