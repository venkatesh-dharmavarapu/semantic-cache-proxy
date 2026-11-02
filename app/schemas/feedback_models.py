from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class FeedbackSubmitRequest(BaseModel):
    completion_id: Optional[str] = Field(
        default=None, description="Completion ID returned in the chat completion response"
    )
    cache_key: Optional[str] = Field(
        default=None, description="Direct cache key (e.g., cache:llm:... or cache:exact:...)"
    )
    rating: Literal["thumbs_up", "thumbs_down", 1, -1] = Field(
        ..., description="Rating indicator (+1/thumbs_up for positive, -1/thumbs_down for negative)"
    )
    feedback_reason: Optional[str] = Field(
        default=None, description="Optional reason category (e.g., 'hallucination', 'outdated', 'unhelpful')"
    )
    user_comment: Optional[str] = Field(
        default=None, description="Optional free-form user comment"
    )


class FeedbackSubmitResponse(BaseModel):
    status: str
    message: str
    target_key: Optional[str]
    current_net_score: int
    evicted: bool


class FeedbackStatsResponse(BaseModel):
    total_feedback: int
    positive_count: int
    negative_count: int
    satisfaction_rate_percent: float
    auto_evictions_count: int
    recent_reasons: Dict[str, int]