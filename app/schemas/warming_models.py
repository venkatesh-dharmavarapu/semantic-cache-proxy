from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CacheWarmItem(BaseModel):
    prompt: str
    response_content: str
    model: str = "llama3.2:1b"
    system_prompt: Optional[str] = ""
    temperature: Optional[float] = 0.0
    context_hash: Optional[str] = "stateless"


class CacheWarmRequest(BaseModel):
    items: List[CacheWarmItem] = Field(..., min_length=1)


class CacheWarmResponse(BaseModel):
    total_submitted: int
    successfully_warmed: int
    failures: int