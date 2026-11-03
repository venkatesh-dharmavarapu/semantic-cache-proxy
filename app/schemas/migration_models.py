from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class IngestionRecord(BaseModel):
    prompt: str = Field(..., min_length=1)
    response: Any = Field(..., description="Response payload as dictionary or completion text")
    model: str = "llama3.2:1b"
    system_prompt: Optional[str] = ""
    temperature: Optional[float] = 0.0
    context_hash: Optional[str] = "stateless"


class BatchIngestionRequest(BaseModel):
    records: List[IngestionRecord] = Field(..., min_length=1)
    chunk_size: int = Field(default=50, ge=1, le=500)


class BatchIngestionResponse(BaseModel):
    status: str
    total_received: int
    successfully_indexed: int
    failures: int


class SchemaMigrationRequest(BaseModel):
    target_index_name: str = Field(..., min_length=3)


class SchemaMigrationResponse(BaseModel):
    status: str
    current_index: str
    target_index: str
    total_keys_ready_for_cutover: int