from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime


class ChunkStrategy(str, Enum):
    fixed = "fixed"
    sentence = "sentence"
    paragraph = "paragraph"


class SearchStrategy(str, Enum):
    vector = "vector"
    hybrid = "hybrid"


#  Collections ───────────────────────────────────────────────────────────────

class CollectionCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None


class CollectionResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    created_at: datetime


#  Documents ─────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    text: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1)
    collection: str = Field(default="default")
    chunk_strategy: ChunkStrategy = ChunkStrategy.sentence
    chunk_size: Optional[int] = Field(default=None, ge=64, le=2048)
    chunk_overlap: Optional[int] = Field(default=None, ge=0, le=256)
    metadata: dict = Field(default_factory=dict)


class IngestResponse(BaseModel):
    document_id: str
    collection: str
    title: str
    chunk_count: int
    char_count: int
    chunk_strategy: str


class DocumentResponse(BaseModel):
    id: str
    collection_id: str
    title: str
    source: str
    char_count: int
    chunk_count: int
    chunk_strategy: str
    status: str
    metadata: dict
    created_at: datetime


#  Search ────────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    collection: str = Field(default="default")
    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    rerank_k: Optional[int] = Field(default=None, ge=1, le=20)
    strategy: SearchStrategy = SearchStrategy.hybrid
    source_filter: Optional[str] = None


class ChunkResult(BaseModel):
    chunk_id: str
    document_id: str
    document_title: str
    source: str
    content: str
    chunk_index: int
    vector_score: float
    bm25_score: Optional[float] = None
    hybrid_score: Optional[float] = None
    rerank_score: Optional[float] = None
    final_score: float


class SearchResponse(BaseModel):
    query: str
    collection: str
    strategy: str
    results: list[ChunkResult]
    result_count: int
    latency_ms: int
