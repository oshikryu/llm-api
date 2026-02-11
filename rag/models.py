from pydantic import BaseModel, Field
from typing import Optional


class DocumentInput(BaseModel):
    content: str = Field(..., description="Document text content")
    metadata: Optional[dict] = Field(default=None, description="Optional metadata")
    id: Optional[str] = Field(default=None, description="Optional document ID")


class IngestRequest(BaseModel):
    documents: list[DocumentInput] = Field(..., description="Documents to ingest")
    collection: str = Field(default="default", description="Collection name")
    chunk_size: int = Field(default=500, ge=50, le=5000, description="Target chunk size in characters")
    chunk_overlap: int = Field(default=50, ge=0, le=500, description="Overlap between chunks")


class IngestResponse(BaseModel):
    documents_processed: int
    chunks_created: int
    chunk_ids: list[str]
    collection: str


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    collection: str = Field(default="default", description="Collection to search")
    n_results: int = Field(default=5, ge=1, le=100, description="Number of results")
    where: Optional[dict] = Field(default=None, description="Metadata filter")


class SearchResult(BaseModel):
    chunk_id: str
    content: str
    metadata: dict
    distance: float


class SearchResponse(BaseModel):
    query: str
    collection: str
    results: list[SearchResult]


class RAGRequest(BaseModel):
    query: str = Field(..., description="Question to answer")
    collection: str = Field(default="default", description="Collection to search")
    n_results: int = Field(default=5, ge=1, le=100, description="Number of context chunks")
    system_prompt: Optional[str] = Field(default=None, description="Override default RAG system prompt")
    max_tokens: int = Field(default=1024, ge=1, le=4096, description="Max tokens for LLM response")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0, description="Sampling temperature")
    include_sources: bool = Field(default=True, description="Include source chunks in response")


class RAGResponse(BaseModel):
    query: str
    answer: str
    sources: Optional[list[SearchResult]] = None
    tokens_generated: int
    tokens_prompt: int


class CollectionInfo(BaseModel):
    name: str
    count: int
    metadata: Optional[dict] = None


class CollectionListResponse(BaseModel):
    collections: list[CollectionInfo]
