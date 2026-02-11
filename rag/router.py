import uuid

from fastapi import APIRouter, HTTPException

from .models import (
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    RAGRequest,
    RAGResponse,
    CollectionInfo,
    CollectionListResponse,
)
from .chunking import chunk_text
from . import vectorstore

router = APIRouter()

RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
Answer ONLY based on the provided context. If the context doesn't contain enough information, say so."""


@router.post("/documents", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest):
    """Ingest documents: chunk, embed, and store in vector database."""
    all_chunks: list[str] = []
    all_metadatas: list[dict] = []
    all_ids: list[str] = []

    for i, doc in enumerate(request.documents):
        doc_id = doc.id or str(uuid.uuid4())
        chunks = chunk_text(doc.content, request.chunk_size, request.chunk_overlap)

        for j, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{j}"
            metadata = {"document_id": doc_id, "chunk_index": j}
            if doc.metadata:
                metadata.update(doc.metadata)
            all_chunks.append(chunk)
            all_metadatas.append(metadata)
            all_ids.append(chunk_id)

    if not all_chunks:
        raise HTTPException(status_code=400, detail="No content to ingest")

    vectorstore.add_chunks(request.collection, all_chunks, all_metadatas, all_ids)

    return IngestResponse(
        documents_processed=len(request.documents),
        chunks_created=len(all_chunks),
        chunk_ids=all_ids,
        collection=request.collection,
    )


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Semantic search across stored documents."""
    try:
        results = vectorstore.query_collection(
            request.collection,
            request.query,
            request.n_results,
            request.where,
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Collection error: {str(e)}")

    search_results = []
    if results["ids"] and results["ids"][0]:
        for i in range(len(results["ids"][0])):
            search_results.append(SearchResult(
                chunk_id=results["ids"][0][i],
                content=results["documents"][0][i],
                metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                distance=results["distances"][0][i] if results["distances"] else 0.0,
            ))

    return SearchResponse(
        query=request.query,
        collection=request.collection,
        results=search_results,
    )


@router.post("/rag", response_model=RAGResponse)
async def rag(request: RAGRequest):
    """Full RAG: retrieve relevant chunks then generate answer with LLM."""
    try:
        results = vectorstore.query_collection(
            request.collection,
            request.query,
            request.n_results,
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Collection error: {str(e)}")

    context_parts = []
    sources = []
    if results["ids"] and results["ids"][0]:
        for i in range(len(results["ids"][0])):
            content = results["documents"][0][i]
            context_parts.append(content)
            sources.append(SearchResult(
                chunk_id=results["ids"][0][i],
                content=content,
                metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                distance=results["distances"][0][i] if results["distances"] else 0.0,
            ))

    context_block = "\n\n---\n\n".join(context_parts)
    system_prompt = request.system_prompt or RAG_SYSTEM_PROMPT

    user_message = f"Context:\n{context_block}\n\nQuestion: {request.query}"

    from api import create_chat_completion, ChatRequest, ChatMessage

    chat_request = ChatRequest(
        messages=[
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_message),
        ],
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        continue_until_done=True,
    )
    result = await create_chat_completion(chat_request)

    return RAGResponse(
        query=request.query,
        answer=result.message.content,
        sources=sources if request.include_sources else None,
        tokens_generated=result.tokens_generated,
        tokens_prompt=result.tokens_prompt,
    )


@router.get("/collections", response_model=CollectionListResponse)
async def list_collections():
    """List all collections."""
    collections = vectorstore.list_collections()
    return CollectionListResponse(
        collections=[CollectionInfo(**c) for c in collections]
    )


@router.get("/collections/{name}", response_model=CollectionInfo)
async def get_collection(name: str):
    """Get collection info."""
    try:
        info = vectorstore.get_collection_info(name)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Collection not found: {str(e)}")
    return CollectionInfo(**info)


@router.delete("/collections/{name}")
async def delete_collection(name: str):
    """Delete a collection."""
    try:
        vectorstore.delete_collection(name)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Collection not found: {str(e)}")
    return {"message": f"Collection '{name}' deleted"}
