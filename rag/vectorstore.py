from typing import Optional

_chroma_client = None
_embedding_model = None


def _get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        _chroma_client = chromadb.PersistentClient(path="./chroma_db")
    return _chroma_client


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = _get_embedding_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


def add_chunks(
    collection_name: str,
    chunks: list[str],
    metadatas: list[dict],
    ids: list[str],
) -> None:
    client = _get_chroma_client()
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    embeddings = embed_texts(chunks)
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )


def query_collection(
    collection_name: str,
    query_text: str,
    n_results: int = 5,
    where: Optional[dict] = None,
) -> dict:
    client = _get_chroma_client()
    collection = client.get_collection(name=collection_name)
    query_embedding = embed_texts([query_text])
    kwargs = {
        "query_embeddings": query_embedding,
        "n_results": min(n_results, collection.count()),
    }
    if where:
        kwargs["where"] = where
    results = collection.query(**kwargs)
    return results


def list_collections() -> list[dict]:
    client = _get_chroma_client()
    collections = client.list_collections()
    result = []
    for col in collections:
        collection = client.get_collection(name=col.name)
        result.append({
            "name": col.name,
            "count": collection.count(),
            "metadata": col.metadata,
        })
    return result


def get_collection_info(name: str) -> dict:
    client = _get_chroma_client()
    collection = client.get_collection(name=name)
    return {
        "name": name,
        "count": collection.count(),
        "metadata": collection.metadata,
    }


def delete_collection(name: str) -> None:
    client = _get_chroma_client()
    client.delete_collection(name=name)
