"""Tests for rag/router.py endpoints."""

from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from tests.conftest import make_chat_response


@pytest.fixture()
def client():
    from api import app
    return TestClient(app)


def _mock_httpx_for_rag(mock_httpx_cls, chat_response):
    """Set up httpx AsyncClient mock so the RAG endpoint can call the LLM."""
    mock_client = AsyncMock()
    mock_httpx_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
    mock_httpx_cls.return_value.__aexit__ = AsyncMock(return_value=False)
    mock_client.post.return_value = chat_response
    return mock_client


# ---------------------------------------------------------------------------
# POST /documents  (ingest)
# ---------------------------------------------------------------------------


class TestIngestDocuments:
    @patch("rag.router.vectorstore")
    def test_ingest_single_document(self, mock_vs, client):
        mock_vs.add_chunks = MagicMock()
        resp = client.post(
            "/documents",
            json={
                "documents": [
                    {"content": "This is a test document about AI."}
                ],
                "collection": "test-col",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["documents_processed"] == 1
        assert data["chunks_created"] >= 1
        assert data["collection"] == "test-col"
        assert len(data["chunk_ids"]) >= 1
        mock_vs.add_chunks.assert_called_once()

    @patch("rag.router.vectorstore")
    def test_ingest_multiple_documents(self, mock_vs, client):
        mock_vs.add_chunks = MagicMock()
        resp = client.post(
            "/documents",
            json={
                "documents": [
                    {"content": "Document one."},
                    {"content": "Document two."},
                    {"content": "Document three."},
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["documents_processed"] == 3
        assert data["chunks_created"] >= 3

    @patch("rag.router.vectorstore")
    def test_ingest_with_metadata_and_id(self, mock_vs, client):
        mock_vs.add_chunks = MagicMock()
        resp = client.post(
            "/documents",
            json={
                "documents": [
                    {
                        "content": "Test content.",
                        "metadata": {"source": "test", "topic": "ai"},
                        "id": "custom-id-1",
                    }
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["chunk_ids"][0].startswith("custom-id-1")
        # Verify metadata was passed through
        call_args = mock_vs.add_chunks.call_args
        metadatas = call_args[0][2]  # 3rd positional arg
        assert metadatas[0]["source"] == "test"
        assert metadatas[0]["topic"] == "ai"

    @patch("rag.router.vectorstore")
    def test_ingest_empty_content(self, mock_vs, client):
        resp = client.post(
            "/documents",
            json={
                "documents": [{"content": "   "}],
            },
        )
        assert resp.status_code == 400

    @patch("rag.router.vectorstore")
    def test_ingest_custom_chunk_size(self, mock_vs, client):
        mock_vs.add_chunks = MagicMock()
        # Use sentences so the chunker can split on sentence boundaries
        long_text = ". ".join([f"Sentence number {i} with some extra words" for i in range(50)])
        resp = client.post(
            "/documents",
            json={
                "documents": [{"content": long_text}],
                "chunk_size": 200,
                "chunk_overlap": 20,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["chunks_created"] > 1

    def test_ingest_missing_documents_field(self, client):
        resp = client.post("/documents", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /search
# ---------------------------------------------------------------------------


class TestSearch:
    @patch("rag.router.vectorstore")
    def test_basic_search(self, mock_vs, client):
        mock_vs.query_collection.return_value = {
            "ids": [["chunk_1", "chunk_2"]],
            "documents": [["First result content.", "Second result content."]],
            "metadatas": [[{"source": "a"}, {"source": "b"}]],
            "distances": [[0.1, 0.3]],
        }
        resp = client.post(
            "/search",
            json={"query": "test query", "collection": "my-col"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "test query"
        assert data["collection"] == "my-col"
        assert len(data["results"]) == 2
        assert data["results"][0]["chunk_id"] == "chunk_1"
        assert data["results"][0]["distance"] == 0.1

    @patch("rag.router.vectorstore")
    def test_search_empty_results(self, mock_vs, client):
        mock_vs.query_collection.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        resp = client.post("/search", json={"query": "nonexistent"})
        assert resp.status_code == 200
        assert resp.json()["results"] == []

    @patch("rag.router.vectorstore")
    def test_search_with_filter(self, mock_vs, client):
        mock_vs.query_collection.return_value = {
            "ids": [["chunk_1"]],
            "documents": [["Filtered result."]],
            "metadatas": [[{"source": "a"}]],
            "distances": [[0.2]],
        }
        resp = client.post(
            "/search",
            json={
                "query": "test",
                "where": {"source": "a"},
                "n_results": 10,
            },
        )
        assert resp.status_code == 200
        mock_vs.query_collection.assert_called_once_with("default", "test", 10, {"source": "a"})

    @patch("rag.router.vectorstore")
    def test_search_collection_not_found(self, mock_vs, client):
        mock_vs.query_collection.side_effect = Exception("Collection not found")
        resp = client.post(
            "/search",
            json={"query": "test", "collection": "nonexistent"},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /rag
# ---------------------------------------------------------------------------


class TestRAG:
    @patch("rag.router.vectorstore")
    @patch("api.httpx.AsyncClient")
    def test_basic_rag(self, mock_httpx_cls, mock_vs, client):
        mock_vs.query_collection.return_value = {
            "ids": [["c1"]],
            "documents": [["The O-1A visa requires extraordinary ability."]],
            "metadatas": [[{"source": "guide"}]],
            "distances": [[0.15]],
        }
        _mock_httpx_for_rag(
            mock_httpx_cls,
            make_chat_response(
                content="Based on the context, the O-1A visa requires extraordinary ability.",
                completion_tokens=20,
                prompt_tokens=50,
            ),
        )

        resp = client.post(
            "/rag",
            json={
                "query": "What does O-1A require?",
                "collection": "visa-docs",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "What does O-1A require?"
        assert "O-1A" in data["answer"]
        assert data["sources"] is not None
        assert len(data["sources"]) == 1
        assert data["sources"][0]["chunk_id"] == "c1"

    @patch("rag.router.vectorstore")
    @patch("api.httpx.AsyncClient")
    def test_rag_without_sources(self, mock_httpx_cls, mock_vs, client):
        mock_vs.query_collection.return_value = {
            "ids": [["c1"]],
            "documents": [["Some context."]],
            "metadatas": [[{}]],
            "distances": [[0.2]],
        }
        _mock_httpx_for_rag(mock_httpx_cls, make_chat_response(content="Answer."))

        resp = client.post(
            "/rag",
            json={
                "query": "test",
                "include_sources": False,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["sources"] is None

    @patch("rag.router.vectorstore")
    def test_rag_collection_not_found(self, mock_vs, client):
        mock_vs.query_collection.side_effect = Exception("not found")
        resp = client.post(
            "/rag",
            json={"query": "test", "collection": "missing"},
        )
        assert resp.status_code == 404

    @patch("rag.router.vectorstore")
    @patch("api.httpx.AsyncClient")
    def test_rag_custom_system_prompt(self, mock_httpx_cls, mock_vs, client):
        mock_vs.query_collection.return_value = {
            "ids": [["c1"]],
            "documents": [["Context text."]],
            "metadatas": [[{}]],
            "distances": [[0.1]],
        }
        mock_client = _mock_httpx_for_rag(
            mock_httpx_cls, make_chat_response(content="Custom.")
        )

        resp = client.post(
            "/rag",
            json={
                "query": "test",
                "system_prompt": "Answer like a pirate.",
                "temperature": 0.9,
                "max_tokens": 256,
            },
        )
        assert resp.status_code == 200
        # Verify the custom system prompt was sent to the LLM
        call_payload = mock_client.post.call_args[1]["json"]
        assert call_payload["messages"][0]["content"] == "Answer like a pirate."


# ---------------------------------------------------------------------------
# GET /collections
# ---------------------------------------------------------------------------


class TestCollections:
    @patch("rag.router.vectorstore")
    def test_list_collections(self, mock_vs, client):
        mock_vs.list_collections.return_value = [
            {"name": "col-a", "count": 10, "metadata": None},
            {"name": "col-b", "count": 5, "metadata": {"hnsw:space": "cosine"}},
        ]
        resp = client.get("/collections")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["collections"]) == 2
        assert data["collections"][0]["name"] == "col-a"
        assert data["collections"][0]["count"] == 10

    @patch("rag.router.vectorstore")
    def test_list_collections_empty(self, mock_vs, client):
        mock_vs.list_collections.return_value = []
        resp = client.get("/collections")
        assert resp.status_code == 200
        assert resp.json()["collections"] == []

    @patch("rag.router.vectorstore")
    def test_get_collection(self, mock_vs, client):
        mock_vs.get_collection_info.return_value = {
            "name": "test-col",
            "count": 42,
            "metadata": {"hnsw:space": "cosine"},
        }
        resp = client.get("/collections/test-col")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "test-col"
        assert data["count"] == 42

    @patch("rag.router.vectorstore")
    def test_get_collection_not_found(self, mock_vs, client):
        mock_vs.get_collection_info.side_effect = Exception("not found")
        resp = client.get("/collections/nonexistent")
        assert resp.status_code == 404

    @patch("rag.router.vectorstore")
    def test_delete_collection(self, mock_vs, client):
        mock_vs.delete_collection = MagicMock()
        resp = client.delete("/collections/test-col")
        assert resp.status_code == 200
        mock_vs.delete_collection.assert_called_once_with("test-col")

    @patch("rag.router.vectorstore")
    def test_delete_collection_not_found(self, mock_vs, client):
        mock_vs.delete_collection.side_effect = Exception("not found")
        resp = client.delete("/collections/nonexistent")
        assert resp.status_code == 404
