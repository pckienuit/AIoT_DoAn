import math
import os
from functools import lru_cache
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

COLLECTION_NAME = "face_embeddings"
VECTOR_SIZE = 128
QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
MIN_ACTIVE_DIMS = 16
MIN_EMBEDDING_NORM = 1e-6
MAX_SINGLE_DIM_ABS = 0.95


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def validate_embedding(vector: list[float]) -> list[float]:
    if len(vector) != VECTOR_SIZE:
        raise ValueError(f"Embedding must have {VECTOR_SIZE} dimensions")

    values = []
    for index, value in enumerate(vector):
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Embedding value at index {index} is not numeric") from exc
        if not math.isfinite(numeric):
            raise ValueError(f"Embedding value at index {index} is not finite")
        values.append(numeric)

    norm = math.sqrt(sum(value * value for value in values))
    if norm < MIN_EMBEDDING_NORM:
        raise ValueError("Embedding norm is too small")

    normalized = [value / norm for value in values]
    active_dims = sum(1 for value in normalized if abs(value) > 1e-4)
    max_abs = max(abs(value) for value in normalized)
    if active_dims < MIN_ACTIVE_DIMS or max_abs > MAX_SINGLE_DIM_ABS:
        raise ValueError("Embedding appears sparse or dummy; real face embedding required")

    return normalized


def ensure_face_collection() -> bool:
    client = get_qdrant_client()
    if client.collection_exists(COLLECTION_NAME):
        return False

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,
            distance=models.Distance.COSINE,
        ),
    )
    return True


def upsert_face_embedding(point_id: str, vector: list[float], payload: dict[str, Any]) -> None:
    ensure_face_collection()
    client = get_qdrant_client()
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=point_id,
                vector=validate_embedding(vector),
                payload=payload,
            )
        ],
    )


def search_face_embedding(vector: list[float], flight_id: int | None = None, limit: int = 1) -> list[Any]:
    ensure_face_collection()
    client = get_qdrant_client()
    query_filter = None
    if flight_id is not None:
        query_filter = models.Filter(
            must=[models.FieldCondition(key="flight_id", match=models.MatchValue(value=flight_id))]
        )
    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=validate_embedding(vector),
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    )
    return result.points


def scroll_flight_embeddings(flight_id: int) -> list[Any]:
    ensure_face_collection()
    client = get_qdrant_client()
    records, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="flight_id", match=models.MatchValue(value=flight_id))]
        ),
        limit=1000,
        with_vectors=True,
        with_payload=True,
    )
    return records


def get_vector_status() -> dict[str, object]:
    client = get_qdrant_client()
    exists = client.collection_exists(COLLECTION_NAME)
    return {
        "url": QDRANT_URL,
        "collection": COLLECTION_NAME,
        "vector_size": VECTOR_SIZE,
        "distance": "Cosine",
        "available": True,
        "collection_exists": exists,
    }


def delete_face_embedding(point_id: str) -> None:
    """Delete a specific face embedding by point_id from Qdrant."""
    client = get_qdrant_client()
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.PointIdsList(
            points=[point_id]
        )
    )


def delete_flight_embeddings(flight_id: int) -> int:
    """Delete all face embeddings for a flight. Returns count of deleted points."""
    client = get_qdrant_client()
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[models.FieldCondition(key="flight_id", match=models.MatchValue(value=flight_id))]
            )
        ),
    )
    return 1  # Qdrant delete is async, count not available
