import os
from typing import List, Dict, Any, Optional, Tuple
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter,
    FieldCondition, MatchAny,
)
import numpy as np


# ── Collection names ──────────────────────────────────────────────────────────
PRIMARY_COLLECTION        = "journal_primary_si"
ASSOC_COLLECTION          = "journal_associate_editor"
FLEX_COLLECTION           = "journal_flexible"
HISTORY_PUB_J_COLLECTION  = "history_pub_j"
HISTORY_REJ_J_COLLECTION  = "history_rej_j"
HISTORY_PUB_SI_COLLECTION = "history_pub_si"
HISTORY_REJ_SI_COLLECTION = "history_rej_si"

# ── Stable namespace for deterministic UUIDs ──────────────────────────────────
QDRANT_NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    if v.ndim == 1:
        n = np.linalg.norm(v) + 1e-12
        return v / n
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / n


def get_qdrant_client() -> QdrantClient:
    url     = os.getenv("QDRANT_URL",     "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY", None)
    return QdrantClient(url=url, api_key=api_key)


def recreate_collection(client: QdrantClient, name: str, dim: int) -> None:
    """Drop and recreate a collection, then create keyword indexes on domain fields."""
    try:
        client.delete_collection(name)
    except Exception:
        pass

    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    # Keyword indexes required for domain filter queries on Qdrant Cloud
    client.create_payload_index(
        collection_name=name,
        field_name="domain_tag",
        field_schema="keyword",
    )
    client.create_payload_index(
        collection_name=name,
        field_name="domain_tag_2",
        field_schema="keyword",
    )
    client.create_payload_index(
        collection_name=name,
        field_name="domain_tag_3",
        field_schema="keyword",
    )

def to_qdrant_point_id(raw_id: str) -> str:
    """Convert any string id to a stable deterministic UUID for Qdrant."""
    return str(uuid.uuid5(QDRANT_NAMESPACE, str(raw_id).strip()))


def upsert_points(
    client: QdrantClient,
    collection: str,
    ids: List[str],
    vectors: np.ndarray,
    payloads: List[Dict[str, Any]],
) -> None:
    points = []
    for i, raw_id in enumerate(ids):
        qid     = to_qdrant_point_id(raw_id)
        payload = payloads[i]
        payload["_id"]      = str(payload.get("_id", raw_id))
        payload["qdrant_id"] = qid
        points.append(
            PointStruct(id=qid, vector=vectors[i].tolist(), payload=payload)
        )
    client.upsert(collection_name=collection, points=points)


def search(
    client: QdrantClient,
    collection: str,
    query_vector: np.ndarray,
    topk: int,
    qfilter: Optional[Filter] = None,
):
    q   = l2_normalize(query_vector)
    res = client.query_points(
        collection_name=collection,
        query=q.tolist(),
        limit=topk,
        query_filter=qfilter,
        with_payload=True,
        with_vectors=False,
    )
    return res.points


def build_domain_filter(domains: List[str]) -> Filter:
    """
    Build a Qdrant filter matching journals whose domain_tag, domain_tag_2,
    OR domain_tag_3 is in the given list.
    Three tags per journal allows interdisciplinary journals to appear
    in up to three relevant domain buckets.
    """
    return Filter(
        should=[
            FieldCondition(key="domain_tag",   match=MatchAny(any=domains)),
            FieldCondition(key="domain_tag_2", match=MatchAny(any=domains)),
            FieldCondition(key="domain_tag_3", match=MatchAny(any=domains)),
        ]
    )


def search_with_domain_filter(
    client: QdrantClient,
    collection: str,
    query_vector: np.ndarray,
    topk: int,
    domains: List[str],
    min_results: int = 3,
) -> Tuple[list, bool]:
    """
    Search with domain filter first.
    If fewer than min_results come back, automatically fall back to
    unfiltered search so the response is never empty.
    Returns (hits, filter_was_applied).
    """
    domain_filter = build_domain_filter(domains)
    hits = search(client, collection, query_vector, topk=topk, qfilter=domain_filter)

    if len(hits) >= min_results:
        return hits, True

    # Not enough results in filtered bucket — widen to full collection
    hits = search(client, collection, query_vector, topk=topk)
    return hits, False