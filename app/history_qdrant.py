import os
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import pandas as pd

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from .core_logic import normalize_text, normalize_key
from .qdrant_store import (
    get_qdrant_client,
    recreate_collection,
    to_qdrant_point_id,
    HISTORY_PUB_J_COLLECTION,
    HISTORY_REJ_J_COLLECTION,
    HISTORY_PUB_SI_COLLECTION,
    HISTORY_REJ_SI_COLLECTION,
)


# We store "aggregate points" in Qdrant using a 1-D dummy vector
# because Qdrant requires vectors in standard collections.
_AGG_VECTOR_DIM = 1


def _dummy_vectors(n: int) -> np.ndarray:
    return np.zeros((n, _AGG_VECTOR_DIM), dtype=np.float32)


def _pick_journal_key(row: pd.Series) -> str:
    """
    Prefer Journal_Short_Name for history calculation.
    Fallback to Journal_Name.
    """
    js = normalize_text(row.get("Journal_Short_Name", ""))
    if js:
        return js
    return normalize_text(row.get("Journal_Name", ""))


def _ensure_history_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make the function resilient to missing columns.
    """
    df = df.copy()
    if "Journal_Name" not in df.columns:
        df["Journal_Name"] = ""
    if "Journal_Short_Name" not in df.columns:
        df["Journal_Short_Name"] = ""
    if "Special_Issue_Name" not in df.columns:
        df["Special_Issue_Name"] = ""
    return df


def build_history_aggregates(
    published_df: pd.DataFrame,
    rejected_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build 4 aggregate tables equivalent to the old SQLite aggregates:
      - pub_j: by Journal_Name_norm
      - rej_j: by Journal_Name_norm
      - pub_si: by Journal_Name_norm + Special_Issue_Name_norm
      - rej_si: by Journal_Name_norm + Special_Issue_Name_norm

    IMPORTANT: Uses Journal_Short_Name as journal key when available,
    else Journal_Name. This affects Journal_Name_norm in outputs.

    Returns dataframes with columns expected by add_history_scores_from_aggregates():
      pub_j:  Journal_Name_norm, pub_count_j
      rej_j:  Journal_Name_norm, rej_count_j
      pub_si: Journal_Name_norm, Special_Issue_Name_norm, pub_count_si
      rej_si: Journal_Name_norm, Special_Issue_Name_norm, rej_count_si
    """
    pub = _ensure_history_cols(published_df)
    rej = _ensure_history_cols(rejected_df)

    # Journal key and norms
    pub["Journal_Key"] = pub.apply(_pick_journal_key, axis=1)
    rej["Journal_Key"] = rej.apply(_pick_journal_key, axis=1)

    pub["Journal_Name_norm"] = pub["Journal_Key"].map(normalize_key)
    rej["Journal_Name_norm"] = rej["Journal_Key"].map(normalize_key)

    pub["Special_Issue_Name_norm"] = pub["Special_Issue_Name"].map(normalize_key)
    rej["Special_Issue_Name_norm"] = rej["Special_Issue_Name"].map(normalize_key)

    # Drop empty journal norms
    pub = pub[pub["Journal_Name_norm"] != ""]
    rej = rej[rej["Journal_Name_norm"] != ""]

    pub_j = (
        pub.groupby(["Journal_Name_norm"], as_index=False)
        .size()
        .rename(columns={"size": "pub_count_j"})
    )

    rej_j = (
        rej.groupby(["Journal_Name_norm"], as_index=False)
        .size()
        .rename(columns={"size": "rej_count_j"})
    )

    pub_si = (
        pub.groupby(["Journal_Name_norm", "Special_Issue_Name_norm"], as_index=False)
        .size()
        .rename(columns={"size": "pub_count_si"})
    )

    rej_si = (
        rej.groupby(["Journal_Name_norm", "Special_Issue_Name_norm"], as_index=False)
        .size()
        .rename(columns={"size": "rej_count_si"})
    )

    # Ensure types
    for c in ["pub_count_j"]:
        if c in pub_j.columns:
            pub_j[c] = pub_j[c].astype(int)
    for c in ["rej_count_j"]:
        if c in rej_j.columns:
            rej_j[c] = rej_j[c].astype(int)
    for c in ["pub_count_si"]:
        if c in pub_si.columns:
            pub_si[c] = pub_si[c].astype(int)
    for c in ["rej_count_si"]:
        if c in rej_si.columns:
            rej_si[c] = rej_si[c].astype(int)

    return pub_j, rej_j, pub_si, rej_si


def recreate_history_collections(client: QdrantClient) -> None:
    """
    Recreate 4 small collections to store aggregate points.
    Safe: does not touch PRIMARY/ASSOC/FLEX collections.
    """
    for name in [
        HISTORY_PUB_J_COLLECTION,
        HISTORY_REJ_J_COLLECTION,
        HISTORY_PUB_SI_COLLECTION,
        HISTORY_REJ_SI_COLLECTION,
    ]:
        recreate_collection(client, name, dim=_AGG_VECTOR_DIM)


def _upsert_agg_df(
    client: QdrantClient,
    collection: str,
    df: pd.DataFrame,
    id_cols: List[str],
) -> None:
    """
    Store aggregate rows as Qdrant points with 1-D dummy vectors.
    Each point's payload contains the aggregate row.
    """
    if df is None or df.empty:
        return

    vecs = _dummy_vectors(len(df))
    points: List[PointStruct] = []

    for i, row in df.reset_index(drop=True).iterrows():
        # Stable point id based on collection + key columns
        raw_id = collection + "::" + "::".join([str(row.get(c, "")) for c in id_cols])
        qid = to_qdrant_point_id(raw_id)

        payload = {k: row[k] for k in df.columns}
        payload["_id"] = raw_id
        payload["qdrant_id"] = qid

        points.append(
            PointStruct(
                id=qid,
                vector=vecs[i].tolist(),
                payload=payload,
            )
        )

    client.upsert(collection_name=collection, points=points)


def upsert_history_aggregates_to_qdrant(
    client: QdrantClient,
    pub_j: pd.DataFrame,
    rej_j: pd.DataFrame,
    pub_si: pd.DataFrame,
    rej_si: pd.DataFrame,
    reset: bool = True,
) -> None:
    """
    Write aggregates into Qdrant.
    If reset=True, collections are recreated (old data cleared).
    """
    if reset:
        recreate_history_collections(client)

    _upsert_agg_df(client, HISTORY_PUB_J_COLLECTION, pub_j, id_cols=["Journal_Name_norm"])
    _upsert_agg_df(client, HISTORY_REJ_J_COLLECTION, rej_j, id_cols=["Journal_Name_norm"])
    _upsert_agg_df(client, HISTORY_PUB_SI_COLLECTION, pub_si, id_cols=["Journal_Name_norm", "Special_Issue_Name_norm"])
    _upsert_agg_df(client, HISTORY_REJ_SI_COLLECTION, rej_si, id_cols=["Journal_Name_norm", "Special_Issue_Name_norm"])


def _scroll_all_payloads(client: QdrantClient, collection: str) -> List[Dict[str, Any]]:
    """
    Scroll all points (payload only) from a small collection.
    """
    payloads: List[Dict[str, Any]] = []
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            payloads.append(p.payload or {})
        if offset is None:
            break

    return payloads


def load_history_aggregates_from_qdrant(
    client: Optional[QdrantClient] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load aggregates from Qdrant and return the 4 dataframes
    expected by add_history_scores_from_aggregates().
    """
    if client is None:
        client = get_qdrant_client()

    try:
        pub_j_payloads = _scroll_all_payloads(client, HISTORY_PUB_J_COLLECTION)
        rej_j_payloads = _scroll_all_payloads(client, HISTORY_REJ_J_COLLECTION)
        pub_si_payloads = _scroll_all_payloads(client, HISTORY_PUB_SI_COLLECTION)
        rej_si_payloads = _scroll_all_payloads(client, HISTORY_REJ_SI_COLLECTION)
    except Exception as e:
        # Match old behavior: main.py expects missing aggregates to be treated as "not ingested"
        raise RuntimeError("History aggregates not found. Call /ingest/history first.") from e

    pub_j = pd.DataFrame(pub_j_payloads)
    rej_j = pd.DataFrame(rej_j_payloads)
    pub_si = pd.DataFrame(pub_si_payloads)
    rej_si = pd.DataFrame(rej_si_payloads)

    # Keep only needed columns, ensure presence
    def _keep(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=cols)
        out = df.copy()
        for c in cols:
            if c not in out.columns:
                out[c] = 0 if c.endswith("_count_j") or c.endswith("_count_si") else ""
        return out[cols]

    pub_j = _keep(pub_j, ["Journal_Name_norm", "pub_count_j"])
    rej_j = _keep(rej_j, ["Journal_Name_norm", "rej_count_j"])
    pub_si = _keep(pub_si, ["Journal_Name_norm", "Special_Issue_Name_norm", "pub_count_si"])
    rej_si = _keep(rej_si, ["Journal_Name_norm", "Special_Issue_Name_norm", "rej_count_si"])

    # Types
    for df, col in [(pub_j, "pub_count_j"), (rej_j, "rej_count_j"), (pub_si, "pub_count_si"), (rej_si, "rej_count_si")]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    return pub_j, rej_j, pub_si, rej_si
