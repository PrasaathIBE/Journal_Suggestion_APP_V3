import io
import os
import re
from typing import Dict, Any, List

import requests
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

from .schemas import SuggestRequest, SuggestResponse
from .core_logic import (
    prepare_primary, prepare_fallback,
    embed_text_primary, embed_text_fallback,
    score_domains, score_domains_text, build_query_text, normalize_key,
    add_history_scores_from_aggregates,
    build_query_text_v2,
    DOMAIN_DESCRIPTIONS,
    score_domains_semantic,
    resolve_domain_ensemble,
)
from .qdrant_store import (
    get_qdrant_client, recreate_collection, upsert_points, search,
    search_with_domain_filter,
    PRIMARY_COLLECTION, ASSOC_COLLECTION, l2_normalize,
    FLEX_COLLECTION,
)
from .history_db import get_conn, reset_and_load_history, load_aggregates
from .history_qdrant import (
    build_history_aggregates,
    upsert_history_aggregates_to_qdrant,
    load_history_aggregates_from_qdrant,
)

load_dotenv()

EMBEDDING_API_URL      = os.getenv("EMBEDDING_API_URL", "").strip()
DEFAULT_WEAK_THRESHOLD = 0.42


# ─────────────────────────────────────────────────────────────────────────────
# Middleware — sanitize JSON body
# ─────────────────────────────────────────────────────────────────────────────
class SanitizeJSONBodyASGIMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        method = scope.get("method", "").upper()
        if method not in ("POST", "PUT", "PATCH"):
            await self.app(scope, receive, send)
            return
        headers      = dict(scope.get("headers") or [])
        content_type = (headers.get(b"content-type") or b"").decode("latin-1").lower()
        if not content_type.startswith("application/json"):
            await self.app(scope, receive, send)
            return
        body = b""
        more_body = True
        while more_body:
            message   = await receive()
            body     += message.get("body", b"")
            more_body = message.get("more_body", False)
        if body:
            text = body.decode("utf-8", errors="ignore")
            text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
            body = text.encode("utf-8")

        async def receive2():
            return {"type": "http.request", "body": body, "more_body": False}

        await self.app(scope, receive2, send)


# ─────────────────────────────────────────────────────────────────────────────
# Lazy domain embeddings cache — module-level singleton
# First request triggers load. Subsequent requests use cached version.
# On cold start: 1 wake-up call first, then 22 parallel calls.
# ─────────────────────────────────────────────────────────────────────────────
_domain_embeddings: Dict[str, np.ndarray] = {}


def get_domain_embeddings() -> Dict[str, np.ndarray]:
    global _domain_embeddings
    if _domain_embeddings:
        return _domain_embeddings

    from concurrent.futures import ThreadPoolExecutor, as_completed

    print("Building domain embeddings cache (parallel)...")

    def fetch_one(dom_desc):
        dom, desc = dom_desc
        return dom, embed_remote(desc, timeout=120)

    result = {}
    items = list(DOMAIN_DESCRIPTIONS.items())

    # First call wakes up HF Space — do it alone with extra timeout
    first_dom, first_desc = items[0]
    print(f"  Waking HF Space with: {first_dom}")
    result[first_dom] = embed_remote(first_desc, timeout=180)
    print(f"  HF Space awake. Loading remaining {len(items)-1} domains in parallel...")

    # Remaining calls in parallel — HF Space is now warm
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(fetch_one, item): item
            for item in items[1:]
        }
        for future in as_completed(futures):
            dom, vec = future.result()
            result[dom] = vec
            print(f"  Cached: {dom}")

    _domain_embeddings = result
    print(f"Domain embeddings ready: {len(_domain_embeddings)} domains.")
    return _domain_embeddings


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Journal Suggestion API",
    version="9.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SanitizeJSONBodyASGIMiddleware)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", str(c).strip()) for c in df.columns]
    return df


def read_any_upload(up: UploadFile) -> pd.DataFrame:
    name = (up.filename or "").lower()
    raw  = up.file.read()
    if name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(raw), dtype=str)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(io.BytesIO(raw), dtype=str)
    else:
        raise HTTPException(400, "Unsupported file type. Upload CSV or Excel.")
    return _normalize_cols(df)


def df_to_payloads(df: pd.DataFrame) -> List[Dict[str, Any]]:
    out = []
    for _, r in df.iterrows():
        d = {}
        for c in df.columns:
            v = r[c]
            d[c] = "" if pd.isna(v) else str(v)
        out.append(d)
    return out


def embed_remote(text: str, timeout: int = 120) -> np.ndarray:
    if not EMBEDDING_API_URL:
        raise HTTPException(status_code=500, detail="EMBEDDING_API_URL not set")
    resp = requests.post(
        EMBEDDING_API_URL,
        json={"text": text},
        timeout=timeout,
    )
    resp.raise_for_status()
    vec = np.array(resp.json()["vector"], dtype=np.float32)
    return l2_normalize(vec)


def _tag_domains_to_payloads(
    vectors: np.ndarray,
    payloads: List[Dict[str, Any]],
) -> None:
    """
    For every journal vector, detect the top-3 domains using the cached
    domain embeddings and store them as domain_tag, domain_tag_2, domain_tag_3
    in the payload.

    Storing three domain tags allows interdisciplinary journals to be found
    by any of their three closest domains during filtered search.
    For example a journal covering AI + Healthcare + Data Science will appear
    correctly in all three domain buckets.
    """
    domain_embs = get_domain_embeddings()
    for i, vec in enumerate(vectors):
        _, scores = score_domains_semantic(vec, domain_embs, topn=3)
        payloads[i]["domain_tag"]   = scores[0][0] if len(scores) > 0 else "Interdisciplinary & General"
        payloads[i]["domain_tag_2"] = scores[1][0] if len(scores) > 1 else "Interdisciplinary & General"
        payloads[i]["domain_tag_3"] = scores[2][0] if len(scores) > 2 else "Interdisciplinary & General"


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "version": "3.0.0"}


# ─────────────────────────────────────────────────────────────────────────────
# Ingest primary (SI-level)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/ingest/primary")
def ingest_primary(
    file: UploadFile = File(...),
    reset: bool = Query(True, description="Drop & recreate collection before ingest"),
):
    df_raw = read_any_upload(file)
    df     = prepare_primary(df_raw)

    if "_id" not in df.columns:
        raise HTTPException(400, "Primary file must contain '_id' column.")

    texts    = df.apply(embed_text_primary, axis=1).tolist()
    vectors  = np.vstack([embed_remote(t).reshape(1, -1) for t in texts]).astype(np.float32)
    ids      = df["_id"].astype(str).tolist()
    payloads = df_to_payloads(df.assign(candidate_text=texts))

    _tag_domains_to_payloads(vectors, payloads)

    client = get_qdrant_client()
    if reset:
        recreate_collection(client, PRIMARY_COLLECTION, dim=vectors.shape[1])
    upsert_points(client, PRIMARY_COLLECTION, ids, vectors, payloads)

    return {"collection": PRIMARY_COLLECTION, "rows_ingested": len(df), "reset": reset}


# ─────────────────────────────────────────────────────────────────────────────
# Ingest associate editor (journal-level)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/ingest/associate")
def ingest_associate(
    file: UploadFile = File(...),
    reset: bool = Query(True, description="Drop & recreate collection before ingest"),
):
    print("Reading associate editor file...")
    df_raw = read_any_upload(file)
    df     = prepare_fallback(df_raw)

    if "_id" in df_raw.columns:
        tmp = df_raw.copy()
        tmp["Journal_Name"]      = tmp.get("Journal_Name", tmp.get("Journal", "")).astype(str)
        tmp["Journal_Name_norm"] = tmp["Journal_Name"].map(lambda x: normalize_key(x))
        id_map = tmp.dropna(subset=["_id"]).groupby("Journal_Name_norm")["_id"].first().reset_index()
        df = df.merge(id_map, on="Journal_Name_norm", how="left")

    if "_id" not in df.columns:
        raise HTTPException(400, "Associate editor file must contain '_id'.")

    texts    = df.apply(embed_text_fallback, axis=1).tolist()
    vectors  = np.vstack([embed_remote(t).reshape(1, -1) for t in texts]).astype(np.float32)
    ids      = df["_id"].astype(str).tolist()
    payloads = df_to_payloads(df.assign(candidate_text=texts))

    _tag_domains_to_payloads(vectors, payloads)

    client = get_qdrant_client()
    if reset:
        recreate_collection(client, ASSOC_COLLECTION, dim=vectors.shape[1])
    upsert_points(client, ASSOC_COLLECTION, ids, vectors, payloads)

    return {"collection": ASSOC_COLLECTION, "rows_ingested": len(df), "reset": reset}


# ─────────────────────────────────────────────────────────────────────────────
# Ingest flexible journals (journal-level)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/ingest/flexible")
def ingest_flexible(
    file: UploadFile = File(...),
    reset: bool = Query(True, description="Drop & recreate collection before ingest"),
):
    print("Reading flexible journals file...")
    df_raw = read_any_upload(file)

    if "Special_Issue_keywords" not in df_raw.columns:
        df_raw["Special_Issue_keywords"] = df_raw.get("Aim_and_Scope", "")
    if "Journal_Short_Name" not in df_raw.columns:
        df_raw["Journal_Short_Name"] = df_raw.get("Short_Name", "")

    df = prepare_fallback(df_raw)

    if "_id" in df_raw.columns:
        tmp = df_raw.copy()
        tmp["Journal_Name"]      = tmp.get("Journal_Name", tmp.get("Journal", "")).astype(str)
        tmp["Journal_Name_norm"] = tmp["Journal_Name"].map(lambda x: normalize_key(x))
        id_map = tmp.dropna(subset=["_id"]).groupby("Journal_Name_norm")["_id"].first().reset_index()
        df = df.merge(id_map, on="Journal_Name_norm", how="left")

    if "_id" not in df.columns:
        raise HTTPException(400, "Flexible journals file must contain '_id'.")

    texts    = df.apply(embed_text_fallback, axis=1).tolist()
    vectors  = np.vstack([embed_remote(t).reshape(1, -1) for t in texts]).astype(np.float32)
    ids      = df["_id"].astype(str).tolist()
    payloads = df_to_payloads(df.assign(candidate_text=texts))

    _tag_domains_to_payloads(vectors, payloads)

    client = get_qdrant_client()
    if reset:
        recreate_collection(client, FLEX_COLLECTION, dim=vectors.shape[1])
    upsert_points(client, FLEX_COLLECTION, ids, vectors, payloads)

    return {"collection": FLEX_COLLECTION, "rows_ingested": len(df), "reset": reset}


# ─────────────────────────────────────────────────────────────────────────────
# Ingest history
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/ingest/history")
def ingest_history(
    published_file: UploadFile = File(...),
    rejected_file:  UploadFile = File(...),
    reset: bool = Query(True, description="Rebuild history tables from scratch"),
):
    if not reset:
        raise HTTPException(400, "This endpoint is designed for daily reset=true ingest.")

    pub_df = read_any_upload(published_file)
    rej_df = read_any_upload(rejected_file)

    pub_j, rej_j, pub_si, rej_si = build_history_aggregates(pub_df, rej_df)

    client = get_qdrant_client()
    upsert_history_aggregates_to_qdrant(
        client=client,
        pub_j=pub_j, rej_j=rej_j,
        pub_si=pub_si, rej_si=rej_si,
        reset=True,
    )

    return {
        "history_backend": "qdrant",
        "published_rows":  len(pub_df),
        "rejected_rows":   len(rej_df),
        "reset": True,
        "collections": {
            "pub_j":  "history_pub_j",
            "rej_j":  "history_rej_j",
            "pub_si": "history_pub_si",
            "rej_si": "history_rej_si",
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Suggest endpoint
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/suggest", response_model=SuggestResponse)
def suggest(req: SuggestRequest):
    title = req.title.strip()
    if not title:
        raise HTTPException(400, "Title cannot be empty.")

    abstract = (req.abstract or "").strip()
    if len(abstract) > 1200:
        abstract = abstract[:1200] + "..."

    domain_topn = getattr(req, "domain_topn", 3)

    # --- Step 1: Title-only domain detection ---
    title_vec = embed_remote(title)
    domain_embs = get_domain_embeddings()
    title_only_domain, title_only_scores = score_domains_semantic(
        title_vec, domain_embs, topn=domain_topn
    )

    # --- Step 2: Combined domain detection ---
    combined_text   = (title + " " + abstract).strip()
    combined_vec    = embed_remote(combined_text)
    combined_domain, combined_scores = score_domains_semantic(
        combined_vec, domain_embs, topn=domain_topn
    )

    # --- Step 3: Ensemble resolver ---
    primary_domain, merged_domain_scores, domain_confidence = resolve_domain_ensemble(
        title_domain=title_only_domain,
        title_scores=title_only_scores,
        combined_domain=combined_domain,
        combined_scores=combined_scores,
    )

    top3_scores = merged_domain_scores[:3]

    # --- Step 4: Build query vector ---
    qtext = build_query_text_v2(title, abstract, top3_scores)
    qvec  = embed_remote(qtext)

    client = get_qdrant_client()

    # --- Step 5: Load history aggregates ---
    try:
        pub_j, rej_j, pub_si, rej_si = load_history_aggregates_from_qdrant(client)
    except Exception:
        raise HTTPException(400, "History aggregates not found. Call /ingest/history first.")

    mode           = req.mode
    topk           = req.topk
    weak_threshold = req.weak_threshold if req.weak_threshold != 0.35 else DEFAULT_WEAK_THRESHOLD
    low_confidence = domain_confidence < 0.4

    # --- Step 6: Domain-filtered search ---
    # Use top-2 domains for the filter so interdisciplinary papers
    # can match journals from either of their closest domains.
    filter_domains = [d for d, _ in top3_scores[:2]]

    primary_hits = []
    assoc_hits   = []
    flex_hits    = []
    fallback_used = False

    if mode in ("AUTO", "PRIMARY_ONLY", "BOTH"):
        primary_hits, _ = search_with_domain_filter(
            client, PRIMARY_COLLECTION, qvec,
            topk=topk,
            domains=filter_domains,
            min_results=3,
        )

    primary_top1_sim = float(primary_hits[0].score) if primary_hits else 0.0

    if mode == "AUTO":
        fallback_used = (
            (not primary_hits)
            or (primary_top1_sim < weak_threshold)
            or low_confidence
        )
        if fallback_used:
            assoc_hits, _ = search_with_domain_filter(
                client, ASSOC_COLLECTION, qvec,
                topk=topk,
                domains=filter_domains,
                min_results=3,
            )
            if low_confidence or primary_top1_sim < weak_threshold:
                flex_hits, _ = search_with_domain_filter(
                    client, FLEX_COLLECTION, qvec,
                    topk=topk,
                    domains=filter_domains,
                    min_results=2,
                )

    elif mode == "ASSOCIATE_ONLY":
        fallback_used = True
        assoc_hits, _ = search_with_domain_filter(
            client, ASSOC_COLLECTION, qvec,
            topk=topk, domains=filter_domains, min_results=3,
        )

    elif mode == "BOTH":
        fallback_used = True
        assoc_hits, _ = search_with_domain_filter(
            client, ASSOC_COLLECTION, qvec,
            topk=topk, domains=filter_domains, min_results=3,
        )

    elif mode == "FLEXIBLE_ONLY":
        fallback_used = True
        flex_hits, _  = search_with_domain_filter(
            client, FLEX_COLLECTION, qvec,
            topk=topk, domains=filter_domains, min_results=2,
        )

    def hits_to_rows(hits, source: str):
        rows = []
        for h in hits:
            payload = h.payload or {}
            row     = dict(payload)
            row["sim"]            = float(h.score)
            row["source"]         = source
            row["candidate_text"] = payload.get("candidate_text", "")
            js = row.get("Journal_Short_Name", "") or row.get("Short_Name", "")
            row["Journal_Name_norm"]       = normalize_key(js) if js and str(js).strip() else normalize_key(row.get("Journal_Name", ""))
            row["Special_Issue_Name_norm"] = normalize_key(row.get("Special_Issue_Name", ""))
            rows.append(row)
        return rows

    cand_rows = []
    cand_rows.extend(hits_to_rows(primary_hits, "PRIMARY"))
    cand_rows.extend(hits_to_rows(assoc_hits,   "ASSOC"))
    cand_rows.extend(hits_to_rows(flex_hits,    "FLEX"))

    if not cand_rows:
        return SuggestResponse(
            title=title,
            primary_domain=primary_domain,
            top3_domains=[{"domain": d, "score": float(s)} for d, s in top3_scores[:3]],
            mode=mode,
            fallback_used=fallback_used,
            primary_top1_sim=primary_top1_sim,
            results=[],
            title_only_domain=title_only_domain,
            title_only_domain_scores=[{"domain": d, "score": float(s)} for d, s in title_only_scores],
            combined_domain=combined_domain,
            combined_domain_scores=[{"domain": d, "score": float(s)} for d, s in combined_scores],
            domain_confidence=domain_confidence,
            low_confidence_warning=low_confidence,
        )

    cand_df = pd.DataFrame(cand_rows)

    ranked = add_history_scores_from_aggregates(
        cand_df=cand_df,
        pub_j=pub_j, rej_j=rej_j,
        pub_si=pub_si, rej_si=rej_si,
        title=title,
        title_domain=primary_domain,
        domain_weights=top3_scores,
    )

    dedup_col    = "Journal_Short_Name" if "Journal_Short_Name" in ranked.columns else "Journal_Name"
    ranked_dedup = ranked.drop_duplicates(subset=[dedup_col], keep="first")
    results      = ranked_dedup.head(topk).to_dict(orient="records")

    return SuggestResponse(
        title=title,
        primary_domain=primary_domain,
        top3_domains=[{"domain": d, "score": float(s)} for d, s in top3_scores[:3]],
        mode=mode,
        fallback_used=fallback_used,
        primary_top1_sim=primary_top1_sim,
        results=results,
        title_only_domain=title_only_domain,
        title_only_domain_scores=[{"domain": d, "score": float(s)} for d, s in title_only_scores],
        combined_domain=combined_domain,
        combined_domain_scores=[{"domain": d, "score": float(s)} for d, s in combined_scores],
        domain_confidence=domain_confidence,
        low_confidence_warning=low_confidence,
    )