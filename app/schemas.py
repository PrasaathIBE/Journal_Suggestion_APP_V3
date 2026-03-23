from typing import Literal, Optional, Any, Dict, List
from pydantic import BaseModel, Field


# Backward-compatible: keep all existing modes, add FLEXIBLE_ONLY.
SearchMode = Literal["AUTO", "PRIMARY_ONLY", "ASSOCIATE_ONLY", "BOTH", "FLEXIBLE_ONLY"]


class SuggestRequest(BaseModel):
    title: str = Field(..., min_length=3)

    # New (optional): abstract as second input.
    # Backward-compatible: default empty string so existing clients don't need to send it.
    abstract: Optional[str] = Field("", description="Optional abstract to improve matching")

    topk: int = Field(10, ge=1, le=50)
    weak_threshold: float = Field(0.35, ge=0.0, le=1.0)
    mode: SearchMode = "AUTO"

    # ✅ NEW: how many domain scores to return
    domain_topn: int = Field(3, ge=1, le=30)

class SuggestResponse(BaseModel):
    title: str
    primary_domain: str
    top3_domains: List[Dict[str, Any]]
    mode: SearchMode
    fallback_used: bool
    primary_top1_sim: float
    results: List[Dict[str, Any]]

    # ✅ NEW: title-only domain results
    title_only_domain: str
    title_only_domain_scores: List[Dict[str, Any]]

    # ✅ NEW: combined (title + abstract) domain results
    combined_domain: str
    combined_domain_scores: List[Dict[str, Any]]

    # NEW: ensemble output
    domain_confidence: float = Field(0.0, description="Confidence in domain detection (0-1)")
    low_confidence_warning: bool = Field(False, description="True when domain detection is uncertain")