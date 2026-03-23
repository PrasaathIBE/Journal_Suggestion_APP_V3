"""
DEPRECATED MODULE ⚠️

This module is intentionally kept for backward compatibility only.

History storage has been migrated from SQLite to Qdrant.
DO NOT use this module in new code.

Active history logic now lives in:
    history_qdrant.py
"""

from fastapi import HTTPException


def get_conn(*args, **kwargs):
    raise HTTPException(
        status_code=410,
        detail="SQLite-based history is deprecated. Use Qdrant history instead."
    )


def reset_and_load_history(*args, **kwargs):
    raise HTTPException(
        status_code=410,
        detail="SQLite-based history ingest is deprecated. Use /ingest/history with Qdrant."
    )


def load_aggregates(*args, **kwargs):
    raise HTTPException(
        status_code=410,
        detail="SQLite-based history aggregates are deprecated. Use Qdrant history."
    )
