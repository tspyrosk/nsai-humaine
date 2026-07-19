"""
Logging Service - Builds and publishes HAIC benchmarking event logs.
"""
import json
import os
import tempfile
import time

import streamlit as st

from dataset import minio_utils

PILOT_TAG = "smart-healthcare-diabetes"
AI_MODEL_VERSION = "neurosymbolic-v0"


def _elapsed() -> float:
    start = st.session_state.get("session_start_time", time.time())
    return round(time.time() - start, 3)


def make_event(agent: str, actor_type: str, action: str,
               interaction_id: str = None, **kwargs) -> dict:
    event = {
        "t": _elapsed(),
        "actor_type": actor_type,
        "action": action,
    }
    if interaction_id:
        event["interaction_id"] = interaction_id
    for key in ("latency_ms", "duration_s", "correct",
                "surrogate_probs", "surrogate_action"):
        if key in kwargs:
            event[key] = kwargs[key]
    # Fields outside the benchmarking-suite schema go in payload so the
    # evaluation pipeline preserves them instead of silently dropping them.
    payload = {"agent": agent}
    for key in ("event_type", "probs"):
        if key in kwargs:
            payload[key] = kwargs[key]
    event["payload"] = payload
    return event


def append_event(event: dict):
    if "log_events" not in st.session_state:
        st.session_state.log_events = []
    st.session_state.log_events.append(event)


def publish_events(token: str, bucket: str, run_id: str):
    events = st.session_state.get("log_events", [])
    if not events:
        return
    if not token:
        print("publish_events: no MinIO token available, skipping.", flush=True)
        return

    def _default(obj):
        import numpy as np
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    # Every event needs an interaction_id for per-case metrics. Setup events
    # (predicates, rules, training) are part of the cycle that leads to the
    # first predicted case, so they inherit its id; if no prediction was made
    # yet, fall back to a session-level case id.
    default_id = next(
        (e["interaction_id"] for e in events if e.get("interaction_id")),
        f"diabetes_case_{run_id}",
    )
    for event in events:
        event.setdefault("interaction_id", default_id)

    document = {
        "logs": [
            {
                "session_id": run_id,
                "pilot_tag": PILOT_TAG,
                "ai_model_version": AI_MODEL_VERSION,
                "decisions": events,
            }
        ]
    }

    tmp_path = os.path.join(tempfile.gettempdir(), f"{run_id}_events.json")
    with open(tmp_path, "w") as f:
        json.dump(document, f, default=_default, indent=2)

    object_name = f"benchmarking-logs/{run_id}/events.json"
    minio_utils.minio_upload(token, bucket, object_name, tmp_path)
