"""
Logging Service - Builds and publishes HAIC benchmarking event logs.
"""
import json
import os
import tempfile
import time

import streamlit as st

from dataset import minio_utils


def _elapsed() -> float:
    start = st.session_state.get("session_start_time", time.time())
    return round(time.time() - start, 3)


def make_event(agent: str, actor_type: str, action: str, **kwargs) -> dict:
    event = {
        "t": _elapsed(),
        "agent": agent,
        "actor_type": actor_type,
        "action": action,
    }
    for key in ("latency_ms", "duration_s", "correct", "probs",
                "surrogate_probs", "surrogate_action", "event_type"):
        if key in kwargs:
            event[key] = kwargs[key]
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

    tmp_path = os.path.join(tempfile.gettempdir(), f"{run_id}_events.jsonl")
    with open(tmp_path, "w") as f:
        for event in events:
            f.write(json.dumps(event, default=_default) + "\n")

    object_name = f"benchmarking-logs/{run_id}/events.jsonl"
    minio_utils.minio_upload(token, bucket, object_name, tmp_path)
