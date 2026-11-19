"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}
        self._inputs: dict[str, dict] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store the (redacted) input and its start time, keyed by request_id/user_id."""
        key = request_id or user_id
        self._open[key] = time.perf_counter()
        self._inputs[key] = {
            "request_id": request_id,
            "user_id": user_id,
            "started_at": utc_now_iso(),
            "input": _scrub(text),
        }
        return key

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
        **extra,
    ):
        """Store output, layer decision and latency; append one entry to self.logs."""
        key = request_id or user_id
        started = self._open.pop(key, None)
        entry = self._inputs.pop(key, None) or {
            "request_id": request_id,
            "user_id": user_id,
            "started_at": None,
            "input": None,
        }
        entry.update(
            {
                "finished_at": utc_now_iso(),
                "output": _scrub(text),
                "blocked": bool(blocked),
                "layer": layer,
                "latency_ms": (
                    round((time.perf_counter() - started) * 1000, 1)
                    if started is not None
                    else None
                ),
            }
        )
        entry.update(extra)
        self.logs.append(entry)
        return entry

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return str(path)


def _scrub(text: str | None, limit: int = 500) -> str | None:
    """Audit logs get committed to GitHub — never store raw PII / secrets in them.

    Reuses the Checkpoint 2 content_filter and truncates very long inputs.
    """
    if text is None:
        return None
    from guardrails.output_guardrails import content_filter

    clean = content_filter(text)["redacted"]
    return clean if len(clean) <= limit else clean[:limit] + f"... [truncated {len(clean) - limit} chars]"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
