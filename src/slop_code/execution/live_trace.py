"""Append-only JSONL trace for watching agent runs in near-real time.

Artifacts are only flushed at checkpoint end, so mid-run there is nothing to
tail. ``LiveTrace`` fills that gap: one ``live.jsonl`` per problem run,
host-side (never inside the workspace, which gets snapshotted and graded),
truncated at open, flushed after every record.

Record shape: one JSON object per line with ``ts`` (UTC ISO-8601) and
``kind`` plus kind-specific fields:

- ``run``/``checkpoint``: lifecycle markers from the runner
- ``exec``: agent command start/finish from the streaming runtimes
- ``raw``: one line of agent stdout/stderr, exactly as the harness parser
  sees it (post-setup), truncated to ``MAX_RAW_LINE_CHARS``
- ``usage``: step/cost/token deltas from the progress watcher
- ``heartbeat``: emitted when the agent has been quiet for a while
- ``model``/``tool``: in-process agent steps (mini_swe)
- ``note``: retries, errors, and other annotations

This file is a debug view, not an artifact of record; ``messages.jsonl``
and friends remain authoritative.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC
from datetime import datetime
from pathlib import Path

LIVE_TRACE_FILENAME = "live.jsonl"
MAX_RAW_LINE_CHARS = 16_384


class LiveTrace:
    """Thread-safe, flush-per-record JSONL writer.

    Writes come from two threads per problem (the worker thread's stream
    loop and the progress-watcher thread); a lock serializes them. Tracing
    must never break a run, so serialization errors are swallowed and
    writes after ``close()`` are no-ops.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w", encoding="utf-8")
        # Raw output is chunked, not line-aligned; buffer per stream and
        # only emit complete lines. Only the stream-consuming thread
        # touches these buffers.
        self._raw_buffers: dict[str, str] = {}

    def emit(self, kind: str, **fields) -> None:
        """Write one record and flush so it is immediately tailable."""
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "kind": kind,
            **fields,
        }
        try:
            line = json.dumps(record, default=str)
        except (TypeError, ValueError):
            return
        with self._lock:
            if self._file.closed:
                return
            try:
                self._file.write(line + "\n")
                self._file.flush()
            except OSError:
                return

    def raw(self, stream: str, text: str) -> None:
        """Buffer raw stdout/stderr and emit one ``raw`` record per line."""
        buffer = self._raw_buffers.get(stream, "") + text
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            self._emit_raw_line(stream, line)
        self._raw_buffers[stream] = buffer

    def flush_raw(self) -> None:
        """Emit any buffered partial lines (call when a command finishes)."""
        buffers, self._raw_buffers = self._raw_buffers, {}
        for stream, buffer in buffers.items():
            self._emit_raw_line(stream, buffer)

    def _emit_raw_line(self, stream: str, line: str) -> None:
        line = line.rstrip("\r")
        if not line.strip():
            return
        truncated = len(line) > MAX_RAW_LINE_CHARS
        if truncated:
            line = line[:MAX_RAW_LINE_CHARS]
        record: dict[str, object] = {"stream": stream, "line": line}
        if truncated:
            record["truncated"] = True
        self.emit("raw", **record)

    def close(self) -> None:
        self.flush_raw()
        with self._lock:
            if not self._file.closed:
                self._file.close()


__all__ = ["LIVE_TRACE_FILENAME", "LiveTrace", "MAX_RAW_LINE_CHARS"]
