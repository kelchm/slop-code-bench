"""Tests for the live trace writer and its process_stream tee."""

from __future__ import annotations

import json
from pathlib import Path

from slop_code.execution.live_trace import MAX_RAW_LINE_CHARS
from slop_code.execution.live_trace import LiveTrace
from slop_code.execution.stream_processor import process_stream


def read_records(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def test_emit_writes_flushed_jsonl_records(tmp_path: Path) -> None:
    trace = LiveTrace(tmp_path / "live.jsonl")
    trace.emit("usage", steps=3, cost=0.5)

    # Readable before close because every record is flushed.
    records = read_records(trace.path)
    assert len(records) == 1
    assert records[0]["kind"] == "usage"
    assert records[0]["steps"] == 3
    assert "ts" in records[0]
    trace.close()


def test_raw_buffers_partial_lines_until_newline(tmp_path: Path) -> None:
    trace = LiveTrace(tmp_path / "live.jsonl")
    trace.raw("stdout", '{"type": "st')
    assert read_records(trace.path) == []

    trace.raw("stdout", 'ep"}\npartial')
    records = read_records(trace.path)
    assert len(records) == 1
    assert records[0]["line"] == '{"type": "st' + 'ep"}'

    # close() flushes the buffered partial line.
    trace.close()
    records = read_records(trace.path)
    assert len(records) == 2
    assert records[1]["line"] == "partial"


def test_raw_truncates_huge_lines(tmp_path: Path) -> None:
    trace = LiveTrace(tmp_path / "live.jsonl")
    trace.raw("stdout", "x" * (MAX_RAW_LINE_CHARS + 100) + "\n")
    trace.close()

    records = read_records(trace.path)
    assert len(records) == 1
    assert len(records[0]["line"]) == MAX_RAW_LINE_CHARS
    assert records[0]["truncated"] is True


def test_emit_after_close_is_noop(tmp_path: Path) -> None:
    trace = LiveTrace(tmp_path / "live.jsonl")
    trace.close()
    trace.emit("note", text="late")

    assert read_records(trace.path) == []


def make_stream(chunks: list[tuple[bytes, bytes]]):
    """Yield chunks, exposing a poll_fn that completes once exhausted.

    Events are queued by the pump thread before the final chunk is
    consumed, so once poll reports an exit code every event is drainable.
    """
    state = {"done": False}

    def gen():
        yield from chunks
        state["done"] = True

    return gen(), lambda: 0 if state["done"] else None


def test_process_stream_tees_output_and_exec_finish(tmp_path: Path) -> None:
    trace = LiveTrace(tmp_path / "live.jsonl")
    stream, poll = make_stream(
        [
            (b'{"step": 1}\n{"ste', b""),
            (b'p": 2}\n', b"warning\n"),
        ]
    )

    events = list(process_stream(stream, None, poll, live_trace=trace))
    trace.close()

    assert [e.kind for e in events if e.kind == "stdout"]
    records = read_records(trace.path)
    stdout_lines = [
        r["line"] for r in records if r["kind"] == "raw" and r["stream"] == "stdout"
    ]
    stderr_lines = [
        r["line"] for r in records if r["kind"] == "raw" and r["stream"] == "stderr"
    ]
    assert stdout_lines == ['{"step": 1}', '{"ste' + 'p": 2}']
    assert stderr_lines == ["warning"]
    exec_records = [r for r in records if r["kind"] == "exec"]
    assert len(exec_records) == 1
    assert exec_records[0]["event"] == "finished"
    assert exec_records[0]["exit_code"] == 0


def test_process_stream_tee_skips_setup_output(tmp_path: Path) -> None:
    trace = LiveTrace(tmp_path / "live.jsonl")
    stream, poll = make_stream(
        [(b"setup noise\n---SPLIT---\nagent line\n", b"")]
    )

    list(
        process_stream(
            stream,
            None,
            poll,
            yield_only_after="---SPLIT---\n",
            live_trace=trace,
        )
    )
    trace.close()

    raw_lines = [
        r["line"] for r in read_records(trace.path) if r["kind"] == "raw"
    ]
    assert raw_lines == ["agent line"]
