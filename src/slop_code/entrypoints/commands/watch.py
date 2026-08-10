"""Follow a run's live.jsonl and pretty-print agent activity in real time.

Companion to the live trace written by the agent runner (see
``execution/live_trace.py``). Point it at a run directory (or ``outputs/``)
in a second terminal while ``slop-code run`` is going:

    slop-code watch outputs/
    slop-code watch outputs/<run-dir> --problem file_backup

It follows the most recently modified ``live.jsonl``, prettifies known agent
event formats, and shows a ticking idle clock for stall detection.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.live import Live
from rich.text import Text

from slop_code.execution.live_trace import LIVE_TRACE_FILENAME

IDLE_WARN_SECONDS = 30.0
POLL_INTERVAL_SECONDS = 0.25


def register(app: typer.Typer, name: str):
    """Register the watch command."""
    app.command(
        name,
        help="Follow a running agent's live.jsonl and pretty-print activity.",
    )(watch)


def _find_trace_file(path: Path, problem: str | None) -> Path:
    """Resolve a live.jsonl from a file, problem dir, or ancestor dir."""
    if path.is_file():
        return path
    direct = path / LIVE_TRACE_FILENAME
    if direct.is_file():
        return direct
    candidates = [
        p
        for p in path.rglob(LIVE_TRACE_FILENAME)
        if problem is None or p.parent.name == problem
    ]
    if not candidates:
        raise typer.BadParameter(
            f"No {LIVE_TRACE_FILENAME} found under {path}"
            + (f" for problem '{problem}'" if problem else "")
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _first_of(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value:
            return value
    return None


def _truncate(text: str, limit: int) -> str:
    text = text.replace("\n", " ⏎ ").strip()
    if len(text) > limit:
        return text[:limit] + "…"
    return text


def _summarize_claude_message(message: dict[str, Any]) -> list[Text]:
    """Summarize claude-code/grok stream-json assistant/user messages."""
    lines: list[Text] = []
    content = message.get("content")
    if isinstance(content, str):
        return [Text(f"✦ {_truncate(content, 400)}", style="green")]
    if not isinstance(content, list):
        return lines
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            lines.append(
                Text(f"✦ {_truncate(block.get('text', ''), 400)}", style="green")
            )
        elif block_type == "thinking":
            lines.append(
                Text(
                    f"∴ {_truncate(block.get('thinking', ''), 200)}",
                    style="dim green",
                )
            )
        elif block_type == "tool_use":
            tool_input = block.get("input") or {}
            target = _first_of(
                tool_input,
                "file_path",
                "path",
                "command",
                "pattern",
                "url",
                "prompt",
            )
            detail = f" {_truncate(str(target), 200)}" if target else ""
            lines.append(
                Text(f"⚙ {block.get('name', 'tool')}{detail}", style="blue")
            )
        elif block_type == "tool_result":
            result = block.get("content")
            if isinstance(result, list):
                result = " ".join(
                    part.get("text", "")
                    for part in result
                    if isinstance(part, dict)
                )
            lines.append(
                Text(f"  ↳ {_truncate(str(result or ''), 200)}", style="dim")
            )
    return lines


def _summarize_agent_event(obj: dict[str, Any]) -> list[Text]:
    """Best-effort prettification of one agent-native JSON event."""
    event_type = _first_of(obj, "type", "event", "method", "msg_type")
    # claude-code / grok-cli stream-json wire format
    if event_type in {"assistant", "user"} and isinstance(
        obj.get("message"), dict
    ):
        return _summarize_claude_message(obj["message"])
    if event_type == "result":
        cost = _first_of(obj, "total_cost_usd", "cost_usd")
        cost_str = f" cost=${cost:.4f}" if isinstance(cost, float) else ""
        return [
            Text(
                f"■ result: {obj.get('subtype', '')}{cost_str}",
                style="bold cyan",
            )
        ]
    # codex JSONL wraps events in "msg"
    if isinstance(obj.get("msg"), dict):
        inner = obj["msg"]
        inner_type = inner.get("type", "event")
        detail = _first_of(inner, "command", "text", "message", "path")
        detail_str = f" {_truncate(str(detail), 200)}" if detail else ""
        return [Text(f"· {inner_type}{detail_str}", style="cyan")]
    # Generic: type plus the most interesting-looking field
    if event_type is not None:
        part = obj.get("part") if isinstance(obj.get("part"), dict) else obj
        detail = _first_of(
            part, "tool", "name", "command", "file_path", "path", "text"
        )
        detail_str = f" {_truncate(str(detail), 200)}" if detail else ""
        return [Text(f"· {event_type}{detail_str}", style="cyan")]
    return [Text(_truncate(json.dumps(obj), 200), style="dim")]


def _render_record(record: dict[str, Any]) -> list[Text]:
    kind = record.get("kind")
    if kind == "run":
        return [
            Text(
                f"▶ run {record.get('event')} · problem={record.get('problem', '')}"
                f" agent={record.get('agent', '')} model={record.get('model', '')}"
                if record.get("event") == "start"
                else f"■ run end · state={record.get('state', '')}",
                style="bold cyan",
            )
        ]
    if kind == "checkpoint":
        if record.get("event") == "start":
            text = (
                f"━━ {record.get('name')} started "
                f"({record.get('index')}/{record.get('total')}) ━━"
            )
        else:
            passed = record.get("passed_policy")
            verdict = "?" if passed is None else ("PASS" if passed else "FAIL")
            text = (
                f"━━ {record.get('name')} finished · {verdict} · "
                f"steps={record.get('steps')} cost=${record.get('cost')} ━━"
            )
        return [Text(text, style="bold magenta")]
    if kind == "exec":
        if record.get("event") == "start":
            return [
                Text(
                    f"$ {_truncate(record.get('command', ''), 200)}",
                    style="dim",
                )
            ]
        return [
            Text(
                f"↳ exit={record.get('exit_code')} "
                f"elapsed={record.get('elapsed')}s"
                + (" TIMED OUT" if record.get("timed_out") else ""),
                style="dim",
            )
        ]
    if kind == "usage":
        return [
            Text(
                f"step {record.get('steps')} · ${record.get('cost')} · "
                f"in {record.get('tokens_in'):,} out {record.get('tokens_out'):,}"
                f" · {record.get('state', '')}",
                style="dim cyan",
            )
        ]
    if kind == "heartbeat":
        return [
            Text(
                f"… idle {record.get('idle_s')}s ({record.get('state', '')})",
                style="yellow",
            )
        ]
    if kind == "model":
        return [
            Text(
                f"✦ [{record.get('step')}] "
                f"{_truncate(record.get('content', ''), 400)}",
                style="green",
            )
        ]
    if kind == "tool":
        return [
            Text(
                f"⚙ {record.get('tool')}: "
                f"{_truncate(record.get('command', ''), 200)}",
                style="blue",
            ),
            Text(
                f"  ↳ {_truncate(record.get('result', ''), 200)}",
                style="dim",
            ),
        ]
    if kind == "note":
        return [Text(f"⚠ {record.get('text', '')}", style="bold red")]
    if kind == "raw":
        line = record.get("line", "")
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            obj = None
        if isinstance(obj, dict):
            return _summarize_agent_event(obj)
        prefix = "!" if record.get("stream") == "stderr" else "|"
        style = "red" if record.get("stream") == "stderr" else "dim"
        return [Text(f"{prefix} {_truncate(str(line), 200)}", style=style)]
    return [Text(_truncate(json.dumps(record), 200), style="dim")]


def _idle_status(idle: float, trace_path: Path) -> Text:
    style = "bold yellow" if idle >= IDLE_WARN_SECONDS else "dim"
    return Text(
        f"[{trace_path.parent.name}] last event {idle:.0f}s ago", style=style
    )


def watch(
    path: Annotated[
        Path,
        typer.Argument(
            help="A live.jsonl, a problem run dir, or any ancestor "
            "(e.g. outputs/). The newest trace underneath is followed.",
        ),
    ] = Path("outputs"),
    problem: Annotated[
        str | None,
        typer.Option("--problem", "-p", help="Only consider this problem."),
    ] = None,
    tail: Annotated[
        int,
        typer.Option("--tail", "-n", help="Render this many trailing records on attach."),
    ] = 30,
    from_start: Annotated[  # noqa: FBT002
        bool,
        typer.Option("--from-start", help="Render the whole trace, not just the tail."),
    ] = False,
    raw: Annotated[  # noqa: FBT002
        bool,
        typer.Option("--raw", help="Print raw JSON records without prettifying."),
    ] = False,
    follow: Annotated[  # noqa: FBT002
        bool,
        typer.Option("--follow/--no-follow", help="Keep following for new records."),
    ] = True,
) -> None:
    """Follow a running agent's live trace."""
    console = Console()
    trace_path = _find_trace_file(path, problem)
    console.print(f"[bold]Watching[/bold] {trace_path}")

    def print_record(line: str) -> None:
        line = line.strip()
        if not line:
            return
        if raw:
            console.print(line, markup=False, highlight=False)
            return
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            console.print(Text(_truncate(line, 200), style="dim"))
            return
        for rendered in _render_record(record):
            console.print(rendered)

    handle = trace_path.open("r", encoding="utf-8")
    try:
        backlog = handle.readlines()
        if not from_start and len(backlog) > tail:
            console.print(
                Text(
                    f"… skipping {len(backlog) - tail} earlier records "
                    "(use --from-start for all)",
                    style="dim",
                )
            )
            backlog = backlog[-tail:]
        for line in backlog:
            print_record(line)
        if not follow:
            return

        last_event = time.monotonic()
        with Live(
            _idle_status(0, trace_path),
            console=console,
            refresh_per_second=4,
            transient=True,
        ) as live:
            while True:
                line = handle.readline()
                if line:
                    if line.endswith("\n"):
                        print_record(line)
                        last_event = time.monotonic()
                    else:
                        # Partial line: rewind and wait for the rest.
                        handle.seek(handle.tell() - len(line))
                        time.sleep(POLL_INTERVAL_SECONDS)
                    continue
                # A new run truncates the file; reopen to follow it.
                try:
                    if trace_path.stat().st_size < handle.tell():
                        handle.close()
                        handle = trace_path.open("r", encoding="utf-8")
                        console.print(
                            Text("— trace restarted —", style="bold yellow")
                        )
                        last_event = time.monotonic()
                        continue
                except FileNotFoundError:
                    pass
                live.update(
                    _idle_status(time.monotonic() - last_event, trace_path)
                )
                time.sleep(POLL_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        pass
    finally:
        handle.close()
