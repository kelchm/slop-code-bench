"""Grok CLI trajectory parser.

``grok --output-format streaming-messages-json`` emits NDJSON in the Anthropic
Messages API wire format, which is the same shape Claude Code writes. The event
walk is therefore identical; only the reported ``agent_type`` differs.
"""

from __future__ import annotations

import json
from pathlib import Path

from slop_code.agent_runner.agents.claude_code.parser import ClaudeCodeParser
from slop_code.agent_runner.trajectory import Trajectory


class GrokCliParser(ClaudeCodeParser):
    """Parser for Grok CLI headless trajectories."""

    def can_parse(self, artifact_dir: Path) -> bool:
        """Check if directory contains a Grok CLI trajectory."""
        jsonl_file = self._find_jsonl_file(artifact_dir)
        if not jsonl_file:
            return False

        try:
            with jsonl_file.open() as f:
                first_line = f.readline()
                if not first_line:
                    return False
                data = json.loads(first_line)
        except (json.JSONDecodeError, OSError):
            return False

        if data.get("type") != "system":
            return False
        # Grok emits the same init envelope as Claude Code; disambiguate on the
        # model id so the two parsers do not both claim a directory.
        return str(data.get("model", "")).startswith("grok")

    def parse(self, artifact_dir: Path) -> Trajectory:
        """Parse a Grok CLI trajectory."""
        trajectory = super().parse(artifact_dir)
        return Trajectory(
            agent_type="grok_cli",
            steps=trajectory.steps,
            metadata=trajectory.metadata,
        )
