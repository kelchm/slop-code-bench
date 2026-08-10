"""Unit tests for the Grok CLI agent."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slop_code.agent_runner.agents.grok_cli import GrokCliAgent
from slop_code.agent_runner.agents.grok_cli import GrokCliConfig
from slop_code.agent_runner.agents.grok_cli import GrokCliParser
from slop_code.agent_runner.credentials import CredentialType
from slop_code.agent_runner.credentials import ProviderCredential
from slop_code.agent_runner.models import AgentCostLimits
from slop_code.agent_runner.models import AgentError
from slop_code.common.llms import APIPricing
from slop_code.common.llms import ModelDefinition


@pytest.fixture
def mock_pricing() -> APIPricing:
    """Grok 4.5 pricing: the over-200k tier doubles the base rates."""
    return APIPricing(
        input=4.0,
        output=12.0,
        cache_read=0.60,
        cache_write=0,
        prompt_tiers=[
            {
                "max_input_tokens": 200000,
                "input": 2.0,
                "output": 6.0,
                "cache_read": 0.30,
                "cache_write": 0,
            }
        ],
    )


@pytest.fixture
def mock_cost_limits() -> AgentCostLimits:
    return AgentCostLimits(
        step_limit=10,
        cost_limit=100.0,
        net_cost_limit=200.0,
    )


@pytest.fixture
def mock_model_def(mock_pricing: APIPricing) -> ModelDefinition:
    return ModelDefinition(
        internal_name="grok-4.5",
        provider="grok_auth",
        pricing=mock_pricing,
        aliases=["grok-4.5"],
    )


@pytest.fixture
def mock_credential(tmp_path: Path) -> ProviderCredential:
    grok_home = tmp_path / ".grok"
    grok_home.mkdir()
    auth = grok_home / "auth.json"
    auth.write_text('{"session": "fake"}', encoding="utf-8")
    return ProviderCredential(
        provider="grok_auth",
        credential_type=CredentialType.FILE,
        value=auth.read_text(),
        source=str(auth),
        destination_key="GROK_AUTH",
    )


def _assistant_event(
    input_tokens: int, output_tokens: int, cache_read: int
) -> str:
    return json.dumps(
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "model": "grok-4.5",
                "content": [{"type": "text", "text": "working"}],
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cache_read_input_tokens": cache_read,
                    "cache_creation_input_tokens": 0,
                },
            },
        }
    )


class TestGrokCliConfig:
    def test_version_is_required(
        self, mock_cost_limits: AgentCostLimits
    ) -> None:
        with pytest.raises(Exception):  # Pydantic validation error
            GrokCliConfig(type="grok_cli", cost_limits=mock_cost_limits)

    def test_get_docker_file_renders_versioned_url(
        self, mock_cost_limits: AgentCostLimits
    ) -> None:
        config = GrokCliConfig(
            type="grok_cli",
            version="1.0.0",
            cost_limits=mock_cost_limits,
        )
        dockerfile = config.get_docker_file("base-image:latest")
        assert dockerfile is not None
        assert "FROM base-image:latest" in dockerfile
        assert "https://x.ai/cli/grok-1.0.0-linux-${arch}" in dockerfile


class TestGrokCliAgentConstruction:
    def test_from_config_defaults_grok_home_to_credential_parent(
        self,
        mock_cost_limits: AgentCostLimits,
        mock_model_def: ModelDefinition,
        mock_credential: ProviderCredential,
    ) -> None:
        config = GrokCliConfig(
            type="grok_cli",
            version="1.0.0",
            cost_limits=mock_cost_limits,
        )
        agent = GrokCliAgent._from_config(
            config=config,
            model=mock_model_def,
            credential=mock_credential,
            problem_name="demo-problem",
            verbose=False,
            image="grok-image",
        )
        assert isinstance(agent, GrokCliAgent)
        assert agent.model == "grok-4.5"
        assert agent.grok_home == Path(mock_credential.source).parent

    def test_from_config_rejects_env_var_credential(
        self,
        mock_cost_limits: AgentCostLimits,
        mock_model_def: ModelDefinition,
    ) -> None:
        env_credential = ProviderCredential(
            provider="xai",
            credential_type=CredentialType.ENV_VAR,
            value="xai-key",
            source="XAI_API_KEY",
            destination_key="XAI_API_KEY",
        )
        config = GrokCliConfig(
            type="grok_cli",
            version="1.0.0",
            cost_limits=mock_cost_limits,
        )
        with pytest.raises(AgentError, match="file-based credential"):
            GrokCliAgent._from_config(
                config=config,
                model=mock_model_def,
                credential=env_credential,
                problem_name="demo-problem",
                verbose=False,
                image="grok-image",
            )

    def test_from_config_requires_image(
        self,
        mock_cost_limits: AgentCostLimits,
        mock_model_def: ModelDefinition,
        mock_credential: ProviderCredential,
    ) -> None:
        config = GrokCliConfig(
            type="grok_cli",
            version="1.0.0",
            cost_limits=mock_cost_limits,
        )
        with pytest.raises(ValueError, match="requires an image"):
            GrokCliAgent._from_config(
                config=config,
                model=mock_model_def,
                credential=mock_credential,
                problem_name="demo-problem",
                verbose=False,
                image=None,
            )

    def test_thinking_preset_maps_to_reasoning_effort(
        self,
        mock_cost_limits: AgentCostLimits,
        mock_model_def: ModelDefinition,
        mock_credential: ProviderCredential,
    ) -> None:
        config = GrokCliConfig(
            type="grok_cli",
            version="1.0.0",
            cost_limits=mock_cost_limits,
        )
        agent = GrokCliAgent._from_config(
            config=config,
            model=mock_model_def,
            credential=mock_credential,
            problem_name="demo-problem",
            verbose=False,
            image="grok-image",
            thinking_preset="xhigh",
        )
        # Grok tops out at "high"; xhigh has no distinct tier.
        assert agent.reasoning_effort == "high"


class TestGrokCliCommand:
    def _agent(
        self,
        mock_cost_limits: AgentCostLimits,
        mock_pricing: APIPricing,
        **overrides: object,
    ) -> GrokCliAgent:
        kwargs: dict = {
            "problem_name": "demo-problem",
            "verbose": False,
            "image": "grok-image",
            "cost_limits": mock_cost_limits,
            "pricing": mock_pricing,
            "credential": None,
            "binary": "grok",
            "model": "grok-4.5",
            "reasoning_effort": "high",
            "max_turns": None,
            "disable_web_search": True,
            "disable_subagents": False,
            "grok_home": Path.home() / ".grok",
            "sync_auth_back": False,
            "timeout": None,
            "extra_args": ["--no-auto-update"],
            "env": {},
        }
        kwargs.update(overrides)
        return GrokCliAgent(**kwargs)  # type: ignore[arg-type]

    def test_build_command_includes_headless_flags(
        self, mock_cost_limits: AgentCostLimits, mock_pricing: APIPricing
    ) -> None:
        agent = self._agent(mock_cost_limits, mock_pricing)
        command_text = " ".join(agent._build_command(resume=False))

        assert "--output-format=streaming-messages-json" in command_text
        assert "--permission-mode=bypassPermissions" in command_text
        assert "--always-approve" in command_text
        assert "--verbatim" in command_text
        assert "--model=grok-4.5" in command_text
        assert "--reasoning-effort=high" in command_text
        assert "--disable-web-search" in command_text
        assert "--no-auto-update" in command_text
        assert "--prompt-file=" in command_text
        assert "--continue" not in command_text

    def test_build_command_resume_adds_continue(
        self, mock_cost_limits: AgentCostLimits, mock_pricing: APIPricing
    ) -> None:
        agent = self._agent(mock_cost_limits, mock_pricing)
        assert "--continue" in " ".join(agent._build_command(resume=True))

    def test_build_command_optional_toggles(
        self, mock_cost_limits: AgentCostLimits, mock_pricing: APIPricing
    ) -> None:
        agent = self._agent(
            mock_cost_limits,
            mock_pricing,
            max_turns=25,
            disable_web_search=False,
            disable_subagents=True,
        )
        command_text = " ".join(agent._build_command(resume=False))

        assert "--max-turns=25" in command_text
        assert "--no-subagents" in command_text
        assert "--disable-web-search" not in command_text


class TestGrokCliUsageParsing:
    def test_parse_line_reads_assistant_usage(
        self, mock_pricing: APIPricing
    ) -> None:
        cost, tokens, payload = GrokCliAgent.parse_line(
            _assistant_event(10612, 138, 4864), pricing=mock_pricing
        )
        assert payload is not None
        assert tokens is not None
        # input is folded to include cache_read so get_cost recovers the
        # uncached count.
        assert tokens.input == 10612 + 4864
        assert tokens.output == 138
        assert tokens.cache_read == 4864
        assert cost == pytest.approx(
            (10612 * 2.0 + 138 * 6.0 + 4864 * 0.30) / 1_000_000
        )

    def test_parse_line_ignores_result_event_usage(
        self, mock_pricing: APIPricing
    ) -> None:
        line = json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "total_cost_usd": 0.5,
                "usage": {"input_tokens": 999, "output_tokens": 99},
            }
        )
        cost, tokens, payload = GrokCliAgent.parse_line(
            line, pricing=mock_pricing
        )
        assert cost is None
        assert tokens is None
        assert payload is not None and payload["type"] == "result"

    def test_parse_line_tolerates_non_json(
        self, mock_pricing: APIPricing
    ) -> None:
        assert GrokCliAgent.parse_line("not json", pricing=mock_pricing) == (
            None,
            None,
            None,
        )

    def test_per_message_pricing_avoids_the_high_tier(
        self, mock_pricing: APIPricing
    ) -> None:
        """Run totals can exceed 200k while every prompt stays under it.

        Pricing the aggregate would push the whole run into the double-rate
        tier, so each message must be priced on its own.
        """
        messages = [_assistant_event(50_000, 500, 20_000) for _ in range(6)]

        per_message = sum(
            GrokCliAgent.parse_line(m, pricing=mock_pricing)[0] or 0.0
            for m in messages
        )

        summed = GrokCliAgent._usage_to_tokens(
            {
                "input_tokens": 6 * 50_000,
                "output_tokens": 6 * 500,
                "cache_read_input_tokens": 6 * 20_000,
                "cache_creation_input_tokens": 0,
            }
        )
        aggregate = mock_pricing.get_cost(summed)

        assert per_message == pytest.approx(
            6 * (50_000 * 2.0 + 500 * 6.0 + 20_000 * 0.30) / 1_000_000
        )
        assert aggregate == pytest.approx(2 * per_message)


class TestGrokCliParser:
    def _write(self, tmp_path: Path, lines: list[str]) -> Path:
        (tmp_path / "stdout.jsonl").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        return tmp_path

    def test_can_parse_grok_init_event(self, tmp_path: Path) -> None:
        artifact_dir = self._write(
            tmp_path,
            [
                json.dumps(
                    {
                        "type": "system",
                        "subtype": "init",
                        "model": "grok-4.5",
                        "tools": [],
                    }
                ),
                _assistant_event(10, 1, 0),
            ],
        )
        assert GrokCliParser().can_parse(artifact_dir) is True

    def test_does_not_claim_claude_code_output(self, tmp_path: Path) -> None:
        artifact_dir = self._write(
            tmp_path,
            [
                json.dumps(
                    {
                        "type": "system",
                        "subtype": "init",
                        "model": "claude-opus-5",
                        "tools": [],
                    }
                )
            ],
        )
        assert GrokCliParser().can_parse(artifact_dir) is False

    def test_parse_reports_grok_agent_type(self, tmp_path: Path) -> None:
        artifact_dir = self._write(
            tmp_path,
            [
                json.dumps(
                    {
                        "type": "system",
                        "subtype": "init",
                        "model": "grok-4.5",
                        "tools": [],
                    }
                ),
                _assistant_event(10, 1, 0),
            ],
        )
        trajectory = GrokCliParser().parse(artifact_dir)
        assert trajectory.agent_type == "grok_cli"
        assert trajectory.metadata["model"] == "grok-4.5"
        assert [step.step_type for step in trajectory.steps] == ["agent"]
