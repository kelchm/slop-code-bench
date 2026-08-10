"""Grok CLI agent implementation.

Runs xAI's ``grok`` CLI headlessly inside the submission container. Grok
authenticates with the OIDC session credentials the CLI writes to
``~/.grok/auth.json`` rather than an API key, so a staged copy of that file is
mounted into the container's home instead of exporting an environment
variable.

``--output-format streaming-messages-json`` emits NDJSON in the Anthropic
Messages API wire format. Usage is accumulated per ``assistant`` message so
that prompt-size pricing tiers apply per prompt; the terminal ``result`` event
reports the same totals and is used to cross-check the walk.
"""

from __future__ import annotations

import functools
import json
import shutil
import tempfile
import typing as tp
from pathlib import Path

from jinja2 import Template
from pydantic import Field

from slop_code.agent_runner.agent import Agent
from slop_code.agent_runner.agent import AgentConfigBase
from slop_code.agent_runner.agents.cli_utils import AgentCommandResult
from slop_code.agent_runner.agents.cli_utils import stream_cli_command
from slop_code.agent_runner.agents.utils import HOME_PATH
from slop_code.agent_runner.credentials import CredentialType
from slop_code.agent_runner.credentials import ProviderCredential
from slop_code.agent_runner.models import AgentCostLimits
from slop_code.agent_runner.models import AgentError
from slop_code.agent_runner.registry import register_agent
from slop_code.common.llms import APIPricing
from slop_code.common.llms import ModelDefinition
from slop_code.common.llms import ThinkingPreset
from slop_code.common.llms import TokenUsage
from slop_code.execution import DockerEnvironmentSpec
from slop_code.execution import EnvironmentSpec
from slop_code.execution import Session
from slop_code.execution import StreamingRuntime

# Grok exposes low/medium/high; SCB's xhigh has no distinct tier.
THINKING_TO_EFFORT: dict[str, str] = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
}

# Written into the staged home so the CLI never prompts for folder trust and
# never tries to reach the plugin marketplace from inside the container.
GROK_CONFIG_TOML = """\
[ui]
permission_mode = "always-approve"
yolo = true

[cli]
installer = "none"
"""

TRUSTED_FOLDERS_TOML = """\
[folders."{workdir}"]
trusted = true
decided_at = 0
"""


class GrokCliConfig(AgentConfigBase):
    """Configuration for ``GrokCliAgent`` instances."""

    type: tp.Literal["grok_cli"] = "grok_cli"
    version: str
    binary: str = "grok"
    docker_template: Path = Path(__file__).parent / "docker.j2"
    max_turns: int | None = Field(
        default=None,
        description="Cap on agent turns per checkpoint (grok --max-turns).",
    )
    disable_web_search: bool = Field(
        default=True,
        description="Disable grok's web search/fetch tools for reproducibility.",
    )
    disable_subagents: bool = Field(
        default=False,
        description="Disable grok's subagent spawning.",
    )
    grok_home: Path | None = Field(
        default=None,
        description=(
            "Host directory holding grok CLI state. Defaults to the parent of "
            "the resolved credential file (normally ~/.grok)."
        ),
    )
    sync_auth_back: bool = Field(
        default=False,
        description=(
            "Copy the container's refreshed auth.json back to the host after "
            "each run. Keeps long runs authenticated but races across parallel "
            "workers; leave off unless running a single worker."
        ),
    )
    extra_args: list[str] = Field(
        default_factory=lambda: ["--no-auto-update"],
        description="Additional arguments appended to the CLI invocation.",
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variable overrides applied to the invocation.",
    )
    timeout: int | None = Field(
        default=None,
        description="Optional timeout (in seconds) for the CLI invocation.",
    )

    def get_docker_file(self, base_image: str) -> str | None:
        """Render the Docker template with version."""
        if self.docker_template is None:
            return None
        template = self.docker_template.read_text()
        return Template(template).render(
            base_image=base_image, version=self.version
        )


class GrokCliAgent(Agent):
    """Agent implementation built on top of the Grok CLI."""

    PROMPT_FILENAME = "prompt.txt"
    COMMAND_FILENAME = "command.txt"
    STDOUT_FILENAME = "stdout.jsonl"
    STDERR_FILENAME = "stderr.log"

    # Mount points inside the container.
    GROK_HOME_BIND = f"{HOME_PATH}/.grok"
    RUN_DIR_BIND = f"{HOME_PATH}/scb_run"

    def __init__(
        self,
        problem_name: str,
        verbose: bool,  # noqa: FBT001
        image: str,
        # From base config
        cost_limits: AgentCostLimits,
        pricing: APIPricing | None,
        credential: ProviderCredential | None,
        # Grok specific
        binary: str,
        model: str,
        reasoning_effort: str | None,
        max_turns: int | None,
        disable_web_search: bool,  # noqa: FBT001
        disable_subagents: bool,  # noqa: FBT001
        grok_home: Path,
        sync_auth_back: bool,  # noqa: FBT001
        timeout: int | None,
        extra_args: list[str],
        env: dict[str, str],
    ) -> None:
        super().__init__(
            agent_name="grok_cli",
            problem_name=problem_name,
            cost_limits=cost_limits,
            pricing=pricing,
            verbose=verbose,
        )

        self.credential = credential
        self.binary = binary
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_turns = max_turns
        self.disable_web_search = disable_web_search
        self.disable_subagents = disable_subagents
        self.grok_home = grok_home
        self.sync_auth_back = sync_auth_back
        self.timeout = timeout
        self.extra_args = extra_args
        self.env = env

        self._image = image
        self._session: Session | None = None
        self._environment: EnvironmentSpec | None = None
        self._runtime: StreamingRuntime | None = None
        self._tmp_dir: tempfile.TemporaryDirectory | None = None

        self._resume_next_run = False
        self._last_prompt: str = ""
        self._last_command_text: str = ""
        self._last_command_stdout: str = ""
        self._last_command_stderr: str = ""
        self._last_reported_cost: float | None = None
        self._last_computed_cost: float = 0.0

    @classmethod
    def _from_config(
        cls,
        config: AgentConfigBase,
        model: ModelDefinition,
        credential: ProviderCredential,
        problem_name: str,
        verbose: bool,  # noqa: FBT001
        image: str | None,
        thinking_preset: ThinkingPreset | None = None,
        thinking_max_tokens: int | None = None,
    ) -> Agent:
        """Create a GrokCliAgent from a GrokCliConfig."""
        if not isinstance(config, GrokCliConfig):
            raise TypeError(
                f"Expected GrokCliConfig, got {type(config).__name__}"
            )
        if image is None:
            raise ValueError("GrokCliAgent requires an image")

        agent_settings = model.get_agent_settings("grok_cli") or {}
        configured_model = agent_settings.get("model_name")
        if not isinstance(configured_model, str) or not configured_model:
            configured_model = model.get_model_slug(credential.provider)

        env = dict(config.env)
        env_overrides = agent_settings.get("env_overrides")
        if isinstance(env_overrides, dict):
            env = {**{k: str(v) for k, v in env_overrides.items()}, **env}

        # Resolve reasoning effort: CLI/config override > model default.
        thinking: ThinkingPreset | None = thinking_preset
        if thinking is None and thinking_max_tokens is None:
            thinking, _ = model.get_thinking_config("grok_cli")
        reasoning_effort = (
            THINKING_TO_EFFORT.get(thinking) if thinking is not None else None
        )

        grok_home = config.grok_home
        if grok_home is None:
            if credential.credential_type != CredentialType.FILE:
                raise AgentError(
                    "GrokCliAgent requires a file-based credential (the grok "
                    "CLI's auth.json) or an explicit grok_home in the agent "
                    f"config; got a {credential.credential_type.value} "
                    f"credential for provider '{credential.provider}'."
                )
            grok_home = Path(credential.source).expanduser().parent
        grok_home = grok_home.expanduser()

        return cls(
            problem_name=problem_name,
            verbose=verbose,
            image=image,
            cost_limits=config.cost_limits,
            pricing=model.pricing,
            credential=credential,
            binary=config.binary,
            model=configured_model,
            reasoning_effort=reasoning_effort,
            max_turns=config.max_turns,
            disable_web_search=config.disable_web_search,
            disable_subagents=config.disable_subagents,
            grok_home=grok_home,
            sync_auth_back=config.sync_auth_back,
            timeout=config.timeout,
            extra_args=config.extra_args,
            env=env,
        )

    @staticmethod
    def _usage_to_tokens(usage: dict) -> TokenUsage:
        """Convert one Anthropic-wire usage block to SCB's TokenUsage.

        Anthropic-wire ``input_tokens`` excludes cached reads, while
        ``APIPricing.get_cost`` subtracts ``cache_read`` from ``input``. Fold
        them back together so the uncached count survives the round trip.
        """
        cache_read = int(usage.get("cache_read_input_tokens") or 0)
        return TokenUsage(
            input=int(usage.get("input_tokens") or 0) + cache_read,
            output=int(usage.get("output_tokens") or 0),
            cache_read=cache_read,
            cache_write=int(usage.get("cache_creation_input_tokens") or 0),
            reasoning=0,
        )

    @classmethod
    def parse_line(
        cls,
        line: str,
        pricing: APIPricing | None = None,
    ) -> tuple[float | None, TokenUsage | None, dict | None]:
        """Parse one NDJSON line of grok's streaming-messages-json output.

        Tokens come from each ``assistant`` message rather than the terminal
        ``result`` event, even though the latter reports the same totals.
        Grok's pricing tiers key off the size of an individual prompt, so
        pricing the run-total input at once would push every multi-turn
        checkpoint into the over-200k tier and roughly double the cost.
        """
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            return None, None, None

        if not isinstance(payload, dict):
            return None, None, None
        if payload.get("type") != "assistant":
            return None, None, payload

        message = payload.get("message")
        if not isinstance(message, dict):
            return None, None, payload

        tokens = cls._usage_to_tokens(message.get("usage") or {})
        cost = pricing.get_cost(tokens) if pricing else 0.0
        return cost, tokens, payload

    @property
    def session(self) -> Session:
        if self._session is None:
            raise AgentError("GrokCliAgent has not been set up with a session")
        return self._session

    @property
    def spec(self) -> EnvironmentSpec:
        if self._environment is None:
            raise AgentError("GrokCliAgent has not been set up with a session")
        return self._environment

    @property
    def runtime(self) -> StreamingRuntime:
        if self._runtime is None:
            raise AgentError("GrokCliAgent has not been set up with a runtime")
        return self._runtime

    @property
    def tmp_dir(self) -> Path:
        if self._tmp_dir is None:
            raise AgentError("GrokCliAgent has not been set up")
        return Path(self._tmp_dir.name)

    def setup(self, session: Session) -> None:
        self._tmp_dir = tempfile.TemporaryDirectory()
        self._session = session
        self._environment = session.spec

        workdir = "/workspace"
        if isinstance(session.spec, DockerEnvironmentSpec):
            workdir = session.spec.docker.workdir

        staged_home = self._stage_grok_home(workdir)
        run_dir = self.tmp_dir / "run"
        run_dir.mkdir(parents=True, exist_ok=True)

        self._runtime = session.spawn(
            mounts={
                str(staged_home.absolute()): {
                    "bind": self.GROK_HOME_BIND,
                    "mode": "rw",
                },
                str(run_dir.absolute()): {
                    "bind": self.RUN_DIR_BIND,
                    "mode": "rw",
                },
            },
            env_vars={"HOME": HOME_PATH},
            image=self._image,
            user="agent",
            disable_setup=True,
        )

    def _stage_grok_home(self, workdir: str) -> Path:
        """Build a minimal ~/.grok to mount, isolated from the host's state.

        Only the credential file is copied; sessions, logs, plugins, and caches
        are deliberately left behind so container runs neither read nor grow
        the host's grok state.
        """
        staged = self.tmp_dir / "grok_home"
        staged.mkdir(parents=True, exist_ok=True)

        auth_src = self.grok_home / "auth.json"
        if not auth_src.is_file():
            raise AgentError(
                f"Grok CLI credentials not found at {auth_src}. "
                "Run `grok login` on the host first."
            )
        shutil.copy2(auth_src, staged / "auth.json")

        (staged / "config.toml").write_text(GROK_CONFIG_TOML, encoding="utf-8")
        (staged / "trusted_folders.toml").write_text(
            TRUSTED_FOLDERS_TOML.format(workdir=workdir), encoding="utf-8"
        )
        return staged

    def run(self, task: str) -> None:
        resume = self._resume_next_run
        self._resume_next_run = False

        self._last_prompt = task
        self._last_command_stdout = ""
        self._last_command_stderr = ""
        self._last_reported_cost = None
        self._last_computed_cost = 0.0

        log_kwargs: dict[str, tp.Any] = {
            "workspace": str(self.session.working_dir),
            "prompt_chars": len(task),
            "environment": self.session.spec.type,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "resume": resume,
        }
        if isinstance(self.session.spec, DockerEnvironmentSpec):
            log_kwargs["image"] = self.session.spec.docker.image
        self.log.info("agent.grok_cli.start", **log_kwargs)

        command_result = self._run_invocation(task, resume=resume)
        self._last_command_stdout = command_result.stdout or ""
        self._last_command_stderr = command_result.stderr or ""

        if self.sync_auth_back:
            self._sync_auth_back()

        self._sync_usage(command_result.usage_totals)

        runtime_result = command_result.result
        if runtime_result is None:
            message = "Grok process failed to start"
            self.log.error("agent.grok_cli.start_failed", error_message=message)
            raise AgentError(message)

        if runtime_result.timed_out:
            message = (
                f"Grok process timed out after {self.timeout}s."
                if self.timeout is not None
                else "Grok process timed out."
            )
            self.log.error(
                "agent.grok_cli.timeout",
                error_message=message,
                timeout=self.timeout,
            )
            raise AgentError(message)

        if runtime_result.exit_code != 0:
            message = (
                f"Grok process failed with exit code {runtime_result.exit_code}"
            )
            if runtime_result.stderr:
                message = f"{message}\n--- Stderr ---\n{runtime_result.stderr.strip()}"
            self.log.error(
                "agent.grok_cli.exit",
                error_message=message,
                exit_code=runtime_result.exit_code,
            )
            raise AgentError(message)

    def retry(self) -> None:
        """Continue the existing grok session after a failed run."""
        self._resume_next_run = True
        super().retry()

    def _run_invocation(self, task: str, *, resume: bool) -> AgentCommandResult:
        """Execute a Grok CLI invocation and return results."""
        prompt_path = self.tmp_dir / "run" / self.PROMPT_FILENAME
        prompt_path.write_text(task, encoding="utf-8")

        command_text = " ".join(self._build_command(resume=resume))
        self._last_command_text = command_text
        env_overrides = {key: str(value) for key, value in self.env.items()}
        parser = functools.partial(self.parse_line, pricing=self.pricing)

        total_tokens = TokenUsage()
        total_cost = 0.0
        step_count = 0
        runtime_result = None
        reported_tokens: TokenUsage | None = None

        for item in stream_cli_command(
            runtime=self.runtime,
            command=command_text,
            parser=parser,
            env=env_overrides,
            timeout=(float(self.timeout) if self.timeout is not None else None),
        ):
            if not isinstance(item, tuple):
                runtime_result = item
                break

            cost, tokens, payload = item
            if tokens is not None:
                total_tokens = total_tokens + tokens
                total_cost += cost or 0.0

            if payload is None:
                continue
            if payload.get("type") == "assistant":
                step_count += 1
            elif payload.get("type") == "result":
                reported = payload.get("total_cost_usd")
                if isinstance(reported, int | float):
                    self._last_reported_cost = float(reported)
                reported_tokens = self._usage_to_tokens(
                    payload.get("usage") or {}
                )
                if payload.get("is_error"):
                    self.log.warning(
                        "agent.grok_cli.result_error",
                        subtype=payload.get("subtype"),
                        stop_reason=payload.get("stop_reason"),
                    )

        # The result event reports the same totals the assistant messages sum
        # to. If they disagree, the per-message walk missed something (a
        # dropped line, an unseen event shape), so trust the reported totals
        # and fall back to pricing them in one go.
        if reported_tokens is not None and (
            reported_tokens.input != total_tokens.input
            or reported_tokens.output != total_tokens.output
        ):
            self.log.warning(
                "agent.grok_cli.usage_mismatch",
                summed_input=total_tokens.input,
                summed_output=total_tokens.output,
                reported_input=reported_tokens.input,
                reported_output=reported_tokens.output,
            )
            total_tokens = reported_tokens
            total_cost = (
                self.pricing.get_cost(reported_tokens) if self.pricing else 0.0
            )

        self._last_computed_cost = total_cost
        self.usage.steps += step_count
        stdout = runtime_result.stdout if runtime_result else ""
        stderr = runtime_result.stderr if runtime_result else ""

        return AgentCommandResult(
            result=runtime_result,
            steps=[],
            usage_totals={
                "input_tokens": total_tokens.input,
                "output_tokens": total_tokens.output,
                "cached_input_tokens": total_tokens.cache_read,
                "cached_write_tokens": total_tokens.cache_write,
                "total_tokens": total_tokens.input + total_tokens.output,
                "steps": step_count,
            },
            stdout=stdout,
            stderr=stderr,
        )

    def _build_command(self, *, resume: bool) -> list[str]:
        command = [
            self.binary,
            "--output-format=streaming-messages-json",
            "--permission-mode=bypassPermissions",
            "--always-approve",
            "--verbatim",
            f"--model={self.model}",
        ]

        if self.reasoning_effort:
            command.append(f"--reasoning-effort={self.reasoning_effort}")
        if self.max_turns is not None:
            command.append(f"--max-turns={self.max_turns}")
        if self.disable_web_search:
            command.append("--disable-web-search")
        if self.disable_subagents:
            command.append("--no-subagents")
        if resume:
            command.append("--continue")

        command.extend(self.extra_args)
        command.append(
            f"--prompt-file={self.RUN_DIR_BIND}/{self.PROMPT_FILENAME}"
        )
        return command

    def _sync_auth_back(self) -> None:
        """Copy a refreshed auth.json out of the staged home onto the host."""
        staged_auth = self.tmp_dir / "grok_home" / "auth.json"
        host_auth = self.grok_home / "auth.json"
        try:
            if staged_auth.is_file():
                shutil.copy2(staged_auth, host_auth)
        except OSError as exc:
            self.log.warning(
                "agent.grok_cli.auth_sync_failed",
                error_message=str(exc),
                destination=str(host_auth),
            )

    def _sync_usage(self, totals: dict[str, int]) -> None:
        totals = totals or {}
        tokens = TokenUsage(
            input=int(totals.get("input_tokens") or 0),
            output=int(totals.get("output_tokens") or 0),
            cache_read=int(totals.get("cached_input_tokens") or 0),
            cache_write=int(totals.get("cached_write_tokens") or 0),
        )
        # Priced per assistant message during the stream so prompt-size tiers
        # apply per prompt rather than to the run total.
        cost = self._last_computed_cost
        self.usage.cost += cost
        self.usage.net_tokens += tokens
        self.usage.current_tokens = tokens

        if self._last_reported_cost is not None:
            self.log.debug(
                "agent.grok_cli.cost",
                catalog_cost=cost,
                grok_reported_cost=self._last_reported_cost,
            )

        if self.cost_limits.is_above_limits(
            self.usage,
            prior_cost=self.prior_cost,
        ):
            raise AgentError("GrokCliAgent exceeded configured usage limits")

    def reset(self) -> None:
        """Drop conversation state so the next checkpoint starts fresh.

        The workspace is untouched; only the grok session linkage is cleared,
        which happens naturally because runs omit ``--continue``.
        """
        self._resume_next_run = False
        self._last_prompt = ""
        self._last_command_text = ""
        self._last_command_stdout = ""
        self._last_command_stderr = ""
        self._last_reported_cost = None
        self._last_computed_cost = 0.0

    def save_artifacts(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

        if self._last_prompt:
            (path / self.PROMPT_FILENAME).write_text(
                self._last_prompt, encoding="utf-8"
            )
        if self._last_command_text:
            (path / self.COMMAND_FILENAME).write_text(
                self._last_command_text, encoding="utf-8"
            )

        (path / self.STDOUT_FILENAME).write_text(
            self._last_command_stdout, encoding="utf-8"
        )
        (path / self.STDERR_FILENAME).write_text(
            self._last_command_stderr, encoding="utf-8"
        )

    def cleanup(self) -> None:
        if self._runtime is not None:
            self._runtime.cleanup()
            self._runtime = None

        if self._tmp_dir is not None:
            self._tmp_dir.cleanup()
            self._tmp_dir = None

        self._session = None


register_agent("grok_cli", GrokCliAgent)
