---
version: 1.0
last_updated: 2026-08-07
---

# Grok CLI Agent

The Grok CLI agent wraps xAI's [`grok` CLI](https://x.ai/cli) for executing coding tasks.

Unlike the other CLI agents, Grok does not authenticate with an API key. The CLI signs in over OIDC and stores a session in `~/.grok/auth.json`, so the agent stages a minimal copy of that file and mounts it into the container's home rather than exporting an environment variable.

## Quick Start

```bash
grok login          # once, on the host
```

```yaml
# configs/agents/grok_cli.yaml
type: grok_cli
version: 1.0.0
timeout: 7200
disable_web_search: true
extra_args:
  - --no-auto-update
cost_limits:
  cost_limit: 20
  step_limit: 250
  net_cost_limit: 0
```

```bash
slop-code run --agent grok_cli \
  --model grok_auth/grok-4.5 \
  --problem file_backup
```

## Configuration Schema

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `type` | `"grok_cli"` | Must be `grok_cli` |
| `version` | string | Release to install, matching `https://x.ai/cli/grok-<version>-linux-<arch>` (e.g. `"1.0.0"`). Required because the agent uses a custom Docker template. |
| `cost_limits` | AgentCostLimits | Cost and step limits |

### Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `binary` | string | `grok` | CLI binary name on `PATH` |
| `timeout` | int \| null | `null` | Per-checkpoint timeout in seconds |
| `max_turns` | int \| null | `null` | Passed as `--max-turns`; unset lets `cost_limits` bound the run |
| `disable_web_search` | bool | `true` | Passes `--disable-web-search` so runs do not depend on live search results |
| `disable_subagents` | bool | `false` | Passes `--no-subagents` |
| `grok_home` | Path \| null | `null` | Host directory holding CLI state; defaults to the parent of the resolved credential file (`~/.grok`) |
| `sync_auth_back` | bool | `false` | Copy the container's refreshed `auth.json` back to the host after each run |
| `extra_args` | list[str] | `["--no-auto-update"]` | Extra CLI arguments |
| `env` | dict[str, str] | `{}` | Environment variable overrides |

## Credentials

Register the credential file through the `grok_auth` provider:

```yaml
# configs/providers.yaml
providers:
  grok_auth:
    type: file
    file_path: ~/.grok/auth.json
    description: Grok CLI OIDC session credentials file
```

The agent copies only `auth.json` into a staged home, alongside a generated `config.toml` (permissions pre-approved, self-update disabled) and `trusted_folders.toml` (workspace pre-trusted). Sessions, logs, plugins, and caches from the host are deliberately left out, so container runs neither read nor grow host state.

### Token Refresh

The session access token is short-lived (roughly six hours) and is refreshed in-container against the staged copy, which is discarded at the end of the run. For runs long enough to outlive the host's token, set `sync_auth_back: true` so refreshes persist — but only with `--num-workers 1`, since parallel workers would race on the same file.

To bill an `XAI_API_KEY` instead of the CLI session, run the model under a different agent (for example `--agent opencode --model xai/grok-4.5`).

## Cost Accounting

`grok --output-format streaming-messages-json` emits NDJSON in the Anthropic Messages API wire format. Usage is accumulated from each `assistant` message rather than from the terminal `result` event, even though both report the same totals.

That distinction matters because grok's pricing tiers key off the size of an *individual* prompt. A multi-turn checkpoint easily sums past 200k input tokens while every single prompt stays under it, so pricing the run total in one go would push the whole checkpoint into the double-rate tier — in practice, exactly 2x the real cost. Each message is therefore priced on its own, and the `result` event's totals are used to cross-check the walk: if they disagree, the agent logs `agent.grok_cli.usage_mismatch` and falls back to the reported totals.

Anthropic-wire `input_tokens` excludes cached reads, while `APIPricing.get_cost` subtracts `cache_read` from `input`, so the agent folds the two back together to keep the uncached count intact. With `configs/models/grok-4.5.yaml`, the resulting cost reproduces the CLI's own `total_cost_usd` to the cent; grok's figure is also logged at debug level for comparison.

## Trajectories

Artifacts are written as `stdout.jsonl` and parsed by `GrokCliParser`, which reuses the Claude Code event walk (same wire format) and disambiguates on the `grok` model id in the init event.

## Troubleshooting

### `Grok CLI credentials not found at ~/.grok/auth.json`

Run `grok login` on the host.

### `Failed to read '.../prompt.txt': No such file or directory`

The staged mount is not visible inside the VM. On macOS with colima only `$HOME` is shared, so point Python's temp directory somewhere under it:

```bash
export TMPDIR=$HOME/.cache/scbench/tmp
```

### Docker connection errors

The Python Docker SDK ignores `docker context`. Point it at the socket explicitly:

```bash
export DOCKER_HOST=unix://$HOME/.colima/default/docker.sock
```

## See Also

- [Agent Configuration](../agent-config.md)
- [Providers Guide](../providers.md)
- [Models Guide](../models.md)
