# memory-preflight

**Deterministically classify tool and memory risk before an agent loads context or executes tools.**

[![CI](https://github.com/brunopetrovic/memory-preflight/actions/workflows/ci.yml/badge.svg)](https://github.com/brunopetrovic/memory-preflight/actions/workflows/ci.yml)
[![Python versions](https://img.shields.io/pypi/pyversions/memory-preflight.svg)](https://pypi.org/project/memory-preflight/)

## Overview

`memory-preflight` is a pure, synchronous, advisory-only preflight layer for AI agents. It provides:

- **Risk classification** — categorizes tool actions as `high`, `medium`, or `low` risk across domains like credentials, social, cron, config, memory, code, preferences, and external apps.
- **Redaction preview** — sanitizes secret-like values (API keys, tokens, passwords) in advisory text previews to prevent accidental leakage in logs.
- **Memory candidate scoring** — promotes/demotes memory entries based on evidence strength, user preferences, system conventions, procedures, contradictions, and staleness.
- **Bi-temporal fact metadata** — tracks observed/valid time ranges for memory facts with UTC-aware comparisons.

**Safety note:** Classification is 100% local and synchronous — no network calls, no agent execution blocking. The advisory is purely informational; the caller decides what to do with it.

## Installation

```bash
pip install memory-preflight
```

Or install from source:

```bash
pip install -e .
```

## Python API

```python
from memory_preflight import (
    classify_action,
    score_candidate,
    build_memory_preflight_advisory,
    FactMetadata,
    MemoryPreflightAdvisory,
    MemoryPreflightDecision,
)

# Classify a tool action by name + optional file path + text content
decision = classify_action(
    action_name="update_config",
    tool_name="update_config",
    file_path="~/.hermes/config.yaml",
    text_content=None,
)
# decision.risk_level  → "high"
# decision.required   → True
# decision.categories → ["config"]
# decision.recommended_queries → [...]

# Build a full advisory for a tool call
advisory = build_memory_preflight_advisory(
    tool_name="terminal",
    args={"command": "edit cron job schedule"},
)
# advisory.risk_level → "high"
# advisory.categories  → ["cron"]
# advisory._text_preview → "edit cron job schedule"  (secrets redacted)

# Score a memory candidate
result = score_candidate(
    text="User prefers dark theme always",
    user_preference=True,
)
# result["decision"]  → "promote_user_memory"
# result["score"]     → 0.8
# result["suggested_status"] → "active"

# Bi-temporal metadata
from datetime import datetime, timezone, timedelta
now = datetime.now(timezone.utc)
meta = FactMetadata(
    observed_at=now - timedelta(hours=2),
    valid_from=now - timedelta(hours=1),
    valid_until=now + timedelta(hours=2),
    source="session_log",
    confidence=0.9,
    status="active",
)
assert meta.is_current(now) is True
```

## CLI

```bash
# Classify a tool call
memory-preflight classify --tool terminal --args '{"command": "update_config"}'

# Score a memory candidate
memory-preflight score --text "User prefers dark theme" --user-preference
```

### classify output example

```json
{
  "tool_name": "terminal",
  "risk_level": "high",
  "categories": ["config", "cron"],
  "recommended_queries": [
    "hermes configuration user preferences",
    "system environment settings",
    "cron schedule preferences",
    "registered background tasks"
  ],
  "reason": "Preflight detected protected categories: config, cron (Risk: HIGH)",
  "_text_preview": "update_config --set-cron '0 2 * * *'"
}
```

### score output example

```json
{
  "decision": "promote_user_memory",
  "score": 0.8,
  "reasons": [
    "Single observation source",
    "Contains user preferences or boundary declarations"
  ],
  "suggested_status": "active"
}
```

## Risk Categories

| Category | Risk | Description |
|----------|------|-------------|
| `credentials` | HIGH | API keys, tokens, passwords, OAuth, private keys |
| `social` | HIGH | Twitter, Mastodon, LinkedIn, Discord public posting |
| `cron` | HIGH | Cron jobs, scheduled tasks, recurring events |
| `provider` | HIGH | Model routing, provider switching, LLM endpoint config |
| `memory` | HIGH | Memory edits, add/delete memory, soul.md/user.md changes |
| `config` | HIGH | Configuration changes, hermes_config, settings.yaml |
| `external-app` | MEDIUM | GitHub, Notion, Jira, Slack, Trello automation |
| `code` | MEDIUM | Code execution, shell commands, bash, eval |
| `preferences` | MEDIUM | User preferences, theme, UI settings, customization |
| `low-risk` | LOW | Read-only actions, chat-only interactions |

## Policy Examples

### Custom trigger patterns

```python
from memory_preflight._governance import _TRIGGER_PATTERNS, classify_action

# Inspect existing patterns
print(list(_TRIGGER_PATTERNS.keys()))
# ['config', 'cron', 'provider', 'social', 'external-app', 'code', 'memory', 'preferences', 'credentials']

# Override patterns before calling classify_action
import re
_TRIGGER_PATTERNS["my-custom"] = re.compile(r"(?i)my-custom-domain")
```

### Agent integration hook (pseudocode)

```python
def execute_tool(tool_name, args):
    advisory = build_memory_preflight_advisory(tool_name, args)
    if advisory.risk_level == "high":
        context = prefetch_context(advisory.recommended_queries)
        log_info(advisory.log_message())
    elif advisory.risk_level == "low":
        log_debug(advisory.log_message())
    # Advisory never blocks — caller decides
    return do_tool_execution(tool_name, args)
```

## Development

```bash
pip install -e ".[dev]"
pytest
memory-preflight classify --tool terminal --args '{"command": "ls"}'
```

## License

MIT