"""memory-preflight: Deterministically classify tool and memory risk before agent execution.

Public promise: Deterministically classify tool and memory risk before an agent loads
context or executes tools.

This package provides a pure, synchronous, advisory-only preflight layer that:
  - Scores memory candidates for promotion/demotion
  - Classifies tool actions by risk category (high/medium/low)
  - Builds preflight advisories with redaction for secret-like values
  - Tracks bi-temporal fact metadata

All classification is local and synchronous — no network calls, no agent execution blocking.
The advisory is purely informational; caller decides what to do with it.
"""

from memory_preflight._governance import (
    MemoryPreflightAdvisory,
    MemoryPreflightDecision,
    FactMetadata,
    classify_action,
    score_candidate,
    build_memory_preflight_advisory,
)

__version__ = "0.1.0"
__all__ = [
    "MemoryPreflightAdvisory",
    "MemoryPreflightDecision",
    "FactMetadata",
    "classify_action",
    "score_candidate",
    "build_memory_preflight_advisory",
]