"""memory_preflight._governance: Pure Python memory governance layer.

Re-factored from Hermes memory_governance.py.  No Hermes imports, no private paths.
Pure stdlib.  Deterministic and safe-by-default.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

# ---------------------------------------------------------------------------
# Advisory-only preflight helper
# ---------------------------------------------------------------------------

_MAX_TEXT_PREVIEW = 300

# Known secret patterns for preview redaction (conservative, stdlib-only).
# Covers the most common API key / token / password shapes.
# These patterns are checked AFTER classification so they do not affect
# the category determination — they only sanitize what gets stored in
# the advisory's _text_preview for logging/debugging purposes.
_SECRET_PATTERNS = [
    # OpenAI / OpenRouter / Anthropic
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    # GitHub PATs
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    # Slack tokens
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{20,}"),
    # Generic Bearer tokens
    re.compile(r"Bearer\s+[A-Za-z0-9_.-]{20,}", re.IGNORECASE),
    # Generic api_key / token assignments (key=value with long value)
    re.compile(r"(?:api[_-]?key|token|secret|password)\s*[:=]\s*[A-Za-z0-9_./-]{20,}", re.IGNORECASE),
    # AWS-style access keys (20+ alphanumeric)
    re.compile(r"(?:AKIA|A3T|ASIA)[A-Z0-9]{16}"),
]


def _redact_for_preview(text: str) -> str:
    """Redact obvious secrets from text preview using conservative stdlib patterns.

    This runs AFTER classification so it does not affect category detection.
    It only sanitizes what gets stored in MemoryPreflightAdvisory._text_preview
    to prevent accidental secret leakage in logs/debug output.
    """
    if not text:
        return text
    result = text
    for pattern in _SECRET_PATTERNS:
        result = pattern.sub(lambda m: _mask_token(m.group(0)), result)
    return result


def _mask_token(token: str) -> str:
    """Mask a token for preview display — conservatively hides the value."""
    if len(token) <= 8:
        return "***"
    return token[:4] + "***" + token[-4:]


@dataclass
class MemoryPreflightAdvisory:
    """Advisory metadata from memory preflight — never blocks execution."""
    tool_name: str
    risk_level: str  # "high", "medium", "low"
    categories: List[str]
    recommended_queries: List[str]
    reason: str
    _text_preview: str = ""  # sanitized, truncated

    def log_message(self) -> str:
        """Build a one-line advisory log message."""
        cats = ", ".join(self.categories)
        queries = "; ".join(self.recommended_queries[:3]) if self.recommended_queries else ""
        base = f"[MemoryPreflight] {self.tool_name} → risk={self.risk_level.upper()} categories=({cats})"
        if queries:
            base += f" | suggested_queries: {queries}"
        return base

    def to_dict(self) -> Dict[str, Any]:
        """Serialize advisory to dict for JSON output."""
        return {
            "tool_name": self.tool_name,
            "risk_level": self.risk_level,
            "categories": self.categories,
            "recommended_queries": self.recommended_queries,
            "reason": self.reason,
            "_text_preview": self._text_preview,
        }


def build_memory_preflight_advisory(
    tool_name: str,
    args: Optional[Dict[str, Any]],
) -> MemoryPreflightAdvisory:
    """Build an advisory preflight for a tool call — no side effects, no blocking.

    Extracts file_path and text_content hints from common args keys, sanitizes
    text preview to avoid leaking secrets, then calls classify_action().
    Returns an advisory object; caller decides what to do with it (log only).
    """
    if args is None:
        args = {}

    # Extract file_path from common keys
    file_path = (
        args.get("path")
        or args.get("file_path")
        or args.get("target")
        or args.get("file")
        or ""
    )
    if isinstance(file_path, list):
        file_path = str(file_path[0]) if file_path else ""

    # Extract text content from common keys for classification.
    # Keep raw text for classification; sanitize only the stored preview.
    _text_keys = ("content", "text", "command", "query", "prompt", "body", "message", "description")
    text_parts = []
    for key in _text_keys:
        if key in args and isinstance(args[key], str):
            preview = args[key].strip()
            if preview:
                text_parts.append(preview)

    # Extract goal/context from delegate_task for classification signal.
    # These are NOT redacted before classification — delegate_task goal/context
    # may contain operational details the classifier needs to see.
    if tool_name == "delegate_task":
        for dk in ("goal", "context"):
            val = args.get(dk)
            if val and isinstance(val, str) and val.strip():
                text_parts.append(val.strip())

    raw_text = " ".join(text_parts)
    # Use raw text for classification; classify_action does not modify text.
    classification_text = raw_text[:_MAX_TEXT_PREVIEW] if raw_text else None

    decision = classify_action(
        action_name=tool_name,
        tool_name=tool_name,
        file_path=file_path if file_path else None,
        text_content=classification_text,
    )

    # Redact before storing in advisory — prevents secret leakage in logs/debug.
    safe_preview = _redact_for_preview(classification_text) if classification_text else ""

    return MemoryPreflightAdvisory(
        tool_name=tool_name,
        risk_level=decision.risk_level,
        categories=decision.categories,
        recommended_queries=decision.recommended_queries,
        reason=decision.reason,
        _text_preview=safe_preview,
    )


# ---------------------------------------------------------------------------
# 1) Memory Preflight Decision Module
# ---------------------------------------------------------------------------

@dataclass
class MemoryPreflightDecision:
    """Represents a preflight decision on whether memory lookup is required."""
    required: bool
    risk_level: str  # "high", "medium", "low"
    categories: List[str]
    recommended_queries: List[str]
    reason: str


# Triggers matching domains
_TRIGGER_PATTERNS = {
    "config": re.compile(r"(?i)config\.yaml|hermes_config|settings\.yaml|update_config|set_config|configure|config"),
    "cron": re.compile(r"(?i)cron|jobs\.json|schedule|recurring|cronjob|add_job|delete_job"),
    "provider": re.compile(r"(?i)provider|model_routing|set_model|change_provider|api_endpoint|llm_model|model_picker"),
    "social": re.compile(r"(?i)social|twitter|tweet|post_tweet|mastodon|linkedin|publish|send_message_public|telegram|discord"),
    "external-app": re.compile(r"(?i)notion|github|issue|pull_request|pr_number|jira|slack|trello"),
    "code": re.compile(r"(?i)code|run_command|execute_code|bash|shell|python|cmd|eval"),
    "memory": re.compile(r"(?i)memory|edit_memory|add_memory|delete_memory|clear_memory|soul\.md|user\.md|memory\.md|recall|prefetch"),
    "preferences": re.compile(r"(?i)preference|user_preference|theme|settings|customization"),
    "credentials": re.compile(r"(?i)credential|auth|api_key|secret|token|password|oauth|private_key"),
}


def classify_action(
    action_name: str,
    tool_name: Optional[str] = None,
    file_path: Optional[str] = None,
    text_content: Optional[str] = None,
) -> MemoryPreflightDecision:
    """Classify whether an action should require memory/session lookup before execution.

    Pure, deterministic, and easily testable.
    """
    categories: List[str] = []

    # Helper to check match against all inputs
    def matches_pattern(pattern: re.Pattern) -> bool:
        inputs = [action_name, tool_name, file_path, text_content]
        return any(pattern.search(str(val)) for val in inputs if val is not None)

    for cat_name, pattern in _TRIGGER_PATTERNS.items():
        if matches_pattern(pattern):
            categories.append(cat_name)

    # If no categories matched, default to low-risk
    if not categories:
        categories = ["low-risk"]

    # Determine risk level
    # Protected domains/High risk: provider, cron, config, memory edits, social, credentials
    high_risk_cats = {"credentials", "social", "cron", "provider", "memory", "config"}
    medium_risk_cats = {"external-app", "code", "preferences"}

    risk_level = "low"
    for cat in categories:
        if cat in high_risk_cats:
            risk_level = "high"
            break
        elif cat in medium_risk_cats:
            risk_level = "medium"

    # Memory lookup is required for all high and medium risk categories
    required = risk_level in ("high", "medium")

    # Generate deterministic recommended queries
    recommended_queries: List[str] = []
    query_map = {
        "config": ["hermes configuration user preferences", "system environment settings"],
        "cron": ["cron schedule preferences", "registered background tasks"],
        "provider": ["configured providers models routing", "preferred LLM endpoint settings"],
        "social": ["social media credentials", "public posting safety boundaries"],
        "external-app": ["external service tokens", "automation workflow preferences"],
        "code": ["code execution sandbox guidelines", "permitted command boundaries"],
        "memory": ["user memory retention rules", "memory update preferences"],
        "preferences": ["user UI profile settings", "custom preferences"],
        "credentials": ["api credentials secrets format", "auth key storage"],
    }

    for cat in categories:
        if cat in query_map:
            for q in query_map[cat]:
                if q not in recommended_queries:
                    recommended_queries.append(q)

    # Final decision reason summary
    if required:
        reason = f"Preflight detected protected categories: {', '.join(categories)} (Risk: {risk_level.upper()})"
    else:
        reason = "Preflight classified as low-risk read-only/chat action"

    return MemoryPreflightDecision(
        required=required,
        risk_level=risk_level,
        categories=categories,
        recommended_queries=recommended_queries,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# 2) Promotion/Demotion Scoring Module
# ---------------------------------------------------------------------------

def score_candidate(
    text: str,
    evidence_count: int = 1,
    observed_at: Optional[Union[datetime, str]] = None,
    current_time: Optional[Union[datetime, str]] = None,
    user_preference: bool = False,
    system_convention: bool = False,
    procedure: bool = False,
    temporary_task_state: bool = False,
    credential_secret: bool = False,
    stale_soon: bool = False,
    contradiction: bool = False,
) -> Dict[str, Any]:
    """Score memory candidate and return promotion/demotion decisions.

    Outputs:
      decision: promote_user_memory|promote_operational_memory|promote_skill|archive|reject|needs_review
      score: float (0.0 to 1.0)
      reasons: List[str]
      suggested_status: active|historical|deprecated|provisional|contradicted
    """
    reasons: List[str] = []

    # 1. Base Score calculation (starts at 0.5)
    score = 0.5

    if evidence_count > 1:
        score += min((evidence_count - 1) * 0.1, 0.3)
        reasons.append(f"Supported by {evidence_count} evidence sources")
    else:
        reasons.append("Single observation source")

    # Apply flags
    if user_preference:
        score += 0.3
        reasons.append("Contains user preferences or boundary declarations")
    if system_convention:
        score += 0.2
        reasons.append("Identified as stable environment setup or system convention")
    if procedure:
        score += 0.2
        reasons.append("Identified as procedural workflow or skill routine")

    # Apply negative score flags
    if temporary_task_state:
        score -= 0.4
        reasons.append("Contains temporary state, PR numbers, or short-lived SHAs")
    if stale_soon:
        score -= 0.3
        reasons.append("Identified as short-lived or stale-soon context")
    if contradiction:
        score -= 0.5
        reasons.append("Contradicts existing known state or user memory")

    # Keep score inside bounds
    score = max(0.0, min(1.0, score))

    # Priority decision rules
    if credential_secret:
        score = 0.0
        decision = "reject"
        suggested_status = "deprecated"
        reasons = ["Rejected: Contains potential credential or raw secret key"]
    elif contradiction:
        decision = "needs_review"
        suggested_status = "contradicted"
        reasons.append("Needs human/agent review due to active contradiction")
    elif stale_soon:
        decision = "needs_review"
        suggested_status = "deprecated"
        reasons.append("Needs review to safely deprecate stale fact")
    elif temporary_task_state:
        # temporary task status/PR numbers/SHAs/recent completed work => reject/archive
        decision = "archive"
        suggested_status = "provisional"
        reasons.append("Archived: Temporary task state")
    elif user_preference:
        # user preference/boundary => USER memory
        decision = "promote_user_memory"
        suggested_status = "active"
        reasons.append("Promoted to User memory")
    elif system_convention:
        # stable environment/system convention => operational MEMORY
        decision = "promote_operational_memory"
        suggested_status = "active"
        reasons.append("Promoted to Operational memory")
    elif procedure:
        # procedures/workflows => skill
        decision = "promote_skill"
        suggested_status = "active"
        reasons.append("Promoted to Skill bundle")
    else:
        # Fallback based on score threshold
        if score >= 0.7:
            decision = "promote_operational_memory"
            suggested_status = "active"
        elif score >= 0.4:
            decision = "needs_review"
            suggested_status = "provisional"
        else:
            decision = "reject"
            suggested_status = "historical"

    return {
        "decision": decision,
        "score": round(score, 3),
        "reasons": reasons,
        "suggested_status": suggested_status,
    }


# ---------------------------------------------------------------------------
# 3) Bi-temporal Fact Metadata Helpers
# ---------------------------------------------------------------------------

def _ensure_utc_aware(dt: datetime) -> datetime:
    """Coerce naive datetime to UTC-aware; leave already-aware datetimes unchanged."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


@dataclass
class FactMetadata:
    """Represents metadata for a stored memory fact using bi-temporal patterns."""
    observed_at: datetime
    valid_from: datetime
    valid_until: Optional[datetime] = None
    source: str = ""
    confidence: float = 1.0
    status: str = "active"  # active | historical | deprecated | provisional | contradicted

    def is_current(self, now: Optional[datetime] = None) -> bool:
        """Check if the fact is current at the given timestamp.

        Handles mixed naive/aware datetime comparisons safely by coercing
        naive datetimes to UTC-aware.
        """
        if now is None:
            now = datetime.now(timezone.utc)
        else:
            now = _ensure_utc_aware(now)

        valid_from = _ensure_utc_aware(self.valid_from)

        if self.status in ("deprecated", "contradicted"):
            return False

        if now < valid_from:
            return False

        if self.valid_until is not None:
            valid_until = _ensure_utc_aware(self.valid_until)
            if now >= valid_until:
                return False

        return True

    def mark_deprecated(self, reason: str, now: Optional[datetime] = None) -> FactMetadata:
        """Return a deprecate clone of FactMetadata (pure equivalent)."""
        if now is None:
            now = datetime.now(timezone.utc)
        else:
            now = _ensure_utc_aware(now)

        valid_until = _ensure_utc_aware(now)

        return FactMetadata(
            observed_at=self.observed_at,
            valid_from=self.valid_from,
            valid_until=valid_until,
            source=f"{self.source} (Deprecated: {reason})".strip(),
            confidence=self.confidence,
            status="deprecated",
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata object to dict."""
        return {
            "observed_at": self.observed_at.isoformat(),
            "valid_from": self.valid_from.isoformat(),
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "source": self.source,
            "confidence": self.confidence,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> FactMetadata:
        """Deserialize metadata object from dict."""
        observed_at = datetime.fromisoformat(d["observed_at"])
        valid_from = datetime.fromisoformat(d["valid_from"])
        valid_until = datetime.fromisoformat(d["valid_until"]) if d.get("valid_until") else None

        return cls(
            observed_at=observed_at,
            valid_from=valid_from,
            valid_until=valid_until,
            source=d.get("source", ""),
            confidence=d.get("confidence", 1.0),
            status=d.get("status", "active"),
        )