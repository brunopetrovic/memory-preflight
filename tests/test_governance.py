"""Unit tests for memory-preflight."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
import pytest

from memory_preflight import (
    classify_action,
    score_candidate,
    FactMetadata,
    MemoryPreflightDecision,
    build_memory_preflight_advisory,
    MemoryPreflightAdvisory,
)


FAKE_OPENAI_KEY = "sk-" + "1" * 24
FAKE_GITHUB_PAT = "ghp_" + "A" * 36
FAKE_BEARER_TOKEN = "eyJ" + "B" * 37

class TestMemoryPreflightDecision:
    """Tests for the memory preflight decision module."""

    def test_preflight_protected_domains(self) -> None:
        # config
        dec = classify_action("updating config.yaml settings")
        assert dec.required is True
        assert dec.risk_level == "high"
        assert "config" in dec.categories
        assert "hermes configuration user preferences" in dec.recommended_queries

        # cron
        dec = classify_action("cron", tool_name="delete_job")
        assert dec.required is True
        assert dec.risk_level == "high"
        assert "cron" in dec.categories
        assert "cron schedule preferences" in dec.recommended_queries

        # provider
        dec = classify_action("switching provider", file_path="model_routing.py")
        assert dec.required is True
        assert dec.risk_level == "high"
        assert "provider" in dec.categories

        # social
        dec = classify_action("post a tweet on Twitter")
        assert dec.required is True
        assert dec.risk_level == "high"
        assert "social" in dec.categories

        # credentials
        dec = classify_action("store api_key secret")
        assert dec.required is True
        assert dec.risk_level == "high"
        assert "credentials" in dec.categories

        # memory edits
        dec = classify_action("edit_memory", tool_name="edit_memory")
        assert dec.required is True
        assert dec.risk_level == "high"
        assert "memory" in dec.categories

    def test_preflight_medium_risk(self) -> None:
        # Notion/GitHub automation (external-app)
        dec = classify_action("create pull request on github")
        assert dec.required is True
        assert dec.risk_level == "medium"
        assert "external-app" in dec.categories

        # Code execution
        dec = classify_action("run bash command", tool_name="run_command")
        assert dec.required is True
        assert dec.risk_level == "medium"
        assert "code" in dec.categories

        # User preferences
        dec = classify_action("change preference theme")
        assert dec.required is True
        assert dec.risk_level == "medium"
        assert "preferences" in dec.categories

    def test_preflight_low_risk(self) -> None:
        # low-risk read-only/small chat style actions
        dec = classify_action("say hello to the user")
        assert dec.required is False
        assert dec.risk_level == "low"
        assert dec.categories == ["low-risk"]
        assert len(dec.recommended_queries) == 0


class TestPromotionDemotionScoring:
    """Tests for the promotion/demotion scoring module."""

    def test_score_user_preference(self) -> None:
        res = score_candidate(
            text="User prefers dark theme always",
            user_preference=True,
        )
        assert res["decision"] == "promote_user_memory"
        assert res["suggested_status"] == "active"
        assert res["score"] >= 0.8

    def test_score_system_convention(self) -> None:
        res = score_candidate(
            text="Convention is to place scratch scripts in scratch/",
            system_convention=True,
        )
        assert res["decision"] == "promote_operational_memory"
        assert res["suggested_status"] == "active"
        assert res["score"] >= 0.7

    def test_score_procedure(self) -> None:
        res = score_candidate(
            text="To build the project, run npm run build",
            procedure=True,
        )
        assert res["decision"] == "promote_skill"
        assert res["suggested_status"] == "active"
        assert res["score"] >= 0.7

    def test_score_credential_secret(self) -> None:
        res = score_candidate(
            text="export OPENAI_API_KEY=***",
            credential_secret=True,
            user_preference=True,  # preference flag shouldn't override secret rejection
        )
        assert res["decision"] == "reject"
        assert res["suggested_status"] == "deprecated"
        assert res["score"] == 0.0

    def test_score_temporary_task_state(self) -> None:
        res = score_candidate(
            text="Task finished on PR #293 with SHA abc123f",
            temporary_task_state=True,
            evidence_count=5,
        )
        assert res["decision"] == "archive"
        assert res["suggested_status"] == "provisional"
        assert res["score"] <= 0.5

    def test_score_stale_soon(self) -> None:
        res = score_candidate(
            text="Temporary server is running at port 8080",
            stale_soon=True,
        )
        assert res["decision"] == "needs_review"
        assert res["suggested_status"] == "deprecated"

    def test_score_contradiction(self) -> None:
        res = score_candidate(
            text="Actually, user prefers light theme now",
            contradiction=True,
        )
        assert res["decision"] == "needs_review"
        assert res["suggested_status"] == "contradicted"


class TestBiTemporalFactMetadata:
    """Tests for bi-temporal FactMetadata helpers."""

    def test_is_current(self) -> None:
        now = datetime.now(timezone.utc)
        observed = now - timedelta(hours=2)
        valid_from = now - timedelta(hours=1)

        meta = FactMetadata(
            observed_at=observed,
            valid_from=valid_from,
            valid_until=now + timedelta(hours=2),
            source="session_log",
            confidence=0.9,
            status="active",
        )

        assert meta.is_current(now) is True
        assert meta.is_current(now - timedelta(hours=1.5)) is False  # Before valid_from
        assert meta.is_current(now + timedelta(hours=3)) is False   # After valid_until

    def test_mark_deprecated(self) -> None:
        now = datetime.now(timezone.utc)
        meta = FactMetadata(
            observed_at=now - timedelta(days=1),
            valid_from=now - timedelta(days=1),
            source="user_input",
            confidence=1.0,
            status="active",
        )

        dep_meta = meta.mark_deprecated("user preference changed", now=now)

        assert dep_meta.status == "deprecated"
        assert dep_meta.valid_until == now
        assert "Deprecated: user preference changed" in dep_meta.source

        # Original should be untouched (pure function)
        assert meta.status == "active"
        assert meta.valid_until is None

    def test_is_current_naive_aware_mix(self) -> None:
        """is_current handles naive/aware datetime comparisons correctly."""
        now_aware = datetime.now(timezone.utc)
        observed_naive = (now_aware - timedelta(hours=2)).replace(tzinfo=None)
        valid_from = now_aware - timedelta(hours=1)

        meta = FactMetadata(
            observed_at=observed_naive,
            valid_from=valid_from,
            valid_until=now_aware + timedelta(hours=2),
            source="test",
            confidence=0.9,
            status="active",
        )
        assert meta.is_current(now_aware) is True

        now_naive = datetime.now()
        observed_aware = datetime.now(timezone.utc) - timedelta(hours=1)
        valid_from_aware = datetime.now(timezone.utc) - timedelta(minutes=30)

        meta2 = FactMetadata(
            observed_at=observed_aware,
            valid_from=valid_from_aware,
            valid_until=None,
            source="test",
            confidence=0.9,
            status="active",
        )
        assert meta2.is_current(now_naive) is True

    def test_mark_deprecated_naive_aware_mix(self) -> None:
        """mark_deprecated handles naive/aware datetimes correctly."""
        now_aware = datetime.now(timezone.utc)
        observed_naive = (now_aware - timedelta(days=1)).replace(tzinfo=None)

        meta = FactMetadata(
            observed_at=observed_naive,
            valid_from=observed_naive,
            source="user_input",
            confidence=1.0,
            status="active",
        )

        now_naive = datetime.now()
        dep_meta = meta.mark_deprecated("user preference changed", now=now_naive)

        assert dep_meta.status == "deprecated"
        assert dep_meta.valid_until is not None
        assert dep_meta.valid_until.tzinfo is not None  # Should be UTC-aware
        assert "Deprecated: user preference changed" in dep_meta.source

    def test_serialization_roundtrip(self) -> None:
        now = datetime.now(timezone.utc)
        meta = FactMetadata(
            observed_at=now,
            valid_from=now,
            valid_until=now + timedelta(days=30),
            source="dream_review_extraction",
            confidence=0.85,
            status="provisional",
        )

        d = meta.to_dict()
        meta_loaded = FactMetadata.from_dict(d)

        assert meta_loaded.source == meta.source
        assert meta_loaded.confidence == meta.confidence
        assert meta_loaded.status == meta.status
        # Compare ISO format strings to avoid precision floating issues
        assert meta_loaded.observed_at.isoformat() == meta.observed_at.isoformat()
        assert meta_loaded.valid_from.isoformat() == meta.valid_from.isoformat()
        assert meta_loaded.valid_until.isoformat() == meta.valid_until.isoformat()


class TestBuildMemoryPreflightAdvisory:
    """Unit tests for the advisory builder helper."""

    def test_high_risk_config_by_action_name(self) -> None:
        advisory = build_memory_preflight_advisory("update_config", {})
        assert advisory.risk_level == "high"
        assert "config" in advisory.categories

    def test_high_risk_by_file_path_config_yaml(self) -> None:
        advisory = build_memory_preflight_advisory(
            "write_file",
            {"path": "~/.hermes/config.yaml", "content": "provider: openrouter"},
        )
        assert advisory.risk_level == "high"
        assert "config" in advisory.categories

    def test_high_risk_by_cron_in_command(self) -> None:
        advisory = build_memory_preflight_advisory(
            "terminal",
            {"command": "edit cron job schedule"},
        )
        assert advisory.risk_level == "high"
        assert "cron" in advisory.categories

    def test_high_risk_social_by_text(self) -> None:
        advisory = build_memory_preflight_advisory(
            "compose_public",
            {"text": "post a tweet to Twitter for me"},
        )
        assert advisory.risk_level == "high"
        assert "social" in advisory.categories

    def test_high_risk_provider_by_content(self) -> None:
        advisory = build_memory_preflight_advisory(
            "write_file",
            {"path": "/tmp/provider.json", "content": "change provider to anthropic"},
        )
        assert advisory.risk_level == "high"
        assert "provider" in advisory.categories

    def test_medium_risk_external_app_github(self) -> None:
        advisory = build_memory_preflight_advisory(
            "terminal",
            {"command": "create pull_request on github"},
        )
        assert advisory.risk_level == "medium"
        assert "external-app" in advisory.categories

    def test_medium_risk_code_by_tool_name(self) -> None:
        advisory = build_memory_preflight_advisory("run_command", {})
        assert advisory.risk_level == "medium"
        assert "code" in advisory.categories

    def test_low_risk_read_only(self) -> None:
        advisory = build_memory_preflight_advisory(
            "read_file",
            {"path": "/tmp/README.md"},
        )
        assert advisory.risk_level == "low"
        assert advisory.categories == ["low-risk"]

    def test_text_preview_truncated_at_300_chars(self) -> None:
        long_text = "x" * 500
        advisory = build_memory_preflight_advisory(
            "terminal",
            {"command": long_text},
        )
        assert len(advisory._text_preview) <= 300

    def test_file_path_extracted_from_path_key(self) -> None:
        advisory = build_memory_preflight_advisory(
            "patch",
            {"path": "~/.hermes/SOUL.md", "old_string": "old", "new_string": "new"},
        )
        # SOUL.md contains "soul" → memory category triggered
        assert advisory.risk_level in ("high", "medium")

    def test_file_path_extracted_from_file_key(self) -> None:
        advisory = build_memory_preflight_advisory(
            "read_file",
            {"file": "/etc/hosts"},
        )
        assert advisory.risk_level == "low"

    def test_text_extracted_from_content_key(self) -> None:
        advisory = build_memory_preflight_advisory(
            "write_file",
            {"path": "/tmp/notes.md", "content": "update provider model_routing"},
        )
        assert advisory.risk_level == "high"
        assert "provider" in advisory.categories

    def test_text_extracted_from_command_key(self) -> None:
        advisory = build_memory_preflight_advisory(
            "terminal",
            {"command": "update_config --set-model gpt-5"},
        )
        assert advisory.risk_level == "high"
        assert "config" in advisory.categories

    def test_multiple_text_keys_concatenated(self) -> None:
        advisory = build_memory_preflight_advisory(
            "terminal",
            {"command": "schedule cron task; edit recurring event"},
        )
        assert advisory.risk_level == "high"
        assert "cron" in advisory.categories

    def test_list_file_path_takes_first(self) -> None:
        advisory = build_memory_preflight_advisory(
            "read_file",
            {"path": ["/tmp/a.txt", "/tmp/b.txt"]},
        )
        assert advisory.risk_level == "low"

    def test_empty_args_returns_low_risk(self) -> None:
        advisory = build_memory_preflight_advisory("todo", {})
        assert advisory.risk_level == "low"
        assert advisory.categories == ["low-risk"]

    def test_log_message_contains_categories_and_queries(self) -> None:
        advisory = build_memory_preflight_advisory(
            "terminal",
            {"command": "edit_config.sh"},
        )
        msg = advisory.log_message()
        assert "risk=" in msg
        assert "categories=" in msg
        assert "terminal" in msg

    def test_advisory_is_dataclass(self) -> None:
        from dataclasses import is_dataclass
        advisory = build_memory_preflight_advisory("read_file", {"path": "/tmp/x"})
        assert is_dataclass(advisory)
        assert advisory.tool_name == "read_file"
        assert advisory.risk_level == "low"


class TestMemoryPreflightAdvisoryLogMessage:
    """Tests for the log_message() method formatting."""

    def test_log_message_high_risk(self) -> None:
        advisory = MemoryPreflightAdvisory(
            tool_name="terminal",
            risk_level="high",
            categories=["code", "config"],
            recommended_queries=["code execution sandbox guidelines"],
            reason="test",
            _text_preview="rm -rf /tmp",
        )
        msg = advisory.log_message()
        assert "terminal" in msg
        assert "HIGH" in msg
        assert "code" in msg
        assert "code execution sandbox guidelines" in msg

    def test_log_message_medium_risk_no_queries(self) -> None:
        advisory = MemoryPreflightAdvisory(
            tool_name="read_file",
            risk_level="medium",
            categories=["external-app"],
            recommended_queries=[],
            reason="test",
        )
        msg = advisory.log_message()
        assert "MEDIUM" in msg
        assert "external-app" in msg
        assert "suggested_queries" not in msg

    def test_log_message_truncates_queries_to_3(self) -> None:
        advisory = MemoryPreflightAdvisory(
            tool_name="terminal",
            risk_level="high",
            categories=["config"],
            recommended_queries=["q1", "q2", "q3", "q4", "q5"],
            reason="test",
        )
        msg = advisory.log_message()
        assert "q1" in msg
        assert "q4" not in msg


class TestMemoryAdvisoryIntegration:
    """Integration tests proving the advisory does not block execution."""

    def test_high_risk_still_classifies_not_blocks(self) -> None:
        decision = classify_action("update_config", tool_name="update_config")
        assert decision.required is True
        assert decision.risk_level == "high"

    def test_low_risk_classification(self) -> None:
        decision = classify_action("read_file", tool_name="read_file")
        assert decision.risk_level == "low"
        assert decision.required is False

    def test_advisory_never_raises_by_design(self) -> None:
        advisory = build_memory_preflight_advisory("", {})
        assert advisory.risk_level == "low"

        advisory2 = build_memory_preflight_advisory("tool", None)
        assert advisory2.risk_level == "low"

    def test_memory_preflight_module_exports(self) -> None:
        from memory_preflight import (
            build_memory_preflight_advisory,
            MemoryPreflightAdvisory,
            classify_action,
            score_candidate,
            FactMetadata,
            MemoryPreflightDecision,
        )
        assert callable(classify_action)
        assert callable(score_candidate)
        assert callable(build_memory_preflight_advisory)

    def test_log_message_debug_for_low_risk(self) -> None:
        advisory = build_memory_preflight_advisory("read_file", {"path": "/tmp/notes.txt"})
        assert advisory.risk_level == "low"
        msg = advisory.log_message()
        assert "low" in msg.lower()

    def test_concurrent_path_advisory_runs_for_all_tools(self) -> None:
        tools_and_args = [
            ("run_command", {"command": "rm -rf /tmp"}),  # medium risk
            ("read_file", {"path": "/tmp/notes.txt"}),  # low risk
            ("update_config", {}),  # high risk
        ]
        advisories = [build_memory_preflight_advisory(n, a) for n, a in tools_and_args]
        assert len(advisories) == 3
        assert advisories[2].risk_level == "high"
        assert advisories[0].risk_level == "medium"
        assert advisories[1].risk_level == "low"

    def test_text_preview_redacts_openai_sk_key(self) -> None:
        advisory = build_memory_preflight_advisory(
            "terminal",
            {"command": "curl https://api.openai.com/v1 -H 'Authorization: Bearer " + FAKE_OPENAI_KEY + "'"},
        )
        assert FAKE_OPENAI_KEY not in advisory._text_preview
        assert "***" in advisory._text_preview
        assert "credentials" in advisory.categories

    def test_text_preview_redacts_github_pat(self) -> None:
        advisory = build_memory_preflight_advisory(
            "terminal",
            {"command": "export GHP_TOKEN=" + FAKE_GITHUB_PAT},
        )
        assert FAKE_GITHUB_PAT not in advisory._text_preview
        assert "credentials" in advisory.categories

    def test_text_preview_redacts_api_key_eq_value(self) -> None:
        advisory = build_memory_preflight_advisory(
            "terminal",
            {"command": "export API_KEY=" + FAKE_OPENAI_KEY},
        )
        assert FAKE_OPENAI_KEY not in advisory._text_preview
        assert "credentials" in advisory.categories

    def test_text_preview_redacts_bearer_token(self) -> None:
        advisory = build_memory_preflight_advisory(
            "terminal",
            {"command": "curl -H 'Authorization: Bearer " + FAKE_BEARER_TOKEN + "'"},
        )
        assert FAKE_BEARER_TOKEN not in advisory._text_preview
        assert "***" in advisory._text_preview

    def test_text_preview_does_not_redact_short_values(self) -> None:
        advisory = build_memory_preflight_advisory(
            "terminal",
            {"command": "echo hello world"},
        )
        assert "hello" in advisory._text_preview

    def test_classification_still_uses_raw_text_not_redacted(self) -> None:
        advisory = build_memory_preflight_advisory(
            "terminal",
            {"command": "set API_KEY=" + FAKE_OPENAI_KEY},
        )
        assert "credentials" in advisory.categories
        assert FAKE_OPENAI_KEY not in advisory._text_preview


class TestDelegateTaskGoalContextExtraction:
    """Tests for delegate_task goal/context text extraction in preflight."""

    def test_delegate_task_goal_extracted_for_classification(self) -> None:
        advisory = build_memory_preflight_advisory(
            "delegate_task",
            {
                "goal": "update cron schedule for nightly backup",
                "context": "User prefers backups at 2am UTC",
            },
        )
        assert "cron" in advisory.categories
        assert advisory.risk_level == "high"

    def test_delegate_task_context_extracted_for_classification(self) -> None:
        advisory = build_memory_preflight_advisory(
            "delegate_task",
            {
                "goal": "check provider configuration",
                "context": "Switch from openrouter to anthropic for reasoning tasks",
            },
        )
        assert "provider" in advisory.categories
        assert advisory.risk_level == "high"

    def test_delegate_task_goal_context_both_extracted(self) -> None:
        advisory = build_memory_preflight_advisory(
            "delegate_task",
            {
                "goal": "configure new model provider",
                "context": "Use api_key=***... for authentication",
            },
        )
        assert "provider" in advisory.categories or "credentials" in advisory.categories

    def test_delegate_task_credentials_still_redacted_in_preview(self) -> None:
        advisory = build_memory_preflight_advisory(
            "delegate_task",
            {
                "goal": "Set GitHub token for automation",
                "context": "github_pat_11CharHereAXXXXX1234567890",
            },
        )
        # Credentials should be in categories (classification worked on raw text)
        assert "credentials" in advisory.categories
        # But preview should not contain raw token (redacted)
        assert "github_pat_11CharHereAXXXXX1234567890" not in advisory._text_preview

    def test_delegate_task_non_secret_goal_preserved(self) -> None:
        advisory = build_memory_preflight_advisory(
            "delegate_task",
            {
                "goal": "Review project structure and summarize dependencies",
                "context": "Focus on main entry points and their relationships",
            },
        )
        assert "config" not in advisory.categories
        assert "credentials" not in advisory.categories
        assert "cron" not in advisory.categories
        assert "provider" not in advisory.categories
        assert "memory" not in advisory.categories
        assert "social" not in advisory.categories
