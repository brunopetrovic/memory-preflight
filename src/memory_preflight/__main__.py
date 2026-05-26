"""CLI entry point for memory-preflight."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict

from memory_preflight import build_memory_preflight_advisory, classify_action


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="memory-preflight",
        description="Deterministically classify tool and memory risk before agent execution.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # classify sub-command
    cp = sub.add_parser("classify", help="Classify a tool call and emit JSON advisory.")
    cp.add_argument("--tool", "-t", required=True, help="Tool name (e.g. terminal, read_file)")
    cp.add_argument(
        "--args", "-a",
        default="{}",
        help="JSON-encoded args dict, e.g. '{\"command\": \"ls\"}'",
    )

    # score sub-command
    sp = sub.add_parser("score", help="Score a memory candidate.")
    sp.add_argument("--text", required=True, help="Candidate text")
    sp.add_argument("--evidence", type=int, default=1, help="Evidence count")
    sp.add_argument("--user-preference", action="store_true", help="Set user_preference flag")
    sp.add_argument("--system-convention", action="store_true", help="Set system_convention flag")
    sp.add_argument("--procedure", action="store_true", help="Set procedure flag")
    sp.add_argument("--temporary-task-state", action="store_true", help="Set temporary_task_state flag")
    sp.add_argument("--credential-secret", action="store_true", help="Set credential_secret flag")
    sp.add_argument("--stale-soon", action="store_true", help="Set stale_soon flag")
    sp.add_argument("--contradiction", action="store_true", help="Set contradiction flag")

    args = parser.parse_args()

    if args.command == "classify":
        try:
            parsed_args: Dict[str, Any] = json.loads(args.args)
        except json.JSONDecodeError as e:
            print(f"Error: --args must be valid JSON: {e}", file=sys.stderr)
            return 1

        advisory = build_memory_preflight_advisory(args.tool, parsed_args)
        print(json.dumps(advisory.to_dict(), indent=2))
        return 0

    elif args.command == "score":
        from memory_preflight import score_candidate
        result = score_candidate(
            text=args.text,
            evidence_count=args.evidence,
            user_preference=args.user_preference,
            system_convention=args.system_convention,
            procedure=args.procedure,
            temporary_task_state=args.temporary_task_state,
            credential_secret=args.credential_secret,
            stale_soon=args.stale_soon,
            contradiction=args.contradiction,
        )
        print(json.dumps(result, indent=2))
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())