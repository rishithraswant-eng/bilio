#!/usr/bin/env python3
"""Post-run assertion: was every argument / response check in an official report really judged by gpt-4o?

The pinned upstream evaluators (v3/evaluate_tool_calls.py, v3/evaluate_pass_rate.py) silently fall back to
exact matching when a single judge request fails, and score a response 0 on a judge parsing error. A passing
judge *preflight* therefore does not prove the whole report was judged. This script walks the saved report
JSON files and counts argument checks whose explanation is one of the exact-match fallback strings, and
response checks that carry a judge-error explanation.

    python3 scripts/judge_coverage.py results/<stamp>/bilio_evaluation_report.json [...more reports]
    exit 0: every check judged   exit 1: some checks fell back (listed)   exit 2: usage / unreadable report
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# exact_match_args() return strings in the pinned evaluators (upstream commit 3e799c4)
FALLBACK_ARG = re.compile(r"^(All arguments match|Missing argument: .+|Mismatch '.+': expected=.*)$")
JUDGE_ERROR_RESPONSE = re.compile(r"^(LLM parsing error: .*|OpenAI client unavailable\.)$")


def walk(node, path=""):
    if isinstance(node, dict):
        yield path, node
        for k, v in node.items():
            yield from walk(v, f"{path}/{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk(v, f"{path}[{i}]")


def audit(report: dict) -> tuple[int, list[str]]:
    checked, fallbacks = 0, []
    for path, d in walk(report):
        exp = d.get("explanation")
        if not isinstance(exp, str):
            continue
        is_arg = "expected_args" in d or "actual_args" in d
        if is_arg:
            checked += 1
            if FALLBACK_ARG.match(exp.strip()):
                fallbacks.append(f"{path}: argument check fell back to exact match ({exp[:60]})")
        elif "score" in d or "response" in path.lower():
            checked += 1
            if JUDGE_ERROR_RESPONSE.match(exp.strip()):
                fallbacks.append(f"{path}: response judge error ({exp[:60]})")
    return checked, fallbacks


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    total, bad = 0, []
    for p in argv:
        try:
            rep = json.loads(Path(p).read_text())
        except (OSError, ValueError) as exc:
            print(f"cannot read {p}: {exc}")
            return 2
        n, fb = audit(rep)
        total += n
        bad += [f"{Path(p).name}{x}" for x in fb]
    print(f"judge coverage: {total - len(bad)}/{total} checks judged by the LLM")
    for x in bad[:50]:
        print("  UNJUDGED", x)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
