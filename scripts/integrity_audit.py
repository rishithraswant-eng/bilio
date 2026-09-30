#!/usr/bin/env python3
"""Benchmark-integrity audit: does the agent source memorise FDB-v3 items?

The participant guide forbids hard-coding or pattern-matching benchmark items. This script checks
the decision code (agent/, livekit_agent/adapter.py) for literal overlap with the public benchmark:

  1. every expected argument value (ids, names, amounts written as strings, >= 4 chars)
  2. every 5-word sequence from the benchmark user utterances

Sources, in order: --data <fdb_v3_data_released> (official release) or the committed per-example
results (results/<run>/per_example/*.json, which contain the utterances and expected calls).

    python3 scripts/integrity_audit.py                 # exit 1 if any overlap is found
    python3 scripts/integrity_audit.py --data livekit_agent/fdb_v3_data_released

Comments and docstrings are included in the scan and reported separately (they are allowed
examples, but are listed so a reviewer can see them).
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# decision code (a hit here FAILS the audit) and everything else a reviewer reads (comments, fixtures)
DECISION = sorted((ROOT / "agent").glob("*.py")) + [ROOT / "livekit_agent/adapter.py"]
# the upstream manifest/mocks are copied verbatim from FDB-v3 (fidelity), so they are not scanned
UPSTREAM = {"mock_apis.py", "latency_injector.py", "livekit_inference.py", "fdb_tools.py"}
EXTRA = sorted(p for p in (ROOT / "livekit_agent").rglob("*.py")
               if ".fdb_v3_repo" not in p.parts and p.name not in UPSTREAM and p not in DECISION) + \
    sorted((ROOT / "tests").rglob("*.py")) + sorted(p for p in (ROOT / "scripts").glob("*.py")
                                                     if p.name != "integrity_audit.py")
SOURCES = DECISION + EXTRA
GENERIC = {"today", "tomorrow", "true", "false", "none", "null", "usd", "eur", "gbp", "checking", "savings",
           "credit", "walking", "driving", "transit", "passport", "monthly", "weekly", "daily", "express",
           "standard", "economy", "business", "first", "compact", "luxury", "english", "spanish", "chicago",
           "boston", "denver", "seattle", "miami", "austin", "paris", "london", "tokyo", "new york",
           "san francisco", "los angeles", "utilities", "electricity", "water", "internet", "phone", "rent"}


# Reviewed overlaps that are general-language knowledge, not memorised items. Each needs a reason.
ALLOW = {
    "Las Vegas": "city gazetteer (agent/nlu.py CITIES lists ~70 world cities, not benchmark answers)",
    "the name on the ticket": "generic passenger-name cue in NAME_RE, alongside booking/reservation variants",
}


def _words(s: str) -> list:
    return re.findall(r"[a-z0-9']+", s.lower())


def load_items(data: Path | None):
    utterances, values = [], set()
    if data and data.exists():
        for meta in data.glob("*/metadata.json"):
            m = json.loads(meta.read_text())
            # the official metadata stores user speech as dialogue[].user (older drafts: turns/user_turns)
            for t in (m.get("dialogue") or m.get("turns") or m.get("user_turns") or []):
                if isinstance(t, dict):
                    utterances.append(t.get("user") or t.get("user_annotated") or t.get("text") or "")
                else:
                    utterances.append(str(t))
            for c in m.get("expected_tool_calls", []) or []:
                values |= _values(c.get("args", {}))
    else:
        runs = sorted(p for p in list((ROOT / "results").glob("20*/per_example")) + list((ROOT / "results").glob("archive/*/per_example")) if p.is_dir())
        if not runs:
            sys.exit("no benchmark source: pass --data <fdb_v3_data_released>")
        for f in runs[-1].glob("*.json"):
            r = json.loads(f.read_text())
            utterances += [t.get("text", "") for t in r.get("user_turns_asr", [])]
            for c in r.get("expected_tool_calls", []) or []:
                values |= _values(c.get("args", {}))
    return utterances, values


def _values(o) -> set:
    out = set()
    if isinstance(o, dict):
        for v in o.values():
            out |= _values(v)
    elif isinstance(o, list):
        for v in o:
            out |= _values(v)
    elif isinstance(o, str) and _distinctive(o):
        out.add(o)
    return out


def _distinctive(v: str) -> bool:
    """Values that would only appear in code if an answer were memorised: ids containing digits
    (ABC123, BK-7741), or multi-word / capitalised proper names that are not common vocabulary.
    Schema field names (max_price), enum-like lowercase words (biking, gold) and city names are
    everyday vocabulary an NLU must know, so they are not flagged."""
    if len(v) < 4 or v.lower() in GENERIC or re.fullmatch(r"\d{4}-\d\d-\d\d", v):
        return False
    if re.fullmatch(r"[a-z_]+", v):                      # identifiers / lowercase enum values
        return False
    if re.search(r"[A-Za-z]", v) and re.search(r"\d", v):  # alphanumeric ids
        return True
    return bool(re.fullmatch(r"[A-Z][a-z]+(?: [A-Z][a-z]+)+", v)) or bool(re.fullmatch(r"[A-Z]{4,}", v))


def split_source(path: Path):
    """(code_strings, comment_text) — string literals in code vs comments/docstrings."""
    code, comments = [], []
    toks = list(tokenize.generate_tokens(io.StringIO(path.read_text(encoding="utf-8")).readline))
    prev = None
    for t in toks:
        if t.type == tokenize.COMMENT:
            comments.append((t.start[0], t.string))
        elif t.type == tokenize.STRING:
            is_doc = prev is None or prev.type in (tokenize.INDENT, tokenize.NEWLINE, tokenize.DEDENT, tokenize.NL)
            (comments if is_doc else code).append((t.start[0], t.string))
        if t.type not in (tokenize.NL, tokenize.COMMENT):
            prev = t
    return code, comments


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path)
    ap.add_argument("--ngram", type=int, default=5)
    ap.add_argument("--strict-comments", action="store_true",
                    help="also FAIL on overlaps in comments/docstrings/test fixtures and non-decision files")
    a = ap.parse_args()
    if a.data is None:
        for cand in (ROOT / "livekit_agent/.fdb_v3_repo/v3/fdb_v3_data_released", ROOT / "livekit_agent/fdb_v3_data_released"):
            if cand.is_dir():
                a.data = cand
                break
    utterances, values = load_items(a.data)
    if not utterances:
        print("RESULT: FAIL - parsed 0 benchmark utterances (wrong data path or metadata schema); audit would be vacuous")
        return 2
    grams = set()
    for u in utterances:
        w = _words(u)
        grams |= {" ".join(w[i:i + a.ngram]) for i in range(len(w) - a.ngram + 1)}
    hits_code, hits_doc = [], []
    for src in SOURCES:
        code, docs = split_source(src)
        decision = src in DECISION
        for bucket, out in ((code, hits_code if decision else hits_doc), (docs, hits_doc)):
            for line, text in bucket:
                low, ws = text.lower(), _words(text)
                src_grams = {" ".join(ws[i:i + a.ngram]) for i in range(len(ws) - a.ngram + 1)}
                for g in src_grams & grams:
                    out.append((src.relative_to(ROOT), line, "utterance 5-gram", g))
                for v in values:
                    if re.search(r"(?<![a-z0-9])" + re.escape(v.lower()) + r"(?![a-z0-9])", low):
                        out.append((src.relative_to(ROOT), line, "expected arg value", v))
    print(f"benchmark items: {len(utterances)} utterances, {len(grams)} {a.ngram}-grams, "
          f"{len(values)} distinctive expected values; scanned {len(SOURCES)} files")
    for label, hits in (("CODE (decision logic)", hits_code), ("comments/docstrings (examples)", hits_doc)):
        print(f"\n{label}: {len(hits)} overlap(s)")
        for h in sorted(set(hits)):
            print(f"  {h[0]}:{h[1]}  {h[2]}: {h[3]!r}")
    blocking = [h for h in hits_code if h[3] not in ALLOW]
    if a.strict_comments:
        blocking += [h for h in hits_doc if h[3] not in ALLOW]
    if hits_code:
        print("\nreviewed allowlist: " + "; ".join(f"{k!r} ({v})" for k, v in ALLOW.items()))
    scope = "decision code, comments, fixtures" if a.strict_comments else "decision code"
    print(f"\nRESULT: {'FAIL' if blocking else 'PASS'} — {len(blocking)} unreviewed overlap(s) in {scope}")
    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main())
