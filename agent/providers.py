"""LLM provider chain: ordered failover across free/cheap hosted LLMs (stdlib only).

Supported providers (all keys stay server-side):

  gemini      Google AI Studio key (GEMINI_API_KEY or GOOGLE_API_KEY). Native function calling.
  cerebras    CEREBRAS_API_KEY    OpenAI-compatible, free tier, very fast.
  openrouter  OPENROUTER_API_KEY  OpenAI-compatible, ``:free`` models.
  mistral     MISTRAL_API_KEY     OpenAI-compatible, free "experiment" tier.
  openai      OPENAI_API_KEY      optional paid provider.
  groq        GROQ_API_KEY        optional (kept only for backwards compatibility).
  custom      TRIAGELINE_LLM_BASE_URL + TRIAGELINE_LLM_API_KEY (any OpenAI-compatible server,
              e.g. Ollama, vLLM, LM Studio).

Chain selection (first match wins):
  TRIAGELINE_LLM_CHAIN=gemini,cerebras,openrouter   explicit ordered list
  TRIAGELINE_LLM_PROVIDER=gemini                    single provider (legacy variable)
  otherwise: every free-tier provider (AUTO_ORDER) that has a key configured.

Per-provider model override: TRIAGELINE_LLM_MODEL_<PROVIDER>=... (TRIAGELINE_LLM_MODEL applies
to the first provider of the chain, for backwards compatibility).

Failure policy: a transient error (timeout, connection error, HTTP 408/409/429/5xx) is retried
once with jittered backoff; then the next provider is tried. Fatal errors (401/403/404/400) skip
straight to the next provider and put the failing provider into a cooldown so a bad key does not
add latency to every turn. All attempts share one wall-clock deadline. Nothing here ever logs
request bodies, response bodies, headers, or keys.
"""
from __future__ import annotations

import json
import logging
import math
import os
import random
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import quote

log = logging.getLogger("bilio.providers")

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

# Model defaults verified against the providers' public model catalogues on 2026-09-28.
# Google: "For any new projects, use our latest models: 3.5 Flash-Lite or 3.8 Flash" (2.5 access is
# restricted to projects that already used it). Override with env vars when catalogues change.
PROVIDERS: Dict[str, Dict[str, Any]] = {
    "gemini": {"kind": "gemini", "key_env": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
               "model": "gemini-3.5-flash-lite", "base_url": GEMINI_BASE_URL},
    "cerebras": {"kind": "openai", "key_env": ("CEREBRAS_API_KEY",),
                 "model": "gpt-oss-120b", "base_url": "https://api.cerebras.ai/v1"},
    "openrouter": {"kind": "openai", "key_env": ("OPENROUTER_API_KEY",),
                   "model": "google/gemma-4-31b-it:free", "base_url": "https://openrouter.ai/api/v1"},
    "mistral": {"kind": "openai", "key_env": ("MISTRAL_API_KEY",),
                "model": "mistral-small-latest", "base_url": "https://api.mistral.ai/v1"},
    "openai": {"kind": "openai", "key_env": ("OPENAI_API_KEY",),
               "model": "gpt-4o-mini", "base_url": "https://api.openai.com/v1"},
    "groq": {"kind": "openai", "key_env": ("GROQ_API_KEY",),
             "model": "llama-3.3-70b-versatile", "base_url": "https://api.groq.com/openai/v1"},
    "custom": {"kind": "openai", "key_env": ("TRIAGELINE_LLM_API_KEY",),
               "model": "", "base_url": ""},
}
DEFAULT_ORDER = ("gemini", "cerebras", "openrouter", "mistral", "openai", "groq", "custom")
# Auto-discovery only uses the free-tier providers; paid/legacy ones must be named explicitly in
# TRIAGELINE_LLM_CHAIN so an unrelated OPENAI_API_KEY in the shell never changes agent behaviour.
AUTO_ORDER = ("gemini", "cerebras", "openrouter", "mistral")
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
COOLDOWN_S = float(os.environ.get("TRIAGELINE_LLM_COOLDOWN_S", "60"))
SEED = 7


class ProviderError(Exception):
    """Provider failure; ``retryable`` decides retry-same vs. go-to-next."""

    def __init__(self, provider: str, reason: str, status: Optional[int] = None,
                 retryable: bool = False, retry_after: Optional[float] = None):
        super().__init__(f"{provider}: {reason}" + (f" (HTTP {status})" if status else ""))
        self.provider, self.reason, self.status = provider, reason, status
        self.retryable, self.retry_after = retryable, retry_after


@dataclass
class Result:
    calls: List[Dict[str, Any]] = field(default_factory=list)
    text: str = ""
    provider: str = ""
    model: str = ""
    attempts: List[str] = field(default_factory=list)


_lock = threading.Lock()
_cooldown: Dict[str, float] = {}
_last: Dict[str, Dict[str, Any]] = {}


def key_for(name: str) -> str:
    spec = PROVIDERS.get(name)
    if not spec:
        return ""
    for env in spec["key_env"]:
        if os.environ.get(env):
            return os.environ[env]
    return ""


def base_url(name: str) -> str:
    if name == "custom":
        return os.environ.get("TRIAGELINE_LLM_BASE_URL", "").rstrip("/")
    return os.environ.get(f"TRIAGELINE_LLM_BASE_URL_{name.upper()}", PROVIDERS[name]["base_url"]).rstrip("/")


def model_for(name: str, first: bool = False) -> str:
    specific = os.environ.get(f"TRIAGELINE_LLM_MODEL_{name.upper()}")
    if specific:
        return specific
    if first and os.environ.get("TRIAGELINE_LLM_MODEL"):
        return os.environ["TRIAGELINE_LLM_MODEL"]
    return PROVIDERS[name]["model"]


def configured(name: str) -> bool:
    if name == "custom":
        return bool(base_url("custom") and model_for("custom"))
    return bool(key_for(name))


def chain() -> List[str]:
    """Ordered provider names. Raises ValueError for unknown names (config error, surfaced loudly)."""
    raw = os.environ.get("TRIAGELINE_LLM_CHAIN", "").strip()
    if not raw:
        raw = os.environ.get("TRIAGELINE_LLM_PROVIDER", "").strip()
    if raw and raw.lower() != "auto":
        names = [n.strip().lower() for n in raw.split(",") if n.strip()]
        bad = [n for n in names if n not in PROVIDERS]
        if bad:
            raise ValueError(f"unknown LLM provider(s) {bad}; use {list(DEFAULT_ORDER)}")
        return list(dict.fromkeys(names))
    found = [n for n in AUTO_ORDER if configured(n)]
    return found or ["gemini"]


def timeout_s() -> float:
    value = float(os.environ.get("TRIAGELINE_LLM_TIMEOUT_S", "8"))
    if not math.isfinite(value) or value <= 0:
        raise ValueError("TRIAGELINE_LLM_TIMEOUT_S must be finite and positive")
    return value


# ---------------------------------------------------------------- tool schemas

def _schema_node(spec, strict_required: bool = False) -> Dict[str, Any]:
    if isinstance(spec, str):
        spec = {"type": spec}
    out = {k: v for k, v in spec.items() if k in ("type", "description", "enum", "minimum", "maximum")}
    # Gemini's OpenAPI subset rejects `any`: JSON Schema expresses it by omitting `type`.
    if out.get("type") == "any":
        out.pop("type")
    if "items" in spec:
        out["items"] = _schema_node(spec["items"], strict_required)
    if spec.get("properties"):
        out["properties"] = {k: _schema_node(v, strict_required) for k, v in spec["properties"].items()}
        req = [k for k, v in spec["properties"].items() if isinstance(v, dict) and v.get("required")]
        if req or not strict_required:
            out["required"] = req
    return out


def _params(spec: Dict[str, Any], strict_required: bool) -> Dict[str, Any]:
    args = spec.get("args") or {}
    out = {"type": "object", "properties": {k: _schema_node(v, strict_required) for k, v in args.items()}}
    req = [k for k, v in args.items() if isinstance(v, dict) and v.get("required")]
    if req or not strict_required:
        out["required"] = req
    return out


def gemini_tools(tools: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [{"name": n, "description": s.get("description", ""), "parametersJsonSchema": _params(s, False)}
            for n, s in tools.items()]


def openai_tools(tools: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Some OpenAI-compatible servers reject an empty ``required: []``: omit it when empty.
    return [{"type": "function", "function": {"name": n, "description": s.get("description", "")[:1024],
                                              "parameters": _params(s, True)}}
            for n, s in tools.items()]


# ---------------------------------------------------------------- transport

def _post(provider: str, url: str, body: Dict[str, Any], headers: Dict[str, str], timeout: float) -> Dict[str, Any]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        status = exc.code
        retry_after = None
        try:
            retry_after = float(exc.headers.get("Retry-After")) if exc.headers else None
        except (TypeError, ValueError):
            retry_after = None
        raise ProviderError(provider, "http error", status=status, retryable=status in RETRYABLE_STATUS,
                            retry_after=retry_after) from None
    except (TimeoutError, OSError) as exc:  # URLError is an OSError; socket.timeout is TimeoutError
        raise ProviderError(provider, type(exc).__name__, retryable=True) from None
    except (ValueError, json.JSONDecodeError):
        raise ProviderError(provider, "invalid JSON response", retryable=True) from None


def _gemini_thinking(model: str) -> Optional[Dict[str, Any]]:
    """Lowest-latency thinking setting for the model family (3.x rejects thinkingBudget-only calls
    on some models and 2.5 rejects thinkingLevel)."""
    m = model.lower()
    if m.startswith("gemini-2.5-flash"):
        return {"thinkingBudget": 0}
    if m.startswith("gemini-2.5-pro"):
        return {"thinkingBudget": 128}
    if m.startswith("gemini-3"):
        # Levels accepted by every current Gemini 3 text model; MINIMAL where supported.
        minimal = ("flash-lite", "3.6-flash", "3-flash-preview", "3.5-flash")
        return {"thinkingLevel": "MINIMAL" if any(s in m for s in minimal) else "LOW"}
    return None


def _call_gemini(model: str, system: str, user: str, tools: Optional[Dict[str, Any]], timeout: float,
                 temperature: float, max_tokens: int) -> Result:
    key = key_for("gemini")
    if not key:
        raise ProviderError("gemini", "missing GEMINI_API_KEY")
    body: Dict[str, Any] = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens, "seed": SEED},
    }
    if tools:
        body["tools"] = [{"functionDeclarations": gemini_tools(tools)}]
        body["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}
    thinking = _gemini_thinking(model)
    if thinking:
        body["generationConfig"]["thinkingConfig"] = thinking
    url = f"{base_url('gemini')}/models/{quote(model, safe='')}:generateContent"
    data = _post("gemini", url, body, {"x-goog-api-key": key}, timeout)
    candidates = data.get("candidates") or []
    if not candidates:
        raise ProviderError("gemini", "no candidates (blocked or empty)", retryable=False)
    parts = candidates[0].get("content", {}).get("parts", []) or []
    calls = [{"name": p["functionCall"].get("name"), "args": p["functionCall"].get("args", {}) or {}}
             for p in parts if isinstance(p, dict) and "functionCall" in p]
    text = " ".join(p.get("text", "") for p in parts if isinstance(p, dict) and not p.get("thought")).strip()
    return Result(calls=calls, text=text, provider="gemini", model=model)


def _call_openai(name: str, model: str, system: str, user: str, tools: Optional[Dict[str, Any]],
                 timeout: float, temperature: float, max_tokens: int) -> Result:
    key = key_for(name)
    url = base_url(name)
    if not url:
        raise ProviderError(name, "missing base URL")
    if not key and name != "custom":
        raise ProviderError(name, f"missing {PROVIDERS[name]['key_env'][0]}")
    body: Dict[str, Any] = {"model": model, "temperature": temperature, "max_tokens": max_tokens,
                            "messages": [{"role": "system", "content": system},
                                         {"role": "user", "content": user}]}
    if name in ("openai", "groq", "cerebras", "mistral"):
        body["seed" if name != "mistral" else "random_seed"] = SEED
    if tools:
        body["tools"] = openai_tools(tools)
        body["tool_choice"] = "auto"
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    if name == "openrouter":
        headers["HTTP-Referer"] = os.environ.get("TRIAGELINE_PUBLIC_URL", "https://github.com/RomitDeokar/BILIO")
        headers["X-Title"] = "BILIO"
    data = _post(name, url + "/chat/completions", body, headers, timeout)
    choices = data.get("choices") or []
    if not choices:
        raise ProviderError(name, "no choices", retryable=False)
    msg = choices[0].get("message") or {}
    calls = []
    for tc in msg.get("tool_calls") or []:
        fn = (tc or {}).get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except (TypeError, ValueError):
            continue  # malformed arguments are dropped; the validator would reject them anyway
        calls.append({"name": fn.get("name"), "args": args if isinstance(args, dict) else {}})
    content = msg.get("content") or ""
    if isinstance(content, list):
        content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
    return Result(calls=calls, text=str(content).strip(), provider=name, model=model)


def _call_one(name: str, model: str, **kw) -> Result:
    if PROVIDERS[name]["kind"] == "gemini":
        return _call_gemini(model, **kw)
    return _call_openai(name, model, **kw)


def call(system: str, user: str, tools: Optional[Dict[str, Any]] = None, *, timeout: Optional[float] = None,
         temperature: float = 0.0, max_tokens: int = 1024, want: str = "any") -> Result:
    """Run the chain until one provider answers. ``want='calls'`` keeps trying when a provider
    answers with neither calls nor text; any real answer (including "no call") is accepted.

    Raises ProviderError (last error) if every provider failed.
    """
    budget = timeout if timeout is not None else timeout_s()
    deadline = time.monotonic() + budget
    names = chain()
    attempts: List[str] = []
    last: Optional[ProviderError] = None
    now = time.monotonic()
    usable = [n for n in names if configured(n) and _cooldown.get(n, 0) <= now] or \
             [n for n in names if configured(n)]
    if not usable:
        raise ProviderError(names[0] if names else "none", "no provider configured")
    for i, name in enumerate(usable):
        model = model_for(name, first=(name == names[0]))
        for attempt in range(2):
            remaining = deadline - time.monotonic()
            if remaining <= 0.2:
                break
            try:
                res = _call_one(name, model, system=system, user=user, tools=tools, timeout=remaining,
                                temperature=temperature, max_tokens=max_tokens)
                attempts.append(f"{name}:ok")
                res.attempts = attempts
                with _lock:
                    _cooldown.pop(name, None)
                    _last[name] = {"ok": True, "at": time.time(), "status": 200}
                return res
            except ProviderError as exc:
                last = exc
                attempts.append(f"{name}:{exc.status or exc.reason}")
                with _lock:
                    _last[name] = {"ok": False, "at": time.time(), "status": exc.status, "reason": exc.reason}
                log.warning("LLM provider %s failed (%s%s)%s", name, exc.reason,
                            f", HTTP {exc.status}" if exc.status else "",
                            "; retrying" if exc.retryable and attempt == 0 else "; trying next provider")
                if not exc.retryable:
                    with _lock:
                        _cooldown[name] = time.monotonic() + COOLDOWN_S
                    break
                if attempt == 0:
                    pause = min(exc.retry_after or 0.25 + random.random() * 0.35, 1.5)
                    if deadline - time.monotonic() > pause + 0.5:
                        time.sleep(pause)
                        continue
                with _lock:  # repeated transient failure: short cooldown
                    _cooldown[name] = time.monotonic() + min(COOLDOWN_S, 15)
                break
    raise last or ProviderError("chain", "deadline exceeded")


def status() -> Dict[str, Any]:
    """Secret-free provider status for /api/ready and the UI."""
    try:
        names = chain()
        error = None
    except ValueError as exc:
        names, error = [], str(exc)
    now = time.monotonic()
    out = []
    for i, n in enumerate(names):
        out.append({"provider": n, "model": model_for(n, first=(i == 0)), "configured": configured(n),
                    "cooling_down": _cooldown.get(n, 0) > now, "last": _last.get(n)})
    return {"chain": out, "configured": any(p["configured"] for p in out), "error": error}


def reset_state() -> None:
    """Test helper: clear cooldowns and last-result telemetry."""
    with _lock:
        _cooldown.clear()
        _last.clear()
