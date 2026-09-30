"""BILIO gateway — the ONE application server (ASGI).

    python -m uvicorn ui.api:app --host 0.0.0.0 --port 8080 --proxy-headers

Run a single worker/replica (sessions live in process memory). Routes:
  /api/health, /api/ready            liveness / readiness (details require auth in production)
  /api/auth/{config,login,me,logout} pilot access-code sign-in (Bearer for apps, HttpOnly cookie for browsers)
  /api/live/*                        text/camera/voice-clip session + SSE stream (PWA /live.html)
  /api/rtc/token                     LiveKit participant token for a signed-in user (/rtc.html, native apps)
  /api/mobile/token                  backend-to-backend LiveKit token (Bearer TRIAGELINE_API_KEY)
  /api/scenarios, /api/run           evaluation console (development only; / console at /console)

An invite-code gate provides pilot access, not customer identity/SSO. Random, signed owner
credentials isolate conversations. All business action tools remain SIMULATED.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StrictBool
from starlette.middleware.trustedhost import TrustedHostMiddleware

ROOT = Path(__file__).resolve().parents[1]
# Exported process values override both files; the gateway's root .env takes priority over
# worker-only settings in livekit_agent/.env.local (same order as scripts/check_providers.py).
_PRESET_ACCESS_CODE = "TRIAGELINE_ACCESS_CODE" in os.environ


_ENV_FILES = (ROOT / ".env", ROOT / "livekit_agent/.env.local")   # first file wins


def load_gateway_env(root: Path = ROOT) -> None:
    """Load gateway settings first, then use worker settings only as fallback."""
    load_dotenv(root / ".env")
    load_dotenv(root / "livekit_agent/.env.local")


load_gateway_env()
from ui import live, mobile  # noqa: E402

log = logging.getLogger("bilio.api")
MAX_BODY = 8 * 1024 * 1024
TOKEN_TTL = 8 * 3600
STREAM_LIMIT = 90   # SSE (re)connects per owner per minute


class Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Login(Body):
    access_code: str = Field(default="", max_length=256)


class Command(Body):
    text: str | None = Field(default=None, min_length=1, max_length=500)
    speaking: StrictBool = False
    image: str | None = None
    audio: str | None = None
    request_id: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_-]{8,80}$")


class Start(Body):
    request_id: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_-]{8,80}$")


class RtcStart(Body):
    request_id: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_-]{8,80}$")
    flow: str | None = Field(default=None, pattern=r"^(assistant|triage)$")   # closed choice, server maps to a worker


class RateLimit:
    """Bounded single-process fixed windows. Edge limits still required in deployment."""
    def __init__(self):
        self.windows = OrderedDict()

    def check(self, key, limit):
        now = time.monotonic()
        began, count = self.windows.get(key, (now, 0))
        if now - began >= 60:
            began, count = now, 0
        if count >= limit:
            raise HTTPException(429, "Rate limit reached. Try again shortly.", headers={"Retry-After": "60"})
        self.windows[key] = (began, count + 1)
        self.windows.move_to_end(key)
        while len(self.windows) > 10000:
            self.windows.popitem(last=False)


def report_access_code(code: str) -> None:
    """Print where the access code came from and a non-secret fingerprint (length + last char) so an
    operator can see why a login is rejected without the code ever being logged."""
    if not code:
        print("[auth] TRIAGELINE_ACCESS_CODE is empty: sign-in needs no code", flush=True)
        return
    from dotenv import dotenv_values
    found = {}
    for f in _ENV_FILES:
        value = dotenv_values(f).get("TRIAGELINE_ACCESS_CODE") if f.exists() else None
        if value is not None:
            found[f.relative_to(ROOT).as_posix()] = value.strip()
    if _PRESET_ACCESS_CODE:
        source = "an OS/shell environment variable (it overrides BOTH .env files)"
    else:
        source = next(iter(found), "?")
    print(f"[auth] access code loaded from {source}: {len(code)} chars, ends with {code[-1]!r}", flush=True)
    if len(set(found.values()) | {code}) > 1:
        print(f"[auth] WARNING: TRIAGELINE_ACCESS_CODE differs between {', '.join(found) or 'the files'} and what is "
              f"in use. Only the value from {source} counts. Make them identical or delete the extra line.", flush=True)
    if " #" in code or code.startswith(("'", '"')) or code.endswith(("'", '"')):
        print("[auth] WARNING: the access code contains ' #' or quotes. Your env loader kept an inline comment or "
              "quotes as part of the value. Put the code alone on its line: TRIAGELINE_ACCESS_CODE=yourcode", flush=True)


def create_app() -> FastAPI:
    environment = os.getenv("TRIAGELINE_ENV", "development")
    if environment not in {"development", "production"}:
        raise RuntimeError("TRIAGELINE_ENV must be development or production")
    # Fail closed when an operator reuses the older server's production flag.
    production = environment == "production" or os.getenv("TRIAGELINE_PRODUCTION") == "1"
    # Surrounding whitespace is never part of a code (copy/paste, CRLF files, "KEY=value   " lines).
    access_code = os.getenv("TRIAGELINE_ACCESS_CODE", "").strip()
    report_access_code(access_code)
    secret = os.getenv("TRIAGELINE_SESSION_SECRET", "") or secrets.token_hex(32)
    origins = [x.strip().rstrip("/") for x in os.getenv("CORS_ORIGINS", "").split(",") if x.strip()]
    hosts = [x.strip() for x in os.getenv("ALLOWED_HOSTS", "").split(",") if x.strip()]
    if "*" in origins:
        raise RuntimeError("CORS_ORIGINS must list explicit origins, not *")
    if production and (len(access_code) < 16 or len(os.getenv("TRIAGELINE_SESSION_SECRET", "")) < 32 or not hosts or "*" in hosts):
        raise RuntimeError("Production requires ACCESS_CODE (16+), SESSION_SECRET (32+) and explicit ALLOWED_HOSTS")
    api_key = os.getenv("TRIAGELINE_API_KEY", "")
    if production and api_key and len(api_key) < 32:
        raise RuntimeError("TRIAGELINE_API_KEY must be at least 32 characters in production")
    console = not production and os.getenv("TRIAGELINE_CONSOLE", "1") == "1"
    limiter = RateLimit()
    manager = live.Sessions()
    rtc_probe = {"at": 0.0, "ok": None, "error": None}
    starts: OrderedDict = OrderedDict()
    operations = asyncio.Lock()

    async def reap():
        while True:
            await asyncio.sleep(30)
            with manager.lock:
                manager.reap()

    def preload():
        try:
            live.P.load_asr()
            live.P.load_clip()
        except Exception as exc:  # noqa: BLE001 - optional local models
            log.warning("model preload failed: %s", type(exc).__name__)

    @asynccontextmanager
    async def lifespan(_app):
        reaper = asyncio.create_task(reap())
        if os.getenv("PRELOAD", "0") == "1":
            asyncio.get_running_loop().run_in_executor(None, preload)
        try:
            yield
        finally:
            reaper.cancel()
            await asyncio.gather(reaper, return_exceptions=True)
            for sid in list(manager.by_id):
                s = manager.get(sid)
                manager.end(sid)
                if s:
                    await asyncio.to_thread(s.thread.join, 3)

    app = FastAPI(title="BILIO API", version="2.0.0", lifespan=lifespan,
                  docs_url=None if production else "/docs", redoc_url=None,
                  openapi_url=None if production else "/openapi.json")
    app.state.sessions = manager
    app.state.limiter = limiter
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True,
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type", "Authorization", "Last-Event-ID"])
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=hosts or ["*"])

    @app.middleware("http")
    async def protections(request: Request, call_next):
        request_id = secrets.token_hex(8)
        try:
            if request.url.path.startswith("/api/"):
                origin = request.headers.get("origin")
                # Behind a TLS-terminating proxy the app sees http:// while the browser sends https://,
                # so same-origin compares host[:port] (Host / X-Forwarded-Host), never the scheme.
                host = (request.headers.get("x-forwarded-host") or request.headers.get("host") or "").lower()
                origin_host = urlsplit(origin).netloc.lower() if origin else ""
                if origin and origin_host != host and origin.rstrip("/") not in origins:
                    raise HTTPException(403, "Origin not allowed")
                if request.method == "POST":
                    try:
                        size = int(request.headers.get("content-length", "0"))
                    except ValueError:
                        raise HTTPException(400, "Invalid Content-Length")
                    if size < 0 or size > MAX_BODY:
                        raise HTTPException(413, "Request too large")
                    if request.headers.get("content-type", "").split(";")[0].strip().lower() != "application/json":
                        raise HTTPException(415, "Use application/json")
                    async def read_bounded():
                        chunks, total = [], 0
                        async for chunk in request.stream():
                            total += len(chunk)
                            if total > MAX_BODY:
                                raise HTTPException(413, "Request too large")
                            chunks.append(chunk)
                        return b"".join(chunks)
                    request._body = await asyncio.wait_for(read_bounded(), 10)
            response = await call_next(request)
        except HTTPException as exc:
            response = JSONResponse({"error": exc.detail, "code": "request_rejected"}, exc.status_code, headers=exc.headers)
        except asyncio.TimeoutError:
            response = JSONResponse({"error": "Request body timed out", "code": "timeout"}, 408)
        except Exception as exc:
            log.error("request_failed id=%s type=%s", request_id, type(exc).__name__)
            response = JSONResponse({"error": "Server error", "code": "server_error"}, 500)
        response.headers.update({"X-Request-ID": request_id, "X-Content-Type-Options": "nosniff",
                                 "Referrer-Policy": "no-referrer", "X-Frame-Options": "DENY",
                                 "Permissions-Policy": "camera=(self), microphone=(self)",
                                 "Cache-Control": "no-store"})
        if production:
            response.headers["Strict-Transport-Security"] = "max-age=31536000"
        return response

    @app.exception_handler(HTTPException)
    async def http_error(_request, exc):
        code = "no_session" if exc.status_code == 404 else "unauthorized" if exc.status_code == 401 else "request_rejected"
        return JSONResponse({"error": exc.detail, "code": code}, exc.status_code, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def invalid(_request, exc):
        # Do not echo the body (may contain recordings, access codes or personal information).
        fields = [".".join(str(x) for x in e["loc"]) for e in exc.errors()]
        return JSONResponse({"error": "Invalid request fields", "fields": fields, "code": "bad_request"}, 422)

    def issue(owner):
        payload = f"{owner}.{int(time.time()) + TOKEN_TTL}"
        sig = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
        return payload + "." + sig

    async def authenticate(request: Request):
        bearer = request.headers.get("authorization", "")
        token = bearer[7:] if bearer.startswith("Bearer ") else request.cookies.get("tl_auth", "")
        try:
            owner, expires, sig = token.split(".")
            payload = f"{owner}.{expires}"
            expected = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
            if len(owner) != 32 or not hmac.compare_digest(sig, expected) or int(expires) <= time.time():
                raise ValueError()
        except (ValueError, TypeError):
            raise HTTPException(401, "Sign in to start or resume a session")
        request.state.auth_expiry = int(expires)
        request.state.owner = owner
        return owner

    def owned(sid, owner):
        s = manager.get(sid)
        if not s or getattr(s, "owner", None) != owner:
            raise HTTPException(404, "Session expired or unavailable")
        return s

    def is_https(request: Request) -> bool:
        # uvicorn rewrites the scheme from X-Forwarded-Proto only for trusted proxies (FORWARDED_ALLOW_IPS);
        # also honour the header directly so a TLS proxy that is not in that list still works.
        proto = request.headers.get("x-forwarded-proto", "").split(",")[0].strip().lower()
        return request.url.scheme == "https" or proto == "https"

    def is_loopback(request: Request) -> bool:
        host = (request.headers.get("host") or "").rsplit(":", 1)[0].strip("[]").lower()
        return host in {"localhost", "127.0.0.1", "::1"}

    def client_ip(request: Request) -> str:
        # uvicorn --proxy-headers --forwarded-allow-ips=<proxy> rewrites request.client from X-Forwarded-For.
        return request.client.host if request.client else "unknown"

    async def backend(request: Request):
        """Server-to-server auth for /api/mobile/token."""
        provided = request.headers.get("authorization", "")
        if not api_key or not hmac.compare_digest(provided.encode(), ("Bearer " + api_key).encode()):
            raise HTTPException(401, "unauthorized")

    async def probe_livekit() -> dict:
        """Cached (30 s) reachability + credential check against the LiveKit server API."""
        if not mobile.configured():
            return {"configured": False, "reachable": None}
        now = time.monotonic()
        if now - rtc_probe["at"] > 30:
            rtc_probe["at"] = now
            try:
                from livekit import api as lkapi
                url, key, sec = mobile.livekit_config()
                http = url.replace("wss://", "https://").replace("ws://", "http://")
                lk = lkapi.LiveKitAPI(http, key, sec)
                try:
                    await asyncio.wait_for(lk.room.list_rooms(lkapi.ListRoomsRequest()), 3)
                finally:
                    await lk.aclose()
                rtc_probe.update(ok=True, error=None)
            except Exception as exc:  # noqa: BLE001
                rtc_probe.update(ok=False, error=type(exc).__name__)
        return {"configured": True, "reachable": rtc_probe["ok"], "error": rtc_probe["error"],
                "agent_name": mobile.agent_name() or None, "flows": mobile.flows()}

    @app.get("/api/health")
    async def health():
        return {"ok": True, "service": "bilio", "sessions": len(manager.by_id)}

    @app.get("/api/auth/config")
    async def auth_config():
        return {"access_code_required": bool(access_code), "production": production}

    @app.post("/api/auth/login")
    async def login(body: Login, request: Request):
        limiter.check(("login", client_ip(request)), 10)
        if access_code and not hmac.compare_digest(body.access_code.strip().encode(), access_code.encode()):
            raise HTTPException(401, "Invalid access code")
        https = is_https(request)
        # Production marks the cookie Secure. Browsers silently DROP a Secure cookie received over plain
        # http:// (except on localhost), so the code was accepted but the next request was unauthenticated
        # and the user was asked to sign in again forever. Refuse loudly instead of issuing a dead cookie.
        if production and not https and not is_loopback(request):
            raise HTTPException(400, "Access code is correct, but production sign-in needs HTTPS. Open the site "
                                     "via https:// (TLS proxy with X-Forwarded-Proto), or set "
                                     "TRIAGELINE_ENV=development for plain-http testing.")
        token = issue(secrets.token_hex(16))
        response = JSONResponse({"access_token": token, "token_type": "Bearer", "expires_in": TOKEN_TTL})
        response.set_cookie("tl_auth", token, max_age=TOKEN_TTL, httponly=True,
                            secure=production or https, samesite="strict")
        return response

    @app.get("/api/auth/me")
    async def me(request: Request, owner=Depends(authenticate)):
        return {"authenticated": True, "expires_at": request.state.auth_expiry}

    @app.post("/api/auth/logout")
    async def logout(request: Request):
        # Ends this owner's sessions (if the credential is still valid) and clears the browser cookie.
        try:
            owner = await authenticate(request)
            for sid, s in list(manager.by_id.items()):
                if getattr(s, "owner", None) == owner:
                    manager.end(sid)
        except HTTPException:
            pass
        response = JSONResponse({"ok": True})
        response.delete_cookie("tl_auth")
        return response

    @app.get("/api/ready")
    async def ready(request: Request):
        if production:
            try:
                await authenticate(request)
            except HTTPException:
                # public probe in production: no package/asset/session details
                return {"ok": True, "planner": live._planner_status().get("configured"),
                        "rtc": mobile.configured()}
        status = live.readiness()
        status["sessions"] = len(manager.by_id)
        status["rtc"] = await probe_livekit()
        status["limitations"] = ["Business tools are simulated", "Session memory is process-local (run one replica)",
                                 "LiveKit reachability does not prove a voice worker is registered"]
        return status

    @app.post("/api/live/start")
    async def start(body: Start, owner=Depends(authenticate)):
        async with operations:
            key = (owner, body.request_id)
            if body.request_id and key in starts and manager.get(starts[key]):
                s = owned(starts[key], owner)
            else:
                limiter.check(("start", owner), 6)
                if sum(getattr(s, "owner", None) == owner for s in manager.by_id.values()) >= 2:
                    raise HTTPException(429, "End an existing session before starting another")
                try:
                    s = await asyncio.to_thread(manager.start)
                except OverflowError:
                    raise HTTPException(429, "Session capacity reached")
                except RuntimeError:
                    raise HTTPException(503, "Session could not start (models still loading?) — retry shortly",
                                        headers={"Retry-After": "5"})
                s.owner = owner
                s.requests = {}
                s.command_lock = asyncio.Lock()
                if body.request_id:
                    starts[key] = s.sid
                    while len(starts) > 1000:
                        starts.popitem(last=False)
            return {"sid": s.sid, "mode": live.MODE, "audio": live.P.speech_config()}

    @app.post("/api/live/{sid}/{op}")
    async def command(sid: str, op: str, body: Command, owner=Depends(authenticate)):
        s = owned(sid, owner)
        limiter.check(("command", owner), 120)
        if op not in {"say", "audio", "frame", "interrupt", "log", "end"}:
            raise HTTPException(404, "Unknown operation")
        required = {"say": "text", "audio": "audio", "frame": "image"}.get(op)
        if required and not getattr(body, required):
            raise HTTPException(422, f"{required} is required")
        # Barge-in and hang-up must never queue behind a slow upload: no lock, no idempotency store.
        try:
            if op == "interrupt":
                s.interrupt()
                return {"ok": True}
            if op == "end":
                manager.end(sid)
                return {"ok": True}
            if op == "log":
                s.touched = time.time()
                return {"log": s.events_after(0), "mode": live.MODE, "audio": live.P.speech_config()}
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        signature = hashlib.sha256((op + body.model_dump_json()).encode()).hexdigest()
        # The lock covers only idempotency bookkeeping; decoding/validation runs outside it.
        async with s.command_lock:
            if body.request_id and body.request_id in s.requests:
                previous, result = s.requests[body.request_id]
                if previous != signature:
                    raise HTTPException(409, "request_id was already used with different input")
                if result is not None:
                    return result
                raise HTTPException(409, "request_id is still being processed")
            if len(s.requests) >= 500:
                raise HTTPException(429, "Session request limit reached; start a new session")
            if body.request_id:
                s.requests[body.request_id] = (signature, None)   # reserve: duplicates can't double-submit
        try:
            if op == "say":
                result = {"ok": True, "as": s.user_text(body.text, body.speaking)}
            elif op == "audio":
                result = {"ok": True, "ref": await asyncio.to_thread(s.user_audio, body.audio, body.speaking)}
            else:
                result = {"ok": True, "ref": await asyncio.to_thread(s.user_frame, body.image)}
        except ValueError as exc:
            if body.request_id:
                s.requests.pop(body.request_id, None)
            raise HTTPException(400, str(exc))
        except BaseException:
            if body.request_id:
                s.requests.pop(body.request_id, None)
            raise
        if body.request_id:
            s.requests[body.request_id] = (signature, result)
        return result

    @app.get("/api/live/{sid}/stream")
    async def stream(sid: str, request: Request, last: int = 0, owner=Depends(authenticate)):
        s = owned(sid, owner)
        limiter.check(("stream", owner), STREAM_LIMIT)
        try:
            cursor = max(0, int(request.headers.get("last-event-id", last)))
        except ValueError:
            raise HTTPException(400, "Invalid Last-Event-ID")
        if cursor > s.seq:
            raise HTTPException(409, "Event cursor is ahead of this session")
        generation = s.claim_stream()
        initial_seq = s.seq
        expiry = request.state.auth_expiry

        async def events():
            nonlocal cursor
            wake = asyncio.Event()
            loop = asyncio.get_running_loop()
            s.subscribe(loop, wake)
            try:
                yield "retry: 1500\n\n"
                heartbeat = time.monotonic()
                while generation == s.stream_gen and time.time() < expiry:
                    if await request.is_disconnected():
                        break
                    wake.clear()
                    batch = s.events_after(cursor)
                    if batch and batch[0]["id"] > cursor + 1:
                        yield 'event: gap\ndata: {"message":"Older events have expired; showing available history"}\n\n'
                    for ev in batch:
                        cursor = ev["id"]
                        yield f"id: {cursor}\ndata: {json.dumps({**ev, 'replay': cursor <= initial_seq})}\n\n"
                    if s.closed:
                        break
                    if time.monotonic() - heartbeat > 10:
                        yield ": ping\n\n"
                        heartbeat = time.monotonic()
                    # Event-driven: emit() wakes us; the 1 s timeout re-checks disconnect/expiry.
                    # Heartbeats do not extend idle lifetime: a forgotten tab must expire.
                    try:
                        await asyncio.wait_for(wake.wait(), 1.0)
                    except asyncio.TimeoutError:
                        pass
            finally:
                s.unsubscribe(loop, wake)

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"X-Accel-Buffering": "no", "Cache-Control": "no-store"})

    @app.post("/api/rtc/token")
    async def rtc_token(body: RtcStart, owner=Depends(authenticate)):
        limiter.check(("rtc", owner), 10)
        try:
            # Client cannot choose room/identity/agent or gain admin grants. Fresh room per call (A1).
            # `flow` is a closed choice (assistant | triage) mapped to an allow-listed worker name.
            return mobile.issue_token(owner, flow=body.flow)
        except mobile.NotConfigured as exc:
            msg = str(exc) if body.flow == "triage" and mobile.configured() else \
                "Voice calls are not configured on this server (LiveKit credentials missing)"
            raise HTTPException(503, msg)

    @app.post("/api/mobile/token", status_code=201, dependencies=[Depends(backend)])
    async def mobile_token(request: Request):
        """Trusted backend-to-backend issuance: your app server authenticates its user, then calls this."""
        limiter.check(("mobile", client_ip(request)), 120)
        try:
            return mobile.issue_token()
        except mobile.NotConfigured:
            return JSONResponse({"error": "LiveKit is not configured", "code": "unavailable"}, 503)

    if console:
        from ui import console as console_routes
        console_routes.register(app, limiter)

    @app.get("/")
    @app.get("/app")
    @app.get("/live")
    async def home():
        return FileResponse(ROOT / "ui/static/live.html")

    @app.get("/voice")
    async def voice():
        return FileResponse(ROOT / "ui/static/rtc.html")

    app.mount("/", StaticFiles(directory=ROOT / "ui/static", html=True), name="static")
    return app


app = create_app()
