"""LiveKit token contract (both paths), backend auth, production route restrictions — on the ASGI app."""
import base64
import json

import pytest
from fastapi.testclient import TestClient

from ui import api, mobile

LK = {"LIVEKIT_URL": "wss://example.livekit.cloud", "LIVEKIT_API_KEY": "lk_key", "LIVEKIT_API_SECRET": "lk_secret_" + "s" * 32}


def _claims(token):
    payload = token.split(".")[1]
    return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))


@pytest.fixture
def lk(monkeypatch):
    for k, v in LK.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("TRIAGELINE_OFFLINE", "1")


def test_scoped_signed_participant_tokens(lk, monkeypatch):
    first, second = mobile.issue_token("a" * 32), mobile.issue_token("a" * 32)
    assert first["room"] != second["room"]            # fresh room per call: rejoin gets a new agent job
    assert first["identity"] != second["identity"]
    assert first["tools"] == "simulated" and first["expires_in"] == 600
    c = _claims(first["token"])
    assert c["video"]["room"] == first["room"] and c["video"]["roomJoin"] is True
    assert c["video"].get("canPublishSources") == ["microphone"]
    assert "roomAdmin" not in c["video"] and "roomCreate" not in c["video"]
    assert c["exp"] - c["nbf"] == 600
    assert "roomConfig" not in c                        # no agent_name -> automatic dispatch
    monkeypatch.setenv("TRIAGELINE_AGENT_NAME", "bilio-assistant")
    c = _claims(mobile.issue_token()["token"])
    assert c["roomConfig"]["agents"][0]["agentName"] == "bilio-assistant"


def test_rejects_bad_livekit_urls(monkeypatch):
    for url in ("", "https://x.example", "wss://user:pw@x.example", "wss://x.example:notaport", "wss://<your-project>"):
        monkeypatch.setenv("LIVEKIT_URL", url)
        monkeypatch.setenv("LIVEKIT_API_KEY", "k")
        monkeypatch.setenv("LIVEKIT_API_SECRET", "s")
        with pytest.raises(mobile.NotConfigured):
            mobile.issue_token()


def test_backend_token_route_requires_api_key(lk, monkeypatch):
    monkeypatch.setenv("TRIAGELINE_API_KEY", "x" * 36)
    with TestClient(api.create_app()) as c:
        r = c.post("/api/mobile/token", json={})
        assert r.status_code == 401 and r.json()["code"] == "unauthorized"
        r = c.post("/api/mobile/token", json={}, headers={"Authorization": "Bearer " + "y" * 36})
        assert r.status_code == 401
        r = c.post("/api/mobile/token", json={}, headers={"Authorization": "Bearer " + "x" * 36})
        assert r.status_code == 201 and r.json()["url"] == LK["LIVEKIT_URL"]
        assert LK["LIVEKIT_API_SECRET"] not in r.text


def test_backend_route_disabled_without_api_key(lk):
    with TestClient(api.create_app()) as c:
        assert c.post("/api/mobile/token", json={}, headers={"Authorization": "Bearer "}).status_code == 401


def test_unconfigured_livekit_does_not_expose_errors(monkeypatch):
    monkeypatch.setenv("TRIAGELINE_API_KEY", "k" * 36)
    monkeypatch.setenv("TRIAGELINE_OFFLINE", "1")
    with TestClient(api.create_app()) as c:
        r = c.post("/api/mobile/token", json={}, headers={"Authorization": "Bearer " + "k" * 36})
        assert r.status_code == 503 and r.json()["code"] == "unavailable"
        assert "secret" not in r.text.lower()
        c.post("/api/auth/login", json={})
        r = c.post("/api/rtc/token", json={})
        assert r.status_code == 503


def test_rtc_token_for_signed_in_user_is_fresh_per_call(lk):
    with TestClient(api.create_app()) as c:
        assert c.post("/api/rtc/token", json={}).status_code == 401
        c.post("/api/auth/login", json={})
        a, b = c.post("/api/rtc/token", json={}).json(), c.post("/api/rtc/token", json={}).json()
        assert a["room"] != b["room"] and a["room"].startswith("bilio-")


def test_production_hides_console_and_details(lk, monkeypatch):
    monkeypatch.setenv("TRIAGELINE_ENV", "production")
    monkeypatch.setenv("TRIAGELINE_ACCESS_CODE", "invite-code-long-enough")
    monkeypatch.setenv("TRIAGELINE_SESSION_SECRET", "z" * 40)
    monkeypatch.setenv("ALLOWED_HOSTS", "testserver")
    with TestClient(api.create_app()) as c:
        assert c.get("/api/scenarios").status_code in (404, 405)
        assert c.post("/api/run", json={}).status_code in (404, 405)
        r = c.get("/api/ready")
        assert r.status_code == 200 and "packages" not in r.json() and "assets" not in r.json()


def test_production_rejects_short_backend_key(lk, monkeypatch):
    monkeypatch.setenv("TRIAGELINE_ENV", "production")
    monkeypatch.setenv("TRIAGELINE_ACCESS_CODE", "invite-code-long-enough")
    monkeypatch.setenv("TRIAGELINE_SESSION_SECRET", "z" * 40)
    monkeypatch.setenv("ALLOWED_HOSTS", "testserver")
    monkeypatch.setenv("TRIAGELINE_API_KEY", "short")
    with pytest.raises(RuntimeError):
        api.create_app()


def test_console_available_in_development(monkeypatch):
    monkeypatch.setenv("TRIAGELINE_OFFLINE", "1")
    with TestClient(api.create_app()) as c:
        r = c.get("/api/scenarios")
        assert r.status_code == 200 and len(r.json()) >= 1
        path = r.json()[0]["path"]
        r = c.post("/api/run", json={"path": path, "agent": "bilio", "time_scale": 8})
        assert r.status_code == 200, r.text
        assert "trace" in r.json()
        assert c.post("/api/run", json={"path": "../etc/passwd"}).status_code == 400
        assert c.get("/console").status_code == 200


def test_same_origin_behind_tls_proxy_and_cross_origin_rejected(monkeypatch):
    """Browser sends Origin https://host while the app sees http:// behind the proxy: must be allowed."""
    monkeypatch.setenv("TRIAGELINE_OFFLINE", "1")
    with TestClient(api.create_app()) as c:
        ok = c.post("/api/auth/login", json={}, headers={"Origin": "https://testserver"})
        assert ok.status_code == 200
        bad = c.post("/api/auth/login", json={}, headers={"Origin": "https://evil.example"})
        assert bad.status_code == 403
        fwd = c.post("/api/auth/login", json={}, headers={"Origin": "https://app.example",
                                                          "X-Forwarded-Host": "app.example"})
        assert fwd.status_code == 200


def test_triage_flow_dispatches_the_extension_worker(lk, monkeypatch):
    # the extension (Triage Line) is reachable from the same gateway through a closed flow choice
    with pytest.raises(mobile.NotConfigured):
        mobile.issue_token(flow="triage")               # no worker name configured -> refuse, never mis-route
    monkeypatch.setenv("TRIAGELINE_AGENT_NAME", "bilio-assistant")
    monkeypatch.setenv("TRIAGELINE_TRIAGE_AGENT_NAME", "bilio-triage")
    t = mobile.issue_token(flow="triage")
    assert t["flow"] == "triage"
    assert _claims(t["token"])["roomConfig"]["agents"][0]["agentName"] == "bilio-triage"
    assert _claims(mobile.issue_token(flow="bogus")["token"])["roomConfig"]["agents"][0]["agentName"] == \
        "bilio-assistant"                            # unknown flow falls back to the assistant, never a free name
    assert mobile.flows() == {"assistant": True, "triage": True}
