"""Exercise the actual mobile ASGI gateway; no hosted calls or real business actions."""
import base64
import json
import time

import pytest
from fastapi.testclient import TestClient
from ui import api


@pytest.fixture
def client(monkeypatch):
    for name in ('TRIAGELINE_ENV', 'TRIAGELINE_PRODUCTION', 'TRIAGELINE_ACCESS_CODE',
                 'TRIAGELINE_SESSION_SECRET', 'CORS_ORIGINS', 'ALLOWED_HOSTS',
                 'LIVEKIT_URL', 'LIVEKIT_API_KEY', 'LIVEKIT_API_SECRET'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('TRIAGELINE_OFFLINE', '1')
    with TestClient(api.create_app()) as c:
        yield c


def login(c):
    r = c.post('/api/auth/login', json={})
    assert r.status_code == 200
    return r.json()['access_token']


def start(c):
    r = c.post('/api/live/start', json={'request_id': 'start-123456'})
    assert r.status_code == 200, r.text
    return r.json()['sid']


def test_gateway_login_uses_root_env_before_worker_env(tmp_path, monkeypatch):
    worker = tmp_path / 'livekit_agent'
    worker.mkdir()
    (tmp_path / '.env').write_text('TRIAGELINE_ACCESS_CODE=root-invite-code\n')
    (worker / '.env.local').write_text('TRIAGELINE_ACCESS_CODE=stale-worker-code\n')
    monkeypatch.delenv('TRIAGELINE_ACCESS_CODE', raising=False)
    monkeypatch.setenv('TRIAGELINE_OFFLINE', '1')
    monkeypatch.setenv('ALLOWED_HOSTS', 'testserver')

    api.load_gateway_env(tmp_path)
    with TestClient(api.create_app()) as c:
        assert c.post('/api/auth/login', json={'access_code': 'root-invite-code'}).status_code == 200
        assert c.post('/api/auth/login', json={'access_code': 'stale-worker-code'}).status_code == 401

    monkeypatch.setenv('TRIAGELINE_ACCESS_CODE', 'exported-invite-code')
    api.load_gateway_env(tmp_path)
    with TestClient(api.create_app()) as c:
        assert c.post('/api/auth/login', json={'access_code': 'exported-invite-code'}).status_code == 200
        assert c.post('/api/auth/login', json={'access_code': 'root-invite-code'}).status_code == 401


def test_auth_ownership_and_bearer(client):
    assert client.post('/api/live/start', json={}).status_code == 401
    token = login(client)
    sid = start(client)
    client.cookies.clear()
    assert client.post(f'/api/live/{sid}/log', json={}).status_code == 401
    assert client.post(f'/api/live/{sid}/log', json={}, headers={'Authorization': 'Bearer '+token}).status_code == 200
    login(client)  # Another owner cannot inspect, end, or stream the first session.
    for op in ('log', 'end', 'say'):
        assert client.post(f'/api/live/{sid}/{op}', json={'text': 'hello'}).status_code == 404
    assert client.get(f'/api/live/{sid}/stream').status_code == 404


def test_cookie_flags_and_no_secret_echo(client):
    r = client.post('/api/auth/login', json={})
    assert 'HttpOnly' in r.headers['set-cookie']
    assert 'SameSite=strict' in r.headers['set-cookie']
    r = client.post('/api/auth/login', json={'access_code': ['private']})
    assert r.status_code == 422 and 'private' not in r.text
    assert r.headers['cache-control'] == 'no-store'


def test_start_and_command_idempotency(client):
    login(client)
    sid = start(client)
    assert start(client) == sid
    body = {'text': 'hello', 'speaking': True, 'request_id': 'message-1234'}
    a = client.post(f'/api/live/{sid}/say', json=body)
    b = client.post(f'/api/live/{sid}/say', json=body)
    assert a.status_code == b.status_code == 200 and a.json() == b.json()
    assert client.app.state.sessions.get(sid).input_count == 1
    assert client.post(f'/api/live/{sid}/say', json={**body, 'speaking': False}).status_code == 409


def test_closed_and_exhausted_sessions_remain_manageable(client):
    login(client)
    sid = start(client)
    session = client.app.state.sessions.get(sid)
    session.requests = {str(n): ('', {}) for n in range(500)}
    assert client.post(f'/api/live/{sid}/say', json={'text': 'hi'}).status_code == 429
    assert client.post(f'/api/live/{sid}/log', json={}).status_code == 200
    assert client.post(f'/api/live/{sid}/end', json={}).status_code == 200
    session.thread.join(3)
    assert not session.thread.is_alive()
    assert client.post(f'/api/live/{sid}/log', json={}).status_code == 404


@pytest.mark.parametrize('body', [[], {'text': ''}, {'text': 'x'*501}, {'text': 'hi', 'speaking': 'false'}, {'extra': 1}])
def test_strict_commands(client, body):
    login(client)
    sid = start(client)
    assert client.post(f'/api/live/{sid}/say', json=body).status_code == 422


def test_payload_limits_and_origins(client):
    assert client.post('/api/auth/login', content=b'{}', headers={'Content-Type': 'text/plain'}).status_code == 415
    assert client.post('/api/auth/login', content=b'{}', headers={'Content-Type': 'application/json', 'Content-Length': str(api.MAX_BODY+1)}).status_code == 413
    assert client.post('/api/auth/login', json={}, headers={'Origin': 'https://evil.example'}).status_code == 403
    assert client.post('/api/auth/login', content=b'bad', headers={'Content-Type': 'application/json'}).status_code == 422


def test_rate_limits(client):
    for _ in range(10):
        assert client.post('/api/auth/login', json={}).status_code == 200
    r = client.post('/api/auth/login', json={})
    assert r.status_code == 429 and r.headers['retry-after'] == '60'


def test_session_capacity(client):
    login(client)
    for _ in range(2):
        assert client.post('/api/live/start', json={}).status_code == 200
    assert client.post('/api/live/start', json={}).status_code == 429


def test_stream_cursor_validation(client):
    login(client)
    sid = start(client)
    assert client.get(f'/api/live/{sid}/stream?last=999999').status_code == 409
    assert client.get(f'/api/live/{sid}/stream', headers={'Last-Event-ID': 'oops'}).status_code == 400


def test_rtc_least_privilege_and_validation(client, monkeypatch):
    login(client)
    assert client.post('/api/rtc/token', json={}).status_code == 503
    monkeypatch.setenv('LIVEKIT_URL', 'wss://voice.example')
    monkeypatch.setenv('LIVEKIT_API_KEY', 'test-key')
    monkeypatch.setenv('LIVEKIT_API_SECRET', 'test-secret-with-at-least-32-characters')
    r = client.post('/api/rtc/token', json={})
    assert r.status_code == 200, r.text
    data = r.json()
    import jwt
    claims = jwt.decode(data['token'], 'test-secret-with-at-least-32-characters', algorithms=['HS256'])
    assert claims['sub'] == data['identity'] and claims['video']['room'] == data['room']
    assert claims['video']['canPublishSources'] == ['microphone']
    assert not claims['video'].get('roomAdmin') and not claims['video'].get('roomCreate')
    assert data['expires_in'] == 600 and data['tools'] == 'simulated'
    assert client.post('/api/rtc/token', json={'room': 'someone-elses-room'}).status_code == 422
    monkeypatch.setenv('LIVEKIT_URL', 'wss://username:password@voice.example')
    assert client.post('/api/rtc/token', json={}).status_code == 503


@pytest.mark.parametrize('flag', ['TRIAGELINE_ENV', 'TRIAGELINE_PRODUCTION'])
def test_production_fails_closed(client, monkeypatch, flag):
    monkeypatch.setenv(flag, 'production' if flag == 'TRIAGELINE_ENV' else '1')
    with pytest.raises(RuntimeError, match='Production requires'):
        api.create_app()


def test_production_hosts_cors_and_secure_cookie(client, monkeypatch):
    monkeypatch.setenv('TRIAGELINE_ENV', 'production')
    monkeypatch.setenv('TRIAGELINE_ACCESS_CODE', 'invite-code-long-enough')
    monkeypatch.setenv('TRIAGELINE_SESSION_SECRET', 's'*40)
    monkeypatch.setenv('ALLOWED_HOSTS', 'testserver')
    monkeypatch.setenv('CORS_ORIGINS', 'https://mobile.example')
    with TestClient(api.create_app(), base_url='https://testserver') as c:
        assert c.get('/api/health', headers={'Host': 'evil.example'}).status_code == 400
        assert c.get('/openapi.json').status_code == 404
        r = c.post('/api/auth/login', json={'access_code': 'invite-code-long-enough'})
        assert r.status_code == 200 and 'Secure' in r.headers['set-cookie']
        r = c.options('/api/live/start', headers={'Origin': 'https://mobile.example', 'Access-Control-Request-Method': 'POST'})
        assert r.status_code == 200 and r.headers['access-control-allow-origin'] == 'https://mobile.example'


def test_offline_provider_boundaries(client, monkeypatch):
    from agent import llm_planner
    monkeypatch.setenv('GEMINI_API_KEY', 'not-a-real-key')
    def fail(*args, **kwargs):
        pytest.fail('Offline mode must not call the provider')
    monkeypatch.setattr(llm_planner.urllib.request, 'urlopen', fail)
    assert llm_planner.plan('hello', {}) == []
    assert 'Offline' in llm_planner.reply('hello')


def test_production_plain_http_login_fails_loudly_not_silently(monkeypatch):
    """Production + http:// on a public host: a Secure cookie would be dropped by the browser, so the
    user could never sign in although the code was right. The server must say so instead."""
    for name in ('TRIAGELINE_PRODUCTION', 'CORS_ORIGINS'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('TRIAGELINE_OFFLINE', '1')
    monkeypatch.setenv('TRIAGELINE_ENV', 'production')
    monkeypatch.setenv('TRIAGELINE_ACCESS_CODE', 'invite-code-long-enough')
    monkeypatch.setenv('TRIAGELINE_SESSION_SECRET', 's' * 40)
    monkeypatch.setenv('ALLOWED_HOSTS', 'app.example,localhost')
    good = {'access_code': 'invite-code-long-enough'}
    with TestClient(api.create_app(), base_url='http://app.example') as c:
        assert c.post('/api/auth/login', json={'access_code': 'wrong'}).status_code == 401
        r = c.post('/api/auth/login', json=good)
        assert r.status_code == 400 and 'HTTPS' in r.json()['error']
        # behind a TLS proxy (X-Forwarded-Proto: https) it works
        r = c.post('/api/auth/login', json=good, headers={'X-Forwarded-Proto': 'https'})
        assert r.status_code == 200 and 'Secure' in r.headers['set-cookie']
    with TestClient(api.create_app(), base_url='http://localhost') as c:   # browsers keep Secure cookies on localhost
        assert c.post('/api/auth/login', json=good).status_code == 200


def test_access_code_ignores_surrounding_whitespace(monkeypatch):
    for name in ('TRIAGELINE_ENV', 'TRIAGELINE_PRODUCTION', 'ALLOWED_HOSTS', 'CORS_ORIGINS'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('TRIAGELINE_OFFLINE', '1')
    monkeypatch.setenv('TRIAGELINE_ACCESS_CODE', 'BILIODemo2026!   \r')
    with TestClient(api.create_app()) as c:
        assert c.get('/api/auth/config').json()['access_code_required'] is True
        assert c.post('/api/auth/login', json={'access_code': 'BILIODemo2026!'}).status_code == 200
        assert c.post('/api/auth/login', json={'access_code': ' BILIODemo2026! '}).status_code == 200
        assert c.post('/api/auth/login', json={'access_code': 'BILIODemo2026'}).status_code == 401
