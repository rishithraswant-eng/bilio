# Mobile integration (LiveKit voice + authenticated gateway)

The voice assistant is the LiveKit worker `livekit_agent/cascaded_agent.py`. Clients join a LiveKit room with a
short-lived token from the gateway (`python -m ui`, i.e. `ui/api.py`). No provider key ever reaches the device.

## 1. Server side
```bash
# gateway (HTTPS in front, single replica: sessions are in memory)
TRIAGELINE_ENV=production TRIAGELINE_ACCESS_CODE=<16+ chars> TRIAGELINE_SESSION_SECRET=<32+ chars> \
ALLOWED_HOSTS=api.example.com TRIAGELINE_API_KEY=<32+ chars> \
LIVEKIT_URL=wss://... LIVEKIT_API_KEY=... LIVEKIT_API_SECRET=... TRIAGELINE_AGENT_NAME=bilio \
  python -m ui
# worker (same LiveKit credentials and agent name; TRIAGELINE_MODE=assistant asks before acting)
TRIAGELINE_MODE=assistant TRIAGELINE_AGENT_NAME=bilio GEMINI_API_KEY=... \
  python livekit_agent/cascaded_agent.py start
```
Or run both with `docker compose up`.

## 2. Getting a token
**Option A — your backend authenticates the user** (recommended for store apps):
```bash
curl -X POST https://api.example.com/api/mobile/token \
  -H "Authorization: Bearer $TRIAGELINE_API_KEY" -H 'Content-Type: application/json' -d '{}'
```
**Option B — the app signs in to the gateway** with the pilot access code:
`POST /api/auth/login {"access_code": "..."}` returns `access_token`, then
`POST /api/rtc/token {}` with `Authorization: Bearer <access_token>`.

Both return `{url, token, room, identity, expires_in, expires_at, agent_name, mode, tools}`. Every call gets a
fresh room. The token grants only join, publish-microphone and subscribe. It expires after
`TRIAGELINE_RTC_TOKEN_TTL_S` (default 600 s); a room that is already joined keeps working. Never log the token.

## 3. Client (Flutter, `livekit_client`)
```dart
final t = await api.post('/api/rtc/token', {});            // your authenticated call
final room = Room(roomOptions: const RoomOptions(adaptiveStream: true, dynacast: true));
room.events.listen((e) { if (e is TrackSubscribedEvent) {/* agent audio plays automatically */} });
await room.connect(t['url'], t['token']);
await room.localParticipant?.setMicrophoneEnabled(true);    // request mic permission first
// barge-in needs nothing extra: just talk; the worker stops its speech and replans
// on exit:
await room.disconnect();
```
React Native (`@livekit/react-native`), Swift (`LiveKit`) and Kotlin (`io.livekit:livekit-android`) follow the same
four steps: fetch token → connect(url, token) → enable microphone → disconnect.

## 4. Lifecycle rules the web client already follows (`ui/static/rtc-controller.mjs`)
- On app background: mute the mic; hang up after 3 minutes away. Native apps may keep audio alive with a
  platform background-audio entitlement instead.
- On reconnect failure: fetch a **new** token (new room). Never reuse an old one.
- Show "simulated" on every action result. The tools are mocks until you plug real providers into `ToolAdapter.execute`.

## Limits
Pilot access code, not user identity; process-local sessions (one replica); simulated tools; camera frames go
to the PWA session (`/live.html`), not to the voice room.
