#!/usr/bin/env python3
"""Compatibility launcher. The ONE application server is the ASGI app in ui/api.py.

    python ui/server.py [--offline]      ==  python -m ui [--offline]
                                         ==  python -m uvicorn ui.api:app --port 8080

Kept so older docs/scripts that call `python ui/server.py` keep working; it serves the same
routes (/live.html PWA, /rtc.html voice, /console evaluation console, /api/*).
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--offline" in argv:
        os.environ["TRIAGELINE_LLM_PLANNER"] = "0"
        os.environ["TRIAGELINE_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ.setdefault("PRELOAD", "0")
    import uvicorn
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8080"))
    print(f"BILIO on http://{host}:{port}   (/live.html PWA · /rtc.html voice · /console evaluation)", flush=True)
    uvicorn.run("ui.api:app", host=host, port=port, workers=1, proxy_headers=True,
                forwarded_allow_ips=os.environ.get("FORWARDED_ALLOW_IPS", "127.0.0.1"),
                log_level=os.environ.get("LOG_LEVEL", "info"))


if __name__ == "__main__":
    main()
