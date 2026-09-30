"""livekit_agent/fdb_tools.py — the REAL FDB-v3 12-tool manifest, read from the
already-present, byte-for-byte-copied upstream files (per SETUP.md provenance):
  - livekit_agent/cascaded_agent.py  (AssistantFnc: names, arg names/types/defaults,
    tool descriptions, the 4 domain groupings — Travel & Identity / Finance &
    Billing / Housing & Location / E-Commerce Support)
  - livekit_agent/mock_apis.py       (MockAPIRegistry.FUNCTIONS: confirms the same
    12 names and required kwargs; execution backend for local smoke tests)

NOT available in this sandbox (no network, no FDB Google-Drive data bundle):
  - FDB-v3's difficulty-level (1-3) tags and its official chained-call /
    interruption scenario definitions. These are NOT in the repo and are not
    invented here — see livekit_agent/logs/tool_integration.log for exactly
    what was and wasn't found. The scenarios this session runs are
    hand-written, BILIO-style, covering the same shapes FDB-v3 exercises
    (single call, 2-3-call chain, correction mid-turn, duplicate state change)
    but are explicitly NOT "the official FDB-v3 examples."

Schema shape matches harness/mock_env.py's TOOL_REGISTRY convention so
agent/nlu.py's existing schema-driven arg-filling (build_args/_fill/_arg_for)
can be used unmodified — see fdb_compat_check.py for the generalization check.
"kind" (read_only / state_modifying) is inferred from what each mock function
actually mutates; FDB-v3 itself doesn't label this, so BILIO's own
book-keeping convention is applied here to get dedup/op-ledger behavior.
"""

from __future__ import annotations

FDB_TOOLS = {
    # ── Travel & Identity ────────────────────────────────────────────
    "search_flights": {
        "kind": "read_only",
        "description": "Search for available flights to a destination.",
        "args": {
            "destination": {"type": "string", "required": True, "description": "The city or airport, e.g. 'London' or 'LHR'"},
            "date": {"type": "string", "required": True, "description": "The travel date, e.g. '2026-08-20'"},
        },
    },
    "book_flight": {
        "kind": "state_modifying",
        "description": "Book a flight ticket.",
        "args": {
            "passenger_name": {"type": "string", "required": True, "description": "The name of the passenger, e.g. 'John Doe'"},
            "flight_id": {"type": "string", "required": False, "description": "Flight id, defaults to 'FL123' in the mock backend"},
        },
    },
    "update_identity_doc": {
        "kind": "state_modifying",
        "description": "Update simulated user identity document details (passport, driver license).",
        "args": {
            "doc_type": {"type": "string", "required": True, "description": "Type of document, e.g. 'passport' or 'id_card'"},
            "doc_number": {"type": "string", "required": True, "description": "The document identifier string"},
        },
    },
    # ── Finance & Billing ────────────────────────────────────────────
    "get_card_benefits": {
        "kind": "read_only",
        "description": "Get benefits for a credit card.",
        "args": {
            "card_type": {"type": "string", "required": True, "description": "The card type, e.g. 'platinum' or 'gold'"},
        },
    },
    "get_exchange_rate": {
        "kind": "read_only",
        "description": "Fetch the current foreign exchange rate.",
        "args": {
            "amount": {"type": "number", "required": True, "description": "Amount to convert"},
            "from_currency": {"type": "string", "required": True, "description": "3-letter currency code, e.g. 'USD'"},
            "to_currency": {"type": "string", "required": True, "description": "3-letter currency code, e.g. 'EUR'"},
        },
    },
    "modify_autopay": {
        "kind": "state_modifying",
        "description": "Process billing details / modify Autopay.",
        "args": {
            "bill_type": {"type": "string", "required": True, "description": "Type of bill, e.g. 'credit_card' or 'utilities'"},
            "source_account": {"type": "string", "required": True, "description": "Bank account identifier, e.g. 'checking'"},
        },
    },
    # ── Housing & Location ───────────────────────────────────────────
    "search_apartments": {
        "kind": "read_only",
        "description": "Search for available rental apartments.",
        "args": {
            "city": {"type": "string", "required": True, "description": "Destination city"},
            # bedrooms / max_price are not asked for when the caller doesn't state them: the benchmark is
            # single-turn and the upstream prompt forbids clarifying questions. The official mock accepts
            # **kwargs, so optional filters stated in the tool's own domain are passed through (A-12).
            "bedrooms": {"type": "integer", "required": True, "description": "Number of bedrooms"},
            "max_price": {"type": "number", "required": True, "description": "Maximum monthly rent budget"},
            "pets_allowed": {"type": "boolean", "required": False, "description": "Only pet-friendly listings"},
        },
    },
    "calculate_commute": {
        "kind": "read_only",
        "description": "Calculate commute duration.",
        "args": {
            "origin_address": {"type": "string", "required": True, "description": "Starting location"},
            "destination_address": {"type": "string", "required": True, "description": "Destination location"},
            # upstream wrapper signature is mode: str = "driving" and it ALWAYS forwards mode to the mock,
            # so the schema default is applied when the caller doesn't name a mode (no enum upstream)
            "mode": {"type": "string", "required": True, "default": "driving", "description": "Transport mode, defaults to 'driving'"},
        },
    },
    "update_search_filter": {
        "kind": "state_modifying",
        "description": "Instantly update the user's search filter in the backend system.",
        "args": {
            "filter_name": {"type": "string", "required": True, "description": "Filter key to modify"},
            # upstream mock: value: Any (the wrapper annotates str, but numbers/booleans are the natural values)
            "value": {"type": "any", "required": True, "description": "Filter value to apply"},
        },
    },
    # ── E-Commerce Support ───────────────────────────────────────────
    "track_order": {
        "kind": "read_only",
        "description": "Track physical package status.",
        "args": {
            "order_id": {"type": "string", "required": True, "description": "Order identifier to track, e.g. 'BOB12'"},
        },
    },
    "search_products": {
        "kind": "read_only",
        "description": "Search for products in the catalog.",
        "args": {
            "query": {"type": "string", "required": True, "description": "Product search term, e.g. 'headphones'"},
            "max_price": {"type": "number", "required": False, "description": "Optional maximum budget"},
            "category": {"type": "string", "required": False, "description": "Optional catalog section, e.g. 'electronics'"},
        },
    },
    "add_to_cart": {
        "kind": "state_modifying",
        "description": "Add an item to the shopping cart.",
        "args": {
            "product_id": {"type": "string", "required": True, "description": "ID of the product"},
            # official mock_apis.add_to_cart(product_id, quantity) has NO default (TypeError if missing);
            # the upstream agent wrapper defaults it to 1 -> always sent, schema default applied (A-05)
            "quantity": {"type": "integer", "required": True, "default": 1, "minimum": 1, "description": "Amount to add, defaults to 1"},
        },
    },
}

assert len(FDB_TOOLS) == 12, f"expected 12 FDB-v3 tools, found {len(FDB_TOOLS)}"
DOMAINS = {
    "Travel & Identity": ["search_flights", "book_flight", "update_identity_doc"],
    "Finance & Billing": ["get_card_benefits", "get_exchange_rate", "modify_autopay"],
    "Housing & Location": ["search_apartments", "calculate_commute", "update_search_filter"],
    "E-Commerce Support": ["track_order", "search_products", "add_to_cart"],
}
assert sorted(sum(DOMAINS.values(), [])) == sorted(FDB_TOOLS), "domain grouping doesn't match tool set"
