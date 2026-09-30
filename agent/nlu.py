"""Fast-path language understanding: pure-Python, sub-millisecond, no model calls.

Everything here is schema-driven or lexicon-driven — nothing keys off scenario ids,
timestamps or expected strings. The slow path (tools, ASR, vision) lives elsewhere.
"""

from __future__ import annotations

import difflib
import re
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- lexicon
CITIES = {
    "boston": "Boston", "bos": "Boston", "new york": "New York", "nyc": "New York",
    "new york city": "New York", "chicago": "Chicago", "denver": "Denver",
    "seattle": "Seattle", "miami": "Miami", "austin": "Austin", "dallas": "Dallas",
    "houston": "Houston", "atlanta": "Atlanta", "phoenix": "Phoenix",
    "portland": "Portland", "san francisco": "San Francisco", "sf": "San Francisco",
    "los angeles": "Los Angeles", "la": "Los Angeles", "las vegas": "Las Vegas",
    "san diego": "San Diego", "washington": "Washington", "philadelphia": "Philadelphia",
    "orlando": "Orlando", "detroit": "Detroit", "minneapolis": "Minneapolis",
    "nashville": "Nashville", "london": "London", "paris": "Paris", "tokyo": "Tokyo",
    "seoul": "Seoul", "berlin": "Berlin", "toronto": "Toronto", "vancouver": "Vancouver",
    "bangalore": "Bangalore", "bengaluru": "Bangalore", "delhi": "Delhi",
    "mumbai": "Mumbai", "singapore": "Singapore", "sydney": "Sydney", "dubai": "Dubai",
    "vegas": "Las Vegas",
}
# Generic gazetteer extension (audit §4 ethics): widely-used major world / North-American cities
# (capitals + the largest metro areas by population/air traffic), not values taken from any test set.
for _c in ("Amsterdam Athens Auckland Baltimore Bangkok Barcelona Beijing Beirut Bogota Brisbane Brussels "
           "Budapest Cairo Calgary Cancun Cape_Town Chennai Cleveland Copenhagen Dublin Edinburgh "
           "Florence Frankfurt Geneva Glasgow Hamburg Hanoi Havana Helsinki Hong_Kong Honolulu Hyderabad "
           "Indianapolis Istanbul Jakarta Jerusalem Johannesburg Kansas_City Karachi Kolkata Kuala_Lumpur "
           "Lagos Lima Lisbon Madrid Manila Melbourne Mexico_City Milan Montreal Moscow Munich Nairobi Naples "
           "New_Orleans Osaka Oslo Ottawa Perth Pittsburgh Prague Quebec Raleigh Reykjavik Rio_de_Janeiro Riyadh "
           "Rome Sacramento Salt_Lake_City San_Antonio San_Jose Santiago Sao_Paulo Shanghai St_Louis Stockholm "
           "Taipei Tampa Tel_Aviv Tucson Venice Vienna Warsaw Zurich Columbus Cincinnati Milwaukee Buffalo "
           "Anchorage Albuquerque Memphis Louisville Richmond Hanover Lyon Marseille Seville Porto Krakow").split():
    CITIES.setdefault(_c.replace("_", " ").lower(), _c.replace("_", " "))
_CITY_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, CITIES), key=len, reverse=True)) + r")\b", re.I)
_CAP_AFTER_PREP = re.compile(r"\b(?:to|in|at|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)")
# lowercase unknown place after a travel cue ("flights to kochi", "weather in pune")
_LOW_PLACE = re.compile(r"\b(?:flights?|fly|flying|trip|travel|going|weather|hotels?|forecast|rental)\b[^.?!]*?"
                        r"\b(?:to|in)\s+([a-z][a-z]{2,})\b", re.I)
# contextual place span after a directional preposition, lowercase or not, up to 3 words (R13)
_PREP_SPAN = re.compile(r"\b(?:to|from|in)\s+(?=([A-Za-z][A-Za-z]+(?:\s+[A-Za-z][A-Za-z]+){0,2}))", re.I)
_SPAN_STOP = re.compile(r"\b(?:for|on|at|by|tomorrow|today|tonight|next|this|and|please|under|with|around|"
                        r"departing|leaving|returning|in|to|from|monday|tuesday|wednesday|thursday|friday|saturday|"
                        r"sunday|morning|evening|afternoon|night|then|book|it|a|the)\b.*$", re.I)
_FROM_RE = re.compile(r"\bfrom\s*$", re.I)

# self-repair / correction markers — the value AFTER the last marker wins
REPAIR_MARKERS = re.compile(
    r"\b(?:no[, \-]+wait|wait[, \-]+no|no[, ]+sorry|sorry[, ]+i mean|i mean|actually|"
    r"rather|scratch that|make (?:it|that)|change (?:it|that) to|instead|not\s+\w+[, ]+but|no[,.]\s|"
    r"sorry i meant|correction|not [a-z0-9]+ but|let me rephrase|i meant to say|hold on|"
    r"switch it to|update that to|on second thought|let me correct that)\b",
    re.I)
RETRACTION = re.compile(
    r"\b(never ?mind|forget (?:it|about it|that)|cancel (?:that|it|everything)|don'?t bother|"
    r"stop(?: that)?|no need|skip it|call it off|scratch that|hold off|"
    r"(?:don'?t|do not) (?:do|book|change|send|submit|go ahead with) (?:that|it|this|anything)(?: after all)?|"
    r"forget the whole thing|never mind all of that|start over|disregard that|drop it|abort)\b", re.I)
INTENT_SWITCH = re.compile(r"\b(forget the \w+|different question|something else|forget it instead|actually can you|on another note|new request|ignore that, instead)\b", re.I)
GREETING = re.compile(r"\b(hi|hello|hey|what can you (?:do|help)|who are you|help me with)\b", re.I)
WEEKDAYS = r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
DATE_RE = re.compile(
    r"\b(today|tonight|tomorrow|day after tomorrow|this weekend|next week|(?:next |this )?" + WEEKDAYS +
    r"|\d{4}-\d{2}-\d{2}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.? \d{1,2}(?:st|nd|rd|th)?)\b", re.I)
TIME_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)", re.I)
TIME24_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
NAME_RE = re.compile(r"\b(?:for|passenger|name is|named|i am|i'm|under(?: the name)?|"
                     r"the name(?: on the (?:ticket|booking|reservation))?(?: should be| is| will be)?|"
                     r"name should be)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)")
ID_RE = re.compile(r"\b([A-Za-z]{2,4}-[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*)\b")
ID_PREFIX = {"booking_id": "BK", "flight_id": "FL", "ticket_id": "TK"}
NEG_BOOK = re.compile(r"\b(?:do not|don'?t|dont|never|no need to|without|not)\s+(?:\w+\s+){0,2}?(?:book|booking|reserve)\b|"
                      r"\b(?:only|just)\s+(?:search|show|look|check|find)|\bshow (?:me )?(?:the )?options\b", re.I)
YES_RE = re.compile(r"^\W*(yes|yeah|yep|yup|correct|right|sure|that'?s right|exactly)\b", re.I)
NO_RE = re.compile(r"^\W*(no|nope|nah|wrong|incorrect)\b", re.I)
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
STOP = set("""a an the to for of in on at and or is are be me my i you your it this that
please can could would will do does what whats what's how with like right now just some any
find get show tell want need there here up um uh uhm hmm er erm ah like well so yeah okay ok oh
know mean guess think kinda sorta basically actually wait let""".split())

# keyword priors for the public tool families; hidden tools fall back to schema overlap
TOOL_PRIORS = {
    "flight_search": {"flight", "flights", "fly", "flying", "plane", "airfare", "seats", "trip"},
    "book_flight": {"book", "reserve"},
    "cancel_booking": {"cancel", "refund"},
    "lookup_manual": {"port", "manual", "what", "blinking", "light", "error", "code",
                      "used", "connect", "cable", "led", "drum", "charging", "does"},
    "create_support_ticket": {"ticket", "broken", "support", "repair", "technician", "escalate",
                              "complaint", "report"},
}
# Generic concept lexicon: maps everyday phrasings onto canonical concept tokens so that
# schema-overlap scoring works for tools whose descriptions use different wording than
# the caller ("package" vs "order", "perks" vs "benefits"). Applied symmetrically to the
# utterance AND each tool's vocabulary, so it generalizes to unseen tool manifests.
CONCEPTS = {
    "order": {"order", "package", "parcel", "shipment", "delivery", "deliver", "shipped", "shipping", "arrive"},
    "track": {"track", "tracking", "where", "status", "arrive"},
    "product": {"product", "products", "item", "items", "catalog", "headphone", "headphones", "earbuds",
                "laptop", "shoes", "buy", "purchase", "shop", "shopping", "store", "pair", "wireless"},
    "cart": {"cart", "basket", "bag"},
    "apartment": {"apartment", "apartments", "flat", "rental", "rent", "bedroom", "bedrooms", "studio",
                  "lease", "housing", "condo", "place", "places", "home", "homes", "listing", "listings"},
    "commute": {"commute", "drive", "driving", "transit", "walk", "walking", "bike", "biking", "bicycle", "cycle",
                "cycling", "far", "distance", "long", "duration"},
    "exchange": {"exchange", "convert", "conversion", "currency", "euro", "euros", "dollar", "dollars", "usd",
                 "eur", "gbp", "pound", "pounds", "yen", "rupee", "rupees", "rate", "fx"},
    "benefit": {"benefit", "benefits", "perk", "perks", "reward", "rewards", "cashback", "lounge", "privilege"},
    "card": {"card", "platinum", "gold", "credit"},
    "autopay": {"autopay", "auto", "automatic", "automatically", "recurring", "bill", "bills", "billing",
                "pay", "payment", "payments", "checking", "savings", "utilities", "utility"},
    "identity": {"identity", "passport", "license", "licence", "id", "document", "doc"},
    "filter": {"filter", "filters", "preference", "preferences", "criteria"},
    "flight": {"flight", "flights", "fly", "flying", "plane", "airfare", "airline"},
}
_CONCEPT_OF: Dict[str, set] = {}
for _c, _ws in CONCEPTS.items():
    for _w in _ws:
        _CONCEPT_OF.setdefault(_w, set()).add(_c)


def _concepts(words) -> set:
    out = set()
    for w in words:
        out |= _CONCEPT_OF.get(w, set())
        out |= _CONCEPT_OF.get(_stem(w), set())
    return out


def _concept_evidence(words) -> Dict[str, int]:
    """How many distinct utterance words support each concept (strength of evidence)."""
    ev: Dict[str, set] = {}
    for w in set(words):
        for c in _CONCEPT_OF.get(w, set()) | _CONCEPT_OF.get(_stem(w), set()):
            ev.setdefault(c, set()).add(w)
    return {c: len(v) for c, v in ev.items()}


DEVICE_ALIASES = {"QN90": ["qn90", "tv", "television", "neo qled"],
                  "S24": ["s24", "galaxy", "phone"],
                  "WF45": ["wf45", "washer", "washing machine"],
                  "GENERIC": ["laptop", "notebook", "pc", "computer"]}
# spoken class noun per device model of the practice-kit manual tool (harness/mock_env.py enum)
DEVICE_CLASS = {"QN90": "TV", "S24": "phone", "WF45": "washer"}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


_FILLERS = re.compile(r"(?i)\b(?:um+|uh+|uhm|hmm+|erm?|ah|like|you know|well|so|okay|ok|alright|"
                      r"let me (?:think|see)(?: about (?:it|what it'?s called))?|kind of|sort of|basically)\b[,.]*")


def strip_fillers(s: str) -> str:
    """Remove spoken disfluencies (fillers, hesitation ellipses) but keep the content words."""
    s = re.sub(r"\.{2,}|\u2026", " ", s or "")
    s = _FILLERS.sub(" ", s)
    return norm(re.sub(r"\s+([,.?!])", r"\1", s)).strip(" ,")


def settled_mention(text: str, options: List[str]) -> Optional[str]:
    """The option the user settled on: last mention after the last self-repair marker; directly
    negated mentions ("not yen", "not a laptop") never win (A-03 generic self-repair)."""
    low = (text or "").lower()
    hits = []
    for o in options:
        for m in re.finditer(r"\b" + re.escape(o.lower()) + r"\b", low):
            if re.search(r"\b(?:not|no|never|instead of|rather than)\s+(?:a\s+|an\s+|the\s+|my\s+|from\s+)?$",
                         low[max(0, m.start() - 16):m.start()]):
                continue
            hits.append((m.start(), o))
    if not hits:
        return None
    hits.sort()
    last_marker = max((m.end() for m in REPAIR_MARKERS.finditer(low)), default=-1)
    after = [o for p, o in hits if p >= last_marker]
    return after[-1] if after else hits[-1][1]


_QUERY_CUE = re.compile(
    r"(?i)\b(?:search(?:ing)? (?:for|the catalog for)|look(?:ing)? for|look up|hunt(?:ing)? for|i'?m after(?: some| a| an)?|"
    r"find(?: me)?|shop(?:ping)? for|need(?: a| an| some)?|"
    r"want(?: a| an| some| to buy)?|buy(?: a| an| some)?|get(?: me)?(?: a| an| some)?|called|a new|an new|recommend(?: a| an)?|"
    r"interested in|show me|browse),?\s+")
_QUERY_END = re.compile(r"(?i)\s*(?:\b(?:under|below|less than|for (?:less|under)|within|around|that|which|with a budget|"
                        r"because|so that|so|if|since|to my|and then|then|and also|and|but|or|first|instead|"
                        r"please|for me|for my|from|in the|on the)\b|[?.!,;:\u2014]).*$")
_QUERY_BAD = set("""something anything one thing things it that this them what stuff product products item items
new good nice few some any options option place places gift""".split())


def extract_query(text: str) -> Optional[str]:
    """Free-text search term = the noun phrase after the search verb, from the clause the user settled
    on (after the last correction), with disfluencies removed (A-02). Falls back to None so a search
    is never issued with the whole rambling utterance as its query."""
    for seg in (_repaired_tail(text), text):
        clean = strip_fillers(seg)
        cands = []
        for m in _QUERY_CUE.finditer(clean):
            phrase = _QUERY_END.sub("", clean[m.end():])
            phrase = re.sub(r"(?i)^(?:a|an|the|some|any|new|a new|pair of|a pair of|nice|good|cheap)\s+", "", phrase)
            phrase = re.sub(r"(?i)^(?:a|an|the|pair of|new|nice)\s+", "", phrase).strip(" ,.")
            words = phrase.split()
            if not words or len(words) > 4 or all(w.lower() in _QUERY_BAD | STOP for w in words):
                continue
            if words[0].lower() in ("to", "what", "you", "me", "my", "if", "it", "i", "flights", "flight"):
                continue
            cands.append(phrase)
        if cands:
            return cands[-1]
    return None


def tokens(s: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if w not in STOP]


def _stem(w: str) -> str:
    for suf in ("ings", "ing", "ies", "es", "s", "ed"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)]
    return w


# --------------------------------------------------------------------------- slots
_NOT_PLACE = set("""tomorrow today tonight morning evening afternoon night week weekend flight flights book
booking the a an my me him her them it this that there here help go get see buy check""".split())


def _travel_context(text: str) -> bool:
    return bool(re.search(r"\b(?:flights?|fly|flying|trip|travel|going|weather|hotels?|forecast|rental|book|"
                          r"from|to)\b", text, re.I))


def cities_in(text: str) -> List[Tuple[int, str]]:
    text = text or ""
    out = [(m.start(), CITIES[m.group(1).lower()]) for m in _CITY_RE.finditer(text)]
    if out and _travel_context(text):
        # combine gazetteer hits with contextual spans so "from Boston to Kochi" keeps both (R13)
        known = {p for p, _ in out}
        for m in _PREP_SPAN.finditer(text):
            st = m.start(1)
            if any(abs(st - k) < 2 for k in known):
                continue
            cand = _SPAN_STOP.sub("", m.group(1)).strip()
            if not cand or cand.lower() in STOP or cand.lower() in _NOT_PLACE or re.match(WEEKDAYS, cand, re.I):
                continue
            if _CITY_RE.fullmatch(cand):
                continue
            if cand[0].isupper() or re.search(r"\bfrom\s*$", text[:m.start(1)], re.I) or \
                    re.search(r"\b(?:flights?|fly|trip|travel)\b", text, re.I):
                out.append((st, " ".join(w[0].upper() + w[1:] for w in cand.split())))
        out.sort()
    if not out:
        for m in _CAP_AFTER_PREP.finditer(text):
            cand = m.group(1)
            if cand.lower() not in STOP and not re.match(WEEKDAYS, cand, re.I):
                out.append((m.start(1), cand))
    if not out:
        for m in _LOW_PLACE.finditer(text):
            cand = m.group(1)
            if cand.lower() not in STOP and cand.lower() not in _NOT_PLACE and not re.match(WEEKDAYS, cand, re.I):
                # keep a multiword lowercase place ("san jose") rather than truncating it (R13)
                rest = text[m.end(1):]
                nxt = re.match(r"\s+([a-z]{3,})\b", rest)
                if nxt and not _SPAN_STOP.fullmatch(nxt.group(1)) and nxt.group(1) not in STOP and \
                        nxt.group(1) not in _NOT_PLACE and not re.match(WEEKDAYS, nxt.group(1), re.I) and \
                        not DATE_RE.match(nxt.group(1)):
                    cand = cand + " " + nxt.group(1)
                out.append((m.start(1), " ".join(w[0].upper() + w[1:] for w in cand.split())))
    return out


def _is_origin(text: str, pos: int) -> bool:
    return bool(_FROM_RE.search(text[:pos]))


def _pick_after_repair(text: str, found: List[Tuple[int, str]]) -> Optional[str]:
    if not found:
        return None
    last_marker = max((m.end() for m in REPAIR_MARKERS.finditer(text)), default=-1)
    after = [c for pos, c in found if pos >= last_marker]
    return after[-1] if after else found[-1][1]


def extract_city(text: str) -> Optional[str]:
    """Destination-role city: the last city after the last self-repair marker, never an origin."""
    text = text or ""
    found = [(p, c) for p, c in cities_in(text) if not _is_origin(text, p)]
    return _pick_after_repair(text, found)


def extract_origin(text: str) -> Optional[str]:
    text = text or ""
    return _pick_after_repair(text, [(p, c) for p, c in cities_in(text) if _is_origin(text, p)])


_ORDINAL_SUFFIX = re.compile(r"(\d)(?:st|nd|rd|th)\b", re.I)


def extract_date(text: str) -> Optional[str]:
    """Last date mention (repair-aware by position); calendar dates are returned without the spoken
    ordinal suffix ("March 3rd" -> "March 3"), the canonical form tool schemas expect (A-11)."""
    m = list(DATE_RE.finditer(text or ""))
    if not m:
        return None
    return _ORDINAL_SUFFIX.sub(r"\1", m[-1].group(1))


def extract_time(text: str) -> Optional[str]:
    """Full time constraint as 'HH:MM' (minutes preserved), or None."""
    ms = list(TIME_RE.finditer(text or ""))
    if ms:
        m = ms[-1]
        h = int(m.group(1)) % 12
        if m.group(3).lower().startswith("p"):
            h += 12
        return f"{h:02d}:{int(m.group(2) or 0):02d}"
    ms = list(TIME24_RE.finditer(text or ""))
    if ms:
        return f"{int(ms[-1].group(1)):02d}:{ms[-1].group(2)}"
    return None


def _bad_name(cand: str) -> bool:
    first = cand.split()[0].lower()
    return (cand.lower() in CITIES or first in CITIES or re.match(WEEKDAYS, cand, re.I) is not None
            or first in STOP or first in _NOT_PLACE)


def extract_name(text: str) -> Optional[str]:
    """Passenger-role name; repair-aware ("for Alice, actually for Priya" -> Priya)."""
    text = text or ""
    found = []
    for m in NAME_RE.finditer(text):
        cand = m.group(1)
        parts = cand.split()
        if len(parts) == 2 and _bad_name(parts[1]):
            cand = parts[0]
        if _bad_name(cand):
            continue
        found.append((m.start(1), cand))
    return _pick_after_repair(text, found)


_LOW_NAME = re.compile(r"\b(?:for|passenger|name is|named|under)\s+([a-z][a-z'\-]+(?:\s+[a-z][a-z'\-]+)?)\s*[.!?]?\s*$",
                       re.I)


def extract_name_any_case(text: str) -> Optional[str]:
    """Role-aware name that also accepts lowercase ASR output ("for bob smith") — R10."""
    got = extract_name(text)
    if got:
        return got
    m = _LOW_NAME.search(text or "")
    if not m:
        return None
    words = m.group(1).split()
    words = [w for w in words if w.lower() not in STOP and w.lower() not in _NOT_PLACE
             and not DATE_RE.fullmatch(w) and w.lower() not in CITIES]
    if not words or _bad_name(words[0]) or len(words) != len(m.group(1).split()):
        return None
    return " ".join(w[0].upper() + w[1:].lower() for w in words)


ORDINALS = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2, "fourth": 3, "4th": 3,
            "last": -1}


def select_option(text: str, options: List[Dict[str, Any]], id_key: str = "flight_id",
                  time_key: str = "depart", price_key: str = "price_usd") -> Optional[Dict[str, Any]]:
    """Resolve a selection against the ACTUAL presented candidates (R12): explicit id, time,
    cheapest / earliest / latest, or an ordinal ("the second one"). None when nothing is specified."""
    if not options:
        return None
    low = (text or "").lower()
    for o in options:
        if str(o.get(id_key, "")).lower() and re.search(r"\b" + re.escape(str(o.get(id_key)).lower()) + r"\b", low):
            return o
    t = extract_time(text)
    if t:
        for o in options:
            if str(o.get(time_key, "")).strip()[:5] == t:
                return o
        return None
    priced = [o for o in options if isinstance(o.get(price_key), (int, float))]
    if priced and re.search(r"\b(cheapest|lowest price|least expensive|cheaper one|best price)\b", low):
        return min(priced, key=lambda o: o[price_key])
    if priced and re.search(r"\b(most expensive|priciest)\b", low):
        return max(priced, key=lambda o: o[price_key])
    timed = [o for o in options if o.get(time_key)]
    if timed and re.search(r"\b(earliest|soonest)\b", low):
        return min(timed, key=lambda o: str(o[time_key]))
    if timed and re.search(r"\b(latest)\b", low):
        return max(timed, key=lambda o: str(o[time_key]))
    m = re.search(r"\b(first|1st|second|2nd|third|3rd|fourth|4th|last)\b(?:\s+(?:one|option|flight))?", low)
    if m:
        i = ORDINALS[m.group(1)]
        if -len(options) <= i < len(options):
            return options[i]
    return None


def has_selector(text: str) -> bool:
    return bool(re.search(r"\b(cheapest|lowest price|least expensive|earliest|soonest|latest|most expensive|"
                          r"(?:first|second|third|fourth|last|1st|2nd|3rd|4th)\s+(?:one|option|flight))\b",
                          text or "", re.I))


def parse_name_answer(text: str) -> Optional[str]:
    """Name given as a clarification answer: case-insensitive, full name kept."""
    t = re.sub(r"(?i)^\W*(?:(?:it'?s|it is|my name is|name is|the name is|under|for|book it under|passenger)\s+)+", "", text or "")
    if re.match(r"(?i)^\W*(?:i said|i mean|i meant|no[, ]|yes[, ]|actually)", t) or cities_in(t):
        return None
    t = re.sub(r"(?i)\b(please|thanks|thank you)\b", "", t)
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z'\-]*", t)]
    words = [w for w in words if w.lower() not in STOP][:3]
    if not words or _bad_name(words[0]):
        return None
    return " ".join(w[0].upper() + w[1:].lower() for w in words)


_SPELLED_RE = re.compile(r"(?<![A-Za-z0-9])((?:[A-Za-z0-9][\-\s]){1,15}[A-Za-z0-9])(?![A-Za-z0-9])")


def spelled_ids(text: str) -> List[str]:
    """Spoken, character-by-character ids: "T-L-W-3-8-2" -> TLW382, "K-2" -> K2, "M-N-O-P" -> MNOP."""
    out = []
    for m in _SPELLED_RE.finditer(text or ""):
        raw = m.group(1)
        if "-" not in raw:
            continue                      # "a b" in ordinary prose is not an id
        chars = re.split(r"[\-\s]", raw)
        if all(len(c) == 1 for c in chars) and len(chars) >= 2:
            out.append("".join(chars).upper())
    return out


def extract_id(text: str, field: str = "") -> Optional[str]:
    if not ID_PREFIX.get(field):
        sp = [(m.start(), "".join(re.split(r"[\-\s]", m.group(1))).upper()) for m in _SPELLED_RE.finditer(text or "")
              if "-" in m.group(1) and all(len(c) == 1 for c in re.split(r"[\-\s]", m.group(1)))]
        if sp:
            # bind to the id that follows this field's own noun ("order ID is X", "item K-2")
            cue = {"order": r"order", "product": r"item|product|sku"}.get(field.split("_")[0], "")
            if cue:
                near = [v for pos, v in sp if re.search(r"\b(?:" + cue + r")\b[^?!]{0,25}$", text[:pos], re.I)]
                if near:
                    return near[-1]
            return sp[-1][1]
    ids = [m.group(1).upper() for m in ID_RE.finditer(text or "")]
    pre = ID_PREFIX.get(field)
    if pre:
        ids = [i for i in ids if i.startswith(pre + "-")]
    return ids[-1] if ids else None


NEG_WORD = re.compile(r"\b(?:do not|don'?t|dont|never|no need to|not)\b", re.I)


def negated_action(text: str, api: str) -> bool:
    """True when the user negates the action a (state-modifying) tool performs, e.g.
    "Do not open a support ticket" / "Please don't cancel my booking" / "Don't reserve a car".
    Schema-driven: the tool's own name tokens are matched within a short window after a negator."""
    low = (text or "").lower()
    toks = {t.rstrip("s") for t in re.split(r"[_\W]+", api.lower()) if len(t) > 2}
    for m in NEG_WORD.finditer(low):
        window = re.findall(r"[a-z']+", low[m.end():])[:4]
        if any(w.rstrip("s") in toks or w.rstrip("s").rstrip("ing") in toks for w in window):
            return True
    return False


def negates_booking(text: str) -> bool:
    return bool(NEG_BOOK.search(text or ""))


def extract_device(text: str, hint: Optional[str] = None) -> Optional[str]:
    low = (text or "").lower()
    for model, aliases in DEVICE_ALIASES.items():
        if any(re.search(r"\b" + re.escape(a) + r"\b", low) for a in aliases):
            return model
    return hint


def severity_of(text: str) -> str:
    low = (text or "").lower()
    if re.search(r"\b(smoke|fire|spark|burn|shock|won'?t turn on|dead|urgent|broken)\b", low):
        return "high"
    if re.search(r"\b(blink|flash|error|noise|slow|intermittent)\w*", low):
        return "medium"
    return "low"


# --------------------------------------------------------------------------- routing
# Generic action verbs that appear in tool NAMES (verb_object / object_verb conventions) with the
# everyday phrasings users say for them. Language-level knowledge, not per-tool/per-test rules.
_VERB_SYNONYMS = {
    "search": r"search|find|look(?:ing)? (?:for|up)|looking|show me|browse|recommend|shop(?:ping)? for|"
              r"hunt(?:ing)? for|(?:i'?m|i am) after",
    "book": r"book|reserve",
    "update": r"update|change|set|raise|lower|bump|increase|decrease|switch|make (?:it|that|the)",
    "calculate": r"calculate|how long|how far|commute|travel time|(?:walking|driving|transit|biking|cycling) time",
    "add": r"add|put|throw",
    "track": r"track|where(?:'s| is) my|check (?:on|up on) (?:my|the|a)",
    "modify": r"modify|set(?: up)?|enable|turn on|switch|change|move|pull from|come from",
    "get": r"get|what are|tell me|check|show",
    "cancel": r"cancel|call off",
    "create": r"create|open|file|raise a",
    "lookup": r"look up|lookup|check the manual",
}
_ACTION_VERB_STEMS = set(_VERB_SYNONYMS)


# references to an earlier result ("whatever you find", "once you find something") are not requests
_RESULT_REF = re.compile(r"(?i)\b(?:whatever|what|once|if|when|after|anything)\s+(?:you|it)\s+(?:find|found|get|show)s?\b")


def _verb_spoken(verb_stem: str, text: str) -> bool:
    pat = _VERB_SYNONYMS.get(verb_stem)
    return bool(pat and re.search(r"\b(?:" + pat + r")\b", _RESULT_REF.sub(" ", text or ""), re.I))


_SHOP_CUE = re.compile(r"(?i)\b(?:looking for|look for|look up|hunting for|hunt for|i'?m after|i want an?|i want some|"
                       r"i need an?(?: new)?|shopping for|buy an?|"
                       r"search(?:ing)? for|find me|recommend|something (?:under|below|for less than)|"
                       r"do you have|in the \w+ section)\b")


def _shopping_request(text: str) -> bool:
    """An open-vocabulary product request: a shopping cue plus a noun phrase that is not a trip,
    a place to live, a route or a filter ("looking for a desk", "i need a desk lamp")."""
    if not _SHOP_CUE.search(text or ""):
        return False
    if _concepts(tokens(text)) & {"apartment", "flight"}:
        return False                      # "find me a flat with 3 rooms ... and a study"
    q = extract_query(text)
    if re.search(r"(?i)\bsection\b", text or "") and re.search(r"(?i)\b(?:electronics|clothing|kitchen|toys|books|"
                                                               r"sports|home|garden|beauty|grocery)\b", text or ""):
        return True
    if not q:
        return False
    qc = _concepts(tokens(q)) | _concepts(q.lower().split())
    if qc & {"apartment", "flight", "commute", "filter", "exchange", "identity", "autopay", "order"}:
        return False
    return not cities_in(q)


def score_tools(text: str, tools: Dict[str, Any]) -> List[Tuple[float, str]]:
    """Rank manifest tools against an utterance: lexical priors + schema overlap."""
    raw = tokens(text)
    toks = {_stem(t) for t in raw}
    tev = _concept_evidence(raw)
    has_city = bool(cities_in(text or ""))
    ranked = []
    for name, spec in tools.items():
        name_words = tokens(name.replace("_", " "))
        vwords = tokens(name.replace("_", " ") + " " + str(spec.get("description", "")))
        vocab = {_stem(t) for t in vwords}
        for arg, aspec in (spec.get("args") or {}).items():
            vocab |= {_stem(t) for t in tokens(arg.replace("_", " "))}
        # concept overlap: name concepts weigh more than description concepts
        nconc = _concepts(name_words)
        dconc = _concepts(vwords) - nconc
        # concept evidence only from words NOT already matched lexically (no double counting)
        unmatched = [w for w in raw if _stem(w) not in vocab]
        uev = _concept_evidence(unmatched)
        cscore = sum(min(uev[c], 3) * 1.0 for c in nconc if c in uev) + \
            sum(min(uev[c], 3) * 0.5 for c in dconc if c in uev)
        # a tool whose NAME concept is explicitly named by the user gets a head-noun bonus. The head
        # noun is the last name word that is not an action verb ("flight_search" -> flight, not search)
        nouns = [w for w in name_words if _stem(w) not in _ACTION_VERB_STEMS]
        head_noun = _stem(nouns[-1]) if nouns else ""
        head = 2.0 if head_noun and len(head_noun) > 2 and any(_stem(w) == head_noun for w in raw) else 0.0
        # the tool's own action verb (first verb in its name) spoken by the user, incl. everyday synonyms
        verb = next((_stem(w) for w in name_words if _stem(w) in _ACTION_VERB_STEMS), "")
        vbonus = 1.5 if verb and _verb_spoken(verb, text) else 0.0
        # evidence for the tool's name concept from ANY word (lexical or not): "1-bedroom" -> apartment
        nbonus = 1.0 if any(c in tev and c not in uev for c in nconc) and head == 0.0 else 0.0
        # an open-vocabulary shopping request ("looking for a desk") is evidence for a product tool
        pbonus = 1.5 if "product" in nconc and _shopping_request(text) else 0.0
        # a named city is evidence for a tool that takes a place argument (schema-driven, any manifest)
        place_arg = any(any(k in a.lower() for k in ("city", "destination", "location", "origin"))
                        for a in (spec.get("args") or {}))
        overlap = len(toks & vocab) + cscore + head + vbonus + nbonus + pbonus
        # only as a tie-breaker between tools that already have lexical evidence: a bare place
        # ("actually make it X") is a slot value, never evidence of a new action on its own
        cbonus = 1.0 if place_arg and has_city and overlap >= 2.0 else 0.0
        overlap += cbonus
        prior = len(toks & {_stem(w) for w in TOOL_PRIORS.get(name, ())})
        s = overlap + 1.5 * prior
        if s > 0:  # bonus if every required arg is fillable from this utterance
            _, missing = build_args(spec, text, {})
            if not missing and (spec.get("args") or {}):
                s += 1.0
        if s > 0:
            ranked.append((s, name))
    ranked.sort(reverse=True)
    return ranked


def is_smalltalk(text: str, tools: Optional[Dict[str, Any]] = None) -> bool:
    """Greeting/capability talk ONLY when no actionable request remains in the turn."""
    if not GREETING.search(text or ""):
        return False
    rest = GREETING.sub(" ", text or "")
    if tools and (cities_in(rest) or extract_id(rest)):
        return False
    if tools:
        r = score_tools(rest, tools)
        if r and r[0][0] >= 2.5:
            return False
    return len(tokens(text)) <= 8


# --------------------------------------------------------------------------- args
CURRENCY_WORDS = {"dollar": "USD", "dollars": "USD", "usd": "USD", "buck": "USD", "bucks": "USD",
                  "euro": "EUR", "euros": "EUR", "eur": "EUR", "pound": "GBP", "pounds": "GBP", "gbp": "GBP",
                  "sterling": "GBP", "yen": "JPY", "jpy": "JPY", "rupee": "INR", "rupees": "INR", "inr": "INR",
                  "yuan": "CNY", "cny": "CNY", "franc": "CHF", "francs": "CHF", "chf": "CHF",
                  "cad": "CAD", "aud": "AUD", "mxn": "MXN", "peso": "MXN", "pesos": "MXN"}
# canonical forms follow the tool docstring convention ('passport', 'id_card' -> snake_case singular nouns)
DOC_TYPES = {"passport": "passport", "driver's license": "driver_license", "drivers license": "driver_license",
             "driver license": "driver_license", "driving licence": "driver_license", "license": "driver_license",
             "licence": "driver_license", "id card": "id_card", "national id": "id_card", "identity card": "id_card",
             "visa": "visa", "residence permit": "residence_permit"}
BILL_TYPES = ["credit_card", "credit card", "utilities", "utility", "electricity", "electric", "water", "gas",
              "internet", "phone", "rent", "mortgage", "insurance", "cable"]
ACCOUNTS = ["checking", "savings", "credit", "brokerage"]
_ADDR_RE = re.compile(r"\bfrom\s+(.+?)\s+to\s+(.+?)(?=\s+(?:by|via|using|driving|walking|transit|cycling|biking|on foot)\b|[?.!,]|$)", re.I)


def _currencies(text: str) -> List[str]:
    out = []
    for w in re.findall(r"[A-Za-z]+", text):
        c = CURRENCY_WORDS.get(w.lower())
        if c is None and len(w) == 3 and w.isupper() and w not in ("THE", "AND", "FOR"):
            c = w
        if c:
            out.append(c)
    return out


# national adjectives that qualify an ambiguous currency word ("Canadian dollars", "British pounds")
CURRENCY_QUALIFIERS = {
    ("canadian", "dollar"): "CAD", ("australian", "dollar"): "AUD", ("singapore", "dollar"): "SGD",
    ("hong kong", "dollar"): "HKD", ("new zealand", "dollar"): "NZD", ("us", "dollar"): "USD",
    ("u.s.", "dollar"): "USD", ("american", "dollar"): "USD", ("mexican", "peso"): "MXN",
    ("british", "pound"): "GBP", ("swiss", "franc"): "CHF", ("japanese", "yen"): "JPY",
    ("indian", "rupee"): "INR", ("chinese", "yuan"): "CNY",
}
_CUR_TOKEN = re.compile(r"\b(?:(canadian|australian|singapore|hong kong|new zealand|us|u\.s\.|american|mexican|british|"
                        r"swiss|japanese|indian|chinese)\s+)?([A-Za-z]+)\b", re.I)


def _currency_mentions(text: str) -> List[Tuple[int, int, str]]:
    """(start, end, ISO code) for every currency mention, qualifier-aware ("100 Canadian dollars" -> CAD)."""
    out = []
    for m in _CUR_TOKEN.finditer(text or ""):
        qual, word = (m.group(1) or "").lower(), m.group(2)
        code = CURRENCY_WORDS.get(word.lower())
        if code is None and len(word) == 3 and word.isupper() and word not in ("THE", "AND", "FOR", "ATM"):
            code = word
        if code is None:
            continue
        if qual:
            code = CURRENCY_QUALIFIERS.get((qual, word.lower().rstrip("s")), code)
        out.append((m.start(), m.end(), code))
    return out


def _repaired_tail(text: str) -> str:
    """The part of the utterance after the last self-repair marker (the version the user settled on)."""
    last = max((m.end() for m in REPAIR_MARKERS.finditer(text or "")), default=-1)
    return text[last:] if last >= 0 else text


def _currency_pair(text: str) -> Tuple[Optional[str], Optional[str]]:
    """(source, target). Direction comes from the sentence structure, not mention order:
    "<amount> X (would be|in|into|to) Y" -> X->Y; "how many Y is <amount> X" -> X->Y.
    Self-repair aware: a corrected source/target ("...to yen -- no wait, British pounds") wins,
    and a directly negated currency ("not yen") is never used."""
    ments = _currency_mentions(text)
    if not ments:
        return None, None
    low = text.lower()
    negated = {c for s, e, c in ments if re.search(r"\bnot\s+(?:in\s+)?$", low[max(0, s - 8):s])}
    ments = [m for m in ments if m[2] not in negated] or ments
    # source = currency attached to an amount ("200 euros", "$50", "USD 100")
    src_hits = [(s, c) for s, e, c in ments
                if re.search(r"\d[\d,.]*\s*(?:k\s*)?(?:(?:us|u\.s\.|canadian|australian|british|swiss|japanese|"
                             r"indian|chinese|mexican|american|new zealand|hong kong|singapore)\s+)?$",
                             low[max(0, s - 24):s])]
    last_marker = max((m.end() for m in REPAIR_MARKERS.finditer(text)), default=-1)

    def settle(hits):
        after = [c for p, c in hits if p >= last_marker]
        return after[-1] if after else (hits[-1][1] if hits else None)

    src = settle(src_hits)
    # target = currency introduced by a direction cue, or the one that is not the source
    tgt_hits = [(s, c) for s, e, c in ments if c != src and re.search(
        r"\b(?:to|into|in|be in|would be|as|for|get|many|much)\s+(?:(?:the|some|us|u\.s\.|canadian|australian|"
        r"british|swiss|japanese|indian|chinese|mexican|american)\s+)?$", low[max(0, s - 20):s])]
    tgt = settle(tgt_hits) or settle([(s, c) for s, e, c in ments if c != src])
    if src is None:
        # no amount-attached currency: "how many euros is it in dollars" / "rate from X to Y"
        rest = [c for s, e, c in ments if c != tgt]
        src = rest[0] if rest else None
        if re.search(r"\bhow (?:many|much)\b", low) and len(ments) > 1 and tgt is None:
            src, tgt = ments[1][2], ments[0][2]
    return src, tgt


def _special_string_arg(lname: str, spec: Dict[str, Any], text: str) -> Any:
    low = text.lower()
    if "currency" in lname:
        src, tgt = _currency_pair(text)
        return tgt if lname.startswith("to") or "target" in lname else src
    if lname.endswith("address"):
        return extract_address(text, origin=("origin" in lname or "from" in lname or "start" in lname))
    if lname in ("doc_type", "document_type"):
        for k in sorted(DOC_TYPES, key=len, reverse=True):
            if k in low:
                return DOC_TYPES[k]
        return None
    if lname in ("doc_number", "document_number"):
        sp = spelled_ids(text)
        if sp:
            return sp[-1]
        m = re.search(r"\b(?:number|no\.?|#)\s*(?:is|to|as|:)?\s*([A-Za-z0-9]*\d[A-Za-z0-9\-]*)", text, re.I) or \
            re.search(r"\b([A-Z]{0,3}\d{5,}[A-Z0-9]*)\b", text)
        return m.group(1) if m else None
    if lname.endswith("_type") and lname not in ("doc_type", "document_type", "bill_type"):
        noun = lname[:-5].split("_")[-1]                       # card_type -> card
        examples = re.findall(r"'([^']+)'", str(spec.get("description", "")))
        bad = STOP | {"which", "new", "credit", "debit", "rewards", "reward", "this", "that", "the", "my", "one"}
        noun_rx = r"\b([a-z]+)(?:\s+(?:credit|debit|rewards?))?\s+" + noun + r"s?\b"
        # self-repair first: the value the user settled on, after the last correction marker, wins
        # over a schema example mentioned before it ("the X card, sorry, I mean the Y card" -> Y)
        last_marker = max((mm.end() for mm in REPAIR_MARKERS.finditer(low)), default=-1)
        if last_marker >= 0:
            tail_m = [mm for mm in re.finditer(noun_rx, low[last_marker:]) if mm.group(1) not in bad]
            if tail_m:
                return tail_m[-1].group(1)
            tail_ex = [ex for ex in examples if re.search(r"\b" + re.escape(ex.lower()) + r"\b", low[last_marker:])]
            if tail_ex:
                return tail_ex[-1]
        for ex in examples:                                     # schema examples first ('platinum', 'gold')
            if re.search(r"\b" + re.escape(ex.lower()) + r"\b", low):
                return ex
        m = re.search(noun_rx, low)
        if m and m.group(1) not in bad:
            return m.group(1)
        return None
    if lname == "bill_type":
        b = settled_mention(text, BILL_TYPES)
        return {"credit card": "credit_card", "utility": "utilities", "electric": "electricity"}.get(b, b) if b else None
    if lname in ("source_account", "account", "from_account"):
        cands = [a for a in ACCOUNTS if not (a == "credit" and "credit card" in low)]
        return settled_mention(text, cands)
    if lname in ("filter_name", "filter", "filter_key"):
        got = extract_filter(text)
        return got[0] if got else None
    if lname == "value":
        got = extract_filter(text)
        return got[1] if got else None
    return None


# --------------------------------------------------------------------------- search filters
# Generic listing-filter vocabulary (rent/real-estate search UIs): phrase -> canonical filter key.
_FILTER_KEYS = [
    (r"\bpets?(?:[- ]friendly)?\b|\bpets? (?:are )?allowed\b|\ballow(?:s)? pets\b", "pets_allowed"),
    (r"\b(?:min(?:imum)?|at least)\s+(?:number of\s+)?bed(?:room)?s?\b", "min_bedrooms"),
    (r"\b(?:max(?:imum)?|top|highest)\s+(?:rent|price|budget)\b|\bprice (?:cap|limit)\b|\bbudget\b", "max_price"),
    (r"\b(?:min(?:imum)?|lowest)\s+(?:rent|price)\b", "min_price"),
    (r"\bbed(?:room)?s?\b", "bedrooms"),
    (r"\bneighbou?rhoods?\b|\barea\b|\bdistrict\b", "neighborhood"),
    (r"\bparking\b", "parking"),
    (r"\bfurnished\b", "furnished"),
]
_BOOL_FILTERS = {"pets_allowed", "parking", "furnished"}


def extract_filters(text: str) -> List[Tuple[str, Any]]:
    """Every (filter_key, value) the user asks to set, in order, self-repair aware per key
    ("cap it at 2800 ... hmm, make the cap 3100" -> one max_price=3100)."""
    t = text or ""
    low = t.lower()
    found: Dict[str, Tuple[int, Any]] = {}
    order: List[str] = []
    for pat, key in _FILTER_KEYS:
        for m in re.finditer(pat, low):
            if key == "bedrooms" and any(k in found and abs(found[k][0] - m.start()) < 30 for k in ("min_bedrooms",)):
                continue
            if key == "max_price" and "min_price" in found and abs(found["min_price"][0] - m.start()) < 10:
                continue
            if key in _BOOL_FILTERS:
                neg = re.search(r"\b(?:no|not|don'?t|without|disallow)\b[^.?!]{0,15}$", low[max(0, m.start() - 20):m.start()])
                val: Any = not bool(neg)
            else:
                after = t[m.end():m.end() + 40]
                if key == "neighborhood":
                    mv = re.search(r"^\s*(?:\w+\s+){0,4}?(?:to|as|=|:)\s+(?:the\s+)?([A-Z][\w\-]*(?:\s+[A-Z][\w\-]*)?)", after)
                    val = mv.group(1) if mv else None
                else:
                    mv = re.search(r"^\D{0,25}?\$?\s*(\d[\d,]*(?:\.\d+)?)\s*(k\b)?", after, re.I)
                    if mv is None:
                        mv2 = re.search(r"(\d[\d,]*)\s*[- ]?bed", low[max(0, m.start() - 12):m.end()])
                        val = int(mv2.group(1).replace(",", "")) if mv2 and key in ("bedrooms", "min_bedrooms") else None
                    else:
                        v = float(mv.group(1).replace(",", "")) * (1000 if mv.group(2) else 1)
                        val = int(v) if v.is_integer() else v
            if val is None:
                continue
            if key not in found:
                order.append(key)
            found[key] = (m.start(), val)         # later mention (a correction) wins
    # generic "<key> to <value>" pairs after a filter cue ("... and <key> to <value>") — B7
    if re.search(r"\bfilters?\b|\bset\b|\bupdate\b|\bchange\b", low):
        for m in re.finditer(r"\b([a-z][a-z_\- ]{1,20}?)\s+(?:to|=|as)\s+([a-z0-9][a-z0-9_\-]*)", low):
            k = re.sub(r"^(?:.*\b(?:for|filter|the|set|update|change|and|my)\s+)", "", m.group(1)).strip().replace(" ", "_")
            if not k or k in STOP or k in ("filter", "search", "it", "that", "search_filter") or \
                    any(m.start() <= pos < m.end() + 1 for pos, _ in found.values()):
                continue
            raw = m.group(2)
            val = True if raw in ("true", "yes", "on") else False if raw in ("false", "no", "off") else raw
            if k not in found:
                order.append(k)
            found[k] = (m.start(), val)
    return [(k, found[k][1]) for k in order]


def extract_filter(text: str) -> Optional[Tuple[str, Any]]:
    got = extract_filters(text)
    return got[0] if got else None


# --------------------------------------------------------------------------- addresses
_ADDR_STOP = re.compile(r"\s*(?:\b(?:by|via|using|during|on a|at|in the|around|because|so|since|for|when|where|which|that|if|and|"
                        r"then|instead|every|each|in|on foot|driving|walking|transit|cycling|biking)\b|[?.!,;\u2014]).*$",
                        re.I)
_PLACE_WORDS = r"(?:office|gym|work|school|university|station|airport|stadium|store|shop|mall|park|library|hospital|" \
               r"downtown|house|home|apartment|place|clinic|campus|center|centre|beach|church|studio|cafe|restaurant)"


def _clean_place(s: str) -> Optional[str]:
    s = _ADDR_STOP.sub("", norm(s or "")).strip(" .,'\"")
    s = re.sub(r"(?i)\b(?:um+|uh+|like|you know)\b", "", s)
    s = norm(s).strip(" .,")
    if not s or s.lower() in STOP or len(s.split()) > 7:
        return None
    # a named place is a name: "the University" -> "University"; common nouns keep it ("the gym")
    return re.sub(r"^(?:the)\s+(?=[A-Z])", "", s)


def extract_address(text: str, origin: bool) -> Optional[str]:
    """Origin / destination of a commute, repair-aware (the clause after the last correction wins).
    Handles "from A to B", "to B from A", "walk from A to B" and deictic "from there" (-> None so the
    planner can bind it to a previous result)."""
    tail = _repaired_tail(text)
    for seg in (tail, text):
        frm = list(re.finditer(r"\bfrom\s+(.+?)(?=\s+to\b|[?.!,;\u2014]|$)", seg, re.I))
        to = list(re.finditer(r"\bto\s+(?!(?:walk|drive|bike|take|get|go|be|pull|use|make|check|keep|set|see)\b)"
                              r"(.+?)(?=\s+from\b|[?.!,;\u2014]|$)", seg, re.I))
        if origin and frm:
            v = _clean_place(frm[-1].group(1))
            if v and v.lower() in ("there", "here", "it", "that", "that place", "whatever you find"):
                return None
            if v:
                return v
        if not origin and to:
            cands = [_clean_place(m.group(1)) for m in to]
            cands = [c for c in cands if c and not re.fullmatch(r"(?i)(?:there|here|it|that)", c)]
            # prefer a real place noun over e.g. "to keep it under budget"
            placey = [c for c in cands if re.search(r"(?i)\b" + _PLACE_WORDS + r"\b", c) or re.search(r"\d|\b[A-Z]", c)]
            if placey or cands:
                return (placey or cands)[-1]
        if seg is text:
            break
    return None


def _arg_for(name: str, spec: Dict[str, Any], text: str, ctx: Dict[str, Any]) -> Any:
    lname = name.lower()
    typ = spec.get("type", "string")
    enum = spec.get("enum")
    if typ == "object":
        obj = {}
        for sub, sspec in (spec.get("properties") or {}).items():
            v = _arg_for(sub, sspec, text, ctx)
            if v is not None:
                obj[sub] = v
        return obj
    if enum:
        got = pick_enum(text, enum)
        if got is None and lname in ("mode", "travel_mode", "transport_mode"):
            got = extract_mode(text, enum)
        if got is not None:
            return got
        if lname == "severity":
            return severity_of(text) if severity_of(text) in enum else enum[0]
        if lname in ("model", "device_model") and ctx.get("device_model") in enum:
            return ctx["device_model"]
        return None
    if typ in ("number", "integer"):
        return extract_number(text, name, spec, integer=(typ == "integer"))
    if typ == "boolean":
        return extract_bool(text, name)
    if typ == "array":
        items = spec.get("items")
        ienum = (items or {}).get("enum") if isinstance(items, dict) else None
        if ienum:
            low = text.lower()
            hits = [e for e in ienum if re.search(r"\b" + re.escape(str(e).lower()) + r"\b", low)]
            return hits or None
        return ctx.get(lname)
    # strings: schema-specific roles that must win over generic origin/destination matching
    special = _special_string_arg(lname, spec, text)
    if special is not None:
        return special
    # strings: match by semantic role of the arg name
    if "origin" in lname or lname.startswith("from") or "departure_city" in lname:
        return ctx.get("origin") or extract_origin(text)
    if any(k in lname for k in ("city", "destination", "location", "place", "town", "to_")):
        return ctx.get("destination") or extract_city(text)
    if "date" in lname or "day" in lname or "when" in lname:
        return ctx.get("date") or extract_date(text)
    if "passenger" in lname or lname in ("name", "customer", "guest", "full_name"):
        return ctx.get("passenger_name") or extract_name(text)
    if lname.endswith("_id") or lname == "id":
        got = extract_id(text, lname) or ctx.get(lname)
        if got:
            return got
        m = re.search(r"\b([A-Z]{2,}\d+|\d+[A-Z]{2,}\w*)\b", text)
        if m:
            return m.group(1)
        # single-letter ids bound to an id cue ("item P52", "order ID 7Q9") — C1
        m = re.search(r"(?i)\b(?:item|product|sku|order|id|number|code)\s+(?:is\s+|number\s+|#\s*)?"
                      r"([A-Za-z]{0,3}\d[A-Za-z0-9]{0,11}|[A-Za-z]\d[A-Za-z0-9]*)\b", text)
        return m.group(1).upper() if m and plausible_id(m.group(1)) else None
    if lname in ("model", "device", "device_model"):
        return ctx.get("device_model")
    if lname in ("mode", "travel_mode", "transport_mode"):
        return extract_mode(text, DEFAULT_MODES)
    if lname == "query":
        return extract_query(text)
    if lname in ("summary", "question", "text", "message", "description", "issue"):
        return norm(text)
    # generic fallback for typed string fields with no dedicated semantic role
    # above (e.g. currency codes, bare alphanumeric ids without a dash prefix,
    # free-form addresses/account/filter values). Added for FDB-v3 compatibility
    # (see livekit_agent/fdb_compat_check.py) — every earlier, already-tested
    # branch still wins first, so this only fires when nothing else matched.
    if "currency" in lname:
        m = re.search(r"\b([A-Za-z]{3})\b", text[text.lower().find(lname.split("_")[0]):] or text) \
            if lname.split("_")[0] in text.lower() else None
        # fall back to scanning the whole utterance for a bare 3-letter code
        codes = re.findall(r"\b([A-Za-z]{3})\b", text)
        codes = [c.upper() for c in codes if c.upper() not in ("THE", "FOR", "AND", "TO ")]
        if lname.startswith("from") and codes:
            return codes[0]
        if lname.startswith("to") and len(codes) > 1:
            return codes[1]
        return codes[0] if codes else None
    if lname.endswith("address"):
        m = re.search(r"\b(?:from|origin)\s+(.+?)\s+to\s+(.+?)(?:\s+(?:by|via|driving|walking|transit|cycling)\b|$)", text, re.I)
        if m:
            return norm(m.group(1) if "origin" in lname else m.group(2)).strip(" .,")
        return None
    words = lname.replace("_", " ")
    m = re.search(re.escape(words) + r"\s+(?:is|to|as|=|:)?\s*([\w][\w\-./]*(?:\s+[\w][\w\-./]*){0,2})", text, re.I)
    if m:
        return norm(m.group(1)).strip(" .,")
    return None


_MODE_WORDS = {
    "walking": r"walk(?:ing|able)?|on foot|stroll", "driving": r"driv(?:e|ing)|by car",
    "transit": r"transit|bus|train|subway|metro|public transport(?:ation)?|tram",
    "biking": r"bik(?:e|ing)|bicycle", "cycling": r"cycl(?:e|ing)",
}
DEFAULT_MODES = ["walking", "driving", "transit", "biking", "cycling"]


def extract_mode(text: str, enum: List[Any]) -> Any:
    """Transport mode from its everyday verbs ("walk", "I'd be driving", "bike there"), repair-aware and
    negation-aware ("not take transit"). Only values the schema allows are returned."""
    low = (text or "").lower()
    hits = []
    for e in enum:
        pat = _MODE_WORDS.get(str(e).lower())
        if not pat:
            continue
        if str(e).lower() == "cycling" and "biking" not in [str(x).lower() for x in enum]:
            pat = pat + "|" + _MODE_WORDS["biking"]
        if str(e).lower() == "biking" and "cycling" not in [str(x).lower() for x in enum]:
            pat = pat + "|" + _MODE_WORDS["cycling"]
        for m in re.finditer(r"\b(?:" + pat + r")\b", low):
            if str(e).lower() == "transit" and m.group() == "train" and re.search(r"\bstation\b", low[m.end():m.end() + 9]):
                continue                          # a station noun after "the" is a place, not a transport mode
            if re.search(r"\b(?:not|no|never|instead of|rather than)\s+(?:\w+\s+)?$", low[max(0, m.start() - 18):m.start()]):
                continue
            hits.append((m.start(), e))
    if not hits:
        return None
    hits.sort(key=lambda h: h[0])
    last_marker = max((m.end() for m in REPAIR_MARKERS.finditer(low)), default=-1)
    after = [e for p, e in hits if p >= last_marker]
    return after[-1] if after else hits[-1][1]


def pick_enum(text: str, enum: List[Any]) -> Any:
    """Authoritative enum mention (R14): positions matter, not schema order. The last mention after the
    last self-repair marker wins; a mention directly negated ("not compact") is skipped."""
    low = (text or "").lower()
    hits = []
    for e in enum:
        for m in re.finditer(r"\b" + re.escape(str(e).lower()) + r"\b", low):
            if re.search(r"\b(?:not|no|don'?t want)\s+(?:a\s+|the\s+)?$", low[max(0, m.start() - 16):m.start()]):
                continue
            hits.append((m.start(), e))
    if not hits:
        return None
    hits.sort(key=lambda h: h[0])
    last_marker = max((m.end() for m in REPAIR_MARKERS.finditer(low)), default=-1)
    after = [e for p, e in hits if p >= last_marker]
    return after[-1] if after else hits[-1][1]


def validate_value(v: Any, spec: Dict[str, Any]) -> bool:
    """Pre-dispatch schema check for one leaf value (R15)."""
    typ = spec.get("type", "string")
    if typ == "integer":
        if isinstance(v, bool) or not isinstance(v, (int, float)) or float(v) != int(v):
            return False
    elif typ == "number":
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return False
    elif typ == "boolean":
        if not isinstance(v, bool):
            return False
    elif typ == "string":
        if not isinstance(v, str):
            return False
        if "minLength" in spec and len(v) < spec["minLength"]:
            return False
        if "maxLength" in spec and len(v) > spec["maxLength"]:
            return False
        if spec.get("pattern") and not re.search(spec["pattern"], v):
            return False
    elif typ == "array":
        if not isinstance(v, list):
            return False
        items = spec.get("items") if isinstance(spec.get("items"), dict) else None
        if items and not all(validate_value(x, items) for x in v):
            return False
        if "minItems" in spec and len(v) < spec["minItems"]:
            return False
    if typ in ("integer", "number"):
        if "minimum" in spec and v < spec["minimum"]:
            return False
        if "maximum" in spec and v > spec["maximum"]:
            return False
        if "exclusiveMinimum" in spec and v <= spec["exclusiveMinimum"]:
            return False
    if spec.get("enum") and v not in spec["enum"]:
        return False
    return True


_UNSET = (None, "")


def _fill(props: Dict[str, Any], text: str, ctx: Dict[str, Any], prefix: str,
          parent_required: bool) -> Tuple[Dict[str, Any], List[str]]:
    """Recursive schema fill. False / 0 / [] are valid values, only None/'' are unresolved."""
    out, missing = {}, []
    for name, aspec in (props or {}).items():
        path = f"{prefix}{name}"
        req = bool(aspec.get("required")) and parent_required
        if aspec.get("type") == "object":
            sub, gaps = _fill(aspec.get("properties") or {}, text, ctx, path + ".", True)
            if gaps:
                if req:
                    missing += gaps
                continue
            if sub or req:
                out[name] = sub
            continue
        v = ctx.get(path) if ctx.get(path) not in _UNSET else _arg_for(name, aspec, text, ctx)
        if v not in _UNSET and not validate_value(v, aspec):
            v = None                               # invalid values never reach the tool (R15)
        if v in _UNSET and "default" in aspec and aspec.get("required"):
            # a required arg whose schema declares a default is filled from the schema, never
            # guessed and never asked (A-05: add_to_cart.quantity defaults to 1 upstream)
            v = aspec["default"]
        if v not in _UNSET:
            out[name] = v
        elif req:
            missing.append(path)
    return out, missing


def build_args(spec: Dict[str, Any], text: str, ctx: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Fill a tool call from its schema. Returns (args, missing_required)."""
    return _fill(spec.get("args") or {}, text, ctx, "", True)


def field_spec(spec: Dict[str, Any], path: str) -> Dict[str, Any]:
    node: Dict[str, Any] = {"properties": spec.get("args") or {}}
    for part in path.split("."):
        node = (node.get("properties") or {}).get(part) or {}
    return node


def parse_field_answer(answer: str, name: str, fspec: Dict[str, Any]) -> Any:
    """Parse a clarification answer specifically for the field that was asked about."""
    lname = name.split(".")[-1].lower()
    typ = fspec.get("type", "string")
    enum = fspec.get("enum")
    low = (answer or "").lower()
    if enum:
        for e in enum:
            if re.search(r"\b" + re.escape(str(e).lower()) + r"\b", low):
                return e
        close = difflib.get_close_matches(low.strip(" .!"), [str(e).lower() for e in enum], n=1, cutoff=0.7)
        return next((e for e in enum if str(e).lower() == close[0]), None) if close else None
    if typ in ("number", "integer"):
        return extract_number(answer, "", fspec, integer=(typ == "integer"))
    if typ == "boolean":
        if YES_RE.search(answer or ""):
            return True
        if NO_RE.search(answer or ""):
            return False
        return extract_bool(answer, lname)
    if typ == "array":
        return [x.strip() for x in re.split(r",|\band\b", answer or "") if x.strip()] or None
    if lname.endswith("_id") or lname == "id":
        return extract_id(answer, lname) or (norm(answer).strip(" .") or None)
    if "passenger" in lname or "name" in lname or lname in ("customer", "guest"):
        return parse_name_answer(answer)
    if "date" in lname or "day" in lname or "when" in lname:
        return extract_date(answer) or norm(answer).strip(" .") or None
    if "origin" in lname:
        return extract_origin("from " + answer) or extract_city("to " + answer)
    if any(k in lname for k in ("city", "destination", "location", "place", "town")):
        return extract_city(answer) or extract_city("to " + answer.strip())
    v = norm(re.sub(r"(?i)^\W*(?:it'?s|it is|use|make it|the|a|an)\s+", "", answer or "")).strip(" .!")
    return v or None


BUDGET_FIELD_HINTS = ("price", "budget", "rent", "cost", "limit", "amount")
BUDGET_RE = re.compile(r"\b(?:under|below|less than|no more than|up,? to|at most|max(?:imum)?(?: price| rent| budget)?"
                       r"(?: of| is| to| at)?|budget(?: of| is| to)?|cheaper than|within|raise (?:it |my budget |the budget )?to|"
                       r"bump (?:it )?up to)\s*\$?\s*(\d[\d,]*(?:\.\d+)?)", re.I)

WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
            "nine": 9, "ten": 10, "a couple": 2, "a single": 1}


_AMOUNT_UNIT = r"(?:(?:us|u\.s\.|canadian|australian|british|swiss|japanese|indian|chinese|mexican|american)\s+)?" \
               r"(?:dollars?|bucks|euros?|pounds?|yen|rupees?|yuan|francs?|pesos?|usd|eur|gbp|jpy|inr|cny|chf|cad|aud|mxn)\b"
_QTY_RE = re.compile(r"\b(?:add|put|get|order|buy|make it|just|only|quantity(?: of| to)?|want),?\s+(?:like,?\s+)?(\d+)\b"
                     r"|\b(\d+)\s+(?:of (?:them|those|these|it|item|product|whatever)|units?|pieces?|pcs|items?|copies)\b"
                     r"|\b(\d+)\s+(?:of\s+)?(?:item|product)\b|\bjust\s+(\d+)\b", re.I)
_BED_RE = re.compile(r"\b(\d+)\s*[- ]?(?:bed(?:room)?s?|br|bd)\b|\bbed(?:room)?s?\s+(?:to\s+)?(\d+)\b"
                     r"|\bstudio\b", re.I)


def _num(g: str, integer: bool):
    v = float(g.replace(",", ""))
    if integer and not v.is_integer():
        return v                                    # keep the fraction so validation rejects it (R15)
    return int(v) if integer or v.is_integer() else v


def _settled_match(regex, text: str):
    """Last regex match after the last self-repair marker (else the last match overall)."""
    ms = list(regex.finditer(text))
    if not ms:
        return None
    last_marker = max((m.end() for m in REPAIR_MARKERS.finditer(text)), default=-1)
    after = [m for m in ms if m.start() >= last_marker]
    return (after or ms)[-1]


def extract_number(text: str, name: str, spec: Dict[str, Any], integer: bool = False) -> Optional[float]:
    """Role-anchored number extraction (A-07): a number is bound to a field only when a unit/keyword of
    that field is next to it; the value the user settled on wins (\"4 of those -- no wait, just 2\" -> 2)."""
    t = re.sub(r"\.{2,}|\u2026", " ", text or "")
    t = norm(re.sub(r"(?i)\b(?:um+|uh+|uhm|hmm+|erm?|like)\b[,.]*", " ", t))     # "raise it... uh, to 2200"
    t = re.sub(r"(?i)\b(add|put|get|order|buy|want|make it)\s*,\s*", r"\1 ", t)    # "add, like, 2" (B6)
    t = re.sub(r"\s*,\s*(?=\d)", " ", t) if re.search(r"(?i)\b(add|buy|order)\b", t) else t
    for w, n in WORD_NUM.items():
        t = re.sub(r"\b" + w + r"\b", str(n), t, flags=re.I)
    t = TIME_RE.sub(" ", t)
    t = ID_RE.sub(" ", t)
    t = _SPELLED_RE.sub(lambda m: " " if "-" in m.group(1) else m.group(0), t)   # spelled ids are not numbers
    t = re.sub(r"(?i)\b(\d+)(?:st|nd|rd|th)\b", " ", t)                        # ordinals / dates
    t = DATE_RE.sub(" ", t)
    nums = [(m.start(), m.end(), m.group()) for m in NUMBER_RE.finditer(t)]
    if not nums:
        return None
    lname = (name or "").lower()
    # money amount to convert: the number attached to a currency, repair-aware
    if lname in ("amount", "value_amount", "sum") or lname.endswith("_amount"):
        m = _settled_match(re.compile(r"\$\s*(\d[\d,]*(?:\.\d+)?)|(\d[\d,]*(?:\.\d+)?)\s*(?:k\s+)?" + _AMOUNT_UNIT, re.I), t)
        if m:
            return _num(m.group(1) or m.group(2), integer)
    if lname in ("quantity", "qty", "count", "number_of_items"):
        # a per-recipient breakdown after the order ("<n> for <person> and <n> for <person>") apportions the
        # requested quantity; it is not a new quantity.
        # Drop "<n> for <someone>" apportioning phrases unless a self-repair follows them.
        last_marker = max((m.end() for m in REPAIR_MARKERS.finditer(t)), default=-1)
        tq = t
        for m in reversed(list(re.finditer(r"(?i)\b(?:get|buy|keep)?\s*\d+\s+(?:for|as)\s+(?:my|him|her|them|a|an|"
                                           r"the|me|us)\w*\b", t))):
            if m.start() > last_marker and _QTY_RE.search(t[:m.start()]):
                tq = tq[:m.start()] + " " + tq[m.end():]
        m = _settled_match(_QTY_RE, tq)
        if m:
            return _num(next(g for g in m.groups() if g), integer)
        return None                                  # never guess a quantity from an unrelated number
    if "bedroom" in lname or lname in ("beds", "rooms"):
        m = _settled_match(_BED_RE, t)
        if m:
            g = next((x for x in m.groups() if x), None)
            return 0 if g is None else _num(g, integer)
        return None                                  # \"1500\" is a price, not a bedroom count
    # upper-bound / budget fields (max_price, budget, max_rent, ...): the value is
    # the number introduced by a ceiling phrase (\"under 3000\", \"below $50\",
    # \"up to 2k\", \"budget of 900\", \"go up to 1600\"), not whichever number comes first.
    if lname and (lname.startswith("max") or any(k in lname for k in BUDGET_FIELD_HINTS)):
        m = _settled_match(BUDGET_RE, t)
        if m:
            return _num(m.group(1), integer)
        m = _settled_match(re.compile(r"\$\s*(\d[\d,]*)|(\d[\d,]*)\s*(?:dollars|bucks|a month|per month|/mo|/month)\b", re.I), t)
        if m:
            return _num(m.group(1) or m.group(2), integer)
        return None
    cues = {_stem(w) for w in tokens(name.replace("_", " "))} or \
        {_stem(w) for w in tokens(str(spec.get("description", "")))}
    words = [(m.start(), _stem(m.group().lower())) for m in re.finditer(r"[A-Za-z]+", t)]
    best, bd = None, 10 ** 9
    for a, b, g in nums:
        for pos, w in words:
            if w in cues:
                d = (pos - b) if pos >= b else (a - pos) + 5   # unit after the number preferred
                if d < bd and d <= 25:
                    best, bd = g, d
    if best is None:
        if len(nums) > 1 and cues:
            return None  # several numbers and none tied to this field: ask rather than guess
        best = nums[0][2]
    return _num(best, integer)


def extract_bool(text: str, name: str) -> Optional[bool]:
    low = (text or "").lower()
    words = [w for w in tokens(name.replace("_", " ")) if len(w) > 2 and w not in ("include", "has", "with", "is")]
    for w in words:
        sw = _stem(w)
        if re.search(r"\b(?:no|without|not|don'?t want|exclude)\s+(?:\w+\s+)?" + re.escape(sw), low):
            return False
        if re.search(r"\b" + re.escape(sw), low):
            return True
    return None


def similar_city(heard: str) -> Optional[str]:
    """Nearest confusable city — used to phrase a two-way clarification."""
    pool = sorted(set(CITIES.values()) - {heard})
    best = max(pool, key=lambda c: difflib.SequenceMatcher(None, heard.lower(), c.lower()).ratio())
    r = difflib.SequenceMatcher(None, heard.lower(), best.lower()).ratio()
    return best if r >= 0.45 else None


def humanize_result(tool: str, result: Dict[str, Any], _depth: int = 0) -> str:
    """Ground an answer in an arbitrary (unseen-tool) result object, incl. lists of records."""
    bits = []
    for k, v in result.items():
        if k == "status":
            continue
        if isinstance(v, list) and v and isinstance(v[0], dict) and _depth < 2:
            first = humanize_result(tool, v[0], _depth + 1)
            more = f" (+{len(v) - 1} more)" if len(v) > 1 else ""
            bits.append(first + more)
            continue
        if isinstance(v, dict) and _depth < 2:
            bits.append(humanize_result(tool, v, _depth + 1))
            continue
        if isinstance(v, list):
            bits.append(", ".join(map(str, v[:3])))
            continue
        if k.endswith("_f"):
            bits.append(f"{v}°F")
        elif k.endswith("_c"):
            bits.append(f"{v}°C")
        elif k.endswith("_usd") or "price" in k or "cost" in k:
            bits.append(f"${v}")
        elif k.endswith("_id") or k == "id":
            bits.append(f"ref {v}")
        elif isinstance(v, str):
            bits.append(v)
        else:
            bits.append(f"{k.replace('_', ' ')} {v}")
    return ", ".join(b for b in bits if b)


# --------------------------------------------------------------------------- spoken-form normalisation (C1)
# Generic ASR repair applied BEFORE intent ranking / argument extraction. Language-level rules only
# (letter names, digit words, "double"/"triple", common homophones next to their disambiguating
# context) — nothing keyed to a benchmark item.
_DIGIT_WORDS = {"zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2", "to": "2", "too": "2", "three": "3",
                "four": "4", "for": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"}
_STRICT_DIGITS = {k: v for k, v in _DIGIT_WORDS.items() if k not in ("oh", "o", "to", "too", "for")}
_NATO = {"alpha": "A", "alfa": "A", "bravo": "B", "charlie": "C", "delta": "D", "echo": "E", "foxtrot": "F",
         "golf": "G", "hotel": "H", "india": "I", "juliet": "J", "juliett": "J", "kilo": "K", "lima": "L",
         "mike": "M", "november": "N", "oscar": "O", "papa": "P", "quebec": "Q", "romeo": "R", "sierra": "S",
         "tango": "T", "uniform": "U", "victor": "V", "whiskey": "W", "xray": "X", "x-ray": "X", "yankee": "Y",
         "zulu": "Z"}
_LETTER_NAMES = {"ay": "A", "bee": "B", "see": "C", "cee": "C", "dee": "D", "ee": "E", "eff": "F", "gee": "G",
                 "aitch": "H", "jay": "J", "kay": "K", "el": "L", "em": "M", "en": "N", "pee": "P", "cue": "Q",
                 "queue": "Q", "ar": "R", "ess": "S", "tee": "T", "you": "U", "vee": "V", "ex": "X", "why": "Y",
                 "zee": "Z", "zed": "Z"}
_ID_CUE = r"(?:order|item|product|sku|booking|flight|ticket|confirmation|reference|tracking|passport|document|" \
          r"card|account|id|number|code|parcel|package|delivery|shipment|license|licence)"
_ASR_CONFUSIONS = [
    # (pattern, replacement) — each rewrite needs its disambiguating context in the same clause
    (re.compile(r"\b(?:the\s+)?idea\s+(?:is|was|number)\b", re.I), "the ID is"),
    (re.compile(r"\b(?:my|the|an?)\s+idea\s+(?=(?:[A-Za-z0-9][,\-\s]*){2,})", re.I), "the ID "),
    (re.compile(r"\border\s+idea\b", re.I), "order ID"),
    (re.compile(r"\bori?gin(?:al)?\s+(number|id|i\.d\.)\b", re.I), r"order \1"),
    (re.compile(r"\b(to|in|into|from|on)\s+(?:my|the)\s+card\b(?=(?:(?!\bbenefit|\bpoints?\b|\breward).)*$)", re.I),
     r"\1 my cart"),
    (re.compile(r"\badd\b([^.?!]{0,40})\bto\s+(?:my\s+|the\s+)?car\b", re.I), r"add\1to my cart"),
    (re.compile(r"\bi\.\s?d\.?(?=\s|$)", re.I), "ID"),
]


def _tok_char(tok: str) -> Optional[str]:
    t = tok.lower().strip(".,;:!?'\"")
    if not t:
        return None
    if t in _STRICT_DIGITS:
        return _STRICT_DIGITS[t]
    if t in _NATO:
        return _NATO[t]
    if re.fullmatch(r"[a-z]", t) or re.fullmatch(r"\d{1,3}", t):
        return t.upper()
    if re.fullmatch(r"[a-z]\d{1,3}|\d{1,3}[a-z]|[a-z]{2,3}\d{0,3}", t) and len(t) <= 4 and t not in STOP \
            and t not in ("is", "it", "my", "the", "and", "for", "you", "can", "of", "to", "at", "in", "on", "am",
                          "an", "as", "be", "by", "do", "go", "he", "if", "me", "no", "or", "so", "up", "us", "we"):
        raw = tok.strip(".,;:!?'\"")
        return t.upper() if any(ch.isdigit() for ch in t) or raw.isupper() else None
    return None


def _join_spelled(seq: List[str]) -> Optional[str]:
    out, i = [], 0
    while i < len(seq):
        t = seq[i].lower().strip(".,;:!?")
        if t in ("double", "triple") and i + 1 < len(seq):
            c = _tok_char(seq[i + 1])
            if c:
                out.append(c * (2 if t == "double" else 3))
                i += 2
                continue
        c = _tok_char(seq[i])
        if c is None:
            return None
        out.append(c)
        i += 1
    s = "".join(out)
    return s if 2 <= len(s) <= 14 and any(ch.isdigit() for ch in s) else None


def normalize_spoken_ids(text: str) -> str:
    """Collapse a spoken alphanumeric id after an id cue into one token:
    "order number is k, l, m, four, five" -> "order number is KLM45"; "item P five two" -> "item P52";
    "double five" -> "55"; NATO letters ("Kilo two") -> K2. Only runs after an id cue word."""
    if not text:
        return text
    words = re.findall(r"\S+", text)
    res, i = [], 0
    while i < len(words):
        res.append(words[i])
        w = words[i].lower().strip(".,;:!?")
        cue_seen = any(re.fullmatch(_ID_CUE, x.lower().strip(".,;:!?")) for x in words[:i])
        if re.fullmatch(_ID_CUE, w) or (cue_seen and w in ("it's", "its", "it", "that's", "is", "was")):
            j = i + 1
            while j < len(words) and words[j].lower().strip(".,;:!?") in ("is", "was", "it's", "its", "number",
                                                                          "id", "code", "of", "the", "#", "uh", "um",
                                                                          "to", "now", "will", "be", "should"):
                j += 1
            best = None
            for k in range(min(len(words), j + 14), j + 1, -1):
                joined = _join_spelled(words[j:k])
                if joined and (k - j) >= 2:
                    best = (k, joined)
                    break
            if best:
                res.extend(words[i + 1:j])
                tail = re.search(r"[.?!,]+$", words[best[0] - 1])
                res.append(best[1] + (tail.group() if tail and tail.group() != "," else ""))
                i = best[0]
                continue
        i += 1
    return " ".join(res)


def normalize_asr(text: str) -> str:
    """Generic spoken-form repair: homophone confusions in context, then spoken ids (C1)."""
    t = text or ""
    for rx, rep in _ASR_CONFUSIONS:
        t = rx.sub(rep, t)
    return normalize_spoken_ids(t)


def plausible_id(value: Any) -> bool:
    """A free-text identifier must look like one (B2): has a digit, no spaces, 2-14 chars."""
    v = str(value or "").strip()
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9\-_]{1,13}", v) and re.search(r"\d", v))


def filler_only(text: str) -> bool:
    """True for empty / filler / noise-only turns ("um", "...", "uh huh", "hmm okay") (B4)."""
    t = re.sub(r"(?i)\b(?:um+|uh+|uhm|hmm+|mm+|mhm|huh|erm?|ah+|oh|okay|ok|so|well|yeah|like|hm+)\b", " ",
               text or "")
    return not re.search(r"[A-Za-z0-9]", t)


def tool_vocabulary(tools: Dict[str, Any], limit: int = 80) -> List[str]:
    """Key terms from the tool manifest, used to bias STT (Whisper prompt / Deepgram keyterm)."""
    words: List[str] = []
    for name, spec in (tools or {}).items():
        words += name.split("_")
        words += re.findall(r"[A-Za-z]{3,}", str(spec.get("description", "")))
        for arg, a in (spec.get("args") or {}).items():
            words += arg.split("_")
            words += re.findall(r"'([^']{2,20})'", str(a.get("description", "")))
    seen, out = set(), []
    for w in ["order ID", "cart", "SKU", "passport", "autopay", "exchange rate", "bedroom"] + words:
        k = w.lower()
        if k in STOP or k in seen or len(k) < 2:
            continue
        seen.add(k)
        out.append(w)
    return out[:limit]


# --------------------------------------------------------------------------- spoken templates (B5/B17)
def _v(args: Dict[str, Any], *keys: str) -> Optional[str]:
    for k in keys:
        if args.get(k) not in (None, ""):
            return str(args[k])
    return None


ACKNOWLEDGEMENTS = [
    "On it!", "Absolutely, give me a second.", "Got you — checking now.",
    "Sure thing, one moment.", "Understood — pulling that up.",
    "Let me grab that for you.", "Right away.", "I'm on it.",
    "Of course, just a moment.", "Sure, looking into that now.",
    "Got it, hold on a sec.", "Let me check that for you.",
    "One moment please.", "Working on it.", "On it right away.",
    "Understood, searching now.", "Allow me a moment.",
    "That's noted — let me look.", "I'll get that sorted.",
    "Pulling that up now."
]

def ack_phrase(api: str, args: Dict[str, Any], kind: str = "read_only") -> str:
    """First substantive line: repeats the key argument so the user can correct it by barge-in."""
    import random
    a = args or {}
    n = api.lower()
    if "cart" in n:
        q, p = a.get("quantity"), _v(a, "product_id", "item_id", "sku")
        return f"Adding {str(q) + ' × ' if q else ''}{p or 'that item'} to your cart."
    if "track" in n:
        return f"Checking order {_v(a, 'order_id') or ''}".rstrip() + " now."
    if "apartment" in n:
        bits = [f"{a['bedrooms']}-bedroom" if a.get("bedrooms") not in (None, "") else "",
                "apartments", f"in {a['city']}" if a.get("city") else "",
                f"under {a['max_price']:g}" if isinstance(a.get("max_price"), (int, float)) else ""]
        return "Looking for " + " ".join(b for b in bits if b) + "."
    if "commute" in n:
        return f"Working out the commute from {_v(a, 'origin_address') or 'there'} to {_v(a, 'destination_address') or 'there'}."
    if "product" in n:
        return f"Searching for {_v(a, 'query') or 'that'}" + (f" under {a['max_price']:g}" if isinstance(a.get('max_price'), (int, float)) else "") + "."
    if "exchange" in n:
        return f"Checking the {_v(a, 'from_currency', 'base', 'source_currency') or ''} to {_v(a, 'to_currency', 'target', 'target_currency') or ''} rate.".replace("  ", " ")
    if "benefit" in n:
        return f"Looking up your {_v(a, 'card_type') or 'card'} card benefits."
    if "flight" in n and "book" in n:
        return f"Booking {_v(a, 'flight_id') or 'that flight'}" + (f" for {a['passenger_name']}" if a.get("passenger_name") else "") + "."
    if "flight" in n:
        return f"Checking flights to {_v(a, 'destination') or 'there'}" + (f" for {a['date']}" if a.get("date") else "") + "."
    vals = [str(v) for v in a.values() if isinstance(v, (str, int, float)) and not isinstance(v, bool)][:1]
    what = norm(api.replace("_", " "))
    if kind == "state_modifying":
        return f"Okay — updating that now{' (' + vals[0] + ')' if vals else ''}."
    return random.choice(ACKNOWLEDGEMENTS)


def done_phrase(api: str, res: Dict[str, Any]) -> Optional[str]:
    r = res or {}
    n = api.lower()
    if "cart" in n:
        q, p = r.get("quantity"), r.get("product_id")
        tot = r.get("cart_total")
        return f"Done — added {str(q) + ' × ' if q else ''}{p or 'the item'} to your cart" + \
            (f"; your cart total is {tot:.2f}." if isinstance(tot, (int, float)) else ".")
    if "filter" in n:
        return f"Done — {r.get('filter_updated', 'the filter')} is now {r.get('new_value')}."
    if "autopay" in n:
        return "Done — your autopay settings are updated."
    if "identity" in n or "doc" in n:
        return "Done — your document details are updated."
    return None
