import re

_QUOTES = {('"', '"'), ("'", "'"), ("\u201c", "\u201d")}
_CTRL = re.compile(r"[\x00-\x1f\x7f]")
_TAG = re.compile(r"<[^>]*>")
_SPOOF = re.compile(r"^\s*(?:\[(?:narrator|system|game|moderator|host)\]|(?:narrator|system|game|moderator|host):)\s*", re.I)
_CAPS_TAG = re.compile(r"\[[A-Z][A-Z0-9 _-]*\]")
_LEAK = re.compile(r"\bI(?:['\u2019]m| am)\s+(?:the\s+)?mafia\b|\bas the mafia,|\bmy role is mafia", re.I)


def sanitize(text: str) -> str:
    """DRD 5.8 steps 1-3. Word cap is separate (enforce_cap)."""
    t = text.strip()
    if len(t) >= 2 and (t[0], t[-1]) in _QUOTES:
        t = t[1:-1].strip()
    t = _CTRL.sub(lambda m: " " if m[0] in "\t\r\n" else "", t)
    t = _TAG.sub("", t)
    if _SPOOF.match(t):  # before the caps-tag strip so [NARRATOR] is rewritten, not removed
        t = "(says) " + _SPOOF.sub("", t, count=1)
    t = _CAPS_TAG.sub("", t)
    return " ".join(t.split())


def word_count(text: str) -> int:
    return len(text.split())


def enforce_cap(text: str, cap: int) -> tuple[str, bool]:
    words = text.split()
    return (text, False) if len(words) <= cap else (" ".join(words[:cap]), True)


def self_leak(text: str) -> bool:
    return bool(_LEAK.search(text))
