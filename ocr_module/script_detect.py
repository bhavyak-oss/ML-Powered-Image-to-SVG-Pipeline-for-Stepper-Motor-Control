"""
script_detect.py
----------------
Text kis script me hai, ye pata karta hai. Isse export decide karta hai:
  - Latin text      -> Hershey stroke font
  - Non-Latin text  -> original image se seedha trace (trace_export.py)
"""
from __future__ import annotations
import unicodedata
from collections import Counter


def _script_of_char(ch: str):
    """Letter ka Unicode script ('LATIN', 'DEVANAGARI', ...). Space/punct/ASCII digit = None."""
    if not ch.isalnum():
        return None
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return "UNKNOWN"
    first = name.split(" ")[0]
    if first == "DIGIT":          # 0-9 kisi bhi script me chalte hain
        return None
    return first


def dominant_script(text: str) -> str:
    c = Counter(s for s in map(_script_of_char, text) if s)
    return c.most_common(1)[0][0] if c else "LATIN"


def needs_trace(text: str) -> bool:
    """True agar text me koi bhi non-Latin letter hai (Hershey font use nahi kar sakta)."""
    return any(s not in (None, "LATIN") for s in map(_script_of_char, text))


def latin_fold(text: str) -> str:
    """Latin accents hatao (é -> e), kyunki stroke font me accented glyph nahi hain.
    Non-Latin akshar waise hi chhod diye jaate hain."""
    out = []
    for ch in text:
        if _script_of_char(ch) == "LATIN":
            out.append("".join(c for c in unicodedata.normalize("NFKD", ch)
                               if not unicodedata.combining(c)))
        else:
            out.append(ch)
    return "".join(out)
