"""
hershey_font.py
----------------
Converts text strings into single-stroke (centerline) line segments — no
filled outlines, so a plotter never double-traces a character.

Built-in stick font (uppercase, digits, common punctuation) on a unit grid.

CHANGES (patch):
  * text_to_strokes ab x_scale (akshar chaude) aur chhota advance_ratio leta hai,
    isse akshar patle-lambe nahi lagte aur spacing tight hoti hai.
  * text_width() add kiya - layout ke liye rendered width pata chale.
"""

from __future__ import annotations
from typing import Dict, List, Tuple

Point = Tuple[float, float]
Segment = Tuple[Point, Point]

GLYPH_WIDTH = 3.0
GLYPH_HEIGHT = 7.0

_TL, _TM, _TR = (0.0, 7.0), (1.5, 7.0), (3.0, 7.0)
_ML, _MM, _MR = (0.0, 4.0), (1.5, 4.0), (3.0, 4.0)
_BL, _BM, _BR = (0.0, 0.0), (1.5, 0.0), (3.0, 0.0)

STROKE_FONT: Dict[str, List[List[Point]]] = {
    " ": [],
    "A": [[_BL, _TM, _BR], [_ML, _MR]],
    "B": [[_BL, _TL, (2.2, 7.0), (3.0, 6.0), (3.0, 5.0), (2.2, 4.0), _ML], [(2.2, 4.0), (3.0, 3.0), (3.0, 1.0), (2.2, 0.0), _BL]],
    "C": [[_TR, _TL, _BL, _BR]],
    "D": [[_BL, _TL, (2.0, 7.0), (3.0, 5.5), (3.0, 1.5), (2.0, 0.0), _BL]],
    "E": [[_TR, _TL, _BL, _BR], [_ML, _MR]],
    "F": [[_BL, _TL, _TR], [_ML, _MR]],
    "G": [[_TR, _TL, _BL, _BR, _MR, _MM]],
    "H": [[_BL, _TL], [_BR, _TR], [_ML, _MR]],
    "I": [[_TL, _TR], [_TM, _BM], [_BL, _BR]],
    "J": [[_TR, _BR, _BL]],
    "K": [[_BL, _TL], [_TR, _ML, _BR]],
    "L": [[_TL, _BL, _BR]],
    "M": [[_BL, _TL, _MM, _TR, _BR]],
    "N": [[_BL, _TL, _BR, _TR]],
    "O": [[_TL, _TR, _BR, _BL, _TL]],
    "P": [[_BL, _TL, _TR, _MR, _ML]],
    "Q": [[_TL, _TR, _BR, _BL, _TL], [_MM, (3.0, -1.0)]],
    "R": [[_BL, _TL, _TR, _MR, _ML], [_ML, _BR]],
    "S": [[_TR, _TL, _ML, _MR, _BR, _BL]],
    "T": [[_TL, _TR], [_TM, _BM]],
    "U": [[_TL, _BL, _BR, _TR]],
    "V": [[_TL, _BM, _TR]],
    "W": [[_TL, (0.75, 0.0), _MM, (2.25, 0.0), _TR]],
    "X": [[_TL, _BR], [_TR, _BL]],
    "Y": [[_TL, _MM], [_TR, _MM], [_MM, _BM]],
    "Z": [[_TL, _TR, _BL, _BR]],
    "0": [[_TL, _TR, _BR, _BL, _TL]],
    "1": [[_TR, _BR]],
    "2": [[_TL, _TR, _MR, _ML, _BL, _BR]],
    "3": [[_TL, _TR, _MR], [_ML, _MR, _BR, _BL]],
    "4": [[_TL, _ML, _MR], [_TR, _BR]],
    "5": [[_TR, _TL, _ML, _MR, _BR, _BL]],
    "6": [[_TR, _TL, _BL, _BR, _MR, _ML]],
    "7": [[_TL, _TR, _BR]],
    "8": [[_TL, _TR, _MR, _ML, _TL], [_ML, _BL, _BR, _MR]],
    "9": [[_BL, _BR, _TR, _TL, _ML, _MR]],
    ".": [[(1.4, 0.0), (1.6, 0.0)]],
    ",": [[(1.5, 0.0), (1.2, -0.8)]],
    "-": [[(0.5, 3.5), (2.5, 3.5)]],
    ":": [[(1.4, 5.0), (1.6, 5.0)], [(1.4, 2.0), (1.6, 2.0)]],
    "'": [[(1.5, 7.0), (1.7, 5.5)]],
    "/": [[_BL, _TR]],
    '"': [[(0.8, 7.0), (0.8, 5.5)], [(2.2, 7.0), (2.2, 5.5)]],
    "?": [[(0.0, 6.0), (0.0, 7.0), (3.0, 7.0), (3.0, 4.0), (1.5, 4.0), (1.5, 2.5)], [(1.4, 0.0), (1.6, 0.0)]],
    "!": [[(1.5, 7.0), (1.5, 2.0)], [(1.4, 0.0), (1.6, 0.0)]],
    "(": [[(2.2, 7.0), (1.0, 5.5), (1.0, 1.5), (2.2, 0.0)]],
    ")": [[(0.8, 7.0), (2.0, 5.5), (2.0, 1.5), (0.8, 0.0)]],
}

_UNSUPPORTED_WARNED: set = set()
_ALIASES = {'\u201c': '"', '\u201d': '"', '\u2018': "'", '\u2019': "'",
            '\u2013': '-', '\u2014': '-'}

# Layout defaults (patch)
DEFAULT_X_SCALE = 1.35      # glyph width = 3/7 * 1.35 ~ 0.58 * font_size
DEFAULT_ADVANCE = 0.78      # har akshar ke beech ka total step (glyph + gap)


def _polylines_to_segments(polylines: List[List[Point]]) -> List[Segment]:
    segments: List[Segment] = []
    for polyline in polylines:
        for a, b in zip(polyline, polyline[1:]):
            segments.append((a, b))
    return segments


def _normalize(segments: List[Segment]) -> List[Segment]:
    """Scale the grid down so cap-height = 1.0 (unit height)."""
    return [
        ((x1 / GLYPH_HEIGHT, y1 / GLYPH_HEIGHT), (x2 / GLYPH_HEIGHT, y2 / GLYPH_HEIGHT))
        for (x1, y1), (x2, y2) in segments
    ]


# ---------------------------------------------------------------------------
# Optional real Hershey-font backend (UNTESTED - abhi use mat karo)
# ---------------------------------------------------------------------------
_BACKEND = "builtin"
_hershey_instance = None

try:
    from hershey_fonts import HersheyFonts as _HersheyFonts  # type: ignore
    _HAVE_HERSHEY_LIB = True
except ImportError:
    _HAVE_HERSHEY_LIB = False


def set_backend(name: str) -> None:
    global _BACKEND, _hershey_instance

    if name == "builtin":
        _BACKEND = "builtin"
        return

    if name == "hershey":
        if not _HAVE_HERSHEY_LIB:
            raise RuntimeError(
                "hershey-fonts is not installed. Run `pip install hershey-fonts` "
                "or keep using set_backend('builtin')."
            )
        _hershey_instance = _HersheyFonts()
        _hershey_instance.load_default_font()
        _hershey_instance.normalize_rendering(font_height=1.0)
        _BACKEND = "hershey"
        return

    raise ValueError(f"Unknown backend: {name!r}. Use 'builtin' or 'hershey'.")


def char_to_strokes(char: str) -> List[Segment]:
    if _BACKEND == "hershey" and _hershey_instance is not None:
        return list(_hershey_instance.strokes_for_character(char))

    char = _ALIASES.get(char, char)
    glyph = STROKE_FONT.get(char.upper())
    if glyph is None:
        if char not in _UNSUPPORTED_WARNED:
            _UNSUPPORTED_WARNED.add(char)
            print(f"[hershey_font] no glyph for {char!r} — skipping")
        return []
    return _normalize(_polylines_to_segments(glyph))


def text_width(text: str, font_size_mm: float,
               advance_ratio: float = DEFAULT_ADVANCE,
               x_scale: float = DEFAULT_X_SCALE) -> float:
    """Rendered width in mm (last glyph ke baad ka gap nahi)."""
    if not text:
        return 0.0
    glyph_w = (GLYPH_WIDTH / GLYPH_HEIGHT) * x_scale * font_size_mm
    return (len(text) - 1) * advance_ratio * font_size_mm + glyph_w


def text_to_strokes(
    text: str,
    x_mm: float,
    y_mm: float,
    font_size_mm: float,
    advance_ratio: float = DEFAULT_ADVANCE,
    x_scale: float = DEFAULT_X_SCALE,
) -> List[Segment]:
    """
    Lay out `text` left-to-right; (x_mm, y_mm) is the TOP-LEFT corner.
    Glyph cap-height = font_size_mm.
    """
    segments: List[Segment] = []
    cursor_x = x_mm

    for ch in text:
        for (ux1, uy1), (ux2, uy2) in char_to_strokes(ch):
            segments.append((
                (cursor_x + ux1 * x_scale * font_size_mm, y_mm + font_size_mm * (1 - uy1)),
                (cursor_x + ux2 * x_scale * font_size_mm, y_mm + font_size_mm * (1 - uy2)),
            ))
        cursor_x += font_size_mm * advance_ratio

    return segments


if __name__ == "__main__":
    import sys

    sample = sys.argv[1] if len(sys.argv) > 1 else "HELLO 123"
    segs = text_to_strokes(sample, x_mm=0, y_mm=0, font_size_mm=10)
    print(f"'{sample}' -> {len(segs)} line segments")