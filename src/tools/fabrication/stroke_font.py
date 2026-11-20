"""A small single-stroke font for engraving part labels as plain polylines (no fonts, no extra dependency).

Laser software (LightBurn, RDWorks, LaserGRBL) drops DXF TEXT and MTEXT on import, so labels are drawn as open
polylines it always keeps. Each glyph is a list of strokes on a grid GLYPH_WIDTH wide and CAP_HEIGHT tall (units),
baseline at 0. Capitals, digits and the hyphen cover every kit label; a lowercase letter (a split piece's suffix,
`F-03Ua`) is its capital drawn at LOWERCASE_SCALE, so it reads as a small capital. Any other character draws a box,
so a label never silently loses a character. The output is a pure function of the text: deterministic.
"""

from typing import Final

Point = tuple[float, float]
Stroke = tuple[Point, ...]

GLYPH_WIDTH: Final[float] = 4.0  # grid units
CAP_HEIGHT: Final[float] = 6.0
LETTER_GAP: Final[float] = 1.0  # between neighbouring glyphs
LOWERCASE_SCALE: Final[float] = 0.7

_O: Final[Stroke] = ((1, 0), (0, 1), (0, 5), (1, 6), (3, 6), (4, 5), (4, 1), (3, 0), (1, 0))
_P: Final[Stroke] = ((0, 0), (0, 6), (3, 6), (4, 5), (4, 4), (3, 3), (0, 3))
BOX: Final[tuple[Stroke, ...]] = (((0, 0), (4, 0), (4, 6), (0, 6), (0, 0)),)
GLYPHS: Final[dict[str, tuple[Stroke, ...]]] = {
    "A": (((0, 0), (0, 4), (2, 6), (4, 4), (4, 0)), ((0, 3), (4, 3))),
    "B": (((0, 0), (0, 6), (3, 6), (4, 5), (4, 4), (3, 3), (0, 3)), ((3, 3), (4, 2), (4, 1), (3, 0), (0, 0))),
    "C": (((4, 5), (3, 6), (1, 6), (0, 5), (0, 1), (1, 0), (3, 0), (4, 1)),),
    "D": (((0, 0), (0, 6), (2, 6), (4, 4), (4, 2), (2, 0), (0, 0)),),
    "E": (((4, 6), (0, 6), (0, 0), (4, 0)), ((0, 3), (3, 3))),
    "F": (((4, 6), (0, 6), (0, 0)), ((0, 3), (3, 3))),
    "G": (((4, 5), (3, 6), (1, 6), (0, 5), (0, 1), (1, 0), (3, 0), (4, 1), (4, 3), (2, 3)),),
    "H": (((0, 0), (0, 6)), ((4, 0), (4, 6)), ((0, 3), (4, 3))),
    "I": (((1, 6), (3, 6)), ((2, 6), (2, 0)), ((1, 0), (3, 0))),
    "J": (((4, 6), (4, 1), (3, 0), (1, 0), (0, 1)),),
    "K": (((0, 0), (0, 6)), ((4, 6), (0, 2)), ((1, 3), (4, 0))),
    "L": (((0, 6), (0, 0), (4, 0)),),
    "M": (((0, 0), (0, 6), (2, 3), (4, 6), (4, 0)),),
    "N": (((0, 0), (0, 6), (4, 0), (4, 6)),),
    "O": (_O,),
    "P": (_P,),
    "Q": (_O, ((2, 2), (4, 0))),
    "R": (_P, ((2, 3), (4, 0))),
    "S": (((4, 5), (3, 6), (1, 6), (0, 5), (0, 4), (1, 3), (3, 3), (4, 2), (4, 1), (3, 0), (1, 0), (0, 1)),),
    "T": (((0, 6), (4, 6)), ((2, 6), (2, 0))),
    "U": (((0, 6), (0, 1), (1, 0), (3, 0), (4, 1), (4, 6)),),
    "V": (((0, 6), (2, 0), (4, 6)),),
    "W": (((0, 6), (1, 0), (2, 3), (3, 0), (4, 6)),),
    "X": (((0, 0), (4, 6)), ((0, 6), (4, 0))),
    "Y": (((0, 6), (2, 3), (4, 6)), ((2, 3), (2, 0))),
    "Z": (((0, 6), (4, 6), (0, 0), (4, 0)),),
    "0": (_O, ((0, 1), (4, 5))),  # slashed, so it never reads as O
    "1": (((1, 5), (2, 6), (2, 0)), ((1, 0), (3, 0))),
    "2": (((0, 5), (1, 6), (3, 6), (4, 5), (4, 4), (0, 0), (4, 0)),),
    "3": (((0, 5), (1, 6), (3, 6), (4, 5), (4, 4), (3, 3), (4, 2), (4, 1), (3, 0), (1, 0), (0, 1)), ((1, 3), (3, 3))),
    "4": (((3, 0), (3, 6), (0, 2), (4, 2)),),
    "5": (((4, 6), (0, 6), (0, 3), (3, 3), (4, 2), (4, 1), (3, 0), (1, 0), (0, 1)),),
    "6": (((4, 5), (3, 6), (1, 6), (0, 5), (0, 1), (1, 0), (3, 0), (4, 1), (4, 2), (3, 3), (0, 3)),),
    "7": (((0, 6), (4, 6), (1, 0)),),
    "8": (((1, 3), (0, 4), (0, 5), (1, 6), (3, 6), (4, 5), (4, 4), (3, 3), (1, 3), (0, 2), (0, 1), (1, 0), (3, 0),
           (4, 1), (4, 2), (3, 3)),),
    "9": (((0, 1), (1, 0), (3, 0), (4, 1), (4, 5), (3, 6), (1, 6), (0, 5), (0, 4), (1, 3), (4, 3)),),
    "-": (((1, 3), (3, 3)),),
    " ": (),
}


def _glyph(char: str) -> tuple[tuple[Stroke, ...], float]:
    """(strokes, scale) of one character: a lowercase letter is its capital at LOWERCASE_SCALE."""
    if char.islower() and char.upper() in GLYPHS:
        return GLYPHS[char.upper()], LOWERCASE_SCALE
    return GLYPHS.get(char, BOX), 1.0


def _advances(text: str) -> list[float]:
    """Grid-unit advance of each character (its glyph plus the gap after it), at its own scale."""
    return [(GLYPH_WIDTH + LETTER_GAP) * _glyph(char)[1] for char in text]


def text_width(text: str, height_m: float) -> float:
    """Width of `text` drawn with capitals `height_m` tall (no gap after the last glyph)."""
    if not text:
        return 0.0
    gap_after_last = LETTER_GAP * _glyph(text[-1])[1]
    return (sum(_advances(text)) - gap_after_last) * height_m / CAP_HEIGHT


def text_strokes(text: str, height_m: float) -> list[list[Point]]:
    """The text's strokes in metres, centred on (0, 0): the capitals span [-height/2, height/2]."""
    unit = height_m / CAP_HEIGHT
    left, strokes = -text_width(text, height_m) / 2.0, []
    for char, advance in zip(text, _advances(text)):
        glyph, scale = _glyph(char)
        for stroke in glyph:
            strokes.append([(left + x * scale * unit, (y * scale - CAP_HEIGHT / 2.0) * unit) for x, y in stroke])
        left += advance * unit
    return strokes
