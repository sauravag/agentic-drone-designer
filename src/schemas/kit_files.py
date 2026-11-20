"""Where a run's laser-kit files live and how they are named, shared by the exporter (`kit_export`, which needs
build123d) and the UI (`presenters`, which must run on a sizing-only host). No dependencies beyond the standard
library: must not import from `src.tools`.

Layout under a run folder: `kit/build_sheet.md`, `kit/parts.csv` and `kit/sheets/<stock-slug>-<n>.dxf`.
"""

import re
from typing import Final

KIT_SUBDIR: Final[str] = "kit"
SHEETS_DIR: Final[str] = "sheets"
SHEETS_SUBDIR: Final[tuple[str, str]] = (KIT_SUBDIR, SHEETS_DIR)
BUILD_SHEET_FILE: Final[str] = "build_sheet.md"
PARTS_CSV_FILE: Final[str] = "parts.csv"
KIT_DOWNLOADS: Final[tuple[str, ...]] = (BUILD_SHEET_FILE, PARTS_CSV_FILE)
SHEET_SUFFIX: Final[str] = ".dxf"
SHEET_GLOB: Final[str] = f"*{SHEET_SUFFIX}"
NON_SLUG_RUN: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")


def stock_slug(name: str) -> str:
    """The file-name form of a sheet stock name: lower case, every other run of characters one hyphen."""
    return NON_SLUG_RUN.sub("-", name.lower()).strip("-")


def sheet_file_name(stock_name: str, sheet: int) -> str:
    """`<stock-slug>-<n>.dxf` for 1-based sheet `n` of a stock."""
    return f"{stock_slug(stock_name)}-{sheet}{SHEET_SUFFIX}"
