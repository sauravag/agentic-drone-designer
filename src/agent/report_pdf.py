"""design_report.pdf: the Markdown report typeset with reportlab, plus figures after the sections they illustrate.

The Markdown report stays the single source of content: this module converts the small Markdown subset that
`report.render` emits (headings, pipe tables, bullets, paragraphs, inline code and bold) rather than rebuilding the
report, so the two files cannot drift. LLM- and user-authored text arrives defanged by report.py; here it is
un-escaped and then XML-escaped for reportlab's paragraph markup, so it can never inject tags.
"""

import functools
import io
import logging
import re
from pathlib import Path
from typing import Callable, Final, Mapping, Optional, Sequence
from xml.sax.saxutils import escape

import matplotlib
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Flowable, Image, Paragraph, SimpleDocTemplate, Table, TableStyle

from src.agent import report_figures
from src.agent.report import SECTION_ENERGY, SECTION_POLAR, SECTION_SITL, SECTION_SIZING, ReportInputs

logger = logging.getLogger(__name__)

PDF_FILE: Final[str] = "design_report.pdf"
BODY_FONT: Final[str] = "DejaVuSans"  # ships with matplotlib; covers the report's degree, squared and +- signs
MONO_FONT: Final[str] = "DejaVuSansMono"
FONT_FILES: Final[dict[str, str]] = {
    BODY_FONT: "DejaVuSans.ttf", f"{BODY_FONT}-Bold": "DejaVuSans-Bold.ttf",
    f"{BODY_FONT}-Oblique": "DejaVuSans-Oblique.ttf", f"{BODY_FONT}-BoldOblique": "DejaVuSans-BoldOblique.ttf",
    MONO_FONT: "DejaVuSansMono.ttf"}
MARGIN: Final[float] = 18 * mm
FIGURE_MAX_HEIGHT_FRACTION: Final[float] = 0.85  # of the frame height, so a figure never overflows a page
GRID_COLOUR: Final[colors.Color] = colors.HexColor("#b0b7bf")
HEADER_FILL: Final[colors.Color] = colors.HexColor("#e6ebf0")
TABLE_SEPARATOR: Final[re.Pattern[str]] = re.compile(r"^\|(\s*-+\s*\|)+$")
CELL_BOUNDARY: Final[re.Pattern[str]] = re.compile(r"(?<!\\)\|")
MARKDOWN_ESCAPE: Final[re.Pattern[str]] = re.compile(r"\\(.)")
CODE_SPAN: Final[re.Pattern[str]] = re.compile(r"`([^`]+)`")
BOLD_SPAN: Final[re.Pattern[str]] = re.compile(r"\*\*(.+?)\*\*")


@functools.cache
def _register_fonts() -> None:
    folder = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    for name, file in FONT_FILES.items():
        pdfmetrics.registerFont(TTFont(name, str(folder / file)))
    pdfmetrics.registerFontFamily(BODY_FONT, normal=BODY_FONT, bold=f"{BODY_FONT}-Bold",
                                  italic=f"{BODY_FONT}-Oblique", boldItalic=f"{BODY_FONT}-BoldOblique")


@functools.cache
def _styles() -> dict[str, ParagraphStyle]:
    _register_fonts()
    body = ParagraphStyle("body", fontName=BODY_FONT, fontSize=9, leading=12, spaceAfter=4)
    return {"body": body,
            "cell": ParagraphStyle("cell", parent=body, fontSize=7, leading=8.5, spaceAfter=0),
            "bullet": ParagraphStyle("bullet", parent=body, leftIndent=10, bulletIndent=2),
            "#": ParagraphStyle("h1", parent=body, fontSize=16, leading=20, spaceAfter=8),
            "##": ParagraphStyle("h2", parent=body, fontSize=12.5, leading=16, spaceBefore=10, spaceAfter=5),
            "###": ParagraphStyle("h3", parent=body, fontSize=10.5, leading=14, spaceBefore=6, spaceAfter=4)}


def markup(text: str) -> str:
    """reportlab paragraph markup for one line of report Markdown: undo its escapes, XML-escape the result (so no
    text can open a tag), then render `code`, **bold** and a whole-line _italic_."""
    plain = MARKDOWN_ESCAPE.sub(r"\1", text.replace("&lt;", "<"))
    safe = escape(plain)
    safe = CODE_SPAN.sub(rf'<font face="{MONO_FONT}">\1</font>', safe)
    safe = BOLD_SPAN.sub(r"<b>\1</b>", safe)
    if len(safe) > 2 and safe.startswith("_") and safe.endswith("_"):
        safe = f"<i>{safe[1:-1]}</i>"
    return safe


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in CELL_BOUNDARY.split(line.strip())[1:-1]]


def _table(lines: Sequence[str], width: float) -> Table:
    style = _styles()["cell"]
    rows = [_cells(line) for line in lines if not TABLE_SEPARATOR.match(line.strip())]
    data = [[Paragraph(f"<b>{markup(c)}</b>" if r == 0 else markup(c), style) for c in row]
            for r, row in enumerate(rows)]
    table = Table(data, colWidths=[width / len(rows[0])] * len(rows[0]), repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, GRID_COLOUR),
                               ("BACKGROUND", (0, 0), (-1, 0), HEADER_FILL), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    table.spaceAfter = 6
    return table


def _image(png: bytes, width: float, max_height: float) -> Image:
    pixel_width, pixel_height = ImageReader(io.BytesIO(png)).getSize()
    scale: float = min(width / pixel_width, max_height / pixel_height)
    return Image(io.BytesIO(png), width=pixel_width * scale, height=pixel_height * scale)


def _kind(line: str) -> str:
    stripped = line.strip()
    return "table" if stripped.startswith("|") else "bullets" if stripped.startswith("- ") else "text"


def _block(lines: Sequence[str], width: float) -> list[Flowable]:
    styles = _styles()
    kind = _kind(lines[0])
    if kind == "table":
        return [_table(lines, width)]
    if kind == "bullets":
        return [Paragraph(markup(line.strip()[2:]), styles["bullet"], bulletText="•") for line in lines]
    return [Paragraph("<br/>".join(markup(line) for line in lines), styles["body"])]


def _blocks(markdown: str) -> list[tuple[str, list[str]]]:
    """Split the Markdown into ("heading", [line]) and ("block", lines) runs: a block is consecutive non-blank
    lines of one kind (table, bullets or text)."""
    runs: list[tuple[str, list[str]]] = []
    for line in markdown.splitlines():
        if line.startswith("#"):
            runs.append(("heading", [line]))
        elif not line.strip():
            runs.append(("break", []))
        elif runs and runs[-1][0] == "block" and _kind(runs[-1][1][0]) == _kind(line):
            runs[-1][1].append(line)
        else:
            runs.append(("block", [line]))
    return [run for run in runs if run[0] != "break"]


def markdown_flowables(markdown: str, figures: Mapping[str, Sequence[bytes]], width: float = A4[0] - 2 * MARGIN,
                       height: float = A4[1] - 2 * MARGIN) -> list[Flowable]:
    """Flowables for the report Markdown; each section's figures close that section (before the next heading)."""
    max_height: float = height * FIGURE_MAX_HEIGHT_FRACTION
    out: list[Flowable] = []
    pending: Sequence[bytes] = ()
    for kind, lines in _blocks(markdown):
        if kind == "block":
            out.extend(_block(lines, width))
            continue
        out.extend(_image(png, width, max_height) for png in pending)
        level, _, title = lines[0].partition(" ")
        out.append(Paragraph(markup(title.strip()), _styles().get(level, _styles()["###"])))
        pending = figures.get(title.strip(), ())
    out.extend(_image(png, width, max_height) for png in pending)
    return out


def section_figures(inputs: ReportInputs) -> dict[str, list[bytes]]:
    """PNG figures keyed by the report section they illustrate; a figure that fails is left out with a warning."""
    builders: list[tuple[str, str, Callable[[], Optional[bytes]]]] = []
    sizing, aero, cad, fcs = inputs.sizing, inputs.aero, inputs.cad, inputs.fcs
    if sizing and cad and cad.viewer_model_path:
        builders.append((SECTION_SIZING, "three-view", lambda: report_figures.three_view(Path(cad.viewer_model_path))))
    if sizing:
        builders.append((SECTION_ENERGY, "power profile", lambda: report_figures.power_profile(sizing)))
    if sizing and aero:
        builders.append((SECTION_POLAR, "drag polar", lambda: report_figures.drag_polar(aero, sizing)))
    if fcs:
        builders.append((SECTION_SITL, "SITL attitude",
                         lambda: report_figures.step_responses(Path(fcs.sitl_report_path))))
    figures: dict[str, list[bytes]] = {}
    for section, label, build in builders:
        try:
            png = build()
        except Exception as exc:  # a figure must never cost the run its report: name it and carry on
            logger.warning("[Tool] report_pdf: %s figure left out of %s: %s", label, PDF_FILE, exc)
            continue
        if png is not None:
            figures[section] = [png]
    return figures


def _footer(canvas: Canvas, doc: SimpleDocTemplate) -> None:
    canvas.setFont(BODY_FONT, 7)
    canvas.drawRightString(A4[0] - MARGIN, MARGIN / 2, f"{doc.title} · page {doc.page}")


def write_pdf_report(inputs: ReportInputs, markdown_path: Path) -> Optional[Path]:
    """Write design_report.pdf next to the Markdown report. Returns None (and logs why) if it cannot be written:
    the Markdown report is the run's record, so a PDF failure never fails the run."""
    path: Path = inputs.out_dir / PDF_FILE
    try:
        _register_fonts()
        doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN,
                                bottomMargin=MARGIN, title=f"Design report: {inputs.requirements.mission_name}",
                                author="isr-forge")
        doc.build(markdown_flowables(markdown_path.read_text(), section_figures(inputs)),
                  onFirstPage=_footer, onLaterPages=_footer)
    except Exception:  # reportlab and file errors vary; the Markdown report already exists
        logger.exception("[Tool] report_pdf: %s could not be written (the Markdown report is complete)", PDF_FILE)
        path.unlink(missing_ok=True)
        return None
    logger.info("[Tool] report_pdf: wrote %s", path.name)
    return path
