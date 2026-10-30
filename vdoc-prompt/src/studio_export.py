"""
Write a VDOC Studio document out as .docx and .pdf.

The two writers are kept side by side on purpose: the point of Studio is that
what the preview shows, Word and the PDF both reproduce -- same fonts, sizes,
weights, alignment, margins, line spacing and page border. When you change one
writer, change the other.
"""
import os
from html import escape

ALIGN_PDF = {"left": 0, "center": 1, "right": 2, "justify": 4}
WINDOWS_FONTS = r"C:\Windows\Fonts"

# Times is the default and reportlab ships it, so a report needs no font files.
# Anything else is looked for in the Windows font folder, then given up on.
TTF_CANDIDATES = {
    "times new roman": None,                       # built in
    "liberation serif": ["LiberationSerif-Regular.ttf"],
    "arial": ["arial.ttf", "Arial.ttf"],
    "calibri": ["calibri.ttf", "Calibri.ttf"],
    "georgia": ["georgia.ttf"],
    "verdana": ["verdana.ttf"],
}


# ── Word ─────────────────────────────────────────────────────────────────────

def to_docx(doc, path):
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Cm, Pt

    align_map = {"left": WD_ALIGN_PARAGRAPH.LEFT, "center": WD_ALIGN_PARAGRAPH.CENTER,
                 "right": WD_ALIGN_PARAGRAPH.RIGHT, "justify": WD_ALIGN_PARAGRAPH.JUSTIFY}
    page = doc.get("page", {})
    document = Document()

    section = document.sections[0]
    margin = Cm(float(page.get("margins_cm", 2.5)))
    section.left_margin = section.right_margin = margin
    section.top_margin = section.bottom_margin = margin
    _page_border(section, page.get("border", "none"))

    normal = document.styles["Normal"]
    normal.font.name = page.get("font") or "Times New Roman"
    normal.font.size = Pt(int(page.get("size", 12)))

    def write(run_spec, space_after=6):
        text = (run_spec or {}).get("text", "")
        paragraph = document.add_paragraph()
        paragraph.alignment = align_map.get(run_spec.get("align", "left"))
        fmt = paragraph.paragraph_format
        fmt.line_spacing = float(page.get("line_spacing", 1.5))
        fmt.space_after = Pt(space_after)
        piece = paragraph.add_run(text)
        piece.bold = bool(run_spec.get("bold"))
        piece.italic = bool(run_spec.get("italic"))
        piece.underline = bool(run_spec.get("underline"))
        piece.font.name = run_spec.get("font") or page.get("font") or "Times New Roman"
        piece.font.size = Pt(int(run_spec.get("size") or page.get("size", 12)))
        _color(piece, run_spec.get("color"))
        _highlight(piece, run_spec.get("highlight"))
        return paragraph

    if (doc.get("title") or {}).get("text"):
        write(doc["title"], space_after=14)
    for entry in doc.get("sections", []):
        if entry.get("heading", {}).get("text"):
            write(entry["heading"], space_after=4)
        for paragraph in entry.get("paragraphs", []):
            write(paragraph)

    document.save(path)
    return path


def _rgb(name):
    """A colour name or #rrggbb as a (r, g, b) triple, or None."""
    from src.studio_tools import COLORS

    if not name:
        return None
    text = str(name).strip().lower()
    hexed = COLORS.get(text, text if text.startswith("#") else None)
    if not hexed:
        return None
    hexed = hexed.lstrip("#")
    if len(hexed) == 3:
        hexed = "".join(c * 2 for c in hexed)
    try:
        return tuple(int(hexed[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return None


def _color(piece, name):
    from docx.shared import RGBColor

    rgb = _rgb(name)
    if rgb:
        piece.font.color.rgb = RGBColor(*rgb)


def _highlight(piece, name):
    """
    Word highlighting is an enum, not a colour value, so an arbitrary hex has
    to land on the nearest name it does offer; anything unmapped becomes
    yellow rather than silently disappearing.
    """
    from docx.enum.text import WD_COLOR_INDEX

    if not name:
        return
    lookup = {
        "yellow": WD_COLOR_INDEX.YELLOW, "green": WD_COLOR_INDEX.BRIGHT_GREEN,
        "cyan": WD_COLOR_INDEX.TURQUOISE, "magenta": WD_COLOR_INDEX.PINK,
        "pink": WD_COLOR_INDEX.PINK, "blue": WD_COLOR_INDEX.BLUE,
        "red": WD_COLOR_INDEX.RED, "darkblue": WD_COLOR_INDEX.DARK_BLUE,
        "darkred": WD_COLOR_INDEX.DARK_RED, "darkgreen": WD_COLOR_INDEX.DARK_YELLOW,
        "gray": WD_COLOR_INDEX.GRAY_25, "grey": WD_COLOR_INDEX.GRAY_25,
        "black": WD_COLOR_INDEX.BLACK, "white": WD_COLOR_INDEX.WHITE,
        "orange": WD_COLOR_INDEX.YELLOW,
    }
    piece.font.highlight_color = lookup.get(str(name).strip().lower(),
                                            WD_COLOR_INDEX.YELLOW)


def _page_border(section, style):
    """Word keeps page borders in sectPr/w:pgBorders; python-docx has no API."""
    if style in (None, "none"):
        return
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    borders = OxmlElement("w:pgBorders")
    borders.set(qn("w:offsetFrom"), "page")
    for side in ("top", "left", "bottom", "right"):
        edge = OxmlElement(f"w:{side}")
        edge.set(qn("w:val"), "double" if style == "double" else "single")
        edge.set(qn("w:sz"), "18" if style == "double" else "12")
        edge.set(qn("w:space"), "24")
        edge.set(qn("w:color"), "000000")
        borders.append(edge)
    section._sectPr.append(borders)


# ── PDF ──────────────────────────────────────────────────────────────────────

def _register(font_name):
    """Return a reportlab family prefix for this font, falling back to Times."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    wanted = (font_name or "Times New Roman").strip().lower()
    if wanted in ("times new roman", "times", ""):
        return "Times-Roman", "Times-Bold", "Times-Italic", "Times-BoldItalic"

    files = TTF_CANDIDATES.get(wanted)
    if files:
        for name in files:
            full = os.path.join(WINDOWS_FONTS, name)
            if os.path.exists(full):
                try:
                    tag = wanted.replace(" ", "")
                    if tag not in pdfmetrics.getRegisteredFontNames():
                        pdfmetrics.registerFont(TTFont(tag, full))
                    return tag, tag, tag, tag
                except Exception:                 # noqa: BLE001 - fall through
                    break
    return "Times-Roman", "Times-Bold", "Times-Italic", "Times-BoldItalic"


def to_pdf(doc, path):
    from reportlab.lib.colors import Color
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    page = doc.get("page", {})
    base, bold, italic, bold_italic = _register(page.get("font"))
    margin = float(page.get("margins_cm", 2.5)) * cm
    spacing = float(page.get("line_spacing", 1.5))
    border = page.get("border", "none")

    template = SimpleDocTemplate(
        path, pagesize=A4, leftMargin=margin, rightMargin=margin,
        topMargin=margin, bottomMargin=margin,
        title=(doc.get("title") or {}).get("text") or "VDOC Studio document")

    def decorate(canvas, _doc):
        if border in (None, "none"):
            return
        canvas.saveState()
        canvas.setLineWidth(1.2)
        inset = margin * 0.45
        width, height = A4
        canvas.rect(inset, inset, width - 2 * inset, height - 2 * inset)
        if border == "double":
            gap = 4
            canvas.setLineWidth(0.7)
            canvas.rect(inset + gap, inset + gap,
                        width - 2 * (inset + gap), height - 2 * (inset + gap))
        canvas.restoreState()

    flow = []

    def add(run_spec, gap=6):
        text = escape((run_spec or {}).get("text", ""))
        if not text:
            return
        size = int(run_spec.get("size") or page.get("size", 12))
        face = base
        if run_spec.get("bold") and run_spec.get("italic"):
            face = bold_italic
        elif run_spec.get("bold"):
            face = bold
        elif run_spec.get("italic"):
            face = italic
        if run_spec.get("underline"):
            text = f"<u>{text}</u>"
        extra = {}
        ink = _rgb(run_spec.get("color"))
        if ink:
            extra["textColor"] = Color(*[c / 255 for c in ink])
        wash = _rgb(run_spec.get("highlight"))
        if wash:
            # backColor paints the whole paragraph box; the preview draws the
            # same thing, so the two agree even though Word wraps the text.
            extra["backColor"] = Color(*[c / 255 for c in wash])
        style = ParagraphStyle(
            "run", fontName=face, fontSize=size, leading=size * spacing,
            alignment=ALIGN_PDF.get(run_spec.get("align", "left"), 0),
            spaceAfter=gap, **extra)
        flow.append(Paragraph(text, style))

    if (doc.get("title") or {}).get("text"):
        add(doc["title"], gap=14)
    for entry in doc.get("sections", []):
        add(entry.get("heading", {}), gap=4)
        for paragraph in entry.get("paragraphs", []):
            add(paragraph)
    if not flow:
        flow.append(Spacer(1, 1))

    template.build(flow, onFirstPage=decorate, onLaterPages=decorate)
    return path
