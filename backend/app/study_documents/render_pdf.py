"""Fixed ReportLab template for study guides. Model text is data; layout is never model input."""

from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.study_documents.schema import (
    Bullets,
    Callout,
    KeyTerms,
    PracticeQuestions,
    Steps,
    StudyGuide,
    WorkedExample,
    answer_key,
)
from app.study_documents.schema import (
    Paragraph as ParagraphBlock,
)

FONT_DIR = Path(__file__).with_name("fonts")
INK = colors.HexColor("#1f2933")
MUTED = colors.HexColor("#616e7c")
ACCENT = colors.HexColor("#2b5fb4")
TINT = colors.HexColor("#eef3fb")
RULE = colors.HexColor("#cbd2d9")
CALLOUT_COLORS = {
    "Remember": colors.HexColor("#2b5fb4"),
    "Tip": colors.HexColor("#1f7a4d"),
    "Watch out": colors.HexColor("#b4540a"),
}

_fonts_registered = False

# Body text is Latin Modern Roman, the font LaTeX documents use, with Latin Modern Math's Greek
# letters and math symbols merged in. The bundled files are renamed ("Tutor Serif"), as the Latin
# Modern license requires for modified copies. DejaVu Sans only covers the odd character left.
FALLBACKS = {"Body": ("Symbols",), "Body-Bold": ("Symbols-Bold",)}


def _register_fonts() -> None:
    """Register the bundled fonts once per process."""
    global _fonts_registered
    if _fonts_registered:
        return
    for name, file in [
        ("Body", "TutorSerif-Regular.ttf"),
        ("Body-Bold", "TutorSerif-Bold.ttf"),
        ("Body-Italic", "TutorSerif-Italic.ttf"),
        ("Body-BoldItalic", "TutorSerif-BoldItalic.ttf"),
    ]:
        pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / file)))
    pdfmetrics.registerFontFamily(
        "Body",
        normal="Body",
        bold="Body-Bold",
        italic="Body-Italic",
        boldItalic="Body-BoldItalic",
    )
    # The fallback only covers the odd character Tutor Serif lacks, so a missing file must not
    # stop a document from being made.
    for name, file in [("Symbols", "DejaVuSans.ttf"), ("Symbols-Bold", "DejaVuSans-Bold.ttf")]:
        path = FONT_DIR / file
        if path.is_file():
            pdfmetrics.registerFont(TTFont(name, str(path)))
    _fonts_registered = True


def _styles() -> dict[str, ParagraphStyle]:
    base = ParagraphStyle(
        "base", fontName="Body", fontSize=11, leading=15.5, textColor=INK, alignment=TA_LEFT
    )
    return {
        "body": base,
        "small": ParagraphStyle("small", parent=base, fontSize=10, leading=13.5),
        "title": ParagraphStyle(
            "title", parent=base, fontName="Body-Bold", fontSize=24, leading=29, spaceAfter=4
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=base, fontSize=12, leading=16, textColor=MUTED, spaceAfter=8
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base,
            fontName="Body-Bold",
            fontSize=15,
            leading=19,
            textColor=ACCENT,
            spaceBefore=16,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "label": ParagraphStyle(
            "label", parent=base, fontName="Body-Bold", fontSize=9.5, leading=12.5, textColor=ACCENT
        ),
    }


def _is_registered(font: str) -> bool:
    return font in pdfmetrics.getRegisteredFontNames()


def _has_glyph(font: str, char: str) -> bool:
    return ord(char) in pdfmetrics.getFont(font).face.charToGlyph


def _font_for(char: str, primary: str) -> str:
    """Pick the font that draws one character: the style's own, else the first fallback."""
    candidates = [primary, *(font for font in FALLBACKS.get(primary, ()) if _is_registered(font))]
    for font in candidates:
        if _has_glyph(font, char):
            return font
    return candidates[-1]


def _p(text: str, style: ParagraphStyle) -> Paragraph:
    """Escape model text so it can never be read as ReportLab paragraph markup.

    Runs of characters the style's font cannot draw are wrapped in a font tag for a fallback.
    """
    primary = style.fontName
    parts: list[str] = []
    current = primary
    for char in text:
        if char == "\n":
            parts.append("<br/>")
            continue
        font = _font_for(char, primary)
        if font != current:
            parts.append("</font>" if current != primary else "")
            parts.append(f'<font name="{font}">' if font != primary else "")
            current = font
        parts.append(escape(char))
    if current != primary:
        parts.append("</font>")
    return Paragraph("".join(parts), style)


def _bullets(items: list[str], styles: dict[str, ParagraphStyle], numbered: bool) -> ListFlowable:
    return ListFlowable(
        [ListItem(_p(item, styles["body"]), leftIndent=18) for item in items],
        bulletType="1" if numbered else "bullet",
        bulletFormat="%s." if numbered else None,
        start="1" if numbered else "•",
        bulletFontName="Body",
        bulletFontSize=11,
        leftIndent=18,
    )


def _box(rows: list, background, bar=None) -> Table:
    """A shaded, splittable box; one row per element so long ones can break across pages."""
    table = Table([[row] for row in rows], colWidths=[6.5 * inch])
    style = [
        ("BACKGROUND", (0, 0), (-1, -1), background),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, 0), 8),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 8),
    ]
    if bar is not None:
        style.append(("LINEBEFORE", (0, 0), (0, -1), 3, bar))
    table.setStyle(TableStyle(style))
    return table


def _block(block, styles: dict[str, ParagraphStyle], counter: list[int]) -> list:
    body, label = styles["body"], styles["label"]
    if isinstance(block, ParagraphBlock):
        return [_p(block.text, body), Spacer(1, 6)]
    if isinstance(block, Bullets):
        return [_bullets(block.items, styles, numbered=False), Spacer(1, 6)]
    if isinstance(block, Steps):
        return [_bullets(block.items, styles, numbered=True), Spacer(1, 6)]
    if isinstance(block, KeyTerms):
        rows = [
            [
                _p(t.term, ParagraphStyle("t", parent=body, fontName="Body-Bold")),
                _p(t.definition, body),
            ]
            for t in block.terms
        ]
        table = Table(rows, colWidths=[1.7 * inch, 4.8 * inch])
        table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LINEBELOW", (0, 0), (-1, -1), 0.5, RULE),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        return [table, Spacer(1, 8)]
    if isinstance(block, WorkedExample):
        rows = [
            _p("WORKED EXAMPLE", label),
            _p(block.problem, ParagraphStyle("prob", parent=body, fontName="Body-Bold")),
        ]
        rows += [_p(f"{i}.  {step}", body) for i, step in enumerate(block.steps, 1)]
        if block.answer:
            rows.append(
                _p(
                    f"Answer: {block.answer}",
                    ParagraphStyle("ans", parent=body, fontName="Body-Bold"),
                )
            )
        return [_box(rows, TINT, ACCENT), Spacer(1, 10)]
    if isinstance(block, PracticeQuestions):
        flow = []
        for question in block.questions:
            counter[0] += 1
            flow.append(_p(f"{counter[0]}.  {question.question}", body))
            flow.append(Spacer(1, 5))
        return flow
    if isinstance(block, Callout):
        rows = [
            _p(
                block.kind.upper(),
                ParagraphStyle("cl", parent=label, textColor=CALLOUT_COLORS[block.kind]),
            ),
            _p(block.text, body),
        ]
        return [_box(rows, colors.HexColor("#f5f7fa"), CALLOUT_COLORS[block.kind]), Spacer(1, 8)]
    raise TypeError(f"Unsupported block: {type(block).__name__}")


def _footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Body", 8)
    canvas.setFillColor(MUTED)
    canvas.drawRightString(LETTER[0] - 1 * inch, 0.6 * inch, f"Page {doc.page}")
    canvas.restoreState()


def render_pdf(guide: StudyGuide) -> bytes:
    """Render a validated guide into PDF bytes using the fixed template."""
    _register_fonts()
    styles = _styles()
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=LETTER,
        leftMargin=1 * inch,
        rightMargin=1 * inch,
        topMargin=0.9 * inch,
        bottomMargin=0.9 * inch,
        title=guide.title,
        author="Tutor",
        creator="Tutor",
    )
    story: list = [_p(guide.title, styles["title"])]
    if guide.subtitle:
        story.append(_p(guide.subtitle, styles["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=1, color=RULE, spaceAfter=4))

    counter = [0]
    for section in guide.sections:
        story.append(_p(section.heading, styles["h1"]))
        for block in section.blocks:
            story.extend(_block(block, styles, counter))

    key = answer_key(guide)
    if key:
        story.append(_p("Answer key", styles["h1"]))
        story.extend(_p(f"{number}.  {answer}", styles["body"]) for number, answer in key)

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buffer.getvalue()
