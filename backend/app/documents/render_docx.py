"""Fixed python-docx template for study guides. Model text is data; layout is never model input."""

import uuid
import zipfile
from io import BytesIO
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from lxml import etree

from app.documents.schema import (
    Bullets,
    Callout,
    KeyTerms,
    Paragraph,
    PracticeQuestions,
    Steps,
    StudyGuide,
    WorkedExample,
    answer_key,
)

# Latin Modern Roman (the LaTeX font) as a renamed, bundled subset. Word users do not have it
# installed, so it is embedded in every file; see _embed_fonts.
FONT = "Tutor Serif"
FONT_DIR = Path(__file__).parent / "fonts"
EMBEDDED_FONTS = {
    "embedRegular": "TutorSerif-Regular.ttf",
    "embedBold": "TutorSerif-Bold.ttf",
    "embedItalic": "TutorSerif-Italic.ttf",
    "embedBoldItalic": "TutorSerif-BoldItalic.ttf",
}
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
ACCENT = RGBColor(0x2B, 0x5F, 0xB4)
MUTED = RGBColor(0x61, 0x6E, 0x7C)
CALLOUT_COLORS = {
    "Remember": RGBColor(0x2B, 0x5F, 0xB4),
    "Tip": RGBColor(0x1F, 0x7A, 0x4D),
    "Watch out": RGBColor(0xB4, 0x54, 0x0A),
}


def _shade(cell, hex_fill: str) -> None:
    properties = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:val"), "clear")
    shading.set(qn("w:color"), "auto")
    shading.set(qn("w:fill"), hex_fill)
    properties.append(shading)


def _left_bar(cell, hex_color: str) -> None:
    borders = OxmlElement("w:tcBorders")
    edge = OxmlElement("w:left")
    for key, value in (("val", "single"), ("sz", "24"), ("space", "0"), ("color", hex_color)):
        edge.set(qn(f"w:{key}"), value)
    borders.append(edge)
    cell._tc.get_or_add_tcPr().append(borders)


def _set_font(style, name: str) -> None:
    """Set every font slot and drop theme references, which Word prefers over explicit names.

    python-docx's default Title and Heading styles point at the theme's heading font, so
    setting only the font name would leave them in the theme font in Word.
    """
    style.font.name = name
    fonts = style.element.rPr.rFonts
    for slot in ("ascii", "hAnsi", "eastAsia", "cs"):
        fonts.set(qn(f"w:{slot}"), name)
    for slot in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
        fonts.attrib.pop(qn(f"w:{slot}"), None)


def _style_fonts(document) -> None:
    normal = document.styles["Normal"]
    _set_font(normal, FONT)
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(6)
    for name, size in (("Title", 24), ("Heading 1", 15)):
        style = document.styles[name]
        _set_font(style, FONT)
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0x1F, 0x29, 0x33) if name == "Title" else ACCENT
    document.styles["Heading 1"].paragraph_format.space_before = Pt(16)
    document.styles["Heading 1"].paragraph_format.space_after = Pt(6)
    document.styles["Heading 1"].paragraph_format.keep_with_next = True


def _add_page_number(section) -> None:
    paragraph = section.footer.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run("Page ")
    run.font.size, run.font.color.rgb = Pt(9), MUTED
    field = paragraph.add_run()
    field.font.size, field.font.color.rgb = Pt(9), MUTED
    for kind, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if kind:
            element = OxmlElement("w:fldChar")
            element.set(qn("w:fldCharType"), kind)
        else:
            element = OxmlElement("w:instrText")
            element.set(qn("xml:space"), "preserve")
            element.text = text
        field._r.append(element)


def _hanging(paragraph, label: str, text: str) -> None:
    """Manual numbering: Word's built-in numbered list continues across separate lists."""
    paragraph.paragraph_format.left_indent = Inches(0.3)
    paragraph.paragraph_format.first_line_indent = Inches(-0.3)
    paragraph.add_run(f"{label}\t{text}")
    paragraph.paragraph_format.tab_stops.add_tab_stop(Inches(0.3))


def _boxed_table(document, background: str, bar: str | None):
    table = document.add_table(rows=1, cols=1)
    cell = table.rows[0].cells[0]
    # tcPr children are schema-ordered: tcBorders must come before shd.
    if bar:
        _left_bar(cell, bar)
    _shade(cell, background)
    return cell


def _box_paragraph(cell, first: bool):
    """The first paragraph of a cell already exists; add the rest."""
    return cell.paragraphs[0] if first else cell.add_paragraph()


def _gap(document) -> None:
    document.add_paragraph().paragraph_format.space_after = Pt(2)


def _block(document, block, counter: list[int]) -> None:
    if isinstance(block, Paragraph):
        document.add_paragraph(block.text)
    elif isinstance(block, Bullets):
        for item in block.items:
            document.add_paragraph(item, style="List Bullet")
    elif isinstance(block, Steps):
        for number, item in enumerate(block.items, 1):
            _hanging(document.add_paragraph(), f"{number}.", item)
    elif isinstance(block, KeyTerms):
        table = document.add_table(rows=0, cols=2)
        table.style = "Table Grid"
        table.autofit = False
        for term in block.terms:
            left, right = table.add_row().cells
            left.paragraphs[0].add_run(term.term).bold = True
            right.paragraphs[0].add_run(term.definition)
        for column, width in zip(table.columns, (Inches(1.7), Inches(4.8)), strict=True):
            column.width = width
            for cell in column.cells:
                cell.width = width
        _gap(document)
    elif isinstance(block, WorkedExample):
        cell = _boxed_table(document, "EEF3FB", "2B5FB4")
        label = _box_paragraph(cell, first=True).add_run("WORKED EXAMPLE")
        label.bold, label.font.size, label.font.color.rgb = True, Pt(9), ACCENT
        cell.add_paragraph().add_run(block.problem).bold = True
        for number, step in enumerate(block.steps, 1):
            _hanging(cell.add_paragraph(), f"{number}.", step)
        if block.answer:
            cell.add_paragraph().add_run(f"Answer: {block.answer}").bold = True
        _gap(document)
    elif isinstance(block, PracticeQuestions):
        for question in block.questions:
            counter[0] += 1
            _hanging(document.add_paragraph(), f"{counter[0]}.", question.question)
    elif isinstance(block, Callout):
        cell = _boxed_table(document, "F5F7FA", str(CALLOUT_COLORS[block.kind]))
        label = _box_paragraph(cell, first=True).add_run(block.kind.upper())
        label.bold, label.font.size = True, Pt(9)
        label.font.color.rgb = CALLOUT_COLORS[block.kind]
        cell.add_paragraph(block.text)
        _gap(document)
    else:
        raise TypeError(f"Unsupported block: {type(block).__name__}")


def _obfuscate(font: bytes, key: uuid.UUID) -> bytes:
    """Word's embedded-font obfuscation: XOR the first 32 bytes with the reversed GUID bytes."""
    mask = key.bytes[::-1]
    head = bytes(byte ^ mask[index % 16] for index, byte in enumerate(font[:32]))
    return head + font[32:]


def _embed_fonts(data: bytes) -> bytes:
    """Embed the bundled font in the saved package so Word shows it without installation."""
    source = zipfile.ZipFile(BytesIO(data))
    parts = {name: source.read(name) for name in source.namelist()}

    table = etree.fromstring(parts["word/fontTable.xml"])
    font = etree.SubElement(table, f"{{{W_NS}}}font", {f"{{{W_NS}}}name": FONT})
    etree.SubElement(font, f"{{{W_NS}}}charset", {f"{{{W_NS}}}val": "00"})
    etree.SubElement(font, f"{{{W_NS}}}family", {f"{{{W_NS}}}val": "roman"})
    etree.SubElement(font, f"{{{W_NS}}}pitch", {f"{{{W_NS}}}val": "variable"})
    rels = etree.Element(
        "Relationships", xmlns="http://schemas.openxmlformats.org/package/2006/relationships"
    )
    for number, (tag, file) in enumerate(EMBEDDED_FONTS.items(), start=1):
        key = uuid.uuid4()
        rel_id = f"rIdFont{number}"
        parts[f"word/fonts/font{number}.odttf"] = _obfuscate((FONT_DIR / file).read_bytes(), key)
        etree.SubElement(
            font,
            f"{{{W_NS}}}{tag}",
            {f"{{{R_NS}}}id": rel_id, f"{{{W_NS}}}fontKey": "{" + str(key).upper() + "}"},
        )
        etree.SubElement(
            rels,
            "Relationship",
            Id=rel_id,
            Type=f"{R_NS}/font",
            Target=f"fonts/font{number}.odttf",
        )
    parts["word/fontTable.xml"] = etree.tostring(
        table, xml_declaration=True, encoding="UTF-8", standalone=True
    )
    parts["word/_rels/fontTable.xml.rels"] = etree.tostring(
        rels, xml_declaration=True, encoding="UTF-8", standalone=True
    )

    settings = etree.fromstring(parts["word/settings.xml"])
    flag = etree.Element(f"{{{W_NS}}}embedTrueTypeFonts")
    zoom = settings.find(f"{{{W_NS}}}zoom")
    if zoom is not None:
        zoom.addnext(flag)  # schema order: zoom comes before embedTrueTypeFonts
    else:
        settings.insert(0, flag)
    parts["word/settings.xml"] = etree.tostring(
        settings, xml_declaration=True, encoding="UTF-8", standalone=True
    )

    content_types = parts["[Content_Types].xml"].decode("utf-8")
    if 'Extension="odttf"' not in content_types:
        default = (
            '<Default Extension="odttf" '
            'ContentType="application/vnd.openxmlformats-officedocument.obfuscatedFont"/>'
        )
        content_types = content_types.replace("<Override", default + "<Override", 1)
    parts["[Content_Types].xml"] = content_types.encode("utf-8")

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as target:
        target.writestr("[Content_Types].xml", parts.pop("[Content_Types].xml"))
        for name, blob in parts.items():
            target.writestr(name, blob)
    return buffer.getvalue()


def render_docx(guide: StudyGuide) -> bytes:
    """Render a validated guide into DOCX bytes using the fixed template."""
    document = Document()
    # python-docx's template writes <w:zoom/> without the percent attribute the schema requires.
    for zoom in document.settings.element.findall(qn("w:zoom")):
        zoom.set(qn("w:percent"), "100")
    _style_fonts(document)
    section = document.sections[0]
    section.page_width, section.page_height = Inches(8.5), Inches(11)
    section.left_margin = section.right_margin = Inches(1)
    section.top_margin = section.bottom_margin = Inches(0.9)
    _add_page_number(section)

    document.core_properties.title = guide.title
    document.core_properties.author = "Tutor"

    document.add_paragraph(guide.title, style="Title")
    if guide.subtitle:
        run = document.add_paragraph().add_run(guide.subtitle)
        run.font.size, run.font.color.rgb = Pt(12), MUTED

    counter = [0]
    for part in guide.sections:
        document.add_heading(part.heading, level=1)
        for block in part.blocks:
            _block(document, block, counter)

    key = answer_key(guide)
    if key:
        document.add_heading("Answer key", level=1)
        for number, answer in key:
            _hanging(document.add_paragraph(), f"{number}.", answer)

    buffer = BytesIO()
    document.save(buffer)
    return _embed_fonts(buffer.getvalue())
