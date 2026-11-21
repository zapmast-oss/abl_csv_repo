"""Write both reading editions from the same newsroom paragraphs."""
from pathlib import Path
from xml.sax.saxutils import escape


def write_report(path: Path, title: str, sections, edition: str):
    from reportlab.lib.colors import HexColor
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from pypdf import PdfReader

    path.parent.mkdir(parents=True, exist_ok=True)
    path = path.with_suffix(".md")
    lines = ["# " + title, "", edition, ""]
    pdf = path.with_suffix(".pdf")
    doc = SimpleDocTemplate(str(pdf), pagesize=letter, topMargin=43, bottomMargin=47,
                            leftMargin=45, rightMargin=45, title=title, author="ABL Newsroom")
    if "BeatRegular" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("BeatRegular","C:/Windows/Fonts/arial.ttf"))
        pdfmetrics.registerFont(TTFont("BeatBold","C:/Windows/Fonts/arialbd.ttf"))
    heading = ParagraphStyle("section", fontName="BeatBold", fontSize=13, leading=16,
                             spaceBefore=8, spaceAfter=4, keepWithNext=True, textColor=HexColor("#174c56"))
    body = ParagraphStyle("body", fontName="BeatRegular", fontSize=10.5, leading=14, spaceAfter=6)
    display = ParagraphStyle("title", parent=heading, fontSize=21, leading=25, textColor=HexColor("#172e47"))
    story = [Paragraph(escape(title), display), Paragraph(escape(edition), body)]
    for label, paragraphs in sections:
        if not paragraphs:
            continue
        lines.extend(["## " + label, ""])
        story.append(Paragraph(escape(label), heading))
        for paragraph in paragraphs:
            lines.extend([paragraph, ""])
            story.append(Paragraph(escape(paragraph).replace("\n", "<br/>"), body))

    def footer(canvas, document):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(HexColor("#34515c"))
        canvas.drawString(45, 25, "THE BASEBALL OBSERVER  /  INTERNAL NEWSROOM")
        canvas.drawRightString(letter[0]-45, 25, str(document.page))

    path.write_text("\n".join(lines), encoding="utf-8")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    reader = PdfReader(pdf)
    if not reader.pages or not reader.pages[-1].extract_text().strip():
        raise ValueError(f"Incomplete PDF: {pdf}")
