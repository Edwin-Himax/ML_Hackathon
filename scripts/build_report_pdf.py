"""Typeset the generated consolidated report and its actual figures as a PDF."""
from pathlib import Path
import html
import re
import sys
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, Preformatted

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/full/technical_report.md'
OUTPUT = ROOT / 'output/pdf/PathSnap_Technical_Report.pdf'
WIDTH = A4[0] - 36 * mm


def inline(text):
    value = html.escape(text)
    value = re.sub(r'`([^`]+)`', r'<font name="Courier">\1</font>', value)
    value = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', value)
    value = re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)', r'<link color="#176c71" href="\2">\1</link>', value)
    return value


def main():
    if not SOURCE.exists():
        raise SystemExit('Generate the report first: python scripts/run_project.py report')
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='ReportTitle', fontName='Helvetica-Bold', fontSize=27, leading=32, textColor=colors.HexColor('#183b40'), spaceAfter=10))
    styles.add(ParagraphStyle(name='ReportSubtitle', fontSize=12, leading=16, textColor=colors.HexColor('#68817d'), spaceAfter=15))
    styles['BodyText'].fontSize = 9.3; styles['BodyText'].leading = 13.7; styles['BodyText'].spaceAfter = 7
    for name, size, leading in [('Heading1', 15, 19), ('Heading2', 11.5, 15)]:
        styles[name].fontSize = size; styles[name].leading = leading
        styles[name].textColor = colors.HexColor('#183b40'); styles[name].spaceBefore = 14
        styles[name].spaceAfter = 8; styles[name].keepWithNext = True
    styles.add(ParagraphStyle(name='CaptionCustom', fontSize=8, leading=11, textColor=colors.HexColor('#738580'), spaceBefore=5, spaceAfter=10))
    styles.add(ParagraphStyle(name='CodeCustom', fontName='Courier', fontSize=7.5, leading=10.5, spaceAfter=9))
    story = []
    lines = SOURCE.read_text(encoding='utf-8').splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1; continue
        if line.startswith('# '):
            story.append(Paragraph(inline(line[2:]), styles['ReportTitle']))
            story.append(Paragraph('Implementation and measured development results<br/>18 November 2026', styles['ReportSubtitle']))
        elif line.startswith('## '):
            story.append(Paragraph(inline(line[3:]), styles['Heading1']))
        elif line.startswith('### '):
            story.append(Paragraph(inline(line[4:]), styles['Heading2']))
        elif line.startswith('!['):
            match = re.fullmatch(r'!\[([^\]]*)\]\((.+)\)', line)
            if match:
                caption, path = match.groups()
                path = (SOURCE.parent / path).resolve()
                if not path.is_file():
                    raise FileNotFoundError(path)
                with PILImage.open(path) as img:
                    w, h = img.size
                ratio = min(WIDTH / w, 335 / h)
                picture = Image(str(path), width=w * ratio, height=h * ratio)
                picture.hAlign = 'CENTER'
                story.append(KeepTogether([Spacer(1, 5), picture, Paragraph(inline(caption), styles['CaptionCustom'])]))
        elif line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                row = [x.strip() for x in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[:\- ]+', x) for x in row):
                    rows.append(row)
                i += 1
            n = len(rows[0])
            font = 6.8 if n >= 9 else 7.7 if n >= 7 else 8.1
            cell = ParagraphStyle(name=f'Cell{n}', fontSize=font, leading=font + 3, spaceAfter=0)
            head = ParagraphStyle(name=f'Header{n}', parent=cell, fontName='Helvetica-Bold', textColor=colors.white)
            if n == 9:
                weights = [.16, .11, .16, .095, .095, .10, .095, .08, .085]
            elif n == 7:
                weights = [.14, .10, .11, .16, .16, .17, .16]
            elif n == 5:
                weights = [.20, .10, .20, .25, .25]
            elif n == 4:
                weights = [.24, .22, .27, .27]
            elif n == 3:
                weights = [.48, .26, .26]
            else:
                weights = [1 / n] * n
            total = sum(weights)
            t = Table([[Paragraph(inline(x), head if ri == 0 else cell) for x in row] for ri, row in enumerate(rows)], colWidths=[WIDTH * w / total for w in weights], repeatRows=1, hAlign='LEFT')
            t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#235860')), ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f0f5f3'), colors.white]), ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LEFTPADDING', (0, 0), (-1, -1), 5), ('RIGHTPADDING', (0, 0), (-1, -1), 5), ('TOPPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 6), ('LINEBELOW', (0, 1), (-1, -1), .25, colors.HexColor('#d8e4df'))]))
            story.extend([KeepTogether([t]) if len(rows) <= 4 else t, Spacer(1, 9)])
            continue
        elif line.startswith('```'):
            block = []; i += 1
            while i < len(lines) and not lines[i].startswith('```'):
                block.append(lines[i]); i += 1
            story.append(Preformatted('\n'.join(block), styles['CodeCustom']))
        elif line.startswith('- '):
            story.append(Paragraph('&#8226; ' + inline(line[2:]), styles['BodyText']))
        else:
            story.append(Paragraph(inline(line), styles['BodyText']))
        i += 1

    def decorate(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 8); canvas.setFillColor(colors.HexColor('#82938b'))
        if doc.page > 1:
            canvas.drawString(18 * mm, A4[1] - 11 * mm, 'PATHSNAP / TECHNICAL REPORT')
        canvas.drawString(18 * mm, 10 * mm, 'Student-development results / provisional pointwise alignment')
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, str(doc.page))
        canvas.restoreState()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=19 * mm, bottomMargin=21 * mm, title='PathSnap Technical Report', author='PathSnap team')
    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    print(OUTPUT)


if __name__ == '__main__':
    main()
