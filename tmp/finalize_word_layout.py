from pathlib import Path
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "深圳大学-苏杭-面向大模型微调与推理部署的个性化实训多智能体协同系统-18980427165"
SRC = SUB / "面向大模型微调与推理部署的个性化实训多智能体协同系统设计实现方案（版式最终修正版）.docx"
OUT = SUB / "面向大模型微调与推理部署的个性化实训多智能体协同系统设计实现方案（最终提交排版版）.docx"
BLACK = RGBColor(0, 0, 0)


def clear_num(p):
    if p._p.pPr is not None:
        node = p._p.pPr.find(qn('w:numPr'))
        if node is not None:
            p._p.pPr.remove(node)


def set_run_font(run):
    run.font.name = 'Times New Roman'
    run._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), '宋体')
    run.font.color.rgb = BLACK


def set_para_text(p, text):
    # Replace content while retaining the paragraph style and layout.
    for child in list(p._p):
        if child.tag == qn('w:r'):
            p._p.remove(child)
    r = p.add_run(text)
    set_run_font(r)
    p.paragraph_format.left_indent = Cm(0.74)
    p.paragraph_format.first_line_indent = Cm(-0.74)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.25


def clear_toc_cache(doc):
    for p in doc.paragraphs:
        if p._p.xpath('.//w:instrText[contains(., "TOC")]'):
            # Remove cached display text between separate and end markers.
            runs = list(p._p.findall(qn('w:r')))
            for r in runs:
                if r.xpath('.//w:fldChar[@w:fldCharType="separate"]'):
                    continue
                if r.xpath('.//w:fldChar[@w:fldCharType="end"]'):
                    continue
                if r.xpath('.//w:instrText'):
                    continue
                if r.xpath('.//w:t'):
                    for t in list(r.findall(qn('w:t'))):
                        r.remove(t)
    settings = doc.settings.element
    upd = settings.find(qn('w:updateFields'))
    if upd is None:
        upd = OxmlElement('w:updateFields'); settings.append(upd)
    upd.set(qn('w:val'), 'true')


def main():
    doc = Document(SRC)
    active_kind = None
    count = 0
    for p in doc.paragraphs:
        style = p.style.name
        if style.startswith('Heading'):
            clear_num(p)
            active_kind = None
            continue
        if style == 'List Number':
            if active_kind != 'number':
                active_kind = 'number'; count = 1
            else:
                count += 1
            text = p.text.strip()
            # Strip any legacy marker and write a stable visible number.
            import re
            text = re.sub(r'^\d+[\.、]\s*', '', text)
            clear_num(p)
            p.style = doc.styles['Normal']
            set_para_text(p, f'{count}. {text}')
            continue
        if style == 'List Bullet':
            if active_kind != 'bullet':
                active_kind = 'bullet'
            clear_num(p)
            p.style = doc.styles['Normal']
            set_para_text(p, f'• {p.text.strip()}')
            continue
        active_kind = None
        if style.startswith('Heading'):
            clear_num(p)
    clear_toc_cache(doc)
    doc.save(OUT)
    print(OUT)


if __name__ == '__main__':
    main()
