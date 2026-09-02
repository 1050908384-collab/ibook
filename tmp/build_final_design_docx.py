from pathlib import Path
import re

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
MATERIALS = ROOT / "深圳大学-苏杭-面向大模型微调与推理部署的个性化实训多智能体协同系统-18980427165" / "材料文档"
OUTPUT = ROOT / "深圳大学-苏杭-面向大模型微调与推理部署的个性化实训多智能体协同系统-18980427165" / "面向大模型微调与推理部署的个性化实训多智能体协同系统设计实现方案（修订版）.docx"
SOURCE = ROOT / "作品设计实现方案_v2.md"
FONT_BODY = "宋体"
FONT_HEADING = "黑体"
FONT_EN = "Times New Roman"
BLUE = "000000"
GRAY = "000000"


def font(run, name=FONT_BODY, size=12, bold=None, color=None):
    run.font.name = FONT_EN
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), FONT_EN)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), FONT_EN)
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def shade(cell, fill):
    props = cell._tc.get_or_add_tcPr()
    node = OxmlElement("w:shd")
    node.set(qn("w:fill"), fill)
    props.append(node)


def cell_margin(cell, top=80, start=120, bottom=80, end=120):
    props = cell._tc.get_or_add_tcPr()
    mar = OxmlElement("w:tcMar")
    for side, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = OxmlElement(f"w:{side}")
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")
        mar.append(node)
    props.append(mar)


def table_geometry(table, widths):
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_pr = table._tbl.tblPr
    tbl_w = OxmlElement("w:tblW")
    tbl_w.set(qn("w:w"), str(sum(widths)))
    tbl_w.set(qn("w:type"), "dxa")
    tbl_pr.append(tbl_w)
    grid = table._tbl.tblGrid
    for old in list(grid):
        grid.remove(old)
    for width in widths:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)
    for row in table.rows:
        for i, cell in enumerate(row.cells):
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = OxmlElement("w:tcW")
            tc_w.set(qn("w:w"), str(widths[i]))
            tc_w.set(qn("w:type"), "dxa")
            tc_pr.append(tc_w)
            cell_margin(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def inline(p, text, size=12, color=None):
    parts = re.split(r"(\*\*.*?\*\*|`.*?`)", text)
    for part in parts:
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            run = p.add_run(part[2:-2]); font(run, size=size, bold=True, color=color)
        elif part.startswith("`") and part.endswith("`"):
            run = p.add_run(part[1:-1]); font(run, name="Consolas", size=size - 1, color=BLUE)
        else:
            run = p.add_run(part); font(run, size=size, color=color)


def setup(doc):
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    sec.top_margin, sec.bottom_margin = Cm(2.54), Cm(2.54)
    sec.left_margin, sec.right_margin = Cm(2.54), Cm(2.54)
    sec.header_distance, sec.footer_distance = Cm(1.25), Cm(1.25)
    normal = doc.styles["Normal"]
    normal.font.name = FONT_EN
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), FONT_BODY)
    normal.font.size = Pt(12)
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.space_after = Pt(6)
    for name, sz, before, after in (("Heading 1", 16, 18, 10), ("Heading 2", 14, 14, 7), ("Heading 3", 12, 10, 5)):
        style = doc.styles[name]
        style.font.name = FONT_EN
        style._element.rPr.rFonts.set(qn("w:eastAsia"), FONT_HEADING)
        style.font.size = Pt(sz)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(BLUE)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
    header = sec.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.CENTER
    inline(header, "面向大模型微调与推理部署的个性化实训多智能体协同系统", 9, GRAY)
    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer.add_run("第 ")
    font(run, size=9, color=GRAY)
    fld = OxmlElement("w:fldSimple"); fld.set(qn("w:instr"), "PAGE")
    footer._p.append(fld)
    run = footer.add_run(" 页"); font(run, size=9, color=GRAY)


def cover(doc):
    p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(135); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    inline(p, "作品设计实现方案", 18, BLUE)
    p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(28); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    inline(p, "面向大模型微调与推理部署的", 24, "000000")
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    inline(p, "个性化实训多智能体协同系统", 24, "000000")
    p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(38); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    inline(p, "赛题编号：XH-202630", 12, GRAY)
    for label in ("申报作品名称：面向大模型微调与推理部署的个性化实训多智能体协同系统", "申报者姓名：苏杭  黄帆林  邓思雨  林原祎  巴一涵", "指导教师：刘柱钢", "申报单位：深圳大学", "编制日期：2026年9月"):
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after = Pt(7)
        inline(p, label, 12, GRAY)
    doc.add_page_break()


def toc(doc):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_before = Pt(20); p.paragraph_format.space_after = Pt(22)
    inline(p, "目录", 20, BLUE)
    for index in range(1, 10):
        matches = list(MATERIALS.glob(f"{index:02d}_*.md"))
        content = extract_body(matches[0])
        for line in content.splitlines():
            match = re.match(r"^(#{2,3})\s+(.*)$", line)
            if not match:
                continue
            level, text = len(match.group(1)), match.group(2).strip()
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0 if level == 2 else 0.7)
            p.paragraph_format.space_after = Pt(3 if level == 2 else 2)
            inline(p, text, 12 if level == 2 else 11, BLUE if level == 2 else "000000")
    p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(5)
    inline(p, "参考文献", 12, BLUE)
    doc.add_page_break()


def extract_body(path):
    text = path.read_text(encoding="utf-8")
    marker = "## 5. 方案书正式正文"
    return text.split(marker, 1)[1].strip() if marker in text else ""


def add_table(doc, rows):
    cols = max(len(r) for r in rows)
    table = doc.add_table(rows=len(rows), cols=cols)
    widths = [9360 // cols] * cols; widths[-1] += 9360 - sum(widths)
    table_geometry(table, widths)
    for i, row in enumerate(rows):
        for j in range(cols):
            cell = table.cell(i, j); cell.text = ""
            p = cell.paragraphs[0]; p.paragraph_format.space_after = Pt(0); p.paragraph_format.line_spacing = 1.15
            inline(p, row[j] if j < len(row) else "", 9.5)
            if i == 0:
                shade(cell, "E8EEF5")
                for run in p.runs: run.bold = True
    doc.add_paragraph().paragraph_format.space_after = Pt(1)


def add_markdown(doc, content):
    lines = content.splitlines(); i = 0; table = []
    while i < len(lines):
        line = lines[i]; s = line.strip()
        if s.startswith("|"):
            cells = [x.strip() for x in s.strip("|").split("|")]
            if not all(re.fullmatch(r"[:\- ]+", c or " ") for c in cells): table.append(cells)
            i += 1
            if i == len(lines) or not lines[i].strip().startswith("|"):
                if table: add_table(doc, table); table = []
            continue
        if table: add_table(doc, table); table = []
        if not s or s == "---": i += 1; continue
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            raw_level = len(m.group(1)); level = 1 if raw_level == 2 else 2 if raw_level == 3 else 3
            p = doc.add_paragraph(style=f"Heading {level}")
            inline(p, m.group(2), {1:16, 2:14, 3:12}[level]); i += 1; continue
        if s.startswith("-"):
            p = doc.add_paragraph(style="List Bullet"); p.paragraph_format.space_after = Pt(3); inline(p, s[1:].strip(), 11); i += 1; continue
        if re.match(r"^\d+\.\s+", s):
            p = doc.add_paragraph(style="List Number"); p.paragraph_format.space_after = Pt(3); inline(p, re.sub(r"^\d+\.\s+", "", s), 11); i += 1; continue
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY; p.paragraph_format.first_line_indent = Cm(0.74)
        inline(p, s, 12); i += 1


def references():
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("## 参考文献"))
    return "\n".join(lines[start:])


def main():
    doc = Document(); setup(doc); cover(doc); toc(doc)
    for index in range(1, 10):
        matches = list(MATERIALS.glob(f"{index:02d}_*.md"))
        if not matches: raise FileNotFoundError(index)
        add_markdown(doc, extract_body(matches[0]))
    add_markdown(doc, references())
    doc.core_properties.title = "面向大模型微调与推理部署的个性化实训多智能体协同系统"
    doc.core_properties.subject = "XH-202630 作品设计实现方案"
    doc.core_properties.author = "深圳大学-苏杭"
    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__": main()
