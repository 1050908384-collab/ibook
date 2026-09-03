from pathlib import Path
from docx import Document
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "深圳大学-苏杭-面向大模型微调与推理部署的个性化实训多智能体协同系统-18980427165"
SRC = SUB / "面向大模型微调与推理部署的个性化实训多智能体协同系统设计实现方案（完善版）.docx"
OUT = SUB / "面向大模型微调与推理部署的个性化实训多智能体协同系统设计实现方案（目录修订版）.docx"


def remove(el):
    el.getparent().remove(el)


def main():
    doc = Document(SRC)
    paragraphs = list(doc.paragraphs)
    title_idx = next((i for i, p in enumerate(paragraphs) if p.text.strip() == "目录"), None)
    first_idx = next((i for i, p in enumerate(paragraphs) if p.style.name == "Heading 1"), None)
    title = paragraphs[title_idx] if title_idx is not None else None
    first_heading = paragraphs[first_idx] if first_idx is not None else None
    if title is None or first_heading is None:
        raise RuntimeError("目录或正文首章未找到")
    # Keep the field paragraph directly after the title, delete stale static TOC lines.
    field_p = paragraphs[title_idx + 1]._p if title_idx + 1 < len(paragraphs) else None
    for p in paragraphs[title_idx + 2 : first_idx]:
        remove(p._p)
    # TOC field with levels 1-3, without the hyperlink switch. It still displays page numbers.
    instrs = field_p.findall('.//' + qn('w:instrText')) if field_p is not None else []
    if instrs:
        instrs[0].text = ' TOC \\o "1-3" \\z \\u '
    settings = doc.settings.element
    update = settings.find(qn('w:updateFields'))
    if update is None:
        from docx.oxml import OxmlElement
        update = OxmlElement('w:updateFields'); settings.append(update)
    update.set(qn('w:val'), 'true')
    doc.save(OUT)
    print(OUT)


if __name__ == '__main__':
    main()
