from pathlib import Path
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "深圳大学-苏杭-面向大模型微调与推理部署的个性化实训多智能体协同系统-18980427165"
SRC = SUB / "面向大模型微调与推理部署的个性化实训多智能体协同系统设计实现方案（目录修订版）.docx"
OUT = SUB / "面向大模型微调与推理部署的个性化实训多智能体协同系统设计实现方案（版式修正版）.docx"


def new_num_id(doc, bullet=False):
    numbering = doc.part.numbering_part.element
    abstracts = [int(x.get(qn('w:abstractNumId'))) for x in numbering.findall(qn('w:abstractNum'))]
    aid = max(abstracts or [0]) + 1
    abstract = OxmlElement('w:abstractNum'); abstract.set(qn('w:abstractNumId'), str(aid))
    multi = OxmlElement('w:multiLevelType'); multi.set(qn('w:val'), 'singleLevel'); abstract.append(multi)
    lvl = OxmlElement('w:lvl'); lvl.set(qn('w:ilvl'), '0')
    start = OxmlElement('w:start'); start.set(qn('w:val'), '1'); lvl.append(start)
    fmt = OxmlElement('w:numFmt'); fmt.set(qn('w:val'), 'bullet' if bullet else 'decimal'); lvl.append(fmt)
    txt = OxmlElement('w:lvlText'); txt.set(qn('w:val'), '•' if bullet else '%1.'); lvl.append(txt)
    jc = OxmlElement('w:lvlJc'); jc.set(qn('w:val'), 'left'); lvl.append(jc)
    ppr = OxmlElement('w:pPr'); ind = OxmlElement('w:ind'); ind.set(qn('w:left'), '720'); ind.set(qn('w:hanging'), '360'); ppr.append(ind); lvl.append(ppr)
    abstract.append(lvl); numbering.append(abstract)
    nums = [int(x.get(qn('w:numId'))) for x in numbering.findall(qn('w:num'))]
    nid = max(nums or [0]) + 1
    num = OxmlElement('w:num'); num.set(qn('w:numId'), str(nid)); ref = OxmlElement('w:abstractNumId'); ref.set(qn('w:val'), str(aid)); num.append(ref)
    override = OxmlElement('w:lvlOverride'); override.set(qn('w:ilvl'), '0')
    start_override = OxmlElement('w:startOverride'); start_override.set(qn('w:val'), '1'); override.append(start_override)
    num.append(override); numbering.append(num)
    return nid


def set_num(p, num_id):
    ppr = p._p.get_or_add_pPr()
    old = ppr.find(qn('w:numPr'))
    if old is not None: ppr.remove(old)
    numpr = OxmlElement('w:numPr')
    ilvl = OxmlElement('w:ilvl'); ilvl.set(qn('w:val'), '0'); numpr.append(ilvl)
    nid = OxmlElement('w:numId'); nid.set(qn('w:val'), str(num_id)); numpr.append(nid)
    ppr.append(numpr)


def main():
    doc = Document(SRC)
    active_style = None
    active_num = None
    for p in doc.paragraphs:
        style = p.style.name
        if style in ('List Number', 'List Bullet'):
            if style != active_style:
                active_num = new_num_id(doc, bullet=(style == 'List Bullet'))
                active_style = style
            set_num(p, active_num)
        else:
            active_style = None
            active_num = None
    doc.save(OUT)
    print(OUT)


if __name__ == '__main__':
    main()
