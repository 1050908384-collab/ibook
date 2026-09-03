from pathlib import Path
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "深圳大学-苏杭-面向大模型微调与推理部署的个性化实训多智能体协同系统-18980427165"
SRC = SUB / "面向大模型微调与推理部署的个性化实训多智能体协同系统设计实现方案（版式修正版）.docx"
OUT = SUB / "面向大模型微调与推理部署的个性化实训多智能体协同系统设计实现方案（版式最终修正版）.docx"


def new_num_id(doc, bullet):
    numbering = doc.part.numbering_part.element
    aids = [int(x.get(qn('w:abstractNumId'))) for x in numbering.findall(qn('w:abstractNum'))]
    aid = max(aids or [0]) + 1
    a = OxmlElement('w:abstractNum'); a.set(qn('w:abstractNumId'), str(aid))
    m = OxmlElement('w:multiLevelType'); m.set(qn('w:val'), 'singleLevel'); a.append(m)
    l = OxmlElement('w:lvl'); l.set(qn('w:ilvl'), '0')
    s = OxmlElement('w:start'); s.set(qn('w:val'), '1'); l.append(s)
    f = OxmlElement('w:numFmt'); f.set(qn('w:val'), 'bullet' if bullet else 'decimal'); l.append(f)
    t = OxmlElement('w:lvlText'); t.set(qn('w:val'), '•' if bullet else '%1.'); l.append(t)
    j = OxmlElement('w:lvlJc'); j.set(qn('w:val'), 'left'); l.append(j)
    ppr = OxmlElement('w:pPr'); ind = OxmlElement('w:ind'); ind.set(qn('w:left'), '720'); ind.set(qn('w:hanging'), '360'); ppr.append(ind); l.append(ppr)
    a.append(l); numbering.append(a)
    nids = [int(x.get(qn('w:numId'))) for x in numbering.findall(qn('w:num'))]
    nid = max(nids or [0]) + 1
    n = OxmlElement('w:num'); n.set(qn('w:numId'), str(nid)); ref = OxmlElement('w:abstractNumId'); ref.set(qn('w:val'), str(aid)); n.append(ref); numbering.append(n)
    return nid


def set_num(p, nid):
    ppr = p._p.get_or_add_pPr()
    old = ppr.find(qn('w:numPr'))
    if old is not None: ppr.remove(old)
    np = OxmlElement('w:numPr')
    il = OxmlElement('w:ilvl'); il.set(qn('w:val'), '0'); np.append(il)
    ni = OxmlElement('w:numId'); ni.set(qn('w:val'), str(nid)); np.append(ni)
    ppr.append(np)


def clear_num(p):
    if p._p.pPr is not None:
        n = p._p.pPr.find(qn('w:numPr'))
        if n is not None: p._p.pPr.remove(n)


def main():
    doc = Document(SRC)
    active = {}
    for p in doc.paragraphs:
        style = p.style.name
        if style.startswith('Heading'):
            clear_num(p)
            active = {}
            continue
        if style not in ('List Number', 'List Bullet'):
            continue
        if style not in active:
            active[style] = new_num_id(doc, bullet=(style == 'List Bullet'))
        set_num(p, active[style])
    doc.save(OUT)
    print(OUT)


if __name__ == '__main__':
    main()
