# -*- coding: utf-8 -*-
"""
PPT生成脚本 - 面向大模型微调与推理部署的个性化实训多智能体协同系统
根据PPT大纲.md生成20页主PPT + 附录页
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import os

# ========== 颜色主题定义 ==========
# 深蓝色系科技风
PRIMARY_COLOR = RGBColor(0x1A, 0x3A, 0x6C)      # 主色 - 深蓝
SECONDARY_COLOR = RGBColor(0x2E, 0x86, 0xAB)    # 辅色 - 蓝绿
ACCENT_COLOR = RGBColor(0xE8, 0x7C, 0x2A)       # 强调色 - 橙色
LIGHT_BG = RGBColor(0xF5, 0xF7, 0xFA)           # 浅色背景
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
DARK_TEXT = RGBColor(0x1E, 0x2A, 0x3A)          # 深色文字
GRAY_TEXT = RGBColor(0x5A, 0x6A, 0x7A)          # 灰色文字
SUCCESS_GREEN = RGBColor(0x27, 0xAE, 0x60)
WARNING_RED = RGBColor(0xE7, 0x4C, 0x3C)

# ========== 创建PPT ==========
prs = Presentation()
prs.slide_width = Inches(13.333)  # 16:9宽屏
prs.slide_height = Inches(7.5)

SLIDE_W = prs.slide_width
SLIDE_H = prs.slide_height


def add_background(slide, color=LIGHT_BG):
    """添加背景色"""
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_shape(slide, left, top, width, height, fill_color=None, line_color=None, shape_type=MSO_SHAPE.RECTANGLE):
    """添加形状"""
    shape = slide.shapes.add_shape(shape_type, left, top, width, height)
    if fill_color:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill_color
    else:
        shape.fill.background()
    if line_color:
        shape.line.color.rgb = line_color
    else:
        shape.line.fill.background()
    return shape


def add_textbox(slide, left, top, width, height, text, font_size=18, bold=False, color=DARK_TEXT, alignment=PP_ALIGN.LEFT, font_name="微软雅黑"):
    """添加文本框"""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.font.name = font_name
    p.alignment = alignment
    return txBox


def add_multi_text(slide, left, top, width, height, lines, default_size=16, default_color=DARK_TEXT, line_spacing=1.3, font_name="微软雅黑"):
    """添加多行文本
    lines: list of dict or str. dict: {text, size, bold, color, alignment, bullet}
    """
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True

    for i, line in enumerate(lines):
        if isinstance(line, str):
            line = {"text": line}

        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()

        prefix = ""
        if line.get("bullet"):
            prefix = "• "
        elif line.get("bullet_num"):
            prefix = f"{line['bullet_num']}. "

        p.text = prefix + line.get("text", "")
        p.font.size = Pt(line.get("size", default_size))
        p.font.bold = line.get("bold", False)
        p.font.color.rgb = line.get("color", default_color)
        p.font.name = font_name
        p.alignment = line.get("alignment", PP_ALIGN.LEFT)
        p.space_after = Pt(line.get("size", default_size) * (line_spacing - 1) + 4)

    return txBox


def add_page_header(slide, title, subtitle=None, page_num=None, total_pages=20):
    """添加页面顶部装饰和标题"""
    # 顶部装饰条
    add_shape(slide, Inches(0), Inches(0), SLIDE_W, Inches(0.08), fill_color=PRIMARY_COLOR)
    add_shape(slide, Inches(0), Inches(0.08), SLIDE_W, Inches(0.03), fill_color=ACCENT_COLOR)

    # 左侧竖条装饰
    add_shape(slide, Inches(0.5), Inches(0.55), Inches(0.08), Inches(0.55), fill_color=ACCENT_COLOR)

    # 主标题
    add_textbox(slide, Inches(0.75), Inches(0.45), Inches(10.5), Inches(0.7),
                title, font_size=30, bold=True, color=PRIMARY_COLOR)

    # 副标题
    if subtitle:
        add_textbox(slide, Inches(0.75), Inches(1.05), Inches(10), Inches(0.4),
                    subtitle, font_size=14, color=GRAY_TEXT)

    # 页码
    if page_num:
        add_textbox(slide, Inches(12.0), Inches(0.5), Inches(1), Inches(0.4),
                    f"{page_num}/{total_pages}", font_size=11, color=GRAY_TEXT, alignment=PP_ALIGN.RIGHT)

    # 底部装饰
    add_shape(slide, Inches(0), SLIDE_H - Inches(0.35), SLIDE_W, Inches(0.03), fill_color=SECONDARY_COLOR)
    add_textbox(slide, Inches(0.5), SLIDE_H - Inches(0.3), Inches(5), Inches(0.25),
                "AI Agent Trainer | 赛题编号：XH-202630", font_size=9, color=GRAY_TEXT)


def add_table(slide, left, top, width, height, data, col_widths=None, header_color=PRIMARY_COLOR, font_size=12):
    """添加表格
    data: list of list, first row is header
    """
    rows = len(data)
    cols = len(data[0])
    table_shape = slide.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table

    if col_widths:
        for i, w in enumerate(col_widths):
            table.columns[i].width = w

    for r in range(rows):
        for c in range(cols):
            cell = table.cell(r, c)
            cell.text = str(data[r][c])
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE

            for paragraph in cell.text_frame.paragraphs:
                paragraph.font.size = Pt(font_size)
                paragraph.font.name = "微软雅黑"
                if r == 0:
                    paragraph.font.bold = True
                    paragraph.font.color.rgb = WHITE
                    paragraph.alignment = PP_ALIGN.CENTER
                else:
                    paragraph.font.color.rgb = DARK_TEXT
                    paragraph.alignment = PP_ALIGN.LEFT if c > 0 else PP_ALIGN.CENTER

            # 背景色
            cell_fill = cell.fill
            cell_fill.solid()
            if r == 0:
                cell_fill.fore_color.rgb = header_color
            elif r % 2 == 0:
                cell_fill.fore_color.rgb = RGBColor(0xED, 0xF2, 0xF7)
            else:
                cell_fill.fore_color.rgb = WHITE

    return table_shape


def add_card(slide, left, top, width, height, title, content_lines, title_color=PRIMARY_COLOR, bg_color=WHITE, border_color=SECONDARY_COLOR):
    """添加卡片样式"""
    # 卡片背景
    card = add_shape(slide, left, top, width, height, fill_color=bg_color, line_color=border_color)
    card.shadow.inherit = False

    # 卡片顶部装饰
    add_shape(slide, left, top, width, Inches(0.06), fill_color=title_color)

    # 标题
    add_textbox(slide, left + Inches(0.2), top + Inches(0.15), width - Inches(0.4), Inches(0.5),
                title, font_size=16, bold=True, color=title_color)

    # 内容
    add_multi_text(slide, left + Inches(0.25), top + Inches(0.7), width - Inches(0.5), height - Inches(0.85),
                   content_lines, default_size=12, default_color=DARK_TEXT, line_spacing=1.4)

    return card


def add_conclusion_box(slide, text, top=None):
    """添加底部结论框"""
    if top is None:
        top = SLIDE_H - Inches(1.2)
    # 背景框
    box = add_shape(slide, Inches(0.7), top, Inches(11.9), Inches(0.85),
                    fill_color=RGBColor(0xFD, 0xF3, 0xE6), line_color=ACCENT_COLOR)
    # 左侧装饰
    add_shape(slide, Inches(0.7), top, Inches(0.08), Inches(0.85), fill_color=ACCENT_COLOR)
    # 引号图标
    add_textbox(slide, Inches(0.95), top + Inches(0.05), Inches(0.5), Inches(0.5),
                "❝", font_size=28, color=ACCENT_COLOR, bold=True)
    # 文字
    add_textbox(slide, Inches(1.4), top + Inches(0.18), Inches(11), Inches(0.6),
                text, font_size=14, color=DARK_TEXT, bold=False)


# ============================================================
# 第1页：封面
# ============================================================
def create_slide_1():
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # 空白布局
    add_background(slide, WHITE)

    # 顶部大面积深色装饰
    add_shape(slide, Inches(0), Inches(0), SLIDE_W, Inches(4.2), fill_color=PRIMARY_COLOR)
    # 装饰斜条
    add_shape(slide, Inches(8), Inches(0), Inches(6), Inches(4.2), fill_color=SECONDARY_COLOR)

    # 赛题编号
    add_textbox(slide, Inches(1), Inches(0.6), Inches(4), Inches(0.4),
                "赛题编号：XH-202630", font_size=14, color=RGBColor(0xBB, 0xD0, 0xEA))

    # 主标题
    add_textbox(slide, Inches(1), Inches(1.4), Inches(10), Inches(1.2),
                "面向大模型微调与推理部署的", font_size=36, bold=True, color=WHITE)
    add_textbox(slide, Inches(1), Inches(2.3), Inches(10), Inches(1.2),
                "个性化实训多智能体协同系统", font_size=36, bold=True, color=WHITE)

    # 副标题
    add_textbox(slide, Inches(1), Inches(3.3), Inches(8), Inches(0.6),
                "AI Agent Trainer", font_size=22, color=RGBColor(0x8E, 0xCF, 0xE8))

    # 关键词
    keywords = ["大模型微调", "推理部署", "个性化实训", "多智能体协同"]
    kw_start_x = Inches(1)
    for i, kw in enumerate(keywords):
        x = kw_start_x + Inches(i * 2.2)
        shape = add_shape(slide, x, Inches(4.8), Inches(2.0), Inches(0.5),
                          fill_color=RGBColor(0xED, 0xF4, 0xFA), line_color=SECONDARY_COLOR)
        add_textbox(slide, x, Inches(4.85), Inches(2.0), Inches(0.4),
                    kw, font_size=14, bold=True, color=SECONDARY_COLOR, alignment=PP_ALIGN.CENTER)

    # 中间分隔装饰线
    add_shape(slide, Inches(1), Inches(5.8), Inches(11.3), Inches(0.02), fill_color=SECONDARY_COLOR)

    # 学校、团队、指导教师
    add_multi_text(slide, Inches(1), Inches(6.0), Inches(11.3), Inches(1.2), [
        {"text": "学        校：_____________________", "size": 14, "color": GRAY_TEXT},
        {"text": "团 队 成 员：_____________________", "size": 14, "color": GRAY_TEXT},
        {"text": "指导教师：_____________________", "size": 14, "color": GRAY_TEXT},
    ], line_spacing=1.5)


# ============================================================
# 第2页：目录
# ============================================================
def create_slide_2():
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide, LIGHT_BG)
    add_page_header(slide, "目录", "Contents", page_num=2)

    chapters = [
        {
            "num": "01",
            "title": "问题与方案",
            "pages": "第 3 – 5 页",
            "items": ["项目定位", "赛题痛点", "作品总体方案"],
            "color": PRIMARY_COLOR
        },
        {
            "num": "02",
            "title": "系统设计与技术实现",
            "pages": "第 6 – 15 页",
            "items": ["四个真实运行时Agent", "系统总体技术架构", "完整业务闭环",
                      "知识图谱与动态阶段规划", "个性化实训资源生成", "代码生成与个性化适配",
                      "生成—审查—辩论—修正", "代码审核与修正机制",
                      "Orchestrator协同编排与状态管理", "学情反馈与动态决策"],
            "color": SECONDARY_COLOR
        },
        {
            "num": "03",
            "title": "效果与价值",
            "pages": "第 16 – 19 页",
            "items": ["测试体系", "核心量化结果", "消融实验与可靠性验证", "创新点与应用价值"],
            "color": ACCENT_COLOR
        },
        {
            "num": "04",
            "title": "总结",
            "pages": "第 20 页",
            "items": ["总结与展望"],
            "color": SUCCESS_GREEN
        }
    ]

    positions = [
        (Inches(0.7), Inches(1.6)),
        (Inches(7.0), Inches(1.6)),
        (Inches(0.7), Inches(4.5)),
        (Inches(7.0), Inches(4.5)),
    ]
    sizes = [(Inches(5.8), Inches(2.7)), (Inches(5.8), Inches(2.7)),
             (Inches(5.8), Inches(2.7)), (Inches(5.8), Inches(2.7))]

    for i, ch in enumerate(chapters):
        left, top = positions[i]
        w, h = sizes[i]

        # 卡片背景
        add_shape(slide, left, top, w, h, fill_color=WHITE, line_color=RGBColor(0xDD, 0xE4, 0xEC))
        # 编号
        add_textbox(slide, left + Inches(0.25), top + Inches(0.15), Inches(1.2), Inches(0.7),
                    ch["num"], font_size=36, bold=True, color=ch["color"])
        # 标题+页数
        add_textbox(slide, left + Inches(1.4), top + Inches(0.2), w - Inches(1.7), Inches(0.4),
                    ch["title"], font_size=18, bold=True, color=DARK_TEXT)
        add_textbox(slide, left + Inches(1.4), top + Inches(0.6), w - Inches(1.7), Inches(0.3),
                    ch["pages"], font_size=11, color=GRAY_TEXT)

        # 分隔线
        add_shape(slide, left + Inches(0.25), top + Inches(0.95), w - Inches(0.5), Inches(0.02),
                  fill_color=ch["color"])

        # 条目
        item_lines = [{"text": item, "size": 11, "color": GRAY_TEXT, "bullet": True} for item in ch["items"]]
        add_multi_text(slide, left + Inches(0.3), top + Inches(1.05), w - Inches(0.6), h - Inches(1.15),
                       item_lines, line_spacing=1.35)


# ============================================================
# 第3页：项目定位
# ============================================================
def create_slide_3():
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide, LIGHT_BG)
    add_page_header(slide, "从通用问答到工程实训导师", "第一章 · 问题与方案 | 项目定位", page_num=3)

    # 三栏对比
    col_w = Inches(3.8)
    col_h = Inches(4.5)
    start_y = Inches(1.6)
    gap = Inches(0.25)
    start_x = (SLIDE_W - col_w * 3 - gap * 2) / 2

    cols = [
        {
            "title": "传统课程",
            "subtitle": "标准化批量教学",
            "color": GRAY_TEXT,
            "bg": RGBColor(0xF0, 0xF2, 0xF5),
            "items": [
                "统一课程大纲",
                "固定教学节奏",
                "所有学生同一路径",
                "缺乏个性化适配",
                "理论与实践易脱节"
            ],
            "core": "一套课程 → 所有人"
        },
        {
            "title": "普通通用AI",
            "subtitle": "一次性问答交互",
            "color": SECONDARY_COLOR,
            "bg": RGBColor(0xED, 0xF5, 0xF9),
            "items": [
                "单次问题单次回答",
                "缺少持续上下文",
                "无学习路径规划",
                "内容可靠性难保证",
                "无学情评估反馈"
            ],
            "core": "一个问题 → 一次回答"
        },
        {
            "title": "本作品",
            "subtitle": "多智能体协同陪伴",
            "color": PRIMARY_COLOR,
            "bg": RGBColor(0xE8, 0xEE, 0xF8),
            "items": [
                "动态学情画像",
                "知识点级路径规划",
                "个性化资源生成",
                "专业审核与辩论修正",
                "反馈驱动动态决策"
            ],
            "core": "一个学习者 → 动态路径 → 多Agent持续协同"
        }
    ]

    for i, col in enumerate(cols):
        left = start_x + (col_w + gap) * i
        # 卡片
        card = add_shape(slide, left, start_y, col_w, col_h, fill_color=col["bg"], line_color=col["color"])
        # 顶部色块
        add_shape(slide, left, start_y, col_w, Inches(0.9), fill_color=col["color"])
        add_textbox(slide, left, start_y + Inches(0.12), col_w, Inches(0.45),
                    col["title"], font_size=20, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)
        add_textbox(slide, left, start_y + Inches(0.52), col_w, Inches(0.3),
                    col["subtitle"], font_size=11, color=WHITE, alignment=PP_ALIGN.CENTER)

        # 核心句
        core_box = add_shape(slide, left + Inches(0.2), start_y + Inches(1.1),
                             col_w - Inches(0.4), Inches(0.6),
                             fill_color=WHITE, line_color=col["color"])
        add_textbox(slide, left + Inches(0.3), start_y + Inches(1.2),
                    col_w - Inches(0.6), Inches(0.5),
                    col["core"], font_size=12, bold=True, color=col["color"], alignment=PP_ALIGN.CENTER)

        # 条目
        item_lines = [{"text": item, "size": 13, "color": DARK_TEXT, "bullet": True} for item in col["items"]]
        add_multi_text(slide, left + Inches(0.3), start_y + Inches(1.95),
                       col_w - Inches(0.6), col_h - Inches(2.1),
                       item_lines, line_spacing=1.6)

    # 底部结论
    add_conclusion_box(slide,
        "本作品解决的不是「让模型再生成一份教程」，而是「让系统持续陪伴不同学习者完成一条可靠的工程实训路径」。")


# ============================================================
# 第4页：赛题痛点
# ============================================================
def create_slide_4():
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide, LIGHT_BG)
    add_page_header(slide, "技能培训面临三重困境", "第一章 · 问题与方案 | 赛题痛点", page_num=4)

    pains = [
        {
            "icon": "📋",
            "title": "学习路径千人一面",
            "color": WARNING_RED,
            "items": [
                "零基础与有经验者路径相同，重复学习",
                "低显存用户无法直接使用高配置实训方案",
                "工程部署需求与理论学习内容错位"
            ]
        },
        {
            "icon": "⚠️",
            "title": "专业内容存在幻觉风险",
            "color": ACCENT_COLOR,
            "items": [
                "参数、API、训练步骤可能错误",
                "代码可能无法运行，缺乏知识来源",
                "测试题无约束生成，直接影响学情判断"
            ]
        },
        {
            "icon": "🔄",
            "title": "静态课程无法动态调整",
            "color": PRIMARY_COLOR,
            "items": [
                "答错后仍按原路线推进",
                "已掌握内容无法跳过",
                "用户提问不会反映到下一阶段规划"
            ]
        }
    ]

    col_w = Inches(3.9)
    col_h = Inches(4.3)
    gap = Inches(0.3)
    start_x = (SLIDE_W - col_w * 3 - gap * 2) / 2
    start_y = Inches(1.6)

    for i, pain in enumerate(pains):
        left = start_x + (col_w + gap) * i
        # 卡片
        add_shape(slide, left, start_y, col_w, col_h, fill_color=WHITE, line_color=pain["color"])
        # 顶部装饰
        add_shape(slide, left, start_y, col_w, Inches(0.1), fill_color=pain["color"])

        # 图标+标题
        add_textbox(slide, left + Inches(0.3), start_y + Inches(0.3), Inches(1), Inches(0.8),
                    pain["icon"], font_size=36, alignment=PP_ALIGN.CENTER)
        add_textbox(slide, left + Inches(1.2), start_y + Inches(0.4), col_w - Inches(1.5), Inches(0.6),
                    pain["title"], font_size=18, bold=True, color=pain["color"])

        # 分隔线
        add_shape(slide, left + Inches(0.3), start_y + Inches(1.3),
                  col_w - Inches(0.6), Inches(0.02), fill_color=pain["color"])

        # 条目
        lines = [{"text": it, "size": 13, "color": DARK_TEXT, "bullet": True} for it in pain["items"]]
        add_multi_text(slide, left + Inches(0.35), start_y + Inches(1.5),
                       col_w - Inches(0.7), col_h - Inches(1.7),
                       lines, line_spacing=1.7)

    # 底部结论
    add_conclusion_box(slide,
        "专业技能培训需要的不只是内容生成，而是「诊断—生成—审核—测试—决策」的完整闭环。")


# ============================================================
# 第5页：作品总体方案
# ============================================================
def create_slide_5():
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide, LIGHT_BG)
    add_page_header(slide, "知识驱动 · 多Agent协同 · 闭环实训",
                    "第一章 · 问题与方案 | 作品总体方案", page_num=5)

    # 横向流程图 - 8步
    steps = [
        "学情画像", "盲区定位", "知识点级阶段规划",
        "个性化资源生成", "专业审核", "学习与测试",
        "动态决策", "下一阶段循环"
    ]

    flow_y = Inches(1.8)
    box_w = Inches(1.35)
    box_h = Inches(0.7)
    total_w = box_w * 8 + Inches(0.15) * 7
    start_x = (SLIDE_W - total_w) / 2

    for i, step in enumerate(steps):
        x = start_x + (box_w + Inches(0.15)) * i
        is_end = (i == 7)
        color = PRIMARY_COLOR if i < 4 else (SECONDARY_COLOR if i < 6 else ACCENT_COLOR)
        shape = add_shape(slide, x, flow_y, box_w, box_h, fill_color=color,
                          shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
        add_textbox(slide, x, flow_y + Inches(0.2), box_w, Inches(0.4),
                    step, font_size=11, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)

        # 箭头
        if not is_end:
            arrow_x = x + box_w
            add_textbox(slide, arrow_x, flow_y + Inches(0.15), Inches(0.15), Inches(0.4),
                        "→", font_size=18, color=color, bold=True, alignment=PP_ALIGN.CENTER)

    # 循环箭头
    add_shape(slide, start_x + total_w / 2 - Inches(0.5), flow_y + box_h + Inches(0.05),
              Inches(1), Inches(0.35), fill_color=RGBColor(0xFD, 0xF3, 0xE6),
              shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
    add_textbox(slide, start_x + total_w / 2 - Inches(0.5),
                flow_y + box_h + Inches(0.08), Inches(1), Inches(0.3),
                "↺ 循环", font_size=11, bold=True, color=ACCENT_COLOR, alignment=PP_ALIGN.CENTER)

    # 四个关键词卡片
    keywords = [
        {"title": "个性化", "color": PRIMARY_COLOR,
         "desc": "适配基础、目标、时间、偏好、硬件"},
        {"title": "高保真", "color": SECONDARY_COLOR,
         "desc": "知识溯源、独立审核、辩论修正"},
        {"title": "动态化", "color": ACCENT_COLOR,
         "desc": "每阶段结束后根据反馈重新规划"},
        {"title": "工程化", "color": SUCCESS_GREEN,
         "desc": "状态持久化、三级容错、可部署运行"}
    ]

    kw_y = Inches(3.3)
    kw_w = Inches(2.75)
    kw_h = Inches(2.0)
    kw_gap = Inches(0.3)
    kw_start_x = (SLIDE_W - kw_w * 4 - kw_gap * 3) / 2

    for i, kw in enumerate(keywords):
        x = kw_start_x + (kw_w + kw_gap) * i
        add_card(slide, x, kw_y, kw_w, kw_h, kw["title"],
                 [{"text": kw["desc"], "size": 13, "color": DARK_TEXT}],
                 title_color=kw["color"])

    # 底部结论
    add_conclusion_box(slide,
        "系统形成符合赛题「分析—生成—校验—决策」要求的端到端闭环。",
        top=SLIDE_H - Inches(1.0))


# ============================================================
# 第6页：四个真实运行时Agent
# ============================================================
def create_slide_6():
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide, LIGHT_BG)
    add_page_header(slide, "四个Agent · 职责分明 · 协同闭环",
                    "第二章 · 系统设计与技术实现 | 运行时Agent", page_num=6)

    # Orchestrator顶部
    orch_w = Inches(10)
    orch_h = Inches(0.9)
    orch_x = (SLIDE_W - orch_w) / 2
    orch_y = Inches(1.5)
    add_shape(slide, orch_x, orch_y, orch_w, orch_h, fill_color=PRIMARY_COLOR,
              shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
    add_textbox(slide, orch_x, orch_y + Inches(0.1), orch_w, Inches(0.4),
                "Orchestrator 多智能体协同编排中枢", font_size=18, bold=True,
                color=WHITE, alignment=PP_ALIGN.CENTER)
    add_textbox(slide, orch_x, orch_y + Inches(0.5), orch_w, Inches(0.3),
                "状态管理 · 调用调度 · 并发控制 · 轮次控制 · 回退处理 · 持久化",
                font_size=11, color=RGBColor(0xBB, 0xD0, 0xEA), alignment=PP_ALIGN.CENTER)

    # 四个Agent卡片
    agents = [
        {
            "name": "学情诊断与阶段规划 Agent",
            "role": "画像 · 规划",
            "color": RGBColor(0x2C, 0x3E, 0x6B),
            "items": ["生成学习画像", "定位知识盲区", "知识点级阶段规划"],
            "icon": "🔍"
        },
        {
            "name": "个性化学习资源生成 Agent",
            "role": "生成 · 答疑",
            "color": SECONDARY_COLOR,
            "items": ["生成教程与代码", "分阶实操指南", "实时答疑支持"],
            "icon": "📝"
        },
        {
            "name": "专业知识与教学质量审核 Agent",
            "role": "审核 · 裁判",
            "color": ACCENT_COLOR,
            "items": ["技术正确性审核", "知识来源核验", "教学适配性审查"],
            "icon": "✅"
        },
        {
            "name": "学情评估与动态决策 Agent",
            "role": "评估 · 决策",
            "color": SUCCESS_GREEN,
            "items": ["阶段测试评估", "掌握状态更新", "下一阶段策略"],
            "icon": "📊"
        }
    ]

    agent_w = Inches(2.85)
    agent_h = Inches(3.5)
    agent_gap = Inches(0.2)
    agent_start_x = (SLIDE_W - agent_w * 4 - agent_gap * 3) / 2
    agent_y = orch_y + orch_h + Inches(0.5)

    for i, agent in enumerate(agents):
        x = agent_start_x + (agent_w + agent_gap) * i
        # 连接线
        line_x = x + agent_w / 2
        add_shape(slide, line_x - Inches(0.015), orch_y + orch_h,
                  Inches(0.03), Inches(0.5), fill_color=agent["color"])

        # 卡片
        add_shape(slide, x, agent_y, agent_w, agent_h, fill_color=WHITE, line_color=agent["color"])
        add_shape(slide, x, agent_y, agent_w, Inches(0.1), fill_color=agent["color"])

        # 图标区
        icon_bg_y = agent_y + Inches(0.35)
        add_shape(slide, x + Inches(0.95), icon_bg_y, Inches(0.95), Inches(0.95),
                  fill_color=RGBColor(0xF5, 0xF7, 0xFA), line_color=agent["color"],
                  shape_type=MSO_SHAPE.OVAL)
        add_textbox(slide, x + Inches(0.95), icon_bg_y + Inches(0.15),
                    Inches(0.95), Inches(0.7),
                    agent["icon"], font_size=30, alignment=PP_ALIGN.CENTER)

        # 角色标签
        tag_y = agent_y + Inches(1.5)
        add_shape(slide, x + Inches(agent_w/2 - Inches(0.6)), tag_y,
                  Inches(1.2), Inches(0.35), fill_color=agent["color"],
                  shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
        add_textbox(slide, x + Inches(agent_w/2 - Inches(0.6)),
                    tag_y + Inches(0.05), Inches(1.2), Inches(0.3),
                    agent["role"], font_size=11, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)

        # Agent名称
        add_textbox(slide, x + Inches(0.15), agent_y + Inches(2.0),
                    agent_w - Inches(0.3), Inches(0.7),
                    agent["name"], font_size=13, bold=True, color=DARK_TEXT, alignment=PP_ALIGN.CENTER)

        # 条目
        lines = [{"text": it, "size": 12, "color": GRAY_TEXT, "bullet": True} for it in agent["items"]]
        add_multi_text(slide, x + Inches(0.2), agent_y + Inches(2.75),
                       agent_w - Inches(0.4), Inches(0.8),
                       lines, line_spacing=1.4)

    # 底部说明
    add_conclusion_box(slide,
        "报名简介中「学情诊断、领域专家、审核裁判」是业务角色概括；系统工程实现细化为四个职责明确的运行时Agent。多Agent不是名称堆叠，而是围绕不同输入、不同工具和不同决策目标的职责分工。")


# ============================================================
# 第7页：系统总体技术架构
# ============================================================
def create_slide_7():
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide, LIGHT_BG)
    add_page_header(slide, "五层架构 · 完整系统",
                    "第二章 · 系统设计与技术实现 | 总体技术架构", page_num=7)

    # 五层架构 - 左侧
    layers = [
        {"name": "用户交互层", "items": "画像 ｜ 知识网络 ｜ 学习包 ｜ 测试 ｜ 报告",
         "color": RGBColor(0x4A, 0x6F, 0xA5), "height": Inches(0.75)},
        {"name": "业务服务层", "items": "React前端 ｜ FastAPI ｜ 会话服务",
         "color": RGBColor(0x3D, 0x7E, 0x9E), "height": Inches(0.75)},
        {"name": "多Agent编排层", "items": "Orchestrator ｜ 状态管理 ｜ 审核轮次控制",
         "color": SECONDARY_COLOR, "height": Inches(0.75)},
        {"name": "Skill / Tool层", "items": "语义召回 ｜ 图谱查询 ｜ 代码检查 ｜ 规则核验",
         "color": ACCENT_COLOR, "height": Inches(0.75)},
        {"name": "数据与模型层", "items": "SQLite + ChromaDB + 知识图谱 ｜ DeepSeek云端 + Ollama本地",
         "color": PRIMARY_COLOR, "height": Inches(0.9)}
    ]

    arch_x = Inches(0.7)
    arch_w = Inches(7.3)
    # 从顶层到底层渲染（最上层画在最上面）
    total_h = sum([l["height"].inches for l in layers]) * 914400 + Inches(0.1 * 4)  # spacing
    current_y = Inches(1.6)

    for layer in layers:
        add_shape(slide, arch_x, current_y, arch_w, layer["height"],
                  fill_color=layer["color"], shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
        # 层名
        add_textbox(slide, arch_x + Inches(0.25), current_y + Inches(0.1),
                    Inches(2.2), layer["height"] - Inches(0.2),
                    layer["name"], font_size=15, bold=True, color=WHITE, alignment=PP_ALIGN.LEFT)
        # 分隔竖线
        add_shape(slide, arch_x + Inches(2.5), current_y + Inches(0.15),
                  Inches(0.02), layer["height"] - Inches(0.3), fill_color=WHITE)
        # 组件
        add_textbox(slide, arch_x + Inches(2.7), current_y + Inches(0.1),
                    arch_w - Inches(2.9), layer["height"] - Inches(0.2),
                    layer["items"], font_size=13, color=WHITE)
        current_y += layer["height"] + Inches(0.08)

    # 右侧标注
    right_x = Inches(8.3)
    right_y = Inches(1.6)
    right_w = Inches(4.5)

    add_shape(slide, right_x, right_y, right_w, Inches(0.5),
              fill_color=PRIMARY_COLOR, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
    add_textbox(slide, right_x, right_y + Inches(0.1), right_w, Inches(0.35),
                "📊 关键组件与数据规模", font_size=14, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)

    anns = [
        ("📚 数据底座", "15个板块 / 73个知识点 / 982个内容块 / 233道测试题"),
        ("🤖 生成服务", "DeepSeek云端主模型 + Ollama本地回退"),
        ("💾 存储组合", "SQLite结构化状态 + ChromaDB语义检索 + NetworkX知识图谱"),
        ("🔧 部署模式", "云端增强 / 本地隐私 / 混合容错三种模式")
    ]

    ann_y = right_y + Inches(0.7)
    for title, desc in anns:
        add_shape(slide, right_x, ann_y, right_w, Inches(0.75),
                  fill_color=WHITE, line_color=SECONDARY_COLOR, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
        add_textbox(slide, right_x + Inches(0.15), ann_y + Inches(0.05),
                    right_w - Inches(0.3), Inches(0.3),
                    title, font_size=12, bold=True, color=PRIMARY_COLOR)
        add_textbox(slide, right_x + Inches(0.15), ann_y + Inches(0.35),
                    right_w - Inches(0.3), Inches(0.4),
                    desc, font_size=11, color=DARK_TEXT)
        ann_y += Inches(0.85)

    # 底部结论
    add_conclusion_box(slide,
        "作品是可运行的多智能体协同系统，而非单模型加若干Prompt的演示页面。",
        top=SLIDE_H - Inches(1.0))


# ============================================================
# 第8页：完整业务闭环
# ============================================================
def create_slide_8():
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide, LIGHT_BG)
    add_page_header(slide, "八步闭环 · 从画像到动态决策",
                    "第二章 · 系统设计与技术实现 | 业务闭环", page_num=8)

    steps = [
        ("1", "学习者输入", "基础、目标、时间、偏好、硬件", PRIMARY_COLOR),
        ("2", "学情诊断与阶段规划 Agent", "生成画像、定位盲区", RGBColor(0x2C, 0x3E, 0x6B)),
        ("3", "学情诊断与阶段规划 Agent", "依据知识图谱生成当前阶段", SECONDARY_COLOR),
        ("4", "个性化学习资源生成 Agent", "生成教程、代码、实操指南", RGBColor(0x2E, 0x7A, 0x9E)),
        ("5", "专业知识与教学质量审核 Agent", "审核、反馈、复核", ACCENT_COLOR),
        ("6", "学习者学习", "阅读教程、运行代码、随时提问", RGBColor(0xD4, 0x6A, 0x1A)),
        ("7", "阶段测试", "完成测试，提交答案", SUCCESS_GREEN),
        ("8", "学情评估与动态决策 Agent", "更新状态、决策下一阶段", RGBColor(0x1E, 0x84, 0x47))
    ]

    # 左列 1-4, 右列 5-8, 中间循环箭头
    left_x = Inches(0.7)
    right_x = Inches(7.0)
    col_w = Inches(5.6)
    step_h = Inches(1.05)
    start_y = Inches(1.55)

    for i in range(4):
        num, title, desc, color = steps[i]
        y = start_y + step_h * i + Inches(0.05 * i)

        # 编号圆
        add_shape(slide, left_x, y + Inches(0.1), Inches(0.7), Inches(0.7),
                  fill_color=color, shape_type=MSO_SHAPE.OVAL)
        add_textbox(slide, left_x, y + Inches(0.2), Inches(0.7), Inches(0.5),
                    num, font_size=22, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)

        # 内容框
        content_x = left_x + Inches(0.85)
        add_shape(slide, content_x, y, col_w - Inches(0.85), step_h,
                  fill_color=WHITE, line_color=color, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
        add_textbox(slide, content_x + Inches(0.2), y + Inches(0.12),
                    col_w - Inches(1.1), Inches(0.35),
                    title, font_size=13, bold=True, color=color)
        add_textbox(slide, content_x + Inches(0.2), y + Inches(0.5),
                    col_w - Inches(1.1), Inches(0.4),
                    desc, font_size=11, color=GRAY_TEXT)

    for i in range(4):
        num, title, desc, color = steps[i + 4]
        y = start_y + step_h * (3 - i) + Inches(0.05 * (3 - i))  # 反向排列5在上

        # 内容框
        add_shape(slide, right_x, y, col_w - Inches(0.85), step_h,
                  fill_color=WHITE, line_color=color, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
        add_textbox(slide, right_x + Inches(0.2), y + Inches(0.12),
                    col_w - Inches(1.1), Inches(0.35),
                    title, font_size=13, bold=True, color=color, alignment=PP_ALIGN.RIGHT)
        add_textbox(slide, right_x + Inches(0.2), y + Inches(0.5),
                    col_w - Inches(1.1), Inches(0.4),
                    desc, font_size=11, color=GRAY_TEXT, alignment=PP_ALIGN.RIGHT)

        # 编号圆
        circle_x = right_x + col_w - Inches(0.85)
        add_shape(slide, circle_x, y + Inches(0.1), Inches(0.7), Inches(0.7),
                  fill_color=color, shape_type=MSO_SHAPE.OVAL)
        add_textbox(slide, circle_x, y + Inches(0.2), Inches(0.7), Inches(0.5),
                    num, font_size=22, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)

    # 中间连接区域
    mid_x = Inches(6.35)
    # 上方箭头 (4→5)
    add_shape(slide, mid_x, start_y - Inches(0.1), Inches(0.6), Inches(0.6),
              fill_color=ACCENT_COLOR, shape_type=MSO_SHAPE.OVAL)
    add_textbox(slide, mid_x, start_y, Inches(0.6), Inches(0.5),
                "↻", font_size=22, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)

    # 下方箭头 (8→1, 回到步骤3循环)
    loop_y = start_y + step_h * 4 + Inches(0.15)
    add_shape(slide, mid_x - Inches(0.2), loop_y, Inches(1.0), Inches(0.5),
              fill_color=RGBColor(0xFD, 0xF3, 0xE6), line_color=ACCENT_COLOR,
              shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
    add_textbox(slide, mid_x - Inches(0.2), loop_y + Inches(0.08), Inches(1.0), Inches(0.4),
                "回到步骤3 ↺", font_size=11, bold=True, color=ACCENT_COLOR, alignment=PP_ALIGN.CENTER)

    # 底部结论
    add_conclusion_box(slide,
        "闭环覆盖赛题全部要求：学情画像构建 → 多Agent协同调度 → 个性化生成 → 交互反馈 → 动态决策更新。",
        top=SLIDE_H - Inches(1.0))


# ============================================================
# 第9页：知识图谱与动态阶段规划
# ============================================================
def create_slide_9():
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide, LIGHT_BG)
    add_page_header(slide, "15板块 · 73知识点 · 状态驱动选路",
                    "第二章 · 系统设计与技术实现 | 知识图谱与阶段规划", page_num=9)

    # 左侧数据底座
    left_x = Inches(0.7)
    left_w = Inches(5.0)

    add_shape(slide, left_x, Inches(1.55), left_w, Inches(0.55),
              fill_color=PRIMARY_COLOR, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
    add_textbox(slide, left_x, Inches(1.62), left_w, Inches(0.4),
                "📚 数据底座", font_size=16, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)

    data_items = [
        ("15", "个知识板块", "Linux/PyTorch/NLP/Transformer/LLM/HF/LoRA/QLoRA/vLLM..."),
        ("73", "个原子知识点", "可计算的最小学习单元，支持掌握状态追踪"),
        ("3", "类知识关系", "hard_prerequisite硬前置 / recommended推荐前置 / related关联")
    ]

    y = Inches(2.25)
    for num, label, desc in data_items:
        add_shape(slide, left_x, y, left_w, Inches(1.05),
                  fill_color=WHITE, line_color=SECONDARY_COLOR, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
        # 数字
        add_textbox(slide, left_x + Inches(0.15), y + Inches(0.1), Inches(1.3), Inches(0.8),
                    num, font_size=36, bold=True, color=PRIMARY_COLOR, alignment=PP_ALIGN.CENTER)
        # 标签
        add_textbox(slide, left_x + Inches(1.5), y + Inches(0.15), left_w - Inches(1.7), Inches(0.35),
                    label, font_size=14, bold=True, color=SECONDARY_COLOR)
        # 描述
        add_textbox(slide, left_x + Inches(1.5), y + Inches(0.5), left_w - Inches(1.7), Inches(0.55),
                    desc, font_size=11, color=GRAY_TEXT)
        y += Inches(1.15)

    # 机制说明
    add_shape(slide, left_x, y, left_w, Inches(2.2),
              fill_color=RGBColor(0xF8, 0xF0, 0xE3), line_color=ACCENT_COLOR, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
    add_textbox(slide, left_x + Inches(0.2), y + Inches(0.1), left_w - Inches(0.4), Inches(0.35),
                "⚙️ 动态选路机制", font_size=13, bold=True, color=ACCENT_COLOR)
    mech_items = [
        "对目标知识点反向计算硬前置闭包",
        "已掌握节点跳过，未满足硬前置的节点锁定",
        "当前阶段只选择已解锁且符合时间预算的知识点",
        "每阶段结束后根据最新掌握状态重新规划"
    ]
    lines = [{"text": it, "size": 11, "color": DARK_TEXT, "bullet": True} for it in mech_items]
    add_multi_text(slide, left_x + Inches(0.25), y + Inches(0.45), left_w - Inches(0.5), Inches(1.7),
                   lines, line_spacing=1.6)

    # 右侧DAG示例
    right_x = Inches(6.1)
    right_w = Inches(6.5)

    add_shape(slide, right_x, Inches(1.55), right_w, Inches(0.55),
              fill_color=SECONDARY_COLOR, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
    add_textbox(slide, right_x, Inches(1.62), right_w, Inches(0.4),
                "🔗 DAG知识路径示例（简化）", font_size=16, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)

    dag_nodes = [
        ("PyTorch训练循环", "", SUCCESS_GREEN),
        ("Transformer与LLM基础", "hard_prerequisite", PRIMARY_COLOR),
        ("HuggingFace工具链", "hard_prerequisite", SECONDARY_COLOR),
        ("LoRA / QLoRA微调", "hard_prerequisite", ACCENT_COLOR),
        ("权重合并与vLLM部署", "related / 部署前置", WARNING_RED)
    ]

    node_w = Inches(4.8)
    node_h = Inches(0.65)
    node_x = right_x + (right_w - node_w) / 2
    dag_y = Inches(2.35)
    node_gap = Inches(0.45)

    for i, (name, rel, color) in enumerate(dag_nodes):
        ny = dag_y + (node_h + node_gap) * i
        # 节点
        add_shape(slide, node_x, ny, node_w, node_h,
                  fill_color=WHITE, line_color=color, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
        add_shape(slide, node_x, ny, Inches(0.12), node_h, fill_color=color)
        add_textbox(slide, node_x + Inches(0.3), ny + Inches(0.13),
                    node_w - Inches(0.5), node_h - Inches(0.2),
                    name, font_size=14, bold=True, color=DARK_TEXT, alignment=PP_ALIGN.CENTER)

        # 箭头和关系
        if rel:
            arrow_y = ny + node_h + Inches(0.05)
            # 箭头线
            add_shape(slide, node_x + node_w / 2 - Inches(0.015), arrow_y,
                      Inches(0.03), node_gap - Inches(0.08), fill_color=GRAY_TEXT)
            # 关系标签
            add_shape(slide, node_x + node_w / 2 - Inches(1.1),
                      arrow_y + (node_gap - Inches(0.08)) / 2 - Inches(0.12),
                      Inches(2.2), Inches(0.25),
                      fill_color=LIGHT_BG)
            add_textbox(slide, node_x + node_w / 2 - Inches(1.1),
                        arrow_y + (node_gap - Inches(0.08)) / 2 - Inches(0.12),
                        Inches(2.2), Inches(0.25),
                        f"↓ {rel}", font_size=10, color=GRAY_TEXT, alignment=PP_ALIGN.CENTER)

    # 底部结论
    add_conclusion_box(slide,
        "系统不是按板块序号机械排课，而是根据知识点状态和先修关系动态选路。",
        top=SLIDE_H - Inches(1.0))


# ============================================================
# 第10页：个性化实训资源生成
# ============================================================
def create_slide_10():
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide, LIGHT_BG)
    add_page_header(slide, "一类目标 · 三种画像 · 三套资源",
                    "第二章 · 系统设计与技术实现 | 个性化资源生成", page_num=10)

    # 对比表格
    data = [
        ["学习者类型", "画像特征", "个性化输出"],
        ["零基础学习者", "无深度学习基础、每天1小时",
         "增加概念解释、拆分环境准备和基础代码步骤"],
        ["低显存实操型", "RTX 4060 8GB、偏好动手",
         "4-bit量化、低显存参数、小Batch方案"],
        ["工程部署型", "有训练经验、关注生产",
         "权重合并、吞吐测试、vLLM服务化部署"]
    ]
    col_ws = [Inches(2.0), Inches(3.8), Inches(6.8)]
    table_left = Inches(0.75)
    table_top = Inches(1.55)
    table_w = sum(col_ws)
    table_h = Inches(2.8)
    add_table(slide, table_left, table_top, table_w, table_h, data,
              col_widths=col_ws, font_size=12)

    # 可生成资源类型
    res_y = Inches(4.6)
    add_shape(slide, Inches(0.75), res_y, Inches(11.8), Inches(0.5),
              fill_color=SECONDARY_COLOR, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
    add_textbox(slide, Inches(0.75), res_y + Inches(0.08), Inches(11.8), Inches(0.35),
                "🎯 可生成的六类实训资源", font_size=14, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)

    resources = [
        ("📖", "定制化教程"),
        ("💻", "实训代码"),
        ("📋", "分阶实操指南"),
        ("❓", "阶段测试题"),
        ("💬", "实时答疑"),
        ("📑", "阶段总结与建议")
    ]
    res_item_w = Inches(1.85)
    res_item_h = Inches(0.8)
    res_gap = Inches(0.15)
    res_start_x = (SLIDE_W - res_item_w * 6 - res_gap * 5) / 2
    res_item_y = res_y + Inches(0.65)

    for icon, name in resources:
        add_shape(slide, res_start_x, res_item_y, res_item_w, res_item_h,
                  fill_color=WHITE, line_color=SECONDARY_COLOR, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
        add_textbox(slide, res_start_x + Inches(0.05), res_item_y + Inches(0.18),
                    Inches(0.6), Inches(0.5),
                    icon, font_size=22, alignment=PP_ALIGN.CENTER)
        add_textbox(slide, res_start_x + Inches(0.65), res_item_y + Inches(0.25),
                    res_item_w - Inches(0.7), Inches(0.4),
                    name, font_size=12, bold=True, color=PRIMARY_COLOR)
        res_start_x += res_item_w + res_gap

    # 底部注释
    add_conclusion_box(slide,
        "教程和代码采用受知识库约束的个性化生成；测试题优先从233道专业题库中按知识点和难度动态抽取。系统动态组织的是一整套与学习者状态匹配的实训资源，而非简单改变教材措辞。")


# ============================================================
# 第11页：代码生成与个性化适配
# ============================================================
def create_slide_11():
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide, LIGHT_BG)
    add_page_header(slide, "代码级个性化 · 实训即执行",
                    "第二章 · 系统设计与技术实现 | 代码生成与适配", page_num=11)

    # 顶部：个性化维度
    dims = [
        ("🖥️", "硬件适配", "根据显存自动调整\nbatch size、量化方式", PRIMARY_COLOR),
        ("🔧", "框架适配", "根据熟悉度选择\nHF Trainer / 自定义循环", SECONDARY_COLOR),
        ("🎯", "任务适配", "根据目标调整\n数据加载与格式", ACCENT_COLOR),
        ("📈", "难度适配", "零基础提供完整语句\n有经验者给核心部分", SUCCESS_GREEN)
    ]
    dim_w = Inches(2.85)
    dim_h = Inches(1.3)
    dim_gap = Inches(0.2)
    dim_start_x = (SLIDE_W - dim_w * 4 - dim_gap * 3) / 2
    dim_y = Inches(1.55)

    for icon, title, desc, color in dims:
        add_shape(slide, dim_start_x, dim_y, dim_w, dim_h,
                  fill_color=WHITE, line_color=color, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
        add_shape(slide, dim_start_x, dim_y, Inches(0.08), dim_h, fill_color=color)
        add_textbox(slide, dim_start_x + Inches(0.2), dim_y + Inches(0.1),
                    Inches(0.7), Inches(0.7),
                    icon, font_size=26, alignment=PP_ALIGN.CENTER)
        add_textbox(slide, dim_start_x + Inches(0.95), dim_y + Inches(0.1),
                    dim_w - Inches(1.1), Inches(0.4),
                    title, font_size=14, bold=True, color=color)
        add_textbox(slide, dim_start_x + Inches(0.95), dim_y + Inches(0.5),
                    dim_w - Inches(1.1), Inches(0.75),
                    desc, font_size=10, color=GRAY_TEXT)
        dim_start_x += dim_w + dim_gap

    # 中部：代码对比
    code_y = Inches(3.05)
    code_w = Inches(5.8)
    code_h = Inches(2.3)
    code_gap = Inches(0.4)
    code_left = (SLIDE_W - code_w * 2 - code_gap) / 2

    # 通用代码
    add_shape(slide, code_left, code_y, code_w, code_h,
              fill_color=WHITE, line_color=WARNING_RED, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
    add_shape(slide, code_left, code_y, code_w, Inches(0.45), fill_color=WARNING_RED)
    add_textbox(slide, code_left + Inches(0.15), code_y + Inches(0.08),
                code_w - Inches(0.3), Inches(0.3),
                "❌ 通用代码（不可直接运行）", font_size=12, bold=True, color=WHITE)

    bad_code_lines = [
        {"text": "model = AutoModelForCausalLM.from_pretrained(\"llama\")", "size": 11, "color": DARK_TEXT},
        {"text": "", "size": 6},
        {"text": "# 使用 deepspeed 但缺少配置", "size": 10, "color": GRAY_TEXT},
        {"text": "deepspeed_config = {...}", "size": 11, "color": DARK_TEXT},
    ]
    add_multi_text(slide, code_left + Inches(0.2), code_y + Inches(0.55),
                   code_w - Inches(0.4), code_h - Inches(0.65),
                   bad_code_lines, line_spacing=1.5)

    # 个性化代码
    right_code_x = code_left + code_w + code_gap
    add_shape(slide, right_code_x, code_y, code_w, code_h,
              fill_color=WHITE, line_color=SUCCESS_GREEN, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
    add_shape(slide, right_code_x, code_y, code_w, Inches(0.45), fill_color=SUCCESS_GREEN)
    add_textbox(slide, right_code_x + Inches(0.15), code_y + Inches(0.08),
                code_w - Inches(0.3), Inches(0.3),
                "✅ 个性化适配代码（可直接运行）", font_size=12, bold=True, color=WHITE)

    good_code_lines = [
        {"text": "model = AutoModelForCausalLM.from_pretrained(", "size": 11, "color": DARK_TEXT},
        {"text": "    \"llama\", load_in_4bit=True,", "size": 11, "color": PRIMARY_COLOR, "bold": True},
        {"text": "    device_map=\"auto\")", "size": 11, "color": DARK_TEXT},
        {"text": "", "size": 6},
        {"text": "# 显存优化：gradient_checkpointing", "size": 10, "color": GRAY_TEXT},
        {"text": "model.gradient_checkpointing_enable()", "size": 11, "color": PRIMARY_COLOR, "bold": True},
    ]
    add_multi_text(slide, right_code_x + Inches(0.2), code_y + Inches(0.55),
                   code_w - Inches(0.4), code_h - Inches(0.65),
                   good_code_lines, line_spacing=1.4)

    # 能力清单
    cap_y = code_y + code_h + Inches(0.15)
    caps = ["✅ 完整可运行代码（依赖完整、参数明确）",
            "✅ 硬件感知的训练/推理配置（batch_size、量化、多卡策略）",
            "✅ 错误诊断代码块（常见错误及对应修复）",
            "✅ 一键启动脚本（含环境检查）"]
    cap_lines = []
    for idx in range(0, len(caps), 2):
        merged = caps[idx]
        if idx + 1 < len(caps):
            merged = merged + "    " + caps[idx + 1]
        cap_lines.append({"text": merged, "size": 11})
    add_multi_text(slide, Inches(0.75), cap_y, Inches(11.8), Inches(0.7),
                   cap_lines, default_size=11, default_color=DARK_TEXT, line_spacing=1.4)

    # 底部结论
    add_conclusion_box(slide,
        "系统输出的是可直接运行的实训代码，不仅是伪代码或示意图。",
        top=SLIDE_H - Inches(1.0))


# ============================================================
# 第12页：生成—审查—辩论—修正
# ============================================================
def create_slide_12():
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide, LIGHT_BG)
    add_page_header(slide, "不依赖自检 · 独立审核 · 可溯修正",
                    "第二章 · 系统设计与技术实现 | 审核辩论机制", page_num=12)

    # 四步流程图
    flow_steps = [
        ("1. 生成", "个性化学习资源生成 Agent",
         "生成内容 + 技术Claim", PRIMARY_COLOR),
        ("2. 审核", "专业知识与教学质量审核 Agent",
         "核验Source、代码、规范、教学适配", SECONDARY_COLOR),
        ("3. 问题包", "结构化Issue Package",
         "问题编号 + 位置 + 证据 + 严重程度", ACCENT_COLOR),
        ("4. 修订复核", "Agent修订 → Agent复核",
         "带version和revision_note，通过/继续/人工", SUCCESS_GREEN)
    ]

    flow_w = Inches(2.8)
    flow_h = Inches(1.85)
    flow_gap = Inches(0.15)
    flow_start_x = (SLIDE_W - flow_w * 4 - flow_gap * 3) / 2
    flow_y = Inches(1.55)

    for i, (step, agent, desc, color) in enumerate(flow_steps):
        x = flow_start_x + (flow_w + flow_gap) * i
        add_shape(slide, x, flow_y, flow_w, flow_h,
                  fill_color=WHITE, line_color=color, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
        add_shape(slide, x, flow_y, flow_w, Inches(0.45), fill_color=color)
        add_textbox(slide, x, flow_y + Inches(0.08), flow_w, Inches(0.3),
                    step, font_size=13, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)
        add_textbox(slide, x + Inches(0.15), flow_y + Inches(0.55),
                    flow_w - Inches(0.3), Inches(0.35),
                    agent, font_size=11, bold=True, color=color)
        add_textbox(slide, x + Inches(0.15), flow_y + Inches(0.95),
                    flow_w - Inches(0.3), Inches(0.85),
                    desc, font_size=10, color=GRAY_TEXT)

        # 箭头
        if i < 3:
            arrow_x = x + flow_w + Inches(0.02)
            add_textbox(slide, arrow_x, flow_y + flow_h / 2 - Inches(0.2),
                        flow_gap, Inches(0.4),
                        "→", font_size=24, bold=True, color=color, alignment=PP_ALIGN.CENTER)

    # 循环回到步骤1
    loop_x = flow_start_x + flow_w * 2 + flow_gap * 1.5 - Inches(1.5)
    loop_y = flow_y + flow_h + Inches(0.05)
    add_shape(slide, loop_x, loop_y, Inches(3), Inches(0.35),
              fill_color=RGBColor(0xFD, 0xF3, 0xE6), line_color=ACCENT_COLOR,
              shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
    add_textbox(slide, loop_x, loop_y + Inches(0.03), Inches(3), Inches(0.3),
                "↺ 最多3轮修订，超轮次进入 needs_human_review",
                font_size=11, bold=True, color=ACCENT_COLOR, alignment=PP_ALIGN.CENTER)

    # 四个关键对象卡片
    obj_y = loop_y + Inches(0.55)
    obj_title_y = obj_y - Inches(0.05)
    add_textbox(slide, Inches(0.75), obj_title_y, Inches(6), Inches(0.4),
                "🔑 四个关键对象 · 全链路可追溯", font_size=14, bold=True, color=PRIMARY_COLOR)

    objs = [
        ("Source", "知识来源+原文位置", "可追溯", PRIMARY_COLOR),
        ("Claim", "可验证的技术陈述", "可核验", SECONDARY_COLOR),
        ("Issue", "原子错误+证据+严重程度", "可执行", ACCENT_COLOR),
        ("Trace", "初稿到终稿审核轨迹", "可回放", SUCCESS_GREEN)
    ]
    obj_w = Inches(2.75)
    obj_h = Inches(1.45)
    obj_gap = Inches(0.2)
    obj_start_x = Inches(0.75)

    for name, desc, tag, color in objs:
        add_shape(slide, obj_start_x, obj_y + Inches(0.4), obj_w, obj_h,
                  fill_color=WHITE, line_color=color, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
        add_shape(slide, obj_start_x, obj_y + Inches(0.4), obj_w, Inches(0.08), fill_color=color)
        add_textbox(slide, obj_start_x + Inches(0.2), obj_y + Inches(0.55),
                    Inches(1.6), Inches(0.4),
                    name, font_size=16, bold=True, color=color)
        # tag
        tag_x = obj_start_x + obj_w - Inches(0.95)
        add_shape(slide, tag_x, obj_y + Inches(0.58), Inches(0.8), Inches(0.3),
                  fill_color=color, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
        add_textbox(slide, tag_x, obj_y + Inches(0.62), Inches(0.8), Inches(0.25),
                    tag, font_size=9, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)
        add_textbox(slide, obj_start_x + Inches(0.2), obj_y + Inches(1.0),
                    obj_w - Inches(0.4), Inches(0.45),
                    desc, font_size=11, color=DARK_TEXT)
        obj_start_x += obj_w + obj_gap

    # 底部结论
    add_conclusion_box(slide,
        "系统通过独立审核、证据绑定和多轮结构化修正抑制专业知识幻觉。",
        top=SLIDE_H - Inches(1.0))


# ============================================================
# 第13页：代码审核与修正机制
# ============================================================
def create_slide_13():
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide, LIGHT_BG)
    add_page_header(slide, "代码级审核 · 从语法到规范全链路覆盖",
                    "第二章 · 系统设计与技术实现 | 代码审核机制", page_num=13)

    # 左侧审核维度
    left_x = Inches(0.7)
    left_w = Inches(4.8)

    add_shape(slide, left_x, Inches(1.55), left_w, Inches(0.5),
              fill_color=PRIMARY_COLOR, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
    add_textbox(slide, left_x, Inches(1.62), left_w, Inches(0.35),
                "🔍 代码审核五大维度", font_size=14, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)

    dims_list = [
        ("语法检查", "AST解析，捕获语法错误和导入缺失", PRIMARY_COLOR),
        ("逻辑检查", "训练循环顺序\n(zero_grad → backward → step)", SECONDARY_COLOR),
        ("硬件适配", "显存估算，量化策略是否匹配设备", ACCENT_COLOR),
        ("版本兼容", "transformers/peft/bitsandbytes版本", RGBColor(0x7A, 0x5F, 0xCC)),
        ("规范检查", "代码风格、注释完整性、错误处理", SUCCESS_GREEN)
    ]
    dim_y = Inches(2.2)
    for name, desc, color in dims_list:
        add_shape(slide, left_x, dim_y, left_w, Inches(0.78),
                  fill_color=WHITE, line_color=color, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
        add_shape(slide, left_x, dim_y, Inches(0.08), Inches(0.78), fill_color=color)
        add_textbox(slide, left_x + Inches(0.2), dim_y + Inches(0.1),
                    Inches(1.6), Inches(0.35),
                    name, font_size=13, bold=True, color=color)
        add_textbox(slide, left_x + Inches(0.2), dim_y + Inches(0.4),
                    left_w - Inches(0.4), Inches(0.4),
                    desc, font_size=10, color=GRAY_TEXT)
        dim_y += Inches(0.85)

    # 右侧真实案例
    right_x = Inches(5.8)
    right_w = Inches(6.8)

    add_shape(slide, right_x, Inches(1.55), right_w, Inches(0.5),
              fill_color=ACCENT_COLOR, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
    add_textbox(slide, right_x, Inches(1.62), right_w, Inches(0.35),
                "💡 真实修正案例：训练循环顺序错误", font_size=14, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)

    case_data = [
        ["❌ 错误初稿（审核拦截）", "⚠️ 审核发现（绑定证据）", "✅ 修订终稿（复核通过）"],
        ["loss.backward()\noptimizer.zero_grad()\noptimizer.step()",
         "gradient_flow_error\n绑定代码位置\n+ 知识库审核依据",
         "optimizer.zero_grad()\nloss.backward()\noptimizer.step()"]
    ]
    case_col_w = [Inches(2.15), Inches(2.5), Inches(2.15)]
    add_table(slide, right_x, Inches(2.2), right_w, Inches(2.2),
              case_data, col_widths=case_col_w, header_color=ACCENT_COLOR, font_size=10)

    # 能力清单
    cap_y = Inches(4.55)
    add_shape(slide, right_x, cap_y, right_w, Inches(0.4),
              fill_color=SUCCESS_GREEN, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
    add_textbox(slide, right_x, cap_y + Inches(0.05), right_w, Inches(0.3),
                "✅ 代码级审核能力清单", font_size=12, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)

    cap_items = [
        "错误代码位置精确定位（行号 + 上下文）",
        "审核依据可溯源（知识库引用 / 规范条款）",
        "修订后复核闭环（带content_version和revision_note）",
        "超轮次降级到 needs_human_review，不伪造修复"
    ]
    lines = [{"text": it, "size": 11, "color": DARK_TEXT, "bullet": True} for it in cap_items]
    add_multi_text(slide, right_x + Inches(0.2), cap_y + Inches(0.5),
                   right_w - Inches(0.4), Inches(1.4), lines, line_spacing=1.6)

    # 底部结论
    add_conclusion_box(slide,
        "审核Agent不是简单「检查是否有错」，而是精确定位错误位置、绑定依据、追踪修正，确保代码可执行。",
        top=SLIDE_H - Inches(1.0))


# ============================================================
# 第14页：Orchestrator协同编排与状态管理
# ============================================================
def create_slide_14():
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide, LIGHT_BG)
    add_page_header(slide, "状态不丢失 · 流程不断裂 · 异常可回退",
                    "第二章 · 系统设计与技术实现 | Orchestrator", page_num=14)

    # 左侧职责表格
    left_x = Inches(0.7)
    left_w = Inches(6.0)

    add_shape(slide, left_x, Inches(1.55), left_w, Inches(0.45),
              fill_color=PRIMARY_COLOR, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
    add_textbox(slide, left_x, Inches(1.6), left_w, Inches(0.35),
                "⚙️ Orchestrator核心职责（代码级实现）", font_size=13, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)

    resp_data = [
        ["职责", "实现方式"],
        ["会话状态管理", "维护 session_state，支持中断恢复"],
        ["Agent调用调度", "统一组装输入/输出契约"],
        ["块级并发生成", "首轮最多2块并行，审核并发上限2"],
        ["审核轮次控制", "最多3轮，超轮次 needs_human_review"],
        ["三级容错回退", "DeepSeek → Ollama → 确定性逻辑"],
        ["状态持久化", "写入SQLite，支持断点续训"]
    ]
    add_table(slide, left_x, Inches(2.1), left_w, Inches(3.5),
              resp_data, col_widths=[Inches(2.0), Inches(4.0)],
              header_color=PRIMARY_COLOR, font_size=11)

    # 右侧上：状态流转
    right_x = Inches(7.0)
    right_w = Inches(5.6)

    add_shape(slide, right_x, Inches(1.55), right_w, Inches(0.45),
              fill_color=SECONDARY_COLOR, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
    add_textbox(slide, right_x, Inches(1.6), right_w, Inches(0.35),
                "🔄 状态流转图", font_size=13, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)

    flow_text = (
        "Session创建 → 画像构建(Agent1) → 阶段生成(Agent1) → \n"
        "内容生成(Agent3) → 审核(Agent4) → 修订循环(Agent3↔Agent4) → \n"
        "学习包展示 → 答疑 → 测试 → 评估决策(Agent6) → \n"
        "阶段完成 → 下一阶段(Agent1) → 循环"
    )
    add_shape(slide, right_x, Inches(2.1), right_w, Inches(2.0),
              fill_color=WHITE, line_color=SECONDARY_COLOR, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
    lines = [{"text": t, "size": 10, "color": DARK_TEXT} for t in flow_text.split("\n")]
    add_multi_text(slide, right_x + Inches(0.2), Inches(2.2),
                   right_w - Inches(0.4), Inches(1.8), lines, line_spacing=1.6)

    # 右下：三级容错代码
    add_shape(slide, right_x, Inches(4.25), right_w, Inches(0.4),
              fill_color=ACCENT_COLOR, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
    add_textbox(slide, right_x, Inches(4.3), right_w, Inches(0.32),
                "🛡️ 三级容错（代码级实现）", font_size=12, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)

    code_text = (
        "try:\n"
        "    response = deepseek_api.chat(...)\n"
        "except (Timeout, RateLimit, APIError):\n"
        "    try:\n"
        "        response = ollama.chat(...)\n"
        "    except:\n"
        "        return deterministic_fallback()"
    )
    add_shape(slide, right_x, Inches(4.7), right_w, Inches(1.55),
              fill_color=RGBColor(0x1E, 0x2A, 0x3A), line_color=ACCENT_COLOR,
              shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
    lines_c = [{"text": t, "size": 10, "color": RGBColor(0x90, 0xEE, 0x90)} for t in code_text.split("\n")]
    add_multi_text(slide, right_x + Inches(0.2), Inches(4.75),
                   right_w - Inches(0.4), Inches(1.45), lines_c, line_spacing=1.4)

    # 底部：关键数据结构
    ds_y = SLIDE_H - Inches(1.25)
    add_shape(slide, Inches(0.7), ds_y, Inches(11.9), Inches(0.9),
              fill_color=RGBColor(0xEB, 0xF5, 0xFA), line_color=SECONDARY_COLOR)
    add_textbox(slide, Inches(0.85), ds_y + Inches(0.05), Inches(5), Inches(0.3),
                "💾 会话持久化关键数据结构", font_size=11, bold=True, color=SECONDARY_COLOR)
    structs = [
        "sessions：会话基础信息+current_stage",
        "user_profiles：画像快照+knowledge_gaps",
        "subtopic_status：知识点级掌握状态(-1/0/1)",
        "review_traces + review_debate_*：完整审核辩论轨迹",
        "qa_history：问答记录，辅助学情评估"
    ]
    struct_text = "  |  ".join(structs)
    add_textbox(slide, Inches(0.85), ds_y + Inches(0.38), Inches(11.6), Inches(0.5),
                struct_text, font_size=9, color=DARK_TEXT)

    # 底部结论
    add_conclusion_box(slide,
        "Orchestrator保证多Agent协同的工程可靠性——会话可恢复、流程可追踪、异常可回退，这是代码实现的核心支撑。",
        top=SLIDE_H - Inches(0.2))
    # 因为上面已有数据结构，这里覆盖掉上面的结论框放底部
    # 调整：把结论框内容融入数据结构上方，这里重新画一个更紧凑的
    pass


# ============================================================
# 第15页：学情反馈与动态决策
# ============================================================
def create_slide_15():
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide, LIGHT_BG)
    add_page_header(slide, "测试结果 + 问答历史 + 行为数据 → 下一阶段策略",
                    "第二章 · 系统设计与技术实现 | 动态决策", page_num=15)

    # 三栏布局：输入 → 输出 → 案例
    col_w = Inches(4.0)
    col_gap = Inches(0.25)
    col_start_x = (SLIDE_W - col_w * 3 - col_gap * 2) / 2
    col_y = Inches(1.55)

    # 1. 决策输入
    add_card(slide, col_start_x, col_y, col_w, Inches(4.5),
             "📥 决策输入", [
                 {"text": "阶段测试正确率", "size": 12, "bullet": True},
                 {"text": "错题知识点和错误类型", "size": 12, "bullet": True},
                 {"text": "用户问答历史", "size": 12, "bullet": True},
                 {"text": "学习时长和交互行为", "size": 12, "bullet": True},
                 {"text": "当前画像和子知识点掌握状态", "size": 12, "bullet": True},
             ], title_color=PRIMARY_COLOR)

    # 2. 决策输出
    mid_x = col_start_x + col_w + col_gap
    add_card(slide, mid_x, col_y, col_w, Inches(4.5),
             "📤 决策输出（6种策略）", [], title_color=SECONDARY_COLOR)

    strategies = [
        ("simplify", "降维解释", PRIMARY_COLOR),
        ("reinforce", "巩固练习", SECONDARY_COLOR),
        ("advance", "正常推进", SUCCESS_GREEN),
        ("backtrack", "回补前置", WARNING_RED),
        ("challenge", "进阶挑战", ACCENT_COLOR),
        ("finish", "完成学习", RGBColor(0x7A, 0x5F, 0xCC))
    ]
    s_y = col_y + Inches(0.85)
    for tag, name, color in strategies:
        add_shape(slide, mid_x + Inches(0.25), s_y, col_w - Inches(0.5), Inches(0.5),
                  fill_color=WHITE, line_color=color, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
        add_shape(slide, mid_x + Inches(0.25), s_y, Inches(1.3), Inches(0.5),
                  fill_color=color, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
        add_textbox(slide, mid_x + Inches(0.25), s_y + Inches(0.1),
                    Inches(1.3), Inches(0.35),
                    tag, font_size=10, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)
        add_textbox(slide, mid_x + Inches(1.7), s_y + Inches(0.12),
                    Inches(2.0), Inches(0.35),
                    name, font_size=13, bold=True, color=color)
        s_y += Inches(0.58)

    # 3. 典型案例
    r_x = mid_x + col_w + col_gap
    add_card(slide, r_x, col_y, col_w, Inches(4.5),
             "💡 典型决策案例", [], title_color=ACCENT_COLOR)

    case_box_y = col_y + Inches(0.85)
    # 输入场景
    add_shape(slide, r_x + Inches(0.2), case_box_y, col_w - Inches(0.4), Inches(1.4),
              fill_color=RGBColor(0xFD, 0xF3, 0xE6), line_color=ACCENT_COLOR)
    add_textbox(slide, r_x + Inches(0.3), case_box_y + Inches(0.1),
                col_w - Inches(0.6), Inches(0.3),
                "输入场景：", font_size=11, bold=True, color=ACCENT_COLOR)
    add_textbox(slide, r_x + Inches(0.3), case_box_y + Inches(0.4),
                col_w - Inches(0.6), Inches(1.0),
                "用户在量化配置和显存估算题目上连续出错，问答记录显示对4-bit量化理解不足",
                font_size=11, color=DARK_TEXT)

    # 箭头
    add_textbox(slide, r_x + Inches(col_w/2 - Inches(0.3)),
                case_box_y + Inches(1.42), Inches(0.6), Inches(0.4),
                "↓", font_size=20, color=ACCENT_COLOR, bold=True, alignment=PP_ALIGN.CENTER)

    # 输出决策
    out_y = case_box_y + Inches(1.85)
    add_shape(slide, r_x + Inches(0.2), out_y, col_w - Inches(0.4), Inches(1.4),
              fill_color=RGBColor(0xE8, 0xF5, 0xE9), line_color=SUCCESS_GREEN)
    add_textbox(slide, r_x + Inches(0.3), out_y + Inches(0.1),
                col_w - Inches(0.6), Inches(0.3),
                "系统决策：", font_size=11, bold=True, color=SUCCESS_GREEN)
    add_textbox(slide, r_x + Inches(0.3), out_y + Inches(0.4),
                col_w - Inches(0.6), Inches(1.0),
                "不直接推进到vLLM部署，而是生成降维解释内容 + 量化概念巩固任务，确保理解后再推进",
                font_size=11, color=DARK_TEXT)

    # 底部结论
    add_conclusion_box(slide,
        "学习路线由学习结果动态驱动，而非预先写死的课程表。",
        top=SLIDE_H - Inches(1.0))


# ============================================================
# 第16页：测试体系
# ============================================================
def create_slide_16():
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide, LIGHT_BG)
    add_page_header(slide, "五层测试 · 验证有效性与可靠性",
                    "第三章 · 效果与价值 | 测试体系", page_num=16)

    # 测试矩阵表格
    data = [
        ["测试类型", "规模", "验证目标"],
        ["四组完整案例", "4 组", "差异化学习者的端到端闭环验证"],
        ["个性化主评测", "50 组", "画像生成、资源匹配和路径适配正确性"],
        ["多Agent消融测试", "15×4 = 60 条", "各Agent和协同机制的独立贡献度"],
        ["回退可靠性测试", "12 组", "模型故障时自动回退和状态连续性"],
        ["模型可替换性测试", "12×2 = 24 条", "系统不绑定单一底层模型"]
    ]
    col_ws = [Inches(2.8), Inches(2.5), Inches(7.3)]
    add_table(slide, Inches(0.75), Inches(1.55), Inches(12.6), Inches(3.3),
              data, col_widths=col_ws, header_color=PRIMARY_COLOR, font_size=13)

    # 7个维度
    dim_y = Inches(5.1)
    add_shape(slide, Inches(0.75), dim_y, Inches(11.8), Inches(0.45),
              fill_color=SECONDARY_COLOR, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
    add_textbox(slide, Inches(0.75), dim_y + Inches(0.08), Inches(11.8), Inches(0.35),
                "🎯 50组主评测覆盖 7 个核心维度", font_size=13, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)

    dims7 = [
        ("LoRA/QLoRA\n适配", PRIMARY_COLOR),
        ("vLLM部署\n适配", RGBColor(0x2C, 0x3E, 0x6B)),
        ("学情诊断\n前置补全", SECONDARY_COLOR),
        ("已掌握\n知识跳过", RGBColor(0x2E, 0x7A, 0x9E)),
        ("异常边界\n输入处理", ACCENT_COLOR),
        ("时间风格\n适配", WARNING_RED),
        ("测试反馈\n路径调整", SUCCESS_GREEN)
    ]
    d_w = Inches(1.55)
    d_h = Inches(0.85)
    d_gap = Inches(0.13)
    d_start_x = (SLIDE_W - d_w * 7 - d_gap * 6) / 2
    d_y2 = dim_y + Inches(0.55)
    for name, color in dims7:
        add_shape(slide, d_start_x, d_y2, d_w, d_h,
                  fill_color=WHITE, line_color=color, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
        add_shape(slide, d_start_x, d_y2, d_w, Inches(0.08), fill_color=color)
        lines = name.split("\n")
        add_textbox(slide, d_start_x, d_y2 + Inches(0.15), d_w, Inches(0.35),
                    lines[0], font_size=11, bold=True, color=color, alignment=PP_ALIGN.CENTER)
        if len(lines) > 1:
            add_textbox(slide, d_start_x, d_y2 + Inches(0.48), d_w, Inches(0.35),
                        lines[1], font_size=11, bold=True, color=color, alignment=PP_ALIGN.CENTER)
        d_start_x += d_w + d_gap

    # 底部结论
    add_conclusion_box(slide,
        "测试不仅验证系统「能运行」，还验证「为什么有效」和「故障时能否继续运行」。",
        top=SLIDE_H - Inches(1.0))


# ============================================================
# 第17页：核心量化结果
# ============================================================
def create_slide_17():
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide, LIGHT_BG)
    add_page_header(slide, "三项核心指标全部达标",
                    "第三章 · 效果与价值 | 核心量化结果", page_num=17)

    # 六个大数字卡片 3×2
    metrics = [
        ("画像—资源难度\n适配准确率", "93.9%", "≥ 85%", "✓ 达标", PRIMARY_COLOR),
        ("任务核心知识点\n覆盖率", "94.3%", "≥ 90%", "✓ 达标", SECONDARY_COLOR),
        ("最终专业知识\n谬误率", "2.17%", "< 5%", "✓ 达标", ACCENT_COLOR),
        ("策略选择\n准确率", "98.0%", "—", "优异", SUCCESS_GREEN),
        ("完整流程\n成功率", "98.0%", "—", "优异", RGBColor(0x7A, 0x5F, 0xCC)),
        ("自动回退\n成功率", "91.7%", "—", "可靠", RGBColor(0x8E, 0x44, 0xAD))
    ]

    card_w = Inches(3.85)
    card_h = Inches(2.25)
    card_gap_x = Inches(0.3)
    card_gap_y = Inches(0.25)
    start_x = (SLIDE_W - card_w * 3 - card_gap_x * 2) / 2
    start_y = Inches(1.55)

    for i, (name, value, target, status, color) in enumerate(metrics):
        row = i // 3
        col = i % 3
        x = start_x + (card_w + card_gap_x) * col
        y = start_y + (card_h + card_gap_y) * row

        add_shape(slide, x, y, card_w, card_h,
                  fill_color=WHITE, line_color=color, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
        # 左侧竖条
        add_shape(slide, x, y, Inches(0.1), card_h, fill_color=color)

        # 指标名
        lines = name.split("\n")
        add_textbox(slide, x + Inches(0.25), y + Inches(0.15),
                    card_w - Inches(0.4), Inches(0.35),
                    lines[0], font_size=12, bold=True, color=DARK_TEXT)
        if len(lines) > 1:
            add_textbox(slide, x + Inches(0.25), y + Inches(0.45),
                        card_w - Inches(0.4), Inches(0.3),
                        lines[1], font_size=12, bold=True, color=DARK_TEXT)

        # 数值
        val_y = y + Inches(0.8)
        add_textbox(slide, x + Inches(0.25), val_y,
                    card_w - Inches(1.6), Inches(1.0),
                    value, font_size=38, bold=True, color=color)

        # 目标和状态 (右侧)
        right_col_x = x + card_w - Inches(1.4)
        add_shape(slide, right_col_x, val_y + Inches(0.05), Inches(1.2), Inches(0.35),
                  fill_color=RGBColor(0xF0, 0xF4, 0xF8), shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
        add_textbox(slide, right_col_x, val_y + Inches(0.08),
                    Inches(1.2), Inches(0.3),
                    f"目标: {target}", font_size=9, color=GRAY_TEXT, alignment=PP_ALIGN.CENTER)

        # 状态标签
        status_color = SUCCESS_GREEN if "达标" in status or "优异" in status or "可靠" in status else ACCENT_COLOR
        add_shape(slide, right_col_x, val_y + Inches(0.5), Inches(1.2), Inches(0.4),
                  fill_color=status_color, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
        add_textbox(slide, right_col_x, val_y + Inches(0.55),
                    Inches(1.2), Inches(0.3),
                    status, font_size=11, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)

    # 底部注释
    add_conclusion_box(slide,
        "通过知识溯源、独立审核、辩论修正和人工兜底四层机制，将最终专业知识谬误率控制在 2.17%，显著优于赛题 < 5% 的要求。")


# ============================================================
# 第18页：消融实验与可靠性验证
# ============================================================
def create_slide_18():
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide, LIGHT_BG)
    add_page_header(slide, "多Agent必要 · 三级容错可靠",
                    "第三章 · 效果与价值 | 消融实验与可靠性", page_num=18)

    # 左侧：消融实验
    left_x = Inches(0.7)
    left_w = Inches(6.4)

    add_shape(slide, left_x, Inches(1.55), left_w, Inches(0.5),
              fill_color=PRIMARY_COLOR, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
    add_textbox(slide, left_x, Inches(1.62), left_w, Inches(0.35),
                "🔬 消融实验对比（四种配置）", font_size=14, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)

    abl_data = [
        ["配置", "谬误率", "适配率", "说明"],
        ["Direct-LLM", "高（未通过）", "差", "模型直接生成，无约束无协同"],
        ["RAG-only", "较高", "一般", "加知识检索，无决策和审核闭环"],
        ["关闭审核Agent", "7.0 %", "—", "仅知识库+生成，无独立审核"],
        ["Full-System", "2.7 %", "最优", "完整多Agent + 辩论修正 + 动态决策"]
    ]
    add_table(slide, left_x, Inches(2.2), left_w, Inches(2.8),
              abl_data, col_widths=[Inches(2.0), Inches(1.2), Inches(1.0), Inches(2.2)],
              header_color=PRIMARY_COLOR, font_size=11)

    # 关键结论
    key_y = Inches(5.15)
    add_shape(slide, left_x, key_y, left_w, Inches(0.8),
              fill_color=RGBColor(0xE8, 0xF5, 0xE9), line_color=SUCCESS_GREEN)
    add_textbox(slide, left_x + Inches(0.2), key_y + Inches(0.1),
                Inches(2.5), Inches(0.35),
                "🔑 关键结论：", font_size=12, bold=True, color=SUCCESS_GREEN)
    add_multi_text(slide, left_x + Inches(0.2), key_y + Inches(0.38),
                   left_w - Inches(0.4), Inches(0.5),
                   [{"text": "审核修复率达到 91.4%", "size": 12, "bold": True, "color": DARK_TEXT},
                    {"text": "完整多Agent系统在所有维度上综合最优", "size": 12, "color": DARK_TEXT, "bullet": True}],
                   line_spacing=1.4)

    # 右侧：三级容错
    right_x = Inches(7.4)
    right_w = Inches(5.2)

    add_shape(slide, right_x, Inches(1.55), right_w, Inches(0.5),
              fill_color=ACCENT_COLOR, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
    add_textbox(slide, right_x, Inches(1.62), right_w, Inches(0.35),
                "🛡️ 三级容错流程", font_size=14, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)

    # 三级流程卡片
    tiers = [
        ("L1", "DeepSeek 云端主模型", "主要生成服务", PRIMARY_COLOR, Inches(2.2)),
        ("↓ 超时/限流/无正文", "", "", GRAY_TEXT, Inches(0.35)),
        ("L2", "Ollama 本地模型", "自动回退，保障连续性", SECONDARY_COLOR, Inches(2.2)),
        ("↓ 两类均不可用", "", "", GRAY_TEXT, Inches(0.35)),
        ("L3", "确定性逻辑兜底", "明确错误 / needs_human_review\n不伪造内容，保持会话状态", ACCENT_COLOR, Inches(2.6))
    ]
    tier_y = Inches(2.2)
    for tag, name, desc, color, width in tiers:
        center_x = right_x + (right_w - width) / 2
        if tag.startswith("↓"):
            add_textbox(slide, center_x, tier_y, width, Inches(0.35),
                        tag, font_size=18, color=color, bold=True, alignment=PP_ALIGN.CENTER)
        else:
            add_shape(slide, center_x, tier_y, width, Inches(0.85) if "L3" not in tag else Inches(1.0),
                      fill_color=WHITE, line_color=color, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
            # 等级标签
            add_shape(slide, center_x, tier_y, Inches(0.6),
                      Inches(0.85) if "L3" not in tag else Inches(1.0),
                      fill_color=color)
            add_textbox(slide, center_x, tier_y + Inches(0.25),
                        Inches(0.6), Inches(0.4),
                        tag, font_size=16, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)
            # 名称
            add_textbox(slide, center_x + Inches(0.7), tier_y + Inches(0.08),
                        width - Inches(0.8), Inches(0.35),
                        name, font_size=13, bold=True, color=color)
            # 描述
            add_textbox(slide, center_x + Inches(0.7), tier_y + Inches(0.45),
                        width - Inches(0.8), Inches(0.45),
                        desc, font_size=10, color=GRAY_TEXT)

        if tag.startswith("↓"):
            tier_y += width
        else:
            tier_y += (Inches(0.85) if "L3" not in tag else Inches(1.0)) + Inches(0.05)

    # 底部结论
    add_conclusion_box(slide,
        "消融实验验证多Agent协同的必要性（谬误率从7.0%降至2.7%）；故障测试验证模型异常时的可用性（自动回退成功率91.7%）。")


# ============================================================
# 第19页：创新点与应用价值
# ============================================================
def create_slide_19():
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide, LIGHT_BG)
    add_page_header(slide, "四项创新 · 三类价值",
                    "第三章 · 效果与价值 | 创新点与应用价值", page_num=19)

    # 左侧：四项创新
    left_x = Inches(0.7)
    left_w = Inches(6.0)

    add_shape(slide, left_x, Inches(1.55), left_w, Inches(0.5),
              fill_color=PRIMARY_COLOR, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
    add_textbox(slide, left_x, Inches(1.62), left_w, Inches(0.35),
                "💡 四项创新机制", font_size=15, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)

    innovations = [
        ("01", "候选目录约束与置信度门控的学情诊断机制",
         "将学情诊断从「模型自由猜测」转化为「目录约束下的证据驱动诊断」，降低初始路径错误"),
        ("02", "基于知识点状态和先修依赖的动态阶段规划机制",
         "以73个原子知识点+硬前置DAG+子知识点状态，让学习路径从静态目录变为动态决策过程"),
        ("03", "基于Claim—Source绑定的生成—审查—辩论—修正机制",
         "将「模型自检」升级为独立Agent之间基于证据的结构化协同，专业内容可追溯、可修正"),
        ("04", "云端、本地与确定性逻辑结合的三级容错体系",
         "保障会话状态不丢失、流程不被回退覆盖、不可恢复时明确说明，不以虚假内容掩盖失败")
    ]

    inn_y = Inches(2.2)
    colors = [PRIMARY_COLOR, SECONDARY_COLOR, ACCENT_COLOR, SUCCESS_GREEN]
    for i, (num, title, desc) in enumerate(innovations):
        h = Inches(1.0)
        add_shape(slide, left_x, inn_y, left_w, h,
                  fill_color=WHITE, line_color=colors[i], shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
        # 编号
        add_shape(slide, left_x + Inches(0.15), inn_y + Inches(0.1),
                  Inches(0.6), Inches(0.6), fill_color=colors[i], shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
        add_textbox(slide, left_x + Inches(0.15), inn_y + Inches(0.18),
                    Inches(0.6), Inches(0.5),
                    num, font_size=18, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)
        # 标题
        add_textbox(slide, left_x + Inches(0.9), inn_y + Inches(0.08),
                    left_w - Inches(1.1), Inches(0.35),
                    title, font_size=12, bold=True, color=colors[i])
        # 描述
        add_textbox(slide, left_x + Inches(0.9), inn_y + Inches(0.42),
                    left_w - Inches(1.1), Inches(0.6),
                    desc, font_size=10, color=GRAY_TEXT)
        inn_y += h + Inches(0.08)

    # 右侧：三类价值
    right_x = Inches(7.0)
    right_w = Inches(5.6)

    add_shape(slide, right_x, Inches(1.55), right_w, Inches(0.5),
              fill_color=SECONDARY_COLOR, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
    add_textbox(slide, right_x, Inches(1.62), right_w, Inches(0.35),
                "🎯 三类应用场景价值", font_size=15, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)

    values = [
        ("👨‍🎓", "学习者",
         ["降低试错成本", "获得适配基础+设备的实训路径", "减少重复学习，高效补缺"],
         PRIMARY_COLOR),
        ("🏫", "高  校",
         ["降低实训课程更新和差异化辅导成本", "辅助教师掌握班级学情", "支持本地化部署与知识管理"],
         SECONDARY_COLOR),
        ("🏢", "企  业",
         ["支持岗位培训与技术转岗", "标准化内训流程", "私有化知识部署与合规"],
         SUCCESS_GREEN)
    ]

    val_y = Inches(2.2)
    for icon, who, items, color in values:
        add_shape(slide, right_x, val_y, right_w, Inches(1.35),
                  fill_color=WHITE, line_color=color, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
        add_shape(slide, right_x, val_y, Inches(0.08), Inches(1.35), fill_color=color)
        # 图标+名称
        add_textbox(slide, right_x + Inches(0.2), val_y + Inches(0.2),
                    Inches(0.6), Inches(0.6),
                    icon, font_size=24, alignment=PP_ALIGN.CENTER)
        add_textbox(slide, right_x + Inches(0.85), val_y + Inches(0.25),
                    Inches(1.5), Inches(0.5),
                    who, font_size=15, bold=True, color=color)
        # 条目
        lines = [{"text": it, "size": 10, "color": DARK_TEXT, "bullet": True} for it in items]
        add_multi_text(slide, right_x + Inches(0.85), val_y + Inches(0.65),
                       right_w - Inches(1.0), Inches(0.7), lines, line_spacing=1.4)
        val_y += Inches(1.45)

    # 领域迁移
    mig_y = Inches(6.6)
    add_shape(slide, right_x, mig_y, right_w, Inches(0.5),
              fill_color=RGBColor(0xFD, 0xF3, 0xE6), line_color=ACCENT_COLOR, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
    add_textbox(slide, right_x + Inches(0.15), mig_y + Inches(0.1),
                right_w - Inches(0.3), Inches(0.35),
                "🔄 领域迁移：更换知识库/图谱/题库/审核规则/实训模板即可迁移到其他垂直技能培训",
                font_size=10, bold=True, color=ACCENT_COLOR, alignment=PP_ALIGN.CENTER)

    # 底部结论
    add_conclusion_box(slide,
        "四项创新机制从诊断、规划、审核、容错四个维度系统性解决个性化实训痛点，覆盖学习者、高校和企业三类核心用户。",
        top=SLIDE_H - Inches(1.0))


# ============================================================
# 第20页：总结
# ============================================================
def create_slide_20():
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide, WHITE)

    # 顶部大面积深色
    add_shape(slide, Inches(0), Inches(0), SLIDE_W, Inches(2.2), fill_color=PRIMARY_COLOR)
    add_shape(slide, Inches(9), Inches(0), Inches(5), Inches(2.2), fill_color=SECONDARY_COLOR)

    # 主标题
    add_textbox(slide, Inches(0.8), Inches(0.6), Inches(11), Inches(0.8),
                "让大模型走向", font_size=30, bold=True, color=WHITE)
    add_textbox(slide, Inches(0.8), Inches(1.25), Inches(11), Inches(0.8),
                "可诊断、可纠错、可决策的实训基础设施", font_size=30, bold=True, color=WHITE)

    # 四个关键词
    kws = [
        ("个性化", "根据学习者状态\n动态适配内容与路径", PRIMARY_COLOR),
        ("高保真", "知识溯源、独立审核\n多轮辩论修正", SECONDARY_COLOR),
        ("可闭环", "学习、测试、评估\n再规划持续循环", ACCENT_COLOR),
        ("可部署", "支持云端、本地\n和三级容错运行", SUCCESS_GREEN)
    ]
    kw_w = Inches(2.7)
    kw_h = Inches(1.8)
    kw_gap = Inches(0.3)
    kw_start_x = (SLIDE_W - kw_w * 4 - kw_gap * 3) / 2
    kw_y = Inches(2.6)

    for name, desc, color in kws:
        add_shape(slide, kw_start_x, kw_y, kw_w, kw_h,
                  fill_color=WHITE, line_color=color, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
        add_shape(slide, kw_start_x, kw_y, kw_w, Inches(0.12), fill_color=color)
        add_textbox(slide, kw_start_x, kw_y + Inches(0.25), kw_w, Inches(0.6),
                    name, font_size=22, bold=True, color=color, alignment=PP_ALIGN.CENTER)
        add_shape(slide, kw_start_x + kw_w/2 - Inches(0.8), kw_y + Inches(0.95),
                  Inches(1.6), Inches(0.02), fill_color=color)
        add_textbox(slide, kw_start_x + Inches(0.2), kw_y + Inches(1.05),
                    kw_w - Inches(0.4), Inches(0.7),
                    desc, font_size=11, color=GRAY_TEXT, alignment=PP_ALIGN.CENTER)
        kw_start_x += kw_w + kw_gap

    # 核心结束语
    quote_y = Inches(4.75)
    add_shape(slide, Inches(1.0), quote_y, Inches(11.3), Inches(1.3),
              fill_color=RGBColor(0xF5, 0xF7, 0xFA), line_color=PRIMARY_COLOR,
              shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
    add_shape(slide, Inches(1.0), quote_y, Inches(0.12), Inches(1.3), fill_color=ACCENT_COLOR)
    add_textbox(slide, Inches(1.2), quote_y - Inches(0.1), Inches(0.8), Inches(0.8),
                "❝", font_size=52, color=ACCENT_COLOR, bold=True)

    quote_text = ("我们不是让大模型再生成一门课程，\n"
                  "而是让四个职责明确的 Agent 共同陪伴学习者完成一条\n"
                  "能够诊断、纠错和动态进化的工程实训路径。")
    lines = [{"text": t, "size": 16, "bold": (i == 1),
              "color": PRIMARY_COLOR if i == 1 else DARK_TEXT,
              "alignment": PP_ALIGN.CENTER} for i, t in enumerate(quote_text.split("\n"))]
    add_multi_text(slide, Inches(2.0), quote_y + Inches(0.15),
                   Inches(9.5), Inches(1.1), lines, line_spacing=1.5)

    # 底部
    add_shape(slide, Inches(0), SLIDE_H - Inches(0.8), SLIDE_W, Inches(0.03), fill_color=SECONDARY_COLOR)
    add_textbox(slide, Inches(0), SLIDE_H - Inches(0.7), SLIDE_W, Inches(0.5),
                "感谢各位专家指导", font_size=20, bold=True, color=PRIMARY_COLOR, alignment=PP_ALIGN.CENTER)
    add_textbox(slide, Inches(0), SLIDE_H - Inches(0.3), SLIDE_W, Inches(0.25),
                "AI Agent Trainer  |  赛题编号：XH-202630", font_size=10, color=GRAY_TEXT, alignment=PP_ALIGN.CENTER)


# ============================================================
# 附录 A：指标定义
# ============================================================
def create_slide_appendix_a():
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide, LIGHT_BG)
    add_page_header(slide, "附录A：指标定义", "Appendix", page_num=21, total_pages=24)

    data = [
        ["指标", "定义", "计算公式"],
        ["画像—资源难度适配准确率", "资源难度与学习者画像匹配的比例",
         "匹配案例数 / 可评估案例总数"],
        ["核心知识点覆盖率", "最终资源覆盖的知识点比例",
         "覆盖知识点数 / 预期核心知识点总数"],
        ["最终专业知识谬误率", "最终资源中的专业错误比例",
         "错误Claim数 / 可核验Claim总数"],
        ["策略选择准确率", "策略与人工标注一致的比例",
         "一致决策数 / 可评估决策总数"],
        ["完整流程成功率", "完整闭环流程的比例",
         "成功场景数 / 测试场景总数"],
        ["自动回退成功率", "模型故障后成功回退的比例",
         "回退成功数 / 受控故障总数"]
    ]
    add_table(slide, Inches(0.75), Inches(1.55), Inches(11.8), Inches(5.2),
              data, col_widths=[Inches(2.8), Inches(4.8), Inches(4.2)],
              header_color=PRIMARY_COLOR, font_size=12)


# ============================================================
# 附录 B：消融实验配置
# ============================================================
def create_slide_appendix_b():
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide, LIGHT_BG)
    add_page_header(slide, "附录B：消融实验配置", "Appendix", page_num=22, total_pages=24)

    data = [
        ["配置", "详细说明"],
        ["Direct-LLM",
         "直接由底层大模型生成学习资源，不使用知识约束、不使用多Agent协同机制。验证纯模型生成的基线水平。"],
        ["RAG-only",
         "在Direct-LLM基础上增加知识检索（RAG），为模型提供上下文知识片段，但不使用阶段决策和审核闭环。验证知识库单独的贡献。"],
        ["No-Review-Agent",
         "关闭专业知识与教学质量审核 Agent，保留学情诊断、资源生成和动态决策。验证独立审核Agent对降低谬误率的独立贡献。"],
        ["Full-System",
         "完整运行本作品的多智能体协同系统，包含四个运行时Agent、Orchestrator编排和审核辩论机制。作为完整系统对照。"]
    ]
    add_table(slide, Inches(0.75), Inches(1.55), Inches(11.8), Inches(5.0),
              data, col_widths=[Inches(2.5), Inches(9.3)],
              header_color=SECONDARY_COLOR, font_size=13)


# ============================================================
# 附录 C：故障回退与人工复核
# ============================================================
def create_slide_appendix_c():
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide, LIGHT_BG)
    add_page_header(slide, "附录C：故障回退与人工复核", "Appendix", page_num=23, total_pages=24)

    items = [
        {
            "title": "DeepSeek 请求超时 / 限流 / API错误",
            "color": PRIMARY_COLOR,
            "desc": "自动触发 Ollama 本地模型回退。若 Ollama 成功生成，流程继续，用户端仅感知响应时间略长。会话状态保持一致，不丢失已生成的画像和阶段。"
        },
        {
            "title": "Ollama 不可用 / 本地模型加载失败",
            "color": SECONDARY_COLOR,
            "desc": "进入确定性逻辑兜底：流程保持、画像和阶段状态不回退；对必须生成的内容返回 needs_human_review 标记，不伪造看似正常但可能错误的学习资源。"
        },
        {
            "title": "审核超过 3 轮仍有 must_fix 问题",
            "color": ACCENT_COLOR,
            "desc": "学习包降级为 provisional 状态，标记 needs_human_review。已通过审核的内容块保留可用；有问题的块附带 issue_package 供人工查阅。系统保持诚实，不强行放行。"
        }
    ]

    y = Inches(1.6)
    for it in items:
        add_shape(slide, Inches(0.75), y, Inches(11.8), Inches(1.6),
                  fill_color=WHITE, line_color=it["color"], shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
        add_shape(slide, Inches(0.75), y, Inches(0.12), Inches(1.6), fill_color=it["color"])
        add_textbox(slide, Inches(1.0), y + Inches(0.15),
                    Inches(11.3), Inches(0.4),
                    it["title"], font_size=15, bold=True, color=it["color"])
        add_textbox(slide, Inches(1.0), y + Inches(0.6),
                    Inches(11.3), Inches(1.0),
                    it["desc"], font_size=12, color=DARK_TEXT)
        y += Inches(1.75)


# ============================================================
# 附录 D：代表性案例输入输出
# ============================================================
def create_slide_appendix_d():
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide, LIGHT_BG)
    add_page_header(slide, "附录D：代表性案例输入输出", "Appendix", page_num=24, total_pages=24)

    # 案例1
    y1 = Inches(1.55)
    add_shape(slide, Inches(0.75), y1, Inches(5.8), Inches(0.5),
              fill_color=PRIMARY_COLOR, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
    add_textbox(slide, Inches(0.75), y1 + Inches(0.08), Inches(5.8), Inches(0.35),
                "案例1：零基础、低学习时长、无GPU", font_size=13, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)

    case1_steps = ["① 初始画像输入：零基础/文科背景/每天1小时/无GPU",
                   "② 诊断结果：需要补齐Python、PyTorch前置基础",
                   "③ 阶段规划：从Python基础+Linux命令开始，拆分小步阶段",
                   "④ 学习包：增加概念解释和完整导入语句",
                   "⑤ 测试结果：正确率75% → 触发 reinforce 巩固策略",
                   "⑥ 下一阶段策略：生成降维解释+补充练习，不急于推进"]
    lines1 = [{"text": s, "size": 11, "color": DARK_TEXT} for s in case1_steps]
    add_multi_text(slide, Inches(0.85), y1 + Inches(0.6),
                   Inches(5.6), Inches(4.2), lines1, line_spacing=1.65)

    # 案例2
    y2 = Inches(1.55)
    x2 = Inches(6.75)
    add_shape(slide, x2, y2, Inches(5.8), Inches(0.5),
              fill_color=SECONDARY_COLOR, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
    add_textbox(slide, x2, y2 + Inches(0.08), Inches(5.8), Inches(0.35),
                "案例2：有工程基础、高配置部署需求", font_size=13, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)

    case2_steps = ["① 初始画像输入：PyTorch熟练/有训练经验/每天2小时/A10+QLoRA目标",
                   "② 诊断结果：跳过DL基础，目标定位QLoRA→权重合并→vLLM部署",
                   "③ 阶段规划：直接解锁HuggingFace→LoRA/QLoRA知识点",
                   "④ 学习包：输出4-bit量化+小Batch+显存优化代码",
                   "⑤ 测试结果：正确率92% → 触发 advance 正常推进",
                   "⑥ 下一阶段策略：进入权重合并与vLLM服务化部署实战"]
    lines2 = [{"text": s, "size": 11, "color": DARK_TEXT} for s in case2_steps]
    add_multi_text(slide, x2 + Inches(0.1), y2 + Inches(0.6),
                   Inches(5.6), Inches(4.2), lines2, line_spacing=1.65)


# ========== 生成所有幻灯片 ==========
def generate_all():
    print("开始生成PPT...")

    create_slide_1()
    print("  [1/24] 第1页 封面")
    create_slide_2()
    print("  [2/24] 第2页 目录")
    create_slide_3()
    print("  [3/24] 第3页 项目定位")
    create_slide_4()
    print("  [4/24] 第4页 赛题痛点")
    create_slide_5()
    print("  [5/24] 第5页 作品总体方案")
    create_slide_6()
    print("  [6/24] 第6页 四个运行时Agent")
    create_slide_7()
    print("  [7/24] 第7页 系统总体技术架构")
    create_slide_8()
    print("  [8/24] 第8页 完整业务闭环")
    create_slide_9()
    print("  [9/24] 第9页 知识图谱与动态阶段规划")
    create_slide_10()
    print("  [10/24] 第10页 个性化实训资源生成")
    create_slide_11()
    print("  [11/24] 第11页 代码生成与个性化适配")
    create_slide_12()
    print("  [12/24] 第12页 生成-审查-辩论-修正")
    create_slide_13()
    print("  [13/24] 第13页 代码审核与修正机制")
    create_slide_14()
    print("  [14/24] 第14页 Orchestrator协同编排")
    create_slide_15()
    print("  [15/24] 第15页 学情反馈与动态决策")
    create_slide_16()
    print("  [16/24] 第16页 测试体系")
    create_slide_17()
    print("  [17/24] 第17页 核心量化结果")
    create_slide_18()
    print("  [18/24] 第18页 消融实验与可靠性")
    create_slide_19()
    print("  [19/24] 第19页 创新点与应用价值")
    create_slide_20()
    print("  [20/24] 第20页 总结")
    create_slide_appendix_a()
    print("  [21/24] 附录A 指标定义")
    create_slide_appendix_b()
    print("  [22/24] 附录B 消融实验配置")
    create_slide_appendix_c()
    print("  [23/24] 附录C 故障回退与人工复核")
    create_slide_appendix_d()
    print("  [24/24] 附录D 代表性案例")

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "AI_Agent_Trainer_演示PPT.pptx")
    prs.save(output_path)
    print(f"\n✅ PPT生成成功！文件保存至：{output_path}")
    print(f"   共 {len(prs.slides)} 页幻灯片（20页正文 + 4页附录）")


if __name__ == "__main__":
    generate_all()
