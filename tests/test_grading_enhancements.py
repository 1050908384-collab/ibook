# -*- coding: utf-8 -*-
"""
验证本次修复的两大问题：
1. 填空题 Python 代码 # 注释差异不应导致正确答案被判错
2. 跨平台（Linux bash ↔ Windows CMD）命令等价答案应互相匹配

同时验证新增能力：多候选答案(|||)、路径分隔符归一化、LLM兜底仅在规则失败时触发。

执行：
  cd d:\code\ai_agent-trainer
  python -m pytest tests\test_grading_enhancements.py -v
  或  python tests\test_grading_enhancements.py
"""
from __future__ import annotations
import sys, os
import asyncio
import types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from orchestrator.harness import Harness
import unittest


class TestPythonCommentStripping(unittest.TestCase):
    """【问题 1】填空题 # 注释差异判分。

    场景：correct_answer 里写的注释和用户答案里的注释不一致，但代码本体完全相同。
    旧逻辑因注释文本字面不同会判错；新逻辑先剥离注释再比较。
    """

    def test_fill_blank_same_code_diff_comment_cn(self):
        """完全相同代码 + 不同措辞的中文注释 → 应当判对"""
        correct = 'd.get("c", 0)  # 获取键c，默认返回0'
        user    = 'd.get("c", 0) # 当键不存在时返回默认值0'
        self.assertTrue(Harness._match_answer(user, correct, 'fill_blank'))

    def test_fill_blank_no_comment_vs_with_comment(self):
        """用户答案没写注释、参考答案写了注释（或反过来） → 应当判对"""
        correct = 'result = a + b  # 求和'
        user    = 'result = a + b'
        self.assertTrue(Harness._match_answer(user, correct, 'fill_blank'))
        self.assertTrue(Harness._match_answer(correct, user, 'fill_blank'))

    def test_fill_blank_multiline_python_code_comment_only_diff(self):
        """多行代码：注释行内容不同 + 行尾注释不同 → 代码相同即对"""
        correct = (
            'import pandas as pd\n'
            'df = pd.read_csv("data.csv")   # 读取csv\n'
            'print(df.head())  # 打印前5行\n'
        )
        user = (
            'import pandas as pd\n'
            '# 从文件加载数据\n'
            'df = pd.read_csv("data.csv")\n'
            'print(df.head())\n'
        )
        self.assertTrue(Harness._match_answer(user, correct, 'practical'))

    def test_strip_comment_not_affect_url_fragment(self):
        """字符串里的 # / URL 的 # / 非注释字符不应当被误剥"""
        correct = 'url = "http://example.com#section"; token = "a#b"'
        user    = 'url = "http://example.com#section"  ; token = "a#b"'
        # 用 exact / contains 逻辑判断 - 两个字面几乎相同，应匹配
        self.assertTrue(Harness._match_answer(user, correct, 'fill_blank'))

    def test_comment_diff_grading_mode_contains_still_works(self):
        """grading_mode=contains（填空题常用）配合注释剥离应仍有效"""
        correct = 'print("Hello, world!") # 输出问候语'
        user    = 'print("Hello, world!") # 在控制台打印问候'
        self.assertTrue(Harness._match_answer(user, correct, 'fill_blank'))


class TestCrossPlatformMatching(unittest.TestCase):
    """【问题 2】CMD ↔ Linux 跨平台命令 + 路径归一化。

    场景：题目问法写 "CMD/终端"，用户写 CMD 等价命令；参考答案仅 Linux 版。
    旧逻辑直接判错；新逻辑做命令词替换 + 路径分隔符归一化。
    """

    # --- 命令词等价 ---
    def test_ls_vs_dir(self):
        correct = 'ls -la'
        user    = 'dir'
        self.assertTrue(Harness._match_answer(user, correct, 'practical'))
        self.assertTrue(Harness._match_answer('dir /a', correct, 'practical'))

    def test_cp_vs_copy(self):
        self.assertTrue(Harness._match_answer('copy a.txt b.txt', 'cp a.txt b.txt', 'practical'))

    def test_rm_vs_del(self):
        self.assertTrue(Harness._match_answer('del old.log', 'rm old.log', 'practical'))
        self.assertTrue(Harness._match_answer('erase old.log', 'rm old.log', 'practical'))

    def test_cat_vs_type(self):
        self.assertTrue(Harness._match_answer('type data.txt', 'cat data.txt', 'practical'))

    def test_mkdir_p_vs_md(self):
        # CMD 中 mkdir/md 本身就递归创建（无需 -p 参数）
        self.assertTrue(Harness._match_answer('mkdir test_dir', 'mkdir -p test_dir', 'practical'))
        self.assertTrue(Harness._match_answer('md test_dir',      'mkdir -p test_dir', 'practical'))

    def test_pip_install_vs_py_m_pip(self):
        # Windows CMD 上 python launcher 用法
        correct = 'pip install torch'
        user    = 'py -m pip install torch'
        self.assertTrue(Harness._match_answer(user, correct, 'practical'))

    def test_conda_activate_windows_style(self):
        # Windows CMD 旧 conda 用法 activate xxx（不带 conda 前缀）
        correct = 'conda activate myenv'
        user    = 'activate myenv'
        self.assertTrue(Harness._match_answer(user, correct, 'fill_blank'))

    def test_whoami_hostname_both_ampersands(self):
        # CMD 单 & 连接 vs bash 双 &&；linux 命令本身跨平台相同
        correct = 'whoami && hostname'
        user    = 'whoami & hostname'
        self.assertTrue(Harness._match_answer(user, correct, 'practical'))

    # --- 路径分隔符 / vs \ ---
    def test_path_backslash_vs_forward_slash(self):
        correct = 'python scripts/train.py --data /tmp/data'
        user    = 'python scripts\\\\train.py --data C:\\\\tmp\\\\data'
        # 用户真实输入会是原生反斜杠，但在字符串字面量里要转义写
        self.assertTrue(Harness._match_answer(user.replace('\\\\', '\\'), correct, 'practical'))

    def test_backslash_path_in_fill_blank(self):
        correct = 'mkdir -p /tmp/backup'
        user    = 'md C:\\tmp\\backup'
        self.assertTrue(Harness._match_answer(user, correct, 'practical'))


class TestMultiAlternativeAnswers(unittest.TestCase):
    """correct_answer 使用 ||| 分隔多个候选正确答案，命中任一即对。"""

    def test_three_alternatives_first_hit(self):
        correct = 'conda activate myenv ||| activate myenv ||| source activate myenv'
        self.assertTrue(Harness._match_answer('conda activate myenv', correct, 'fill_blank'))

    def test_three_alternatives_middle_hit(self):
        correct = 'conda activate myenv ||| activate myenv ||| source activate myenv'
        self.assertTrue(Harness._match_answer('activate myenv', correct, 'fill_blank'))

    def test_linux_and_cmd_variants(self):
        correct = 'mkdir -p /tmp/test_dir ||| md C:\\tmp\\test_dir'
        self.assertTrue(Harness._match_answer('mkdir -p /tmp/test_dir', correct, 'practical'))
        self.assertTrue(Harness._match_answer('md C:\\tmp\\test_dir',   correct, 'practical'))

    def test_multi_alt_wrong_answer_still_false(self):
        correct = 'python --version ||| python -V ||| py --version'
        self.assertFalse(Harness._match_answer('node --version', correct, 'practical'))

    def test_split_alternatives_function(self):
        alts = Harness._split_alternatives('A ||| B ||| C')
        self.assertEqual(len(alts), 3)
        self.assertIn('A', alts)


class TestRegressionOriginalFeatures(unittest.TestCase):
    """旧版 _match_answer 功能必须仍可用（回归）。"""

    def test_exact_match(self):
        self.assertTrue(Harness._match_answer('B', 'B', 'single_choice'))

    def test_true_false_a_is_correct(self):
        self.assertTrue(Harness._match_answer('A', '正确', 'true_false'))
        self.assertTrue(Harness._match_answer('B', '错误', 'true_false'))
        self.assertTrue(Harness._match_answer('√', 'T', 'true_false'))
        self.assertTrue(Harness._match_answer('×', 'F', 'true_false'))

    def test_single_choice_letter(self):
        self.assertTrue(Harness._match_answer('A', 'A：ls 命令的含义', 'single_choice'))

    def test_contains_match(self):
        self.assertTrue(Harness._match_answer(
            '长格式详细信息列出文件；显示所有文件包括隐藏文件',
            '`-l` 以长格式（详细信息：权限、所有者、大小、修改时间）列出文件；`-a` 显示所有文件（包括以`.`开头的隐藏文件）。',
            'fill_blank',
        ))

    def test_numeric_approx(self):
        self.assertTrue(Harness._match_answer('0.333', '0.33', 'fill_blank'))
        self.assertFalse(Harness._match_answer('0.5', '0.33', 'fill_blank'))


class TestLlmSemanticFallback(unittest.TestCase):
    def test_fill_blank_rule_failure_uses_semantic_grader(self):
        harness = Harness.__new__(Harness)
        called = []

        async def semantic_grader(self, **kwargs):
            called.append(kwargs)
            return True

        harness._grade_semantically_with_llm = types.MethodType(semantic_grader, harness)
        result = asyncio.run(harness._match_answer_async(
            user_answer="采用梯度累加模拟大批量训练",
            correct_answer="optimizer.step()",
            question_type="fill_blank",
            question_text="梯度累积后执行什么操作？",
        ))
        self.assertTrue(result)
        self.assertEqual(1, len(called))


class TestStripCommentPreprocessor(unittest.TestCase):
    """针对 _strip_code_comments 辅助函数的直接单测。"""

    def test_python_inline_comment_removed(self):
        out = Harness._strip_code_comments('x = 1  # 设置x为1', 'fill_blank')
        self.assertNotIn('#', out)
        self.assertIn('x = 1', out)

    def test_bash_full_line_comment_removed(self):
        code = '#!/bin/bash\n# 这是注释\nls -la'
        out = Harness._strip_code_comments(code, 'practical')
        self.assertNotIn('这是注释', out)
        self.assertIn('ls -la', out)

    def test_comment_inside_string_preserved(self):
        code = 's = "hello#world"  # 赋值'
        out = Harness._strip_code_comments(code, 'fill_blank')
        self.assertIn('hello#world', out)
        self.assertNotIn('赋值', out)


class TestCrossPlatformPreprocessor(unittest.TestCase):
    """针对 _normalize_cross_platform 辅助函数的直接单测。"""

    def test_backslash_normalized(self):
        out = Harness._normalize_cross_platform(r'scripts\run.py data\text.csv')
        self.assertNotIn('\\', out)
        # 原始路径是相对路径（scripts/ 无 /前缀），归一化不应凭空增加前导 /
        self.assertIn('scripts/run.py', out)
        self.assertIn('data/text.csv', out)

    def test_cmd_copy_mapped_to_cp(self):
        out = Harness._normalize_cross_platform('copy a.txt b.txt')
        self.assertIn('cp', out)

    def test_cmd_dir_mapped_to_ls(self):
        out = Harness._normalize_cross_platform('dir /a')
        self.assertIn('ls', out)


if __name__ == '__main__':
    unittest.main(verbosity=2)
