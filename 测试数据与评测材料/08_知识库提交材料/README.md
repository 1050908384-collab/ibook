# 知识库比赛提交材料

本目录是由 `scripts/build_knowledge_submission_materials.py` 从项目现有数据生成的可审计材料，不替代原始知识文章和运行数据库。

## 数量口径

- `data/知识库原始数据/` 下有 73 篇按知识点组织的 Markdown 原始文章，作为知识内容与参考网址的第一来源。
- `data/seeds/knowledge_subtopics.json` 有 73 个知识点，用于知识图谱、阶段规划和题目映射。
- `data/seeds/knowledge_docs.json` 有 68 条运行时资料索引，是从知识内容整理出的派生资料，不等于知识点数量。
- `data/seeds/test_questions.json` 有 323 道题，覆盖 73 个知识点。

## 文件说明

- `knowledge_subtopic_catalog.csv`：73 个知识点、前置关系、原始文章路径、运行资料数和题量。
- `knowledge_document_catalog.csv`：68 条运行时资料与原始文章的对应关系。
- `knowledge_source_registry.csv`：从 73 篇原始文章提取的参考网址。作者未在原文明确写出的，统一标记为“原文未声明”，不得自行补写。
- `knowledge_question_coverage.csv`：每个知识点的题量和题型覆盖。
- `knowledge_submission_manifest.json`：数量和映射完整性摘要。
- `knowledge_submission_checklist.md`：提交前人工核验清单。

## 证据链

评委应能按以下链路回溯：

`知识点 -> 原始 Markdown -> 参考网址 -> 运行资料 -> 知识块/向量库 -> 题目 -> 个性化学习包 -> Agent4 审核`

本目录中的 CSV 是索引和审计材料；正文、图谱、向量库、题库和运行日志仍应从提交包的对应目录提供。
