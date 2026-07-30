import os, glob

cwd = os.getcwd()
suspects = [
    "backend/agents/agent_01_diagnostic.py",
    "backend/agents/agent_04_tech_review.py",
    "backend/agents/agent_05_quality_review.py",
    "backend/agents/base_agent.py",
    "backend/db/chroma_client.py",
    "backend/db/graph_client.py",
    "backend/llm/client.py",
    "backend/main.py",
    "backend/models/material.py",
    "backend/models/profile.py",
    "backend/tools/agent3_tools/llm_generate_tool.py",
    "backend/tools/agent4_tools/code_parser_tool.py",
    "backend/tools/agent6_tools/accuracy_calculator_tool.py",
]
for sf in suspects:
    path = os.path.join(cwd, sf)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    has_cn = any("\u4e00" <= c <= "\u9fff" for c in content)
    lines = content.split("\n")
    has_header = "coding: utf-8" in lines[0] if lines else False
    print(f"  {sf}: CN={has_cn} Header={has_header}")