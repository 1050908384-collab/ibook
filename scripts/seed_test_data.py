
# ---------- 路径设置：python scripts\xxx.py 直接运行时能正确找到 backend / data ----------
import sys as _sys_py
from pathlib import Path as _Path_py
_PROJ_ROOT_ = _Path_py(__file__).resolve().parent.parent
if str(_PROJ_ROOT_) not in _sys_py.path:
    _sys_py.path.insert(0, str(_PROJ_ROOT_))
if str(_PROJ_ROOT_ / "backend") not in _sys_py.path:
    _sys_py.path.insert(0, str(_PROJ_ROOT_ / "backend"))
# -------------------------------------------------------------------------------------

# ---------- 路径设置：python scripts\xxx.py 直接运行时能正确找到 backend / data ----------
import sys as _sys_py
from pathlib import Path as _Path_py
_PROJ_ROOT_ = _Path_py(__file__).resolve().parent.parent
if str(_PROJ_ROOT_) not in _sys_py.path:
    _sys_py.path.insert(0, str(_PROJ_ROOT_))
if str(_PROJ_ROOT_ / "backend") not in _sys_py.path:
    _sys_py.path.insert(0, str(_PROJ_ROOT_ / "backend"))
# -------------------------------------------------------------------------------------

import os
import sys
from sqlalchemy import text
import json

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, BACKEND_DIR)

from backend.db.sqlite_client import SQLiteClient

def seed_test_data():
    print("Seeding test data...")
    
    sqlite_client = SQLiteClient()
    
    knowledge_blocks = [
        ("KB-001", "Python基础", "Python编程基础", 2.0, "初级", '["KB-000"]', '["Python"]'),
        ("KB-002", "PyTorch基础", "深度学习框架基础", 3.0, "初级", '["KB-001"]', '["PyTorch"]'),
        ("KB-003", "Transformer架构", "Transformer模型原理", 4.0, "中级", '["KB-002"]', '["Transformer"]'),
        ("KB-004", "LoRA微调原理", "LoRA技术原理与应用", 3.0, "中级", '["KB-002", "KB-003"]', '["LoRA"]'),
        ("KB-005", "模型部署", "大模型部署实践", 3.0, "中级", '["KB-002"]', '["部署"]'),
    ]
    
    with sqlite_client.Session() as session:
        for block in knowledge_blocks:
            session.execute(text("""
                INSERT OR IGNORE INTO knowledge_blocks 
                (block_id, block_name, description, estimated_hours, difficulty, prerequisites, corresponding_gaps)
                VALUES (:block_id, :block_name, :description, :estimated_hours, :difficulty, :prerequisites, :corresponding_gaps)
            """), {
                "block_id": block[0],
                "block_name": block[1],
                "description": block[2],
                "estimated_hours": block[3],
                "difficulty": block[4],
                "prerequisites": block[5],
                "corresponding_gaps": block[6]
            })
        
        test_questions = [
            ("Q-001", "KB-001", "初级", "选择", "Python中哪个关键字用于定义函数？", '["def", "function", "func", "define"]', "def", "函数定义关键字"),
            ("Q-002", "KB-001", "初级", "判断", "Python是一种强类型语言", None, "错误", "类型系统"),
            ("Q-003", "KB-002", "初级", "选择", "PyTorch中创建张量的函数是？", '["torch.array", "torch.Tensor", "torch.create", "torch.make"]', "torch.Tensor", "张量创建"),
            ("Q-004", "KB-003", "中级", "简答", "请简述Transformer的Encoder结构", None, "多层多头注意力+前馈网络", "Transformer结构"),
            ("Q-005", "KB-004", "中级", "选择", "LoRA的全称是什么？", '["Low-Rank Adaptation", "Local Regression Analysis", "Layer-wise Regularization", "Linear Regression Approximation"]', "Low-Rank Adaptation", "LoRA定义"),
        ]
        
        for question in test_questions:
            session.execute(text("""
                INSERT OR IGNORE INTO test_questions 
                (question_id, block_id, difficulty, question_type, question_text, options, correct_answer, hint)
                VALUES (:question_id, :block_id, :difficulty, :question_type, :question_text, :options, :correct_answer, :hint)
            """), {
                "question_id": question[0],
                "block_id": question[1],
                "difficulty": question[2],
                "question_type": question[3],
                "question_text": question[4],
                "options": question[5],
                "correct_answer": question[6],
                "hint": question[7]
            })
        
        session.commit()
    
    print("Test data seeded successfully!")

if __name__ == "__main__":
    seed_test_data()
