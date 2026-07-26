import sqlite3
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import settings

DB_PATH = settings.DATABASE_URL.replace("sqlite:///", "").replace("../", "")

# ============================================
# 1. 定义要插入的数据（严格按学习顺序排列）
# ============================================
blocks_data = [
    {
        "id": "KB-01",
        "name": "Linux环境与命令行操作",
        "desc": "（环境准备：安装Linux系统（或WSL2/云服务器），熟悉终端使用，确保能执行Shell命令）\n- 常用Shell命令（cd/ls/mkdir/cp/mv）\n- 环境变量配置\n- conda/pip虚拟环境管理\n- nvidia-smi显存监控",
        "hours": 4.0,
        "prereq": "[]"
    },
    {
        "id": "KB-02",
        "name": "Python编程基础",
        "desc": "（环境准备：安装Python 3.10+，配置conda虚拟环境，安装IDE如PyCharm或VS Code，配置pip国内镜像源加速下载）\n- 基础语法（变量、运算符、条件判断、循环）\n- 数据结构（列表、元组、字典、集合）\n- 函数与类与对象\n- 文件与异常处理\n- 装饰器与生成器",
        "hours": 14.0,
        "prereq": '["KB-01"]'
    },
    {
        "id": "KB-03",
        "name": "数据处理基础",
        "desc": "（环境准备：在Python环境中安装Pandas、NumPy、Matplotlib、re库）\n- Pandas数据清洗\n- JSON/CSV读写与格式转换\n- NumPy基本数组运算\n- 正则表达式进行文本预处理",
        "hours": 4.5,
        "prereq": '["KB-02"]'
    },
    {
        "id": "KB-04",
        "name": "深度学习框架PyTorch基础",
        "desc": "（环境准备：根据CUDA版本安装PyTorch（如torch 2.0+ with CUDA 11.8/12.1））\n- 张量操作\n- 自动求导机制\n- torch.nn模块\n- Dataset/DataLoader数据装载\n- 优化器（如AdamW）与损失函数\n- 训练循环编写\n- 模型保存与加载（checkpoint）",
        "hours": 8.5,
        "prereq": '["KB-03"]'
    },
    {
        "id": "KB-05",
        "name": "自然语言处理（NLP）基础概念",
        "desc": "- 分词、词性标注、命名实体识别（NER）等基础任务\n- 文本的数值化表示方法（Bag-of-Words、TF-IDF、Word2Vec）",
        "hours": 3.5,
        "prereq": '["KB-04"]'
    },
    {
        "id": "KB-06",
        "name": "注意力机制",
        "desc": "- 注意力机制的核心概念（Query、Key、Value）\n- Self-Attention与Multi-Head Attention的计算原理与公式",
        "hours": 2.5,
        "prereq": '["KB-05"]'
    },
    {
        "id": "KB-07",
        "name": "子词切分与分词器（Tokenizer）",
        "desc": "（环境准备：在Python环境中安装transformers、tokenizers库）\n- BPE、WordPiece、SentencePiece算法原理\n- HuggingFace Tokenizer的加载与使用\n- padding、truncation、max_length等关键参数\n- ChatTemplate对话模板格式",
        "hours": 3.0,
        "prereq": '["KB-06"]'
    },
    {
        "id": "KB-08",
        "name": "Transformer架构（Decoder-Only）",
        "desc": "（环境准备：可选安装transformers库，加载小型模型（如GPT-2）进行前向传播实验）\n- Transformer的Encoder-Decoder结构概述\n- Decoder-Only架构（如GPT/Llama/Qwen）\n- RMSNorm与残差连接\n- 前馈网络（FFN）与SwiGLU激活函数\n- 旋转位置编码（RoPE）\n- KV Cache机制",
        "hours": 5.5,
        "prereq": '["KB-06", "KB-07"]'
    },
    {
        "id": "KB-09",
        "name": "大语言模型（LLM）核心特性",
        "desc": "- 涌现能力\n- 上下文学习（In-Context Learning）\n- 指令遵循（Instruction Following）\n- 逐步推理（Step-by-Step Reasoning）\n- 幻觉问题",
        "hours": 3.0,
        "prereq": '["KB-08"]'
    },
    {
        "id": "KB-10",
        "name": "LLM三阶段训练流程",
        "desc": "- 预训练（Pretrain）整体流程与目标\n- 有监督微调（SFT）整体流程与目标\n- 基于人类反馈的强化学习（RLHF）整体流程与目标",
        "hours": 5.5,
        "prereq": '["KB-09"]'
    },
    {
        "id": "KB-11",
        "name": "HuggingFace生态工具链",
        "desc": "（环境准备：安装transformers、datasets、accelerate库，建议使用huggingface_hub配置缓存路径与Token）\n- transformers库的AutoModel/AutoTokenizer加载方式\n- datasets库的数据处理（map/filter/shuffle）\n- Trainer训练器的使用及TrainingArguments参数配置",
        "hours": 4.5,
        "prereq": '["KB-10"]'
    },
    {
        "id": "KB-12",
        "name": "参数高效微调（LoRA与QLoRA）",
        "desc": "（环境准备：在上述基础上额外安装peft、bitsandbytes库，确保bitsandbytes能与CUDA版本兼容，QLoRA需要GPU显存≥12GB）\n- LoRA的低秩分解原理\n- Rank/Alpha参数作用\n- PEFT库的LoraConfig配置（target_modules、r、lora_alpha、lora_dropout）\n- QLoRA的4-bit量化（NF4、双重量化）与分页优化器原理\n- LoRA权重合并与导出（merge_and_unload）",
        "hours": 6.0,
        "prereq": '["KB-11"]'
    },
    {
        "id": "KB-13",
        "name": "PyTorch进阶训练技巧",
        "desc": "（环境准备：复用上述PyTorch环境，确保torch版本支持AMP混合精度训练）\n- 梯度累积（Gradient Accumulation）\n- 混合精度训练（AMP）\n- 梯度检查点（Gradient Checkpointing）\n- 学习率调度（Warmup + 余弦退火）",
        "hours": 4.5,
        "prereq": '["KB-12"]'
    },
    {
        "id": "KB-14",
        "name": "分布式训练（DeepSpeed）",
        "desc": "（环境准备：安装deepspeed库，配置多GPU环境（至少2卡），编写ds_config.json文件，使用 deepspeed --num_gpus=N 启动训练）\n- ZeRO-1/2/3优化策略配置\n- 数据并行与模型并行的区别\n- 编写ds_config.json并启动deepspeed训练",
        "hours": 4.5,
        "prereq": '["KB-13"]'
    },
    {
        "id": "KB-15",
        "name": "模型推理部署（vLLM）",
        "desc": "（环境准备：安装vLLM库，注意其PyTorch版本兼容性，GPU显存建议≥16GB；安装前需确保CUDA环境正确）\n- 模型生成策略（贪心搜索、束搜索）\n- Top-p/Top-k采样、Temperature温度系数\n- vLLM的PagedAttention机制\n- vLLM的Continuous Batching机制\n- vLLM加载模型并配置gpu_memory_utilization等关键参数进行推理部署",
        "hours": 5.0,
        "prereq": '["KB-14"]'
    }
]

# ============================================
# 2. 执行插入操作
# ============================================

def insert_data():
    # 连接数据库
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 清空旧数据（避免重复插入）
    cursor.execute("DELETE FROM knowledge_blocks")
    print("[INFO] 已清空旧数据")

    # 逐条插入
    for block in blocks_data:
        cursor.execute("""
            INSERT INTO knowledge_blocks (
                block_id, 
                block_name, 
                description, 
                estimated_hours, 
                difficulty, 
                prerequisites,
                corresponding_gaps
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            block["id"],                         # block_id
            block["name"],                       # block_name
            block["desc"],                       # description
            block["hours"],                      # estimated_hours
            "中级",                              # difficulty（先统一填"中级"）
            block["prereq"],                     # prerequisites (JSON数组)
            '["' + block["name"] + '"]'          # corresponding_gaps (对应盲区)
        ))
        print(f"[OK] 已插入: {block['id']} - {block['name']}")

    # 提交并关闭
    conn.commit()
    conn.close()
    print("\n[DONE] 所有数据导入成功！")

if __name__ == "__main__":
    insert_data()