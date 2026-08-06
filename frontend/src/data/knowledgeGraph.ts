export type KnowledgeGraphSubtopic = {
  subtopic_id: string;
  subtopic_name: string;
  sequence_index: number;
  prerequisites: string[];
};

export type KnowledgeGraphBlock = {
  block_id: string;
  block_name: string;
  prerequisites: string[];
  aliases: string[];
  subtopics: KnowledgeGraphSubtopic[];
};

type SubtopicDefinition = [id: string, name: string, prerequisites?: string[]];

const block = (
  block_id: string,
  block_name: string,
  prerequisites: string[],
  aliases: string[],
  definitions: SubtopicDefinition[]
): KnowledgeGraphBlock => ({
  block_id,
  block_name,
  prerequisites,
  aliases,
  subtopics: definitions.map(([subtopic_id, subtopic_name, explicitPrerequisites], index) => ({
    subtopic_id,
    subtopic_name,
    sequence_index: Number(subtopic_id.slice(-2)),
    prerequisites: explicitPrerequisites ?? (index > 0 ? [definitions[index - 1][0]] : [])
  }))
});

// Mirrors the 15 blocks and 73 subtopics in data/seeds/knowledge_subtopics.json.
// Keep this browser-side catalog static so a failed API never becomes fake mastery data.
export const knowledgeGraphBlocks: KnowledgeGraphBlock[] = [
  block('KB-01', 'Linux环境与命令行操作', [], ['Linux基础', '命令行操作'], [
    ['KBS-01-00', '环境准备：安装 Linux 系统（或 WSL2/云服务器），熟悉终端使用'],
    ['KBS-01-01', '常用 Shell 命令（cd/ls/mkdir/cp/mv）'],
    ['KBS-01-02', '环境变量配置'],
    ['KBS-01-03', 'conda/pip 虚拟环境管理'],
    ['KBS-01-04', 'nvidia-smi 显存监控']
  ]),
  block('KB-02', 'Python编程基础', ['KB-01'], ['Python基础', '编程能力'], [
    ['KBS-02-00', '环境准备：安装 Python 3.10+、配置 conda 与开发工具'],
    ['KBS-02-01', '语法'],
    ['KBS-02-02', '数据结构'],
    ['KBS-02-03', '函数与类与对象'],
    ['KBS-02-04', '文件与异常处理'],
    ['KBS-02-05', '装饰器与生成器']
  ]),
  block('KB-03', '数据处理基础', ['KB-02'], ['数据处理', '数据分析'], [
    ['KBS-03-00', '环境准备：安装 Pandas、NumPy、Matplotlib、re 库', ['KBS-02-00']],
    ['KBS-03-01', '使用 Pandas 进行数据清洗'],
    ['KBS-03-02', '格式转换（JSON/CSV 读写）'],
    ['KBS-03-03', 'NumPy 进行基本数组运算'],
    ['KBS-03-04', '正则表达式进行文本预处理']
  ]),
  block('KB-04', '深度学习框架PyTorch基础', ['KB-02', 'KB-03'], ['PyTorch', '深度学习'], [
    ['KBS-04-00', '环境准备：根据 CUDA 版本安装 PyTorch', ['KBS-01-04', 'KBS-02-00']],
    ['KBS-04-01', '张量操作'],
    ['KBS-04-02', '自动求导机制'],
    ['KBS-04-03', 'torch.nn 模块'],
    ['KBS-04-04', 'Dataset/DataLoader 数据装载'],
    ['KBS-04-05', '优化器（如 AdamW）与损失函数'],
    ['KBS-04-06', '训练循环编写'],
    ['KBS-04-07', '模型保存与加载（checkpoint）']
  ]),
  block('KB-05', '自然语言处理（NLP）基础概念', ['KB-02', 'KB-03'], ['NLP基础', '文本表示'], [
    ['KBS-05-01', '理解分词、词性标注、命名实体识别（NER）等基础任务'],
    ['KBS-05-02', '掌握文本的数值化表示方法（Bag-of-Words、TF-IDF、Word2Vec）']
  ]),
  block('KB-06', '注意力机制', ['KB-04'], ['注意力机制'], [
    ['KBS-06-01', '理解注意力机制的核心概念（Query、Key、Value）'],
    ['KBS-06-02', '掌握 Self-Attention 与 Multi-Head Attention 的计算原理与公式']
  ]),
  block('KB-07', '子词切分与分词器（Tokenizer）', ['KB-05'], ['Tokenizer', '子词切分'], [
    ['KBS-07-00', '环境准备：安装 transformers、tokenizers 库', ['KBS-02-00']],
    ['KBS-07-01', '理解 BPE、WordPiece、SentencePiece 算法原理'],
    ['KBS-07-02', '掌握 HuggingFace Tokenizer 的加载与使用'],
    ['KBS-07-03', '熟悉 padding、truncation、max_length 等关键参数'],
    ['KBS-07-04', '了解 ChatTemplate 对话模板格式']
  ]),
  block('KB-08', 'Transformer架构（Decoder-Only）', ['KB-04', 'KB-06'], ['Transformer', 'GPT'], [
    ['KBS-08-00', '环境准备：加载小型模型进行前向传播实验', ['KBS-07-00']],
    ['KBS-08-01', '掌握 Transformer 的 Encoder-Decoder 结构'],
    ['KBS-08-02', '重点理解 Decoder-Only 架构（如 GPT/Llama/Qwen）'],
    ['KBS-08-03', 'RMSNorm 与残差连接'],
    ['KBS-08-04', '前馈网络（FFN）与 SwiGLU 激活函数'],
    ['KBS-08-05', '旋转位置编码（RoPE）'],
    ['KBS-08-06', 'KV Cache 机制'],
    ['KBS-08-07', '前向传播实验（加载 GPT-2 验证理解）']
  ]),
  block('KB-09', '大语言模型（LLM）核心特性', ['KB-08'], ['LLM', '大模型'], [
    ['KBS-09-01', '理解涌现能力'],
    ['KBS-09-02', '上下文学习（In-Context Learning）'],
    ['KBS-09-03', '指令遵循（Instruction Following）'],
    ['KBS-09-04', '逐步推理（Step-by-Step Reasoning）'],
    ['KBS-09-05', '幻觉问题']
  ]),
  block('KB-10', 'LLM三阶段训练流程', ['KB-08', 'KB-09'], ['LLM训练', 'SFT', 'RLHF'], [
    ['KBS-10-01', '掌握预训练（Pretrain）整体流程与目标'],
    ['KBS-10-02', '有监督微调（SFT）整体流程与目标'],
    ['KBS-10-03', '基于人类反馈的强化学习（RLHF）整体流程与目标']
  ]),
  block('KB-11', 'HuggingFace生态工具链', ['KB-04', 'KB-07'], ['HuggingFace', 'transformers'], [
    ['KBS-11-00', '环境准备：安装 transformers、datasets、accelerate 库', ['KBS-07-00']],
    ['KBS-11-01', '掌握 transformers 库的 AutoModel/AutoTokenizer 加载方式'],
    ['KBS-11-02', 'datasets 库的数据处理（map/filter/shuffle）'],
    ['KBS-11-03', 'Trainer 训练器的使用及 TrainingArguments 参数配置']
  ]),
  block('KB-12', '参数高效微调（LoRA与QLoRA）', ['KB-04', 'KB-11'], ['LoRA', 'QLoRA', 'PEFT', '参数高效微调'], [
    ['KBS-12-00', '环境准备：安装 peft、bitsandbytes 库', ['KBS-04-00', 'KBS-11-00']],
    ['KBS-12-01', '理解 LoRA 的低秩分解原理'],
    ['KBS-12-02', 'Rank/Alpha 参数作用'],
    ['KBS-12-03', '掌握 PEFT 库的 LoraConfig 配置'],
    ['KBS-12-04', '理解 QLoRA 的 4-bit 量化与分页优化器原理'],
    ['KBS-12-05', '掌握 LoRA 权重合并与导出（merge_and_unload）']
  ]),
  block('KB-13', 'PyTorch进阶训练技巧', ['KB-04'], ['PyTorch进阶', '训练优化'], [
    ['KBS-13-00', '环境准备：复用 PyTorch 环境并确认支持 AMP', ['KBS-04-00']],
    ['KBS-13-01', '掌握梯度累积（Gradient Accumulation）'],
    ['KBS-13-02', '混合精度训练（AMP）'],
    ['KBS-13-03', '梯度检查点（Gradient Checkpointing）'],
    ['KBS-13-04', '学习率调度（Warmup + 余弦退火）']
  ]),
  block('KB-14', '分布式训练（DeepSpeed）', ['KB-04', 'KB-13'], ['分布式训练', 'DeepSpeed'], [
    ['KBS-14-00', '环境准备：安装 DeepSpeed 并配置多 GPU 环境', ['KBS-13-00']],
    ['KBS-14-01', '掌握 DeepSpeed 的 ZeRO-1/2/3 优化策略配置'],
    ['KBS-14-02', '理解数据并行与模型并行的区别'],
    ['KBS-14-03', '能够编写 ds_config.json 并启动 DeepSpeed 训练']
  ]),
  block('KB-15', '模型推理部署（vLLM）', ['KB-08', 'KB-11'], ['vLLM', '推理部署'], [
    ['KBS-15-00', '环境准备：安装 vLLM 并确认 CUDA 与 PyTorch 兼容', ['KBS-04-00', 'KBS-11-00']],
    ['KBS-15-01', '掌握模型生成策略（贪心搜索、束搜索）'],
    ['KBS-15-02', 'Top-p/Top-k 采样、Temperature 温度系数'],
    ['KBS-15-03', '理解 vLLM 的 PagedAttention 与 Continuous Batching 机制'],
    ['KBS-15-04', '使用 vLLM 加载模型并配置关键参数进行推理部署']
  ])
];

export function findKnowledgeGraphBlock(blockId?: string, ...names: Array<string | undefined>) {
  if (blockId) {
    const byId = knowledgeGraphBlocks.find((item) => item.block_id === blockId);
    if (byId) return byId;
  }
  const normalizedNames = names.filter(Boolean).map((name) => name!.trim().toLowerCase());
  return knowledgeGraphBlocks.find((item) => {
    const candidates = [item.block_name, ...item.aliases].map((name) => name.toLowerCase());
    return normalizedNames.some((name) => candidates.some((candidate) => name === candidate || name.includes(candidate) || candidate.includes(name)));
  });
}

export function getKnowledgeGraphBlockName(blockId: string) {
  return knowledgeGraphBlocks.find((item) => item.block_id === blockId)?.block_name || blockId;
}
