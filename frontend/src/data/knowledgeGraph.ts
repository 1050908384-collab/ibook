export type KnowledgeGraphSubtopic = {
  subtopic_id: string;
  subtopic_name: string;
  sequence_index: number;
};

export type KnowledgeGraphBlock = {
  block_id: string;
  block_name: string;
  prerequisites: string[];
  aliases: string[];
  subtopics: KnowledgeGraphSubtopic[];
};

const block = (
  block_id: string,
  block_name: string,
  prerequisites: string[],
  aliases: string[],
  subtopicNames: string[]
): KnowledgeGraphBlock => ({
  block_id,
  block_name,
  prerequisites,
  aliases,
  subtopics: subtopicNames.map((subtopic_name, index) => ({
    subtopic_id: `KBS-${block_id.slice(3)}-${String(index + 1).padStart(2, '0')}`,
    subtopic_name,
    sequence_index: index + 1
  }))
});

// Frontend preview fallback, mirrored from data/seeds/knowledge_*.json.
export const knowledgeGraphBlocks: KnowledgeGraphBlock[] = [
  block('KB-01', 'Linux环境与命令行操作', [], ['Linux基础', '命令行操作'], [
    '常用Shell命令', '环境变量配置', 'conda/pip虚拟环境管理', 'nvidia-smi显存监控', '实际操作与练习'
  ]),
  block('KB-02', 'Python编程基础', ['KB-01'], ['Python基础', '编程能力'], [
    '语法', '数据结构', '函数与类与对象', '文件与异常处理', '装饰器与生成器', '综合练习与实战'
  ]),
  block('KB-03', '数据处理基础', ['KB-02'], ['数据处理', '数据分析'], [
    '使用Pandas进行数据清洗', '格式转换（JSON/CSV读写）', 'NumPy进行基本数组运算', '正则表达式进行文本预处理', '综合练习'
  ]),
  block('KB-04', '深度学习框架PyTorch基础', ['KB-02', 'KB-03'], ['PyTorch', '深度学习'], [
    '张量操作', '自动求导机制', 'torch.nn模块', 'Dataset/DataLoader数据装载', '优化器（如AdamW）与损失函数', '训练循环编写', '模型保存与加载（checkpoint）'
  ]),
  block('KB-05', '自然语言处理（NLP）基础概念', ['KB-02', 'KB-03'], ['NLP基础', '文本表示'], [
    '理解分词、词性标注、命名实体识别（NER）等基础任务', '掌握文本的数值化表示方法（Bag-of-Words、TF-IDF、Word2Vec）'
  ]),
  block('KB-06', '注意力机制', ['KB-04'], ['注意力机制'], [
    '理解注意力机制的核心概念（Query、Key、Value）', '掌握Self-Attention与Multi-Head Attention的计算原理与公式'
  ]),
  block('KB-07', '子词切分与分词器（Tokenizer）', ['KB-05'], ['Tokenizer', '子词切分'], [
    '理解BPE、WordPiece、SentencePiece算法原理', '掌握HuggingFace Tokenizer的加载与使用', '熟悉padding、truncation、max_length等关键参数', '了解ChatTemplate对话模板格式'
  ]),
  block('KB-08', 'Transformer架构（Decoder-Only）', ['KB-04', 'KB-06'], ['Transformer', 'GPT'], [
    '掌握Transformer的Encoder-Decoder结构', '重点理解Decoder-Only架构（如GPT/Llama/Qwen）', 'RMSNorm与残差连接', '前馈网络（FFN）与SwiGLU激活函数', '旋转位置编码（RoPE）', 'KV Cache机制', '前向传播实验（加载GPT-2验证理解）'
  ]),
  block('KB-09', '大语言模型（LLM）核心特性', ['KB-08'], ['LLM', '大模型'], [
    '理解涌现能力', '上下文学习（In-Context Learning）', '指令遵循（Instruction Following）', '逐步推理（Step-by-Step Reasoning）', '幻觉问题', '综合理解与案例'
  ]),
  block('KB-10', 'LLM三阶段训练流程', ['KB-08', 'KB-09'], ['LLM训练', 'SFT', 'RLHF'], [
    '掌握预训练（Pretrain）整体流程与目标', '有监督微调（SFT）整体流程与目标', '基于人类反馈的强化学习（RLHF）整体流程与目标', '三阶段串联理解'
  ]),
  block('KB-11', 'HuggingFace生态工具链', ['KB-04', 'KB-07'], ['HuggingFace', 'transformers'], [
    '掌握transformers库的AutoModel/AutoTokenizer加载方式', 'datasets库的数据处理（map/filter/shuffle）', 'Trainer训练器的使用及TrainingArguments参数配置'
  ]),
  block('KB-12', '参数高效微调（LoRA与QLoRA）', ['KB-04', 'KB-11'], ['LoRA', 'QLoRA', 'PEFT', '参数高效微调'], [
    '理解LoRA的低秩分解原理', 'Rank/Alpha参数作用', '掌握PEFT库的LoraConfig配置（target_modules、r、lora_alpha、lora_dropout）', '理解QLoRA的4-bit量化（NF4、双重量化）与分页优化器原理', '掌握LoRA权重合并与导出（merge_and_unload）', '实战演练'
  ]),
  block('KB-13', 'PyTorch进阶训练技巧', ['KB-04'], ['PyTorch进阶', '训练优化'], [
    '掌握梯度累积（Gradient Accumulation）', '混合精度训练（AMP）', '梯度检查点（Gradient Checkpointing）', '学习率调度（Warmup + 余弦退火）'
  ]),
  block('KB-14', '分布式训练（DeepSpeed）', ['KB-04', 'KB-13'], ['分布式训练', 'DeepSpeed'], [
    '掌握DeepSpeed的ZeRO-1/2/3优化策略配置', '理解数据并行与模型并行的区别', '能够编写ds_config.json并启动deepspeed训练'
  ]),
  block('KB-15', '模型推理部署（vLLM）', ['KB-08', 'KB-11'], ['vLLM', '推理部署'], [
    '掌握模型生成策略（贪心搜索、束搜索）', 'Top-p/Top-k采样、Temperature温度系数', '理解vLLM的PagedAttention与Continuous Batching机制', '能够使用vLLM加载模型并配置gpu_memory_utilization等关键参数进行推理部署'
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
