import { CSSProperties, FormEvent, PointerEvent as ReactPointerEvent, useEffect, useMemo, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import { normalizeMarkdownMath } from './utils/markdown';
import type { KeyboardEvent as ReactKeyboardEvent } from 'react';
import {
  AlertTriangle,
  Award,
  BookOpen,
  Bot,
  Brain,
  CalendarDays,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ChevronUp,
  Circle,
  Clock,
  Code2,
  Copy,
  ExternalLink,
  FileQuestion,
  Folder,
  Gauge,
  LayoutDashboard,
  ListTree,
  LocateFixed,
  Loader2,
  MessageCircle,
  Minus,
  PanelLeftClose,
  PanelLeftOpen,
  Pause,
  Pencil,
  Play,
  Plus,
  RefreshCw,
  Search,
  Send,
  Settings,
  Sparkles,
  SquarePen,
  Target,
  Trash2,
  UserRound,
  X
} from 'lucide-react';
import {
  apiClient,
  getClientUserId,
  isUiPreviewMode,
  type ApiLearningPackage,
  type ApiProfile,
  type ApiStage,
  type AssessmentAttempt,
  type GoalBlock,
  type SessionListItem
} from './api/client';
import { findKnowledgeGraphBlock, knowledgeGraphBlocks, type KnowledgeGraphBlock } from './data/knowledgeGraph';
import FoundationPickerModal from './components/FoundationPickerModal';
import KnowledgeNetwork from './components/KnowledgeNetwork';
import { useGenerationProgress, type GenStagePreview } from './components/GenerationProgress';
import OnboardingPage from './pages/OnboardingPage';
import ProfileConfirmPage from './pages/ProfileConfirmPage';
import {
  emptyOnboardingForm,
  formDataFromProfile,
  OnboardingField,
  OnboardingFormData
} from './utils/buildDiagnosticMessage';
import {
  createAgentStatusMap,
  finishAgents,
  restoreAgentStatuses,
  startAgents,
  type AgentId,
  type AgentStatusMap
} from './utils/agentFlowState';
import { filterProjectsForDashboard, formatProjectCreatedDate, sortProjectsForDashboard } from './utils/learningProjects';
import { isHistoricalStageReview, resolveStageResultActionMode } from './utils/stageResultActions';

type View = 'chat' | 'learn' | 'settings' | 'result';

function toKnowledgeGraphBlocks(blocks: GoalBlock[]): KnowledgeGraphBlock[] {
  return blocks.map((block) => ({
    block_id: block.block_id,
    block_name: block.block_name,
    aliases: [],
    prerequisites: [],
    subtopics: block.subtopics.map((topic, index) => ({
      subtopic_id: topic.subtopic_id,
      subtopic_name: topic.subtopic_name,
      sequence_index: topic.sequence_index ?? index,
      prerequisites: topic.prerequisites ?? []
    }))
  }));
}

type WorkflowState = 'onboarding' | 'diagnosing' | 'needs_more_info' | 'profile_ready' | 'starting_stage' | 'learning';
type Message = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  note?: string;
  reviewSummary?: {
    technical_passed?: boolean;
    quality_passed?: boolean;
    iterations?: number;
    status?: string;
  };
};
type KnowledgeSubtopic = {
  subtopic_id?: string;
  subtopic_name: string;
  sequence_index?: number;
  status?: string;
  progress?: number;
  prerequisites?: string[];
};
type KnowledgeGap = {
  skill: string;
  block_id?: string;
  block_name?: string;
  target_subtopic_ids?: string[];
  subtopics?: KnowledgeSubtopic[];
  target_subtopics?: KnowledgeSubtopic[];
  level?: string;
  status?: string;
  progress?: number;
  prerequisites?: string[];
};
type Profile = {
  grade?: string;
  skill_tags?: string[];
  foundation_confirmed?: boolean;
  knowledge_gaps?: KnowledgeGap[];
  goal?: string;
  daily_hours?: number;
  learning_style?: string;
  hardware?: string;
  learning_rhythm?: string;
  confidence_index?: number;
  confusion_markers?: string[];
  confidence?: number;
  version?: number;
};
type StageInfo = {
  stage_id?: string;
  stage_num?: number;
  block_id?: string;
  block_name?: string;
  subtopic_id?: string;
  subtopic_name?: string;
  subtopic_ids?: string[];
  subtopic_names?: string[];
  estimated_hours?: number;
  gap_skill?: string;
  prerequisites?: string[];
  status?: string;
  accuracy?: number;
  strategy_used?: string;
};
type TestQuestion = {
  question_id: string;
  question_type?: string;
  question_text: string;
  options?: Array<string | { key?: string; text?: string }>;
  hint?: string;
};
type LearningResource = {
  resource_id?: string;
  resource_type?: 'video' | 'article' | 'documentation' | 'code';
  source_id?: string;
  source_type?: string;
  title?: string;
  url?: string;
  platform?: string;
  description?: string;
  recommendation_level?: string;
  role?: string;
  order?: number;
  content_ref?: string;
  locator?: string;
  content_summary?: string;
  metadata_json?: Record<string, unknown>;
  subtopic_id?: string;
};
type LearningPackage = {
  subtopic_name?: string;
  subtopic_ids?: string[];
  subtopic_names?: string[];
  tutorial?: string;
  code_examples?: Array<{ title?: string; code?: string; language?: string }>;
  questions?: TestQuestion[];
  block_name?: string;
  learning_resources?: LearningResource[];
  review_summary?: Record<string, unknown>;
  sources?: LearningResource[];
  recommended_videos?: LearningResource[];
  recommended_articles?: LearningResource[];
};
type StageResult = {
  status?: string;
  accuracy?: number;
  bypass_assessment?: boolean;
  stage_summary?: string;
  next_strategy?: string;
  final_report?: Record<string, unknown>;
  can_plan_next_stage?: boolean;
  all_mastered?: boolean;
  session_status?: string;
  phase?: string;
  available_actions?: string[];
};
type LearningProject = {
  sessionId: string;
  goal: string;
  stageName: string;
  stageNum: number;
  phase?: string;
  createdAt?: number;
  updatedAt: number;
  profile?: Profile;
  stages?: StageInfo[];
  targetTotalHours?: number;
  targetDailyHours?: number;
  averageAccuracy?: number | null;
  overallProgress?: number;
};
type StoredStageConversation = {
  id: string;
  sessionId: string;
  stage: StageInfo;
  messages: Message[];
  updatedAt: number;
};
type WorkspaceSnapshot = {
  sessionId: string | null;
  profile: Profile;
  stage: StageInfo | null;
  learningPackage: LearningPackage | null;
  messages: Message[];
  selectedConversationId: string;
  input: string;
  result: StageResult | null;
  plannedStage: StageInfo | null;
  timerSeconds: number;
  timerRunning: boolean;
  materialPanelOpen: boolean;
  agentStatuses: AgentStatusMap;
};

type AssessmentState = {
  attemptId: string | null;
  questions: TestQuestion[];
};

type AssessmentReviewQuestion = TestQuestion & {
  user_answer?: string;
  correct_answer?: string;
  explanation?: string;
  is_correct?: boolean;
};

type AssessmentReview = Omit<AssessmentAttempt, 'questions'> & {
  questions: AssessmentReviewQuestion[];
};

const USER_SCOPE = getClientUserId();
const SESSION_STORAGE_KEY = `sessionId:${USER_SCOPE}`;
const LAST_LEARNING_LOCATION_KEY = `lastLearningLocation:${USER_SCOPE}`;
const PROJECTS_STORAGE_KEY = `learningProjects:${USER_SCOPE}`;
const CONVERSATIONS_STORAGE_KEY = `stageConversations:${USER_SCOPE}`;
const OWNED_SESSION_IDS_STORAGE_KEY = `ownedSessionIds:${USER_SCOPE}`;
const DEMO_SESSION_ID = 'demo-preview';

function isLocalSessionId(sessionId?: string) {
  return Boolean(sessionId?.startsWith('local-goal-') || sessionId === DEMO_SESSION_ID);
}

function readOwnedSessionIds() {
  try {
    const stored = localStorage.getItem(OWNED_SESSION_IDS_STORAGE_KEY);
    const parsed = stored ? JSON.parse(stored) : [];
    return new Set(Array.isArray(parsed) ? parsed.filter((item): item is string => typeof item === 'string') : []);
  } catch {
    return new Set<string>();
  }
}

function rememberOwnedSessionId(sessionId: string) {
  if (!sessionId || isLocalSessionId(sessionId)) return;
  const sessionIds = readOwnedSessionIds();
  sessionIds.add(sessionId);
  localStorage.setItem(OWNED_SESSION_IDS_STORAGE_KEY, JSON.stringify([...sessionIds]));
}

function forgetOwnedSessionId(sessionId: string) {
  const sessionIds = readOwnedSessionIds();
  sessionIds.delete(sessionId);
  localStorage.setItem(OWNED_SESSION_IDS_STORAGE_KEY, JSON.stringify([...sessionIds]));
}

function isMissingSessionError(error: unknown) {
  if (!error || typeof error !== 'object' || !('response' in error)) return false;
  const response = (error as { response?: { status?: number; data?: unknown } }).response;
  return response?.status === 404;
}

function readLearningProjects(): LearningProject[] {
  try {
    const stored = localStorage.getItem(PROJECTS_STORAGE_KEY);
    const parsed = stored ? JSON.parse(stored) : [];
    return Array.isArray(parsed) ? parsed.filter((item) => item?.sessionId && item?.goal) : [];
  } catch {
    return [];
  }
}

function readStageConversations(): StoredStageConversation[] {
  try {
    const stored = localStorage.getItem(CONVERSATIONS_STORAGE_KEY);
    const parsed = stored ? JSON.parse(stored) : [];
    return Array.isArray(parsed) ? parsed.filter((item) => item?.id && item?.stage && Array.isArray(item?.messages)) : [];
  } catch {
    return [];
  }
}

const fallbackProfile: Profile = {
  grade: '算法工程师（转向大模型应用）',
  skill_tags: ['PyTorch', 'Transformer', 'Python'],
  knowledge_gaps: [
    { skill: '注意力机制', block_id: 'KB-06', block_name: '注意力机制', status: '学习中', progress: 0.68 },
    { skill: 'PEFT', block_id: 'KB-12', block_name: '参数高效微调（LoRA与QLoRA）', status: '未掌握', progress: 0.34 },
    { skill: 'LoRA', block_id: 'KB-12', block_name: '参数高效微调（LoRA与QLoRA）', status: '未开始', progress: 0.18 },
    { skill: '推理部署', block_id: 'KB-15', block_name: '模型推理部署（vLLM）', status: '未开始', progress: 0.08 }
  ],
  goal: '掌握 LoRA 微调并能独立部署到生产环境',
  daily_hours: 2,
  learning_style: 'practice',
  hardware: 'RTX 4060 8GB',
  learning_rhythm: 'medium',
  confidence_index: 0.76,
  confidence: 0.92,
  version: 2
};

const fallbackStage: StageInfo = {
  stage_id: 'STG-001',
  stage_num: 1,
  block_id: 'KB-06',
  block_name: 'Self-Attention 计算机制',
  subtopic_id: 'KBS-06-02',
  subtopic_name: '掌握Self-Attention与Multi-Head Attention的计算原理与公式',
  estimated_hours: 1,
  gap_skill: '注意力机制',
  prerequisites: ['Python', 'PyTorch 张量基础'],
  status: 'learning'
};

// Used only while a real stage response is still unavailable. It deliberately
// contains no mock topic, resource, or learning goal.
const unavailableStage: StageInfo = { status: 'pending', estimated_hours: 0 };

const fallbackPackage: LearningPackage = {
  block_name: 'Self-Attention 计算机制',
  tutorial:
    '## 核心直觉\n\nSelf-Attention 让序列中的每个位置都能主动寻找与自己最相关的信息。它不是平均读取所有词，而是为不同位置动态计算一组权重。\n\n## 缩放点积注意力\n\n`Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V`\n\n除以 `sqrt(d_k)` 是为了避免维度增大后点积值过大，导致 Softmax 梯度过小。\n\n## 和 LoRA 的关系\n\nLoRA 常被注入到生成 Q、K、V 的线性映射层中，因此理解注意力计算是后续学习参数高效微调的前置基础。',
  code_examples: [
    {
      title: 'scaled_dot_product_attention.py',
      language: 'python',
      code: `import torch

def scaled_dot_product_attention(q, k, v):
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1)
    weights = torch.softmax(scores / d_k ** 0.5, dim=-1)
    return weights @ v, weights`
    }
  ],
  recommended_videos: [],
  recommended_articles: [
    {
      resource_id: 'demo-lora-paper',
      title: 'LoRA: Low-Rank Adaptation of Large Language Models',
      url: 'https://arxiv.org/abs/2106.09685',
      content_summary: 'LoRA 原始论文，介绍低秩适配的核心方法与实验结果。'
    }
  ],
  questions: [
    {
      question_id: 'Q-DEMO-1',
      question_type: '选择',
      question_text: '在缩放点积注意力中，除以 sqrt(d_k) 的主要目的是什么？',
      options: ['减少参数量', '避免点积过大导致 Softmax 梯度过小', '增加注意力头数量', '消除位置编码']
    },
    {
      question_id: 'Q-DEMO-2',
      question_type: '选择',
      question_text: '哪两个矩阵的点积决定注意力权重？',
      options: ['Q 与 K', 'K 与 V', 'Q 与 V', 'V 与输出']
    },
    {
      question_id: 'Q-DEMO-3',
      question_type: '选择',
      question_text: 'LoRA 通常可以注入 Transformer 的哪个位置？',
      options: ['数据加载器', 'Q、K、V 的线性映射层', '损失函数', 'Tokenizer 词表']
    }
  ]
};

const initialMessages: Message[] = [
  {
    id: 'hello',
    role: 'assistant',
    content: '你好，我是你的 AI 实训导师。告诉我你的学习基础、目标、每天能投入的时间，以及你喜欢的学习方式。'
  }
];

const demoMessages: Message[] = [
  {
    id: 'demo-welcome',
    role: 'assistant',
    content: '欢迎回来。当前目标是掌握 LoRA 微调与部署，我们先从 Self-Attention 的计算机制开始。理解 Q、K、V 如何形成注意力权重，会让后面的低秩适配更容易上手。',
    note: '阶段导学'
  },
  {
    id: 'demo-question',
    role: 'user',
    content: '为什么注意力分数需要除以 sqrt(d_k)？',
    note: '已记录'
  },
  {
    id: 'demo-answer',
    role: 'assistant',
    content: '当维度 d_k 增大时，Q 与 K 点积的方差也会变大，Softmax 更容易进入饱和区，梯度随之变小。除以 sqrt(d_k) 可以把分数尺度稳定下来，让训练更平稳。',
    note: '审核通过'
  }
];

function createDemoProject(): LearningProject {
  const now = Date.now();
  return {
    sessionId: DEMO_SESSION_ID,
    goal: fallbackProfile.goal || 'LoRA 微调与部署',
    stageName: fallbackStage.block_name || '第一阶段',
    stageNum: fallbackStage.stage_num ?? 1,
    createdAt: now,
    updatedAt: now,
    profile: fallbackProfile,
    stages: [fallbackStage]
  };
}

function createDemoConversation(): StoredStageConversation {
  return {
    id: `${DEMO_SESSION_ID}:${fallbackStage.stage_num ?? 1}`,
    sessionId: DEMO_SESSION_ID,
    stage: fallbackStage,
    messages: demoMessages,
    updatedAt: Date.now()
  };
}

const statusText: Record<string, string> = {
  ready: '已生成',
  pending: '待开始',
  learning: '学习中',
  tested: '已测试',
  completed: '已完成',
  retreat: '需回顾',
  active: '进行中',
  abandoned: '已放弃'
};

const styleText: Record<string, string> = {
  theory: '理论型',
  practice: '实操型',
  mixed: '混合型'
};

const sidebarAgents = [
  {
    id: 'agent1',
    number: '1',
    title: '学情诊断与规划',
    desc: '画像诊断 / 阶段规划',
    icon: Brain,
    tone: 'blue'
  },
  {
    id: 'agent3',
    number: '3',
    title: '个性化内容生成',
    desc: '生成教程、代码与测试题',
    icon: BookOpen,
    tone: 'violet'
  },
  {
    id: 'agent4',
    number: '4',
    title: '技术与教学双审',
    desc: '技术正确性 / 学习适配度',
    icon: Check,
    tone: 'orange'
  },
  {
    id: 'agent6',
    number: '6',
    title: '学情决策与推进',
    desc: '更新画像 / 确定后续策略',
    icon: Gauge,
    tone: 'green'
  }
] as const;

function pct(value = 0) {
  return Math.round(Math.max(0, Math.min(1, value)) * 100);
}

function shortId() {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

/** 将原始文章中的（①）或（①+②）引用转为带原文来源标题的悬停标记。 */
function addCitationTooltips(markdown: string, resources: LearningResource[]) {
  const markerToIndex: Record<string, number> = {
    '①': 0, '②': 1, '③': 2, '④': 3, '⑤': 4,
    '⑥': 5, '⑦': 6, '⑧': 7, '⑨': 8, '⑩': 9,
  };
  return markdown.replace(/（([①②③④⑤⑥⑦⑧⑨⑩](?:\s*\+\s*[①②③④⑤⑥⑦⑧⑨⑩])*)）/g, (full, markers: string) => {
    const renderedMarkers = markers.split(/(\s*\+\s*)/).map((part) => {
      const index = markerToIndex[part];
      if (index === undefined) return part;
      const title = resources[index]?.title;
      return title ? `[${part}](#source-${index + 1} "来源：${title.replace(/"/g, "'")}")` : part;
    }).join('');
    return `（${renderedMarkers}）`;
  });
}

function createStageConversationMessages(nextStage: StageInfo, nextProfile?: Profile | null): Message[] {
  const stageName = nextStage.block_name || `学习阶段 ${nextStage.stage_num ?? 1}`;
  const subtopicName = nextStage.subtopic_name || nextStage.gap_skill;
  return [{
    id: `stage-welcome-${shortId()}`,
    role: 'assistant',
    content: `欢迎进入${stageName}。${subtopicName ? `本阶段将围绕「${subtopicName}」展开。` : ''}你可以在这里记录问题并获得实时答疑。`,
    note: nextProfile?.goal ? `目标：${nextProfile.goal}` : '阶段导学'
  }];
}

function getErrorMessage(error: unknown) {
  if (typeof error === 'object' && error && 'response' in error) {
    const response = (error as { response?: { data?: { detail?: unknown } } }).response;
    if (response?.data?.detail) return String(response.data.detail);
  }
  if (typeof error === 'object' && error && 'message' in error) {
    return String((error as { message?: unknown }).message);
  }
  return '后端暂时不可用，已切换为本地演示数据。';
}

/**
 * 从 axios 错误中抽取结构化后端业务错误码。
 * FastAPI HTTPException(detail={code, message}) 会包装成
 * `response.data = {detail: {code, message}}`；老接口或参数校验失败时
 * `response.data.detail` 为字符串。两者都要兼容。
 */
function getApiErrorCode(error: unknown): string | null {
  if (typeof error !== 'object' || !error || !('response' in error)) return null;
  const err = error as { response?: { data?: { detail?: unknown } } };
  const detail = err?.response?.data?.detail;
  if (!detail) return null;
  if (typeof detail === 'object' && detail && 'code' in detail) {
    const code = (detail as { code?: unknown }).code;
    return typeof code === 'string' ? code : null;
  }
  return null;
}

function formatTimer(totalSeconds: number) {
  const safeSeconds = Math.max(0, totalSeconds);
  const hours = Math.floor(safeSeconds / 3600);
  const minutes = Math.floor((safeSeconds % 3600) / 60);
  const seconds = safeSeconds % 60;
  if (hours > 0) {
    return `${hours}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
  }
  return `${minutes}:${String(seconds).padStart(2, '0')}`;
}

function Sidebar({
  view,
  setView,
  agentStatuses,
  collapsed,
  onToggleCollapsed
}: {
  view: View;
  setView: (view: View) => void;
  agentStatuses: AgentStatusMap;
  collapsed: boolean;
  onToggleCollapsed: () => void;
}) {
  const nav = [
    { id: 'learn' as View, label: '学习仪表盘', icon: LayoutDashboard },
    { id: 'chat' as View, label: '学习对话', icon: MessageCircle },
    { id: 'settings' as View, label: '设置', icon: Settings }
  ];

  return (
    <aside className={`app-sidebar ${collapsed ? 'collapsed' : ''}`}>
      <div className="brand">
        <div className="brand-mark">
          <Sparkles size={22} />
        </div>
        <div className="brand-copy">
          <strong>AI Agent Trainer</strong>
          <span>个性化学习工作台</span>
        </div>
        <button
          className="sidebar-toggle"
          onClick={onToggleCollapsed}
          aria-label={collapsed ? '展开主导航' : '收起主导航'}
          title={collapsed ? '展开主导航' : '收起主导航'}
        >
          {collapsed ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
        </button>
      </div>

      <nav className="nav-list" aria-label="主导航">
        <span>工作台</span>
        {nav.map((item) => {
          const Icon = item.icon;
          const active = view === item.id || (view === 'result' && item.id === 'learn');
          return (
            <button key={item.id} className={active ? 'active' : ''} onClick={() => setView(item.id)} title={collapsed ? item.label : undefined}>
              <Icon size={18} />
              <span className="nav-label">{item.label}</span>
            </button>
          );
        })}
      </nav>

      <div className="agent-mini">
        <div className="mini-title">
          <Bot size={16} />
          Agent协同流程
        </div>
        <div className="agent-flow">
          {sidebarAgents.map((agent) => {
            const Icon = agent.icon;
            const state = agentStatuses[agent.id];
            const statusLabel = agent.id === 'agent4' && state === 'running'
              ? '审核中：技术正确性审核'
              : agent.id === 'agent4' && state === 'completed'
                ? '已完成：技术正确性、教学质量均已审核'
              : state === 'running'
              ? '正在运行'
              : state === 'completed'
                ? '已完成'
                : state === 'failed'
                  ? '执行失败'
                  : '待调用';

            return (
              <div className={`agent-flow-card ${state}`} key={agent.id}>
                <span className={`agent-icon ${agent.tone}`}>
                  <Icon size={18} />
                </span>
                <div className="agent-copy">
                  <b>Agent{agent.number}</b>
                  <strong>{agent.title}</strong>
                  <small>{agent.desc}</small>
                  <small className={`agent-state-label ${state}`}>{statusLabel}</small>
                </div>
                <span className={`agent-status ${state}`} aria-label={statusLabel} title={statusLabel}>
                  {state === 'running'
                    ? <Loader2 size={17} />
                    : state === 'completed'
                      ? <Check size={15} />
                      : state === 'failed'
                        ? <X size={15} />
                        : <Circle size={12} />}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </aside>
  );
}

function Topbar({
  title,
  subtitle,
  loading,
  error,
  onReturnToChat,
  onOpenLearningPath,
  onOpenProfile
}: {
  title: string;
  subtitle: string;
  loading: boolean;
  error: string | null;
  onReturnToChat?: () => void;
  onOpenLearningPath?: () => void;
  onOpenProfile?: () => void;
}) {
  return (
    <header className="topbar">
      <div>
        <h1>{title}</h1>
        <p>{subtitle}</p>
      </div>
      <div className="top-actions">
        {error && <span className="error-pill">{error}</span>}
        {loading && (
          <span className="session-pill">
            <Loader2 className="spin" size={14} />
            Agent 处理中
          </span>
        )}
        {onReturnToChat && (
          <button className="topbar-tool-button" onClick={onReturnToChat} disabled={loading}>
            <MessageCircle size={15} />
            返回学习对话
          </button>
        )}
        {onOpenLearningPath && (
          <button className="topbar-tool-button" onClick={onOpenLearningPath} disabled={loading}>
            <ListTree size={15} />
            阶段知识点
          </button>
        )}
        {onOpenProfile && (
          <button className="topbar-tool-button" onClick={onOpenProfile} disabled={loading}>
            <UserRound size={15} />
            查看画像
          </button>
        )}
      </div>
    </header>
  );
}

function DurationField({
  label,
  value,
  onChange,
  min,
  max
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  min: number;
  max: number;
}) {
  const numericValue = Number(value);
  const invalid = value !== '' && (!Number.isFinite(numericValue) || numericValue < min || numericValue > max);
  const step = (delta: number) => {
    const current = Number(value);
    const next = Math.min(max, Math.max(min, (Number.isFinite(current) ? current : min) + delta));
    onChange(String(Math.round(next * 2) / 2));
  };
  return (
    <label className="duration-field">
      <span>{label}</span>
      <div>
        <button type="button" onClick={() => step(-0.5)} aria-label={`${label}减少0.5小时`}><Minus size={13} /></button>
        <input type="number" min={min} max={max} step="0.5" value={value} onChange={(event) => onChange(event.target.value)} aria-label={label} />
        <em>小时</em>
        <button type="button" onClick={() => step(0.5)} aria-label={`${label}增加0.5小时`}><Plus size={13} /></button>
      </div>
      {invalid && <small className="field-error">请输入 {min} 到 {max} 小时之间的数值。</small>}
    </label>
  );
}

function ProfilePanel({
  stage,
  projects,
  currentSessionId,
  overallProgress,
  dailyProgress,
  selectedConversationId,
  onSelectProjectStage,
  onCreateGoal,
  loading,
  collapsed,
  onToggleCollapsed,
  timerSeconds,
  timerRunning,
  onToggleTimer,
  onStartStage,
  onDeleteLastStage
}: {
  stage: StageInfo;
  projects: LearningProject[];
  currentSessionId: string | null;
  overallProgress?: number;
  dailyProgress: number;
  selectedConversationId: string;
  onSelectProjectStage: (sessionId: string, stageNum: number, stageId?: string) => void;
  onCreateGoal: () => void;
  loading: boolean;
  collapsed: boolean;
  onToggleCollapsed: () => void;
  timerSeconds: number;
  timerRunning: boolean;
  onToggleTimer: () => void;
  onStartStage: () => void;
  onDeleteLastStage: (sessionId: string) => Promise<void>;
}) {
  const [expandedGoals, setExpandedGoals] = useState<Record<string, boolean>>({});
  const [pendingDeleteStage, setPendingDeleteStage] = useState<{ sessionId: string; stageNum: number; goal: string; blockName: string } | null>(null);
  const timerLabel = timerRunning
    ? '暂停提醒'
    : timerSeconds <= 0
      ? '重新开始'
      : '继续计时';
  const toggleGoal = (sessionId: string) => {
    setExpandedGoals((prev) => ({
      ...prev,
      [sessionId]: !(prev[sessionId] ?? sessionId === currentSessionId)
    }));
  };
  return (
    <>
    <aside className={`context-panel ${collapsed ? 'collapsed' : ''}`}>
      <div className="context-panel-controls">
        {!collapsed && <span>对话导航</span>}
        <button
          onClick={onToggleCollapsed}
          aria-label={collapsed ? '展开学习对话导航' : '收起学习对话导航'}
          title={collapsed ? '展开学习对话导航' : '收起学习对话导航'}
        >
          {collapsed ? <PanelLeftOpen size={17} /> : <PanelLeftClose size={17} />}
        </button>
      </div>
      <div className="context-panel-body">
      <section className="context-card stage-card">
        <div className="stage-eyebrow">
          <span>当前学习包</span>
          <b>阶段 {stage.stage_num ?? 1}</b>
        </div>
        <h3>{stage.block_name || '等待阶段规划'}</h3>
        <p>{stage.gap_skill
          ? `围绕「${stage.gap_skill}」补齐关键知识点，并准备进入后续实操。`
          : stage.subtopic_name
            ? `当前聚焦：${stage.subtopic_name}`
            : '正在同步当前学习阶段。'}</p>
        <div className="stage-meta">
          <span>
            <Target size={15} />
            阶段进度 {pct(dailyProgress)}%
          </span>
          <span>
            <Target size={15} />
            {statusText[stage.status || 'pending'] || stage.status}
          </span>
        </div>
        <button className={`stage-timer-button ${timerRunning ? 'running' : ''}`} onClick={onToggleTimer}>
          {timerRunning ? <Pause size={17} /> : <Play size={17} />}
          <span>{timerLabel}</span>
          <b>{formatTimer(timerSeconds)}</b>
        </button>
      </section>

      <button
        className="context-new-goal-button"
        onClick={onCreateGoal}
        disabled={loading}
        aria-label="新建学习目标"
      >
        <SquarePen size={18} />
        <span>新建学习目标</span>
      </button>

      <section className="goal-stage-nav" aria-label="学习目标与阶段对话">
        <div className="goal-stage-nav-heading">
          <span>学习目标</span>
        </div>
        {sortProjectsForDashboard(projects, currentSessionId).map((project) => {
          const expanded = expandedGoals[project.sessionId] ?? project.sessionId === currentSessionId;
          const stagesByNumber = new Map<number, StageInfo>();
          for (const item of project.stages || []) stagesByNumber.set(item.stage_num ?? 1, item);
          const stages = [...stagesByNumber.values()].sort((a, b) => (a.stage_num ?? 1) - (b.stage_num ?? 1));
          return (
            <div className={`goal-nav-group ${project.sessionId === currentSessionId ? 'current' : ''}`} key={project.sessionId}>
              <button className="goal-nav-title" onClick={() => toggleGoal(project.sessionId)} aria-expanded={expanded}>
                <Folder size={17} />
                <span>{project.goal}</span>
              </button>
              {expanded && (
                <div className="goal-stage-list">
                  {stages.map((item, index) => {
                    const stageNum = item.stage_num ?? 1;
                    const conversationId = `${project.sessionId}:${stageNum}`;
                    const current = project.sessionId === currentSessionId && (
                      (item.stage_id && stage.stage_id) ? item.stage_id === stage.stage_id : stageNum === (stage.stage_num ?? 1)
                    );
                    const isLastStage = index === stages.length - 1;
                    return (
                      <div
                        className={`goal-stage-item ${selectedConversationId === conversationId ? 'active' : ''} ${current ? 'current' : ''}`}
                        key={conversationId}
                      >
                        <button
                          className="goal-stage-button"
                          onClick={() => onSelectProjectStage(project.sessionId, stageNum, item.stage_id)}
                        >
                          <span>
                            第 {stageNum} 阶段 · {item.block_name || '阶段对话'}
                          </span>
                          {current && <i aria-label="当前阶段" />}
                        </button>
                        {isLastStage && (
                          <button
                            className="goal-stage-delete-button"
                            type="button"
                            aria-label={`删除第 ${stageNum} 阶段`}
                            title="删除最后一个学习阶段"
                            disabled={loading}
                            onClick={() => setPendingDeleteStage({
                              sessionId: project.sessionId,
                              stageNum,
                              goal: project.goal,
                              blockName: item.block_name || '阶段对话',
                            })}
                          >
                            <Trash2 size={13} />
                          </button>
                        )}
                      </div>
                    );
                  })}
                  {stages.length === 0 && project.sessionId === currentSessionId ? (
                    <div className="goal-stage-empty">
                      <p>画像已完成，确认后即可生成第一阶段</p>
                      <button type="button" onClick={onStartStage} disabled={loading}>
                        {loading ? '正在生成…' : '确认画像并生成第一阶段'}
                      </button>
                    </div>
                  ) : stages.length === 0 ? (
                    <p>完成画像后生成第一阶段</p>
                  ) : null}
                </div>
              )}
            </div>
          );
        })}
      </section>
      </div>
    </aside>
    {pendingDeleteStage && (
      <div className="stage-delete-modal-overlay" role="dialog" aria-modal="true">
        <div className="stage-delete-modal">
          <div className="stage-delete-modal-header">
            <Trash2 size={18} />
            <span>删除学习阶段</span>
          </div>
          <p className="stage-delete-modal-body">
            确定删除「{pendingDeleteStage.goal}」的
            <strong>第 {pendingDeleteStage.stageNum} 阶段 · {pendingDeleteStage.blockName}</strong> 吗？
          </p>
          <p className="stage-delete-modal-hint">
            仅可删除最后一个学习阶段。若阶段全部删完，该学习目标也会一并移除，且不可恢复。
          </p>
          <div className="stage-delete-modal-actions">
            <button
              type="button"
              className="stage-delete-modal-cancel"
              onClick={() => setPendingDeleteStage(null)}
              disabled={loading}
            >
              取消
            </button>
            <button
              type="button"
              className="stage-delete-modal-confirm"
              onClick={() => {
                const target = pendingDeleteStage;
                setPendingDeleteStage(null);
                void onDeleteLastStage(target.sessionId);
              }}
              disabled={loading}
            >
              {loading ? '删除中…' : '确认删除'}
            </button>
          </div>
        </div>
      </div>
    )}
    </>
  );
}

function getResourceUrl(resource: LearningResource) {
  if (resource.url) return resource.url;
  if (resource.content_ref?.startsWith('http')) return resource.content_ref;
  if (resource.locator?.startsWith('http')) return resource.locator;
  return '';
}

// =====================================================================
// 视频嵌入解析与组件：B 站 BV / YouTube / mp4 直链 / 其他平台回退
// =====================================================================
type VideoKind = 'bilibili' | 'youtube' | 'native' | 'link-only';
type ParsedVideo =
  | { kind: 'bilibili'; embedUrl: string; bvid: string; page?: string; external: string }
  | { kind: 'youtube'; embedUrl: string; videoId: string; external: string }
  | { kind: 'native'; embedUrl: string; external: string }
  | { kind: 'link-only'; external: string };

function parseVideoUrl(rawUrl: string): ParsedVideo {
  if (!rawUrl) return { kind: 'link-only', external: '' };
  let url: URL | null = null;
  try {
    url = new URL(rawUrl);
  } catch {
    return { kind: 'link-only', external: rawUrl };
  }
  const host = url.hostname.replace(/^www\./, '');

  // --- B 站 ---
  if (host === 'bilibili.com' || host === 'm.bilibili.com') {
    const bvMatch = url.pathname.match(/\/video\/(BV[a-zA-Z0-9]+)/);
    const bvid = bvMatch ? bvMatch[1] : '';
    if (bvid) {
      const p = url.searchParams.get('p') || undefined;
      const pageParam = p ? `&page=${encodeURIComponent(p)}` : '';
      return {
        kind: 'bilibili',
        bvid,
        page: p,
        embedUrl: `https://player.bilibili.com/player.html?bvid=${bvid}${pageParam}&high_quality=1&danmaku=0&autoplay=0`,
        external: rawUrl,
      };
    }
  }
  if (host === 'b23.tv') {
    // b23.tv 短链：无法离线拿到 BV，只能回退链接；但仍尝试用 iframe 嵌入可能失败，所以直接 link-only 走外链
    return { kind: 'link-only', external: rawUrl };
  }

  // --- YouTube ---
  if (host === 'youtube.com' || host === 'm.youtube.com' || host === 'music.youtube.com') {
    const v = url.searchParams.get('v');
    if (v) {
      return {
        kind: 'youtube',
        videoId: v,
        embedUrl: `https://www.youtube.com/embed/${v}?rel=0&modestbranding=1`,
        external: rawUrl,
      };
    }
    const shortMatch = url.pathname.match(/^\/(?:embed|shorts|live)\/([A-Za-z0-9_-]{6,})/);
    if (shortMatch) {
      return {
        kind: 'youtube',
        videoId: shortMatch[1],
        embedUrl: `https://www.youtube.com/embed/${shortMatch[1]}?rel=0&modestbranding=1`,
        external: rawUrl,
      };
    }
  }
  if (host === 'youtu.be') {
    const id = url.pathname.replace(/^\//, '').split('/')[0];
    if (id) {
      return {
        kind: 'youtube',
        videoId: id,
        embedUrl: `https://www.youtube.com/embed/${id}?rel=0&modestbranding=1`,
        external: rawUrl,
      };
    }
  }

  // --- 原生 mp4 / webm / ogg ---
  if (/\.(mp4|webm|ogg|m4v|mov)(\?|#|$)/i.test(url.pathname + url.search)) {
    return { kind: 'native', embedUrl: rawUrl, external: rawUrl };
  }

  return { kind: 'link-only', external: rawUrl };
}

function useVideoPoster(parsed: ParsedVideo): string | undefined {
  // B 站没有稳定的离线可算封面 API，返回 undefined，让前端显示占位图标
  if (parsed.kind === 'youtube') {
    // 取 YouTube 标准中质量封面（存在性最好）
    return `https://i.ytimg.com/vi/${parsed.videoId}/hqdefault.jpg`;
  }
  return undefined;
}

function EmbeddedVideoCard({
  resource,
  index,
  getUrl,
}: {
  resource: LearningResource;
  index: number;
  getUrl: (r: LearningResource) => string;
}) {
  const external = getUrl(resource);
  const parsed = parseVideoUrl(external);
  const poster = useVideoPoster(parsed);
  const [activated, setActivated] = useState(false);

  const title = resource.title || `学习视频 ${index + 1}`;
  const summary = resource.content_summary || resource.description || '';

  return (
    <article className="material-video-card">
      <div
        className={`material-video-stage ${activated ? 'is-playing' : ''}`}
        onClick={() => !activated && parsed.kind !== 'link-only' && setActivated(true)}
      >
        {!activated && (
          <div className={`material-video-cover ${parsed.kind === 'link-only' ? 'is-link-only' : ''}`}>
            {parsed.kind !== 'link-only' && poster ? (
              <img src={poster} alt={title} loading="lazy" onError={(e) => { (e.currentTarget as HTMLImageElement).style.display = 'none'; }} />
            ) : (
              <div className="material-video-cover-fallback">
                {parsed.kind === 'bilibili' && <span className="material-video-badge bilibili">B站</span>}
                {parsed.kind === 'youtube' && <span className="material-video-badge youtube">YouTube</span>}
                {parsed.kind === 'native' && <span className="material-video-badge native">MP4</span>}
                {parsed.kind === 'link-only' && <span className="material-video-badge linkonly">外链</span>}
                <Play size={34} strokeWidth={2.4} />
              </div>
            )}
            {parsed.kind !== 'link-only' && (
              <button type="button" className="material-video-play" aria-label={`播放 ${title}`}>
                <Play size={22} fill="currentColor" />
              </button>
            )}
          </div>
        )}

        {activated && parsed.kind === 'bilibili' && (
          <iframe
            src={parsed.embedUrl}
            title={title}
            allow="autoplay; encrypted-media; fullscreen; picture-in-picture"
            allowFullScreen
            loading="lazy"
            referrerPolicy="no-referrer"
          />
        )}
        {activated && parsed.kind === 'youtube' && (
          <iframe
            src={parsed.embedUrl}
            title={title}
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
            allowFullScreen
            loading="lazy"
            referrerPolicy="no-referrer"
          />
        )}
        {activated && parsed.kind === 'native' && (
          <video controls preload="metadata" src={parsed.embedUrl} playsInline poster={poster} />
        )}
      </div>

      <div className="material-video-meta">
        <div className="material-video-title-row">
          <strong>{title}</strong>
          <a href={external || parsed.external || '#'} target="_blank" rel="noreferrer" className="material-video-external" title="新标签打开">
            <ExternalLink size={13} /> 打开
          </a>
        </div>
        {summary && <p>{summary}</p>}
      </div>
    </article>
  );
}

function uniqueResources(resources: LearningResource[]) {
  const seenIds = new Set<string>();
  // ★ 只按 (subtopic_id, URL) 二元组去重，而不是全局 URL 去重。
  // 之前的全局 URL 去重是错误的：同一个 URL 在多个知识点下的参考列表里出现是正常的，
  // 比如"拓展阅读"预告后续章节，此时必须允许每个 subtopic 各有一条。
  const seenSubtopicUrl = new Set<string>();
  return resources.filter((resource) => {
    const key = resource.resource_id || resource.source_id || getResourceUrl(resource) || resource.title;
    const url = getResourceUrl(resource);
    const subtopicId = resource.subtopic_id || '';
    // 先按 resource_id 去重
    if (key && seenIds.has(key)) return false;
    // 再按 (subtopic_id, URL) 去重：同一个知识点下同一个 URL 只保留一条
    if (url) {
      const pair = `${subtopicId}\u0001${url}`;
      if (seenSubtopicUrl.has(pair)) return false;
      seenSubtopicUrl.add(pair);
    }
    if (key) seenIds.add(key);
    return true;
  });
}

function resolveStageSubtopics(profile: Profile, stage: StageInfo): KnowledgeSubtopic[] {
  const directSubtopics = (stage.subtopic_names || []).map((subtopic_name, index) => ({
    subtopic_name,
    subtopic_id: stage.subtopic_ids?.[index],
    sequence_index: index
  })).filter((topic) => topic.subtopic_name);
  if (directSubtopics.length > 0) return directSubtopics;

  const currentGraphBlock = findKnowledgeGraphBlock(stage.block_id, stage.block_name, stage.gap_skill);
  const currentGap = (profile.knowledge_gaps || []).find((gap) => (
    Boolean(stage.block_id && gap.block_id === stage.block_id)
    || gap.skill === stage.gap_skill
    || gap.block_name === stage.block_name
    || Boolean(currentGraphBlock && findKnowledgeGraphBlock(gap.block_id, gap.block_name, gap.skill)?.block_id === currentGraphBlock.block_id)
  ));
  const apiSubtopics = currentGap?.subtopics || currentGap?.target_subtopics || [];
  const targetIds = new Set(currentGap?.target_subtopic_ids || []);
  const catalogSubtopics = currentGraphBlock?.subtopics || [];
  const selectedCatalogSubtopics = targetIds.size
    ? catalogSubtopics.filter((topic) => targetIds.has(topic.subtopic_id))
    : catalogSubtopics;
  return (apiSubtopics.length ? apiSubtopics : selectedCatalogSubtopics)
    .map((topic) => ({ ...topic }))
    .sort((a, b) => (a.sequence_index ?? 0) - (b.sequence_index ?? 0));
}

function topicKey(topic?: KnowledgeSubtopic | null) {
  return topic?.subtopic_id || topic?.subtopic_name || '';
}

// 去掉知识点名称尾部的章节标记（如 "（①）"、"（新增②）"）和开头序号前缀，
// 使前端 selectedTopicName 与后端 backfill_stage_tutorial 生成的 ## heading 对齐。
// 后端 _TRAILING_SECTION_MARK 去尾部括号标记，_HEADING_CRUMB 去面包屑前缀，
// 但 stage.subtopic_names 存的是原始未清洗的名称，不做同样清洗就无法精确匹配。
function _cleanTopicName(name: string): string {
  let s = name.trim().replace(/\s+/g, ' ');
  // 去尾部 （①②③） 等标记
  s = s.replace(/[\s\u3000]*[（(][^）)]*[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳][^）)]*[）)]\s*$/g, '');
  // 去尾部纯全角括号包裹
  if (s.startsWith('（') && s.endsWith('）')) {
    s = s.slice(1, -1).trim();
  }
  // 去开头序号前缀（如 "3 " / "3." / "3、"）
  s = s.replace(/^[\s\d①②③④⑤⑥⑦⑧⑨⑩一二三四五六七八九十]+[.、．:：)\]\s]*/, '').trim();
  return s;
}

function getTopicTutorial(tutorial: string | undefined, topicName: string) {
  if (!tutorial?.trim() || !topicName.trim()) return tutorial;
  const normalizedTopic = topicName.trim().replace(/\s+/g, ' ');
  const cleanedTopic = _cleanTopicName(normalizedTopic);
  const headingPattern = /^##\s+(.+?)\s*$/gm;
  const matches: { index: number; heading: string; contentStart: number; cleaned: string }[] = [];
  let m: RegExpExecArray | null;
  while ((m = headingPattern.exec(tutorial))) {
    const heading = m[1].trim().replace(/\s+/g, ' ');
    matches.push({
      index: m.index,
      heading,
      contentStart: m.index + m[0].length,
      cleaned: _cleanTopicName(heading),
    });
  }
  if (matches.length === 0) return tutorial;

  // 匹配优先级：1) 原始名称精确相等  2) 清洗后精确相等  3) 清洗后去掉括号后缀相等
  let exactIndex = matches.findIndex((x) => x.heading === normalizedTopic);
  if (exactIndex < 0) exactIndex = matches.findIndex((x) => x.cleaned === cleanedTopic);
  if (exactIndex < 0) {
    // 进一步去掉括号后缀再比
    const topicCore = cleanedTopic.replace(/[\s（(【\[].*$/, '').trim();
    if (topicCore && topicCore !== cleanedTopic) {
      exactIndex = matches.findIndex((x) => {
        const core = x.cleaned.replace(/[\s（(【\[].*$/, '').trim();
        return core === topicCore;
      });
    }
  }
  if (exactIndex < 0) return tutorial;
  const start = matches[exactIndex].contentStart;
  // 结束 = 下一个 ## heading 的起点（无论是什么 heading 都停止）
  const endIdx = exactIndex + 1 < matches.length
    ? matches[exactIndex + 1].index
    : tutorial.length;
  let slice = tutorial.slice(start, endIdx).trim();
  // 去掉开头可能存在的同名"空 H2"（外层 wrapper 与内层嵌套 heading 同名导致的"内容为空"段落）
  const dupHeading = matches[exactIndex].heading;
  const duplicateHeading = new RegExp(`^\\s*##\\s+${escapeRegex(dupHeading)}\\s*\\n`, 'm');
  const dupMatch = duplicateHeading.exec(slice);
  if (dupMatch && dupMatch.index === 0) {
    const afterDup = slice.slice(dupMatch.index + dupMatch[0].length);
    const nextHeading = afterDup.search(/^##\s+/m);
    slice = (nextHeading < 0 ? afterDup : afterDup.slice(0, nextHeading)).trim();
  }
  return slice || tutorial;
}

function escapeRegex(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function resolveSessionStage(session: {
  current_stage?: StageInfo | null;
  stage_info?: StageInfo | null;
  next_stage?: StageInfo | null;
}) {
  return session.current_stage || session.stage_info || session.next_stage || null;
}

function isLearningPhase(phase?: string) {
  return ['learning', 'stage_result', 'next_stage_ready', 'completed'].includes(phase || '');
}

function toProfile(profile?: ApiProfile | Profile | null): Profile | null {
  if (!profile) return null;
  return {
    ...profile,
    knowledge_gaps: Array.isArray(profile.knowledge_gaps) ? profile.knowledge_gaps as KnowledgeGap[] : undefined
  };
}

function toStageInfo(stage?: ApiStage | StageInfo | null): StageInfo | null {
  return stage ? stage as StageInfo : null;
}

function toLearningPackage(pkg?: ApiLearningPackage | LearningPackage | null): LearningPackage | null {
  if (!pkg) return null;
  const declaredResources = Array.isArray(pkg.learning_resources) ? pkg.learning_resources : [];
  const recommendedVideos = Array.isArray((pkg as LearningPackage).recommended_videos)
    ? ((pkg as LearningPackage).recommended_videos || []).map((video) => ({
      ...video,
      resource_type: video.resource_type || 'video'
    }))
    : [];
  // 合并（双保险）：后端 schemas.py / stages.py 理论上已经合并过，
  // 但为了兼容旧快照或其它端点，前端这里再合并一次，按 resource_id / url 去重。
  const seenIds = new Set<string>();
  const seenUrls = new Set<string>();
  const merged: LearningResource[] = [];
  for (const r of [...declaredResources, ...recommendedVideos]) {
    if (!r) continue;
    const id = String((r as any).resource_id || '').trim();
    const url = String((r as any).url || getResourceUrl(r as any) || '').trim();
    if (id && seenIds.has(id)) continue;
    if (url && seenUrls.has(url)) continue;
    if (id) seenIds.add(id);
    if (url) seenUrls.add(url);
    merged.push(r as LearningResource);
  }
  return {
    ...pkg,
    tutorial: pkg.tutorial ?? undefined,
    code_examples: Array.isArray(pkg.code_examples) ? pkg.code_examples as LearningPackage['code_examples'] : undefined,
    learning_resources: merged,
    questions: Array.isArray(pkg.questions) ? pkg.questions as TestQuestion[] : undefined
  };
}

function sessionSummaryToProject(item: SessionListItem): LearningProject | null {
  if (!item.session_id) return null;
  const profile = toProfile({
    goal: item.goal,
    grade: item.grade,
    skill_tags: item.skill_tags,
    knowledge_gaps: item.knowledge_gaps,
    daily_hours: item.daily_hours,
    learning_style: item.learning_style,
    version: item.version
  });
  return {
    sessionId: item.session_id,
    goal: item.goal || item.title || '未命名学习目标',
    stageName: item.current_stage_name || '等待阶段规划',
    stageNum: item.current_stage_num || 1,
    // The session-list endpoint is intentionally compact, but it still has
    // enough current-stage metadata to render the project navigation before
    // a browser-local cache or dashboard request is available.
    stages: item.current_stage_name && item.current_stage_num
      ? [{ stage_num: item.current_stage_num, block_name: item.current_stage_name }]
      : [],
    phase: item.phase,
    createdAt: item.created_at ? Date.parse(item.created_at) || undefined : undefined,
    updatedAt: item.updated_at ? Date.parse(item.updated_at) || Date.now() : Date.now(),
    profile: profile || undefined,
    targetTotalHours: item.target_total_hours,
    targetDailyHours: item.target_daily_hours ?? item.daily_hours,
    overallProgress: typeof item.progress === 'number' ? item.progress / 100 : undefined
  };
}

function qaItemsToMessages(items?: Array<{
  qa_id?: string;
  question?: string;
  answer?: string;
  review_summary?: {
    technical_passed?: boolean;
    quality_passed?: boolean;
    status?: string;
  };
}>): Message[] {
  if (!Array.isArray(items) || items.length === 0) return [];
  return items.flatMap((item, index) => {
    const seed = item.qa_id || `qa-${index}`;
    return [
      item.question ? { id: `${seed}-q`, role: 'user' as const, content: item.question } : null,
      item.answer ? {
        id: `${seed}-a`,
        role: 'assistant' as const,
        content: item.answer,
        note: item.review_summary?.technical_passed && item.review_summary?.quality_passed ? '审核通过' : item.review_summary?.status
      } : null
    ].filter(Boolean) as Message[];
  });
}

function matchesSubtopic(resource: LearningResource, subtopic?: KnowledgeSubtopic): boolean {
  if (!subtopic) return true;
  const subtopicId = subtopic.subtopic_id;
  const subtopicName = subtopic.subtopic_name?.trim();
  if (!subtopicId && !subtopicName) return true;
  const resourceSubtopicId = String((resource as any).subtopic_id ?? '').trim();

  // 1. 资源有精确 subtopic_id 时优先按 id 匹配
  if (resourceSubtopicId) {
    if (subtopicId && resourceSubtopicId === subtopicId) return true;
    if (subtopicId && resourceSubtopicId !== subtopicId) return false;
    // 当前 selectedSubtopic.subtopic_id 为空（stage.subtopic_ids 没同步/缺字段），
    // 但资源有精确 subtopic_id。此时前端无法通过 id 判断是否归属当前知识点，
    // 不能直接 return false（否则会把该阶段所有文章都隐藏）。
    return true;
  }
  // 2. 资源 subtopic_id 为空 / ""：语义是"block 级通用资源/兜底资源"，
  //    不是"归属错误的资源"。在 block 级层面推荐给整个 block 下所有知识点是合理的
  //    （例如 Python 阶段的一个"Python 全 3 小时入门"视频，语法/安装/集合小节都能看），
  //    所以不再视为"串台"，总是显示。
  return true;
}

function MaterialDocument({
  stage,
  learningPackage,
  selectedSubtopic,
  compact = false
}: {
  stage: StageInfo;
  learningPackage: LearningPackage;
  selectedSubtopic?: KnowledgeSubtopic;
  compact?: boolean;
}) {
  const codeExamples = learningPackage.code_examples || [];
  const recommendedVideos = (learningPackage.recommended_videos || []).map((video) => ({
    ...video,
    resource_type: video.resource_type || 'video'
  }));
  const learningResources = learningPackage.learning_resources?.length
    ? learningPackage.learning_resources
    : recommendedVideos;
  const scopedResources = learningResources.filter((res) => matchesSubtopic(res, selectedSubtopic));
  const videoResources = uniqueResources(
    scopedResources.filter((resource) => resource.resource_type === 'video' && Boolean(getResourceUrl(resource)))
  );
  const articleResources = uniqueResources(
    scopedResources.filter((resource) =>
      (resource.resource_type === 'article' || resource.resource_type === 'documentation') &&
      Boolean(resource.content_summary || getResourceUrl(resource))
    )
  );

  const codeResourceItems = uniqueResources(
    scopedResources.filter((resource) => resource.resource_type === 'code' && Boolean(resource.content_summary || getResourceUrl(resource)))
  );
  const selectedTopicName = selectedSubtopic?.subtopic_name
    || stage.subtopic_name
    || learningPackage.subtopic_name
    || stage.block_name
    || learningPackage.block_name
    || '未命名学习材料';
  const topicTutorial = getTopicTutorial(learningPackage.tutorial, selectedTopicName);
  const hasTextContent = Boolean(
    topicTutorial?.trim()
    || codeExamples.length
    || articleResources.length
    || codeResourceItems.length
  );
  return (
    <article className={`material-document ${compact ? 'compact' : ''}`}>
      <div className="material-document-meta">
        <span>阶段 {stage.stage_num ?? 1}</span>
        <span><Clock size={13} /> {stage.estimated_hours ?? 1} 小时</span>
      </div>
      <h2>{selectedTopicName}</h2>
      <section className="material-resource-section text-resources">
        <div className="material-section-heading"><BookOpen size={16} /><h3>文章内容</h3></div>
        {hasTextContent ? (
          <>
            {topicTutorial?.trim() && (
              <div className="material-markdown">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm, remarkMath]}
                    rehypePlugins={[rehypeKatex]}
                    components={{
                    a({ href, title, children, ...props }) {
                      if (href?.startsWith('#source-')) {
                        return <span className="material-citation" title={title || undefined}>{children}</span>;
                      }
                      return <a href={href} title={title} {...props}>{children}</a>;
                    },
                    pre({ children }) {
                      return <div className="material-code-block">{children}</div>;
                    },
                    code({ className, children, ...props }) {
                      const isInline = !className;
                      if (isInline) {
                        return <code className="material-inline-code" {...props}>{children}</code>;
                      }
                      const lang = (className || '').replace(/^language-/, '') || 'code';
                      return (
                        <>
                          <div className="material-code-lang">{lang}</div>
                          <code className={className} {...props}>{children}</code>
                        </>
                      );
                    },
                  }}
                >
                  {normalizeMarkdownMath(addCitationTooltips(topicTutorial, articleResources))}
                </ReactMarkdown>
              </div>
            )}
            {codeExamples.length > 0 && (
              <div className="material-code-section">
                <h3>代码示例</h3>
                {codeExamples.map((example, index) => (
                  <div className="material-code" key={`${example.title}-${index}`}>
                    <div><Code2 size={14} /> {example.title || `代码示例 ${index + 1}`}</div>
                    <pre><code>{example.code}</code></pre>
                  </div>
                ))}
              </div>
            )}
            {codeResourceItems.length > 0 && (
              <div className="material-article-list">
                {codeResourceItems.map((resource, index) => {
                  const url = getResourceUrl(resource);
                  return (
                    <article key={resource.resource_id || `${resource.title}-${index}`}>
                      <span>
                        <strong>{resource.title || `代码材料 ${index + 1}`}</strong>
                        {resource.content_summary && <small>{resource.content_summary}</small>}
                      </span>
                      {url && <a href={url} target="_blank" rel="noreferrer"><ExternalLink size={14} />打开链接</a>}
                    </article>
                  );
                })}
              </div>
            )}
          </>
        ) : <p className="material-empty">暂无文章内容</p>}
      </section>
      <section className="material-resource-section article-urls">
        <div className="material-section-heading"><ExternalLink size={16} /><h3>参考网址</h3></div>
        {articleResources.length > 0 ? (
          <div className="material-article-list">
            {articleResources.map((resource, index) => {
              const url = getResourceUrl(resource);
              return (
                <article key={resource.resource_id || `${resource.title}-${index}`}>
                  <span>
                    <strong>{resource.title || `拓展文章 ${index + 1}`}</strong>
                    {resource.content_summary && <small>{resource.content_summary}</small>}
                  </span>
                  {url ? (
                    <a href={url} target="_blank" rel="noreferrer"><ExternalLink size={14} />查看原文</a>
                  ) : (
                    <em className="material-empty-hint">暂无外部链接</em>
                  )}
                </article>
              );
            })}
          </div>
        ) : <p className="material-empty">暂无匹配拓展链接</p>}
      </section>
      <section className="material-resource-section video-resources">
        <div className="material-section-heading"><Play size={16} /><h3>学习视频</h3></div>
        {videoResources.length > 0 ? (
          <div className="material-video-list">
            {videoResources.map((resource, index) => (
              <EmbeddedVideoCard
                key={resource.resource_id || `${resource.title}-${index}`}
                resource={resource}
                index={index}
                getUrl={getResourceUrl}
              />
            ))}
          </div>
        ) : <p className="material-empty">暂无匹配视频</p>}
      </section>
    </article>
  );
}

function CurrentMaterialPanel({
  stage,
  learningPackage,
  selectedSubtopic,
  subtopics = [],
  onSelectSubtopic,
  onTest,
  stageResultReady,
  onOpenResult
}: {
  stage: StageInfo;
  learningPackage: LearningPackage | null;
  selectedSubtopic?: KnowledgeSubtopic;
  subtopics?: KnowledgeSubtopic[];
  onSelectSubtopic?: (subtopicId: string) => void;
  onTest: () => void;
  stageResultReady: boolean;
  onOpenResult: () => void;
}) {
  const content = learningPackage || fallbackPackage;
  const pages: KnowledgeSubtopic[] = subtopics.length > 0
    ? subtopics
    : selectedSubtopic
      ? [selectedSubtopic]
      : [{
          subtopic_id: stage.subtopic_id,
          subtopic_name: stage.subtopic_name || stage.block_name || content.block_name || '当前知识点',
          sequence_index: 0
        }];
  const resolvedIndex = Math.max(0, pages.findIndex((t) => topicKey(t) === topicKey(selectedSubtopic)));
  const currentIndex = resolvedIndex < 0 ? 0 : resolvedIndex;
  const current = pages[currentIndex] || selectedSubtopic;

  // 记忆每个知识点文章的滚动位置：
  // - 切换到从未阅读过的知识点 -> 从顶部开始
  // - 切换到已阅读过的知识点 -> 恢复到上次的滚动位置
  const materialScrollRef = useRef<HTMLDivElement>(null);
  const scrollPositions = useRef<Map<string, number>>(new Map());
  const currentTopicKey = topicKey(current);

  // 滚动时持续记录当前位置（供切回该知识点时恢复）
  useEffect(() => {
    const el = materialScrollRef.current;
    if (!el || !currentTopicKey) return;
    const handleScroll = () => {
      scrollPositions.current.set(currentTopicKey, el.scrollTop);
    };
    el.addEventListener('scroll', handleScroll, { passive: true });
    return () => el.removeEventListener('scroll', handleScroll);
  }, [currentTopicKey]);

  // 切换知识点时：有记忆位置则恢复，否则回到顶部
  useEffect(() => {
    const el = materialScrollRef.current;
    if (!el) return;
    const saved = currentTopicKey ? scrollPositions.current.get(currentTopicKey) : undefined;
    // 用 requestAnimationFrame 等待新文章内容渲染完成后再设置滚动位置，
    // 避免新内容高度还没布局好就被截断
    requestAnimationFrame(() => {
      el.scrollTop = saved ?? 0;
    });
  }, [currentTopicKey]);

  const goPrev = () => {
    if (currentIndex > 0) {
      const prev = pages[currentIndex - 1];
      onSelectSubtopic?.(topicKey(prev));
    }
  };
  const goNext = () => {
    if (currentIndex < pages.length - 1) {
      const next = pages[currentIndex + 1];
      onSelectSubtopic?.(topicKey(next));
    }
  };
  return (
    <aside className="current-material-panel">
      <header>
        <div>
          <span>当前阶段学习包 · 共 {pages.length} 个知识点</span>
          <strong>{stage.block_name || content.block_name}</strong>
        </div>
      </header>
      {pages.length > 1 && (
        <div className="material-page-controls compact">
          <button
            className="material-page-button compact"
            onClick={goPrev}
            disabled={currentIndex === 0}
            aria-label="上一个知识点"
            title="上一个知识点"
          >
            <ChevronLeft size={14} />
          </button>
          <div className="material-page-indicator compact" role="tablist" aria-label="知识点分页">
            {pages.map((topic, i) => (
              <button
                key={topicKey(topic) || `panel-${i}`}
                type="button"
                role="tab"
                aria-selected={i === currentIndex}
                aria-label={`第 ${i + 1} 页：${topic.subtopic_name}`}
                title={topic.subtopic_name}
                onClick={() => onSelectSubtopic?.(topicKey(topic))}
                className={i === currentIndex ? 'is-active' : ''}
              >
                {i + 1}
              </button>
            ))}
          </div>
          <button
            className="material-page-button compact"
            onClick={goNext}
            disabled={currentIndex >= pages.length - 1}
            aria-label="下一个知识点"
            title="下一个知识点"
          >
            <ChevronRight size={14} />
          </button>
        </div>
      )}
      <div className="current-material-scroll" ref={materialScrollRef}>
        <MaterialDocument stage={stage} learningPackage={content} selectedSubtopic={current} compact />
      </div>
      <footer>
        <button onClick={stageResultReady ? onOpenResult : onTest}>
          <FileQuestion size={16} />
          {stageResultReady ? '查看该阶段总结' : '开始阶段测试'}
        </button>
      </footer>
    </aside>
  );
}

function ChatView({
  goalTitle,
  messages,
  input,
  setInput,
  onSend,
  stage,
  conversationStage,
  timerSeconds,
  timerRunning,
  dailyProgress,
  onToggleTimer,
  onCreateGoal,
  onOpenLearningPath,
  onOpenProfile,
  learningPackage,
  stageSubtopics,
  selectedSubtopicId,
  onSelectSubtopic,
  onTest,
  stageResultReady,
  onOpenResult,
  projects,
  currentSessionId,
  selectedConversationId,
  onSelectProjectStage,
  onStartStage,
  onDeleteLastStage,
  readOnly,
  readOnlyNotice,
  loading,
  error
}: {
  goalTitle: string;
  messages: Message[];
  input: string;
  setInput: (input: string) => void;
  onSend: (event: FormEvent) => void;
  stage: StageInfo;
  conversationStage: StageInfo;
  timerSeconds: number;
  timerRunning: boolean;
  dailyProgress: number;
  onToggleTimer: () => void;
  onCreateGoal: () => void;
  onOpenLearningPath: () => void;
  onOpenProfile: () => void;
  learningPackage: LearningPackage | null;
  stageSubtopics: KnowledgeSubtopic[];
  selectedSubtopicId: string | null;
  onSelectSubtopic?: (subtopicId: string) => void;
  onTest: () => void;
  stageResultReady: boolean;
  onOpenResult: () => void;
  projects: LearningProject[];
  currentSessionId: string | null;
  selectedConversationId: string;
  onSelectProjectStage: (sessionId: string, stageNum: number) => void;
  onStartStage: () => void;
  onDeleteLastStage: (sessionId: string) => Promise<void>;
  readOnly: boolean;
  readOnlyNotice: string;
  loading: boolean;
  error: string | null;
}) {
  const [contextPanelCollapsed, setContextPanelCollapsed] = useState(false);
  const [contextPanelWidth, setContextPanelWidth] = useState(260);
  const [materialPanelWidth, setMaterialPanelWidth] = useState(600);

  const clampMaterialWidth = (width: number, layoutWidth: number) => {
    const contextWidth = contextPanelCollapsed ? 48 : contextPanelWidth;
    const maxMaterialWidth = Math.max(440, layoutWidth - contextWidth - 300 - 20);
    return Math.round(Math.max(440, Math.min(maxMaterialWidth, width)));
  };

  const clampContextWidth = (width: number, layoutWidth: number) => {
    const maxContextWidth = Math.max(200, Math.min(420, layoutWidth - 440 - 300 - 20));
    return Math.round(Math.max(200, Math.min(maxContextWidth, width)));
  };

  const resizeContextPanel = (event: ReactPointerEvent<HTMLButtonElement>) => {
    event.preventDefault();
    const layout = event.currentTarget.parentElement;
    if (!layout) return;
    const startX = event.clientX;
    const startWidth = contextPanelWidth;
    const layoutWidth = layout.getBoundingClientRect().width;
    const handlePointerMove = (moveEvent: PointerEvent) => {
      setContextPanelWidth(clampContextWidth(startWidth + moveEvent.clientX - startX, layoutWidth));
    };
    const handlePointerUp = () => {
      document.body.classList.remove('resizing-workspace-panels');
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', handlePointerUp);
    };
    document.body.classList.add('resizing-workspace-panels');
    window.addEventListener('pointermove', handlePointerMove);
    window.addEventListener('pointerup', handlePointerUp, { once: true });
  };

  const resizePanels = (event: ReactPointerEvent<HTMLButtonElement>) => {
    event.preventDefault();
    const layout = event.currentTarget.parentElement;
    if (!layout) return;
    const startX = event.clientX;
    const startWidth = materialPanelWidth;
    const layoutWidth = layout.getBoundingClientRect().width;
    const handlePointerMove = (moveEvent: PointerEvent) => {
      setMaterialPanelWidth(clampMaterialWidth(startWidth + moveEvent.clientX - startX, layoutWidth));
    };
    const handlePointerUp = () => {
      document.body.classList.remove('resizing-workspace-panels');
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', handlePointerUp);
    };
    document.body.classList.add('resizing-workspace-panels');
    window.addEventListener('pointermove', handlePointerMove);
    window.addEventListener('pointerup', handlePointerUp, { once: true });
  };

  const resizePanelsByKeyboard = (event: ReactKeyboardEvent<HTMLButtonElement>) => {
    if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
    event.preventDefault();
    const layoutWidth = event.currentTarget.parentElement?.getBoundingClientRect().width ?? window.innerWidth;
    const delta = event.key === 'ArrowRight' ? 24 : -24;
    setMaterialPanelWidth((width) => clampMaterialWidth(width + delta, layoutWidth));
  };

  const resizeContextPanelByKeyboard = (event: ReactKeyboardEvent<HTMLButtonElement>) => {
    if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
    event.preventDefault();
    const layoutWidth = event.currentTarget.parentElement?.getBoundingClientRect().width ?? window.innerWidth;
    const delta = event.key === 'ArrowRight' ? 24 : -24;
    setContextPanelWidth((width) => clampContextWidth(width + delta, layoutWidth));
  };

  const layoutStyle = {
    '--context-panel-width': `${contextPanelWidth}px`,
    '--material-panel-width': `${materialPanelWidth}px`
  } as CSSProperties;

  // ---- 对话区自动滚动到底部：
  // 1) 首次点开阶段会话 / 切换历史对话时
  // 2) 用户发送消息（回车触发）后
  // 3) AI 流式回复追加内容时
  const conversationRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = conversationRef.current;
    if (!el) return;
    // 使用即时滚动，避免动画时错过底部
    el.scrollTop = el.scrollHeight;
  }, [
    messages,                  // 任何消息内容/长度变化（含 AI 流式追加）
    selectedConversationId,    // 切换阶段/历史会话
    loading                    // AI 回复结束后再对齐一次
  ]);

  return (
    <div className="page-shell">
      <Topbar
        title={goalTitle}
        subtitle={conversationStage.block_name || conversationStage.subtopic_name || conversationStage.gap_skill || '阶段学习与实时答疑'}
        loading={loading}
        error={error}
        onOpenLearningPath={onOpenLearningPath}
        onOpenProfile={onOpenProfile}
      />
      <div
        className={`chat-layout ${contextPanelCollapsed ? 'context-collapsed' : ''}`}
        style={layoutStyle}
      >
        <ProfilePanel
          stage={stage}
          projects={projects}
          currentSessionId={currentSessionId}
          overallProgress={projects.find((project) => project.sessionId === currentSessionId)?.overallProgress}
          dailyProgress={dailyProgress}
          selectedConversationId={selectedConversationId}
          onSelectProjectStage={onSelectProjectStage}
          onCreateGoal={onCreateGoal}
          loading={loading}
          collapsed={contextPanelCollapsed}
          onToggleCollapsed={() => setContextPanelCollapsed((prev) => !prev)}
          timerSeconds={timerSeconds}
          timerRunning={timerRunning}
          onToggleTimer={onToggleTimer}
          onStartStage={onStartStage}
          onDeleteLastStage={onDeleteLastStage}
        />

        <button
          className="context-resize-handle"
          onPointerDown={resizeContextPanel}
          onKeyDown={resizeContextPanelByKeyboard}
          aria-label="拖动调整对话导航栏宽度"
          title="拖动调整对话导航栏宽度"
        />

        <CurrentMaterialPanel
          stage={stage}
          learningPackage={learningPackage}
          selectedSubtopic={stageSubtopics.find((topic) => topicKey(topic) === selectedSubtopicId) || stageSubtopics[0]}
          subtopics={stageSubtopics}
          onSelectSubtopic={onSelectSubtopic}
          onTest={onTest}
          stageResultReady={stageResultReady}
          onOpenResult={onOpenResult}
        />

        <button
          className="workspace-resize-handle"
          onPointerDown={resizePanels}
          onKeyDown={resizePanelsByKeyboard}
          aria-label="拖动调整学习资料与答疑栏宽度"
          title="拖动调整学习资料与答疑栏宽度"
        />

        <main className="chat-main">
          <header className="chat-assistant-header">
            <MessageCircle size={16} />
            <div>
              <span>学习助手</span>
              <strong>阶段内答疑</strong>
            </div>
          </header>
          <div className="conversation" ref={conversationRef}>
            <div className="date-separator">
              <span>今天</span>
            </div>
            {messages.map((message) => (
              <div id={`message-${message.id}`} key={message.id} className={`message-row ${message.role}`}>
                <div className="message-avatar">{message.role === 'assistant' ? <Sparkles size={16} /> : '你'}</div>
                <div className="message-wrap">
                  <div className="message-meta">
                    {message.role === 'assistant' ? 'AI 实训导师' : '你'}
                    <span>{message.note || '已记录'}</span>
                  </div>
                  <div className="message-bubble message-markdown">
                    <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>
                      {normalizeMarkdownMath(message.content)}
                    </ReactMarkdown>
                  </div>
                  {message.role === 'assistant' && message.reviewSummary && (
                    <div className={`review-badge ${message.reviewSummary.status === 'skipped' ? 'skipped' : ''}`}>
                      <Check size={13} />
                      {message.reviewSummary.status === 'skipped'
                        ? '日常问题，无需技术审核'
                        : `Agent4 审核：技术${message.reviewSummary.technical_passed ? '通过' : '待复核'} · 教学${message.reviewSummary.quality_passed ? '通过' : '待复核'}${message.reviewSummary.iterations ? ` · ${message.reviewSummary.iterations} 轮` : ''}`}
                    </div>
                  )}
                </div>
              </div>
            ))}
            {loading && (
              <div className="message-row assistant">
                <div className="message-avatar">
                  <Sparkles size={16} />
                </div>
                <div className="typing">
                  <i />
                  <i />
                  <i />
                </div>
              </div>
            )}
          </div>

          {readOnly ? (
            <div className="historical-conversation-note">{readOnlyNotice}</div>
          ) : (
            <form className="composer" onSubmit={onSend}>
              <textarea
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={(event) => {
                  // Enter 直接发送；Shift+Enter / Ctrl+Enter 保留换行
                  if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
                    event.preventDefault();
                    const target = event.currentTarget;
                    const formEl = target.closest('form');
                    if (formEl && input.trim() && !loading) {
                      formEl.requestSubmit();
                    }
                  }
                }}
                placeholder="针对当前阶段内容提问...（Enter 发送，Shift+Enter 换行）"
                rows={2}
              />
              <div className="composer-bottom">
                <span>问题会记录在当前阶段对话中</span>
                <button disabled={!input.trim() || loading} aria-label="发送">
                  <Send size={17} />
                </button>
              </div>
            </form>
          )}
        </main>
      </div>
    </div>
  );
}

function LearnView({
  stage,
  learningPackage,
  onStart,
  onTest,
  loading,
  error,
  subtopics = []
}: {
  stage: StageInfo;
  learningPackage: LearningPackage | null;
  onStart: () => void;
  onTest: () => void;
  loading: boolean;
  error: string | null;
  subtopics?: KnowledgeSubtopic[];
}) {
  const content = learningPackage || fallbackPackage;
  const pages: KnowledgeSubtopic[] = subtopics.length > 0
    ? subtopics
    : [{
        subtopic_id: stage.subtopic_id,
        subtopic_name: stage.subtopic_name || stage.block_name || content.block_name || '当前知识点',
        sequence_index: 0
      }];
  const [pageIndex, setPageIndex] = useState(0);
  const safeIndex = Math.max(0, Math.min(pageIndex, pages.length - 1));
  const currentSubtopic = pages[safeIndex];
  const progressPct = pages.length > 1 ? ((safeIndex + 1) / pages.length) * 100 : (content.tutorial ? 66 : 18);

  return (
    <div className="page-shell">
      <Topbar title="当前阶段" subtitle={`阶段 ${stage.stage_num ?? 1} · ${stage.block_name || content.block_name}`} loading={loading} error={error} />
      <main className="lesson">
        <div className="lesson-top">
          <div>
            <span className="lesson-kicker">阶段 {stage.stage_num ?? 1} · {statusText[stage.status || 'learning']}</span>
            <h2>{stage.block_name || content.block_name}</h2>
            <p>本阶段共 <b>{pages.length}</b> 个知识点，请逐页完成学习后再进入测试。</p>
          </div>
          <div className="lesson-actions">
            <button className="ghost" onClick={onStart}>
              <RefreshCw size={16} />
              重新生成
            </button>
            <button className="primary" onClick={onTest}>
              <FileQuestion size={17} />
              进入测试
            </button>
          </div>
        </div>

        <div className="reading-progress">
          <span style={{ width: `${progressPct}%` }} />
        </div>

        <div className="lesson-meta">
          <span>
            <Clock size={15} />
            预计 {stage.estimated_hours ?? 1} 小时
          </span>
          <span>
            <Target size={15} />
            第 {safeIndex + 1} / {pages.length} 页 · {currentSubtopic?.subtopic_name || stage.gap_skill || '当前知识点'}
          </span>
          <span className="verified">
            <Check size={14} />
            双重审核展示
          </span>
        </div>

        {pages.length > 1 && (
          <div className="material-page-controls lesson-page-controls">
            <button
              className="material-page-button"
              onClick={() => setPageIndex((i) => Math.max(0, i - 1))}
              disabled={safeIndex === 0}
            >
              <ChevronLeft size={16} /> 上一页
            </button>
            <div className="material-page-indicator" role="tablist" aria-label="知识点分页">
              {pages.map((topic, i) => (
                <button
                  key={topicKey(topic) || i}
                  type="button"
                  role="tab"
                  aria-selected={i === safeIndex}
                  aria-label={`第 ${i + 1} 页：${topic.subtopic_name}`}
                  title={topic.subtopic_name}
                  onClick={() => setPageIndex(i)}
                  className={i === safeIndex ? 'is-active' : ''}
                >
                  {i + 1}
                </button>
              ))}
            </div>
            <button
              className="material-page-button"
              onClick={() => setPageIndex((i) => Math.min(pages.length - 1, i + 1))}
              disabled={safeIndex >= pages.length - 1}
            >
              下一页 <ChevronRight size={16} />
            </button>
          </div>
        )}

        <article className="lesson-article">
          <aside className="lesson-toc">
            <span>知识点分页</span>
            {pages.map((topic, i) => (
              <button
                type="button"
                key={topicKey(topic) || i}
                className={i === safeIndex ? 'is-active' : ''}
                onClick={() => setPageIndex(i)}
                title={topic.subtopic_name}
              >
                {i + 1}. {topic.subtopic_name?.slice(0, 14)}{(topic.subtopic_name?.length || 0) > 14 ? '…' : ''}
              </button>
            ))}
            <a href="#quiz">阶段测试</a>
          </aside>
          <div className="lesson-content">
            <MaterialDocument
              stage={stage}
              learningPackage={content}
              selectedSubtopic={currentSubtopic}
            />

            <section id="quiz">
              <span className="section-index">03</span>
              <h3>阶段检查</h3>
              <p>完成当前阶段全部知识点的学习后，可进入测试。系统会根据正确率更新画像和下一阶段策略。</p>
              <button className="complete-reading" onClick={onTest}>
                <FileQuestion size={17} />
                开始阶段测试
              </button>
            </section>
          </div>
        </article>

        {pages.length > 1 && (
          <div className="material-page-controls lesson-page-controls bottom">
            <button
              className="material-page-button"
              onClick={() => setPageIndex((i) => Math.max(0, i - 1))}
              disabled={safeIndex === 0}
            >
              <ChevronLeft size={16} /> 上一页
            </button>
            <div className="material-page-indicator" role="tablist" aria-label="知识点分页">
              {pages.map((topic, i) => (
                <button
                  key={topicKey(topic) || `b-${i}`}
                  type="button"
                  role="tab"
                  aria-selected={i === safeIndex}
                  aria-label={`第 ${i + 1} 页：${topic.subtopic_name}`}
                  title={topic.subtopic_name}
                  onClick={() => setPageIndex(i)}
                  className={i === safeIndex ? 'is-active' : ''}
                >
                  {i + 1}
                </button>
              ))}
            </div>
            <button
              className="material-page-button"
              onClick={() => setPageIndex((i) => Math.min(pages.length - 1, i + 1))}
              disabled={safeIndex >= pages.length - 1}
            >
              下一页 <ChevronRight size={16} />
            </button>
          </div>
        )}
      </main>
    </div>
  );
}

function TestModal({
  questions,
  onClose,
  onComplete
}: {
  questions: TestQuestion[];
  onClose: () => void;
  onComplete: (answers: Record<string, string>) => void;
}) {
  const [index, setIndex] = useState(0);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const TRUE_FALSE_TYPES = new Set(['true_false', '判断', '判断题', 'yes_no']);
  // 对每个 question 做最终兜底：判断题型若 options 为空，补 [正确/错误]，避免退化成 textarea。
  const hydratedQuestions: TestQuestion[] = questions.map((q) => {
    if (q.options && q.options.length > 0) return q;
    const qt = (q.question_type || '').trim().toLowerCase();
    if (TRUE_FALSE_TYPES.has(qt)) return { ...q, options: ['正确', '错误'] };
    return q;
  });
  const safeQuestions = hydratedQuestions.length > 0 ? hydratedQuestions : questions;
  const current = safeQuestions[index];
  const done = safeQuestions.length > 0 && safeQuestions.every((question) => answers[question.question_id]);

  if (!current) return null;

  return (
    <div className="modal-backdrop">
      <section className="test-modal" role="dialog" aria-modal="true" aria-label="阶段测试">
        <header>
          <div>
            <span>阶段测试</span>
            <strong>{index + 1} / {safeQuestions.length}</strong>
          </div>
          <button onClick={onClose} aria-label="关闭测试">×</button>
        </header>
        <div className="test-body">
          <aside>
            <span>答题进度</span>
            <strong>{Object.keys(answers).length} / {safeQuestions.length}</strong>
            <div className="test-progress">
              <i style={{ width: `${safeQuestions.length ? (Object.keys(answers).length / safeQuestions.length) * 100 : 0}%` }} />
            </div>
            <div className="question-grid">
              {safeQuestions.map((question, questionIndex) => (
                <button
                  type="button"
                  key={question.question_id}
                  className={`${questionIndex === index ? 'current' : ''} ${answers[question.question_id] ? 'answered' : ''}`}
                  onClick={() => setIndex(questionIndex)}
                >
                  {questionIndex + 1}
                </button>
              ))}
            </div>
            <p>答题结果会用于更新知识盲区、信心指数和下一阶段策略。</p>
          </aside>
          <main>
            <div className="question-label">{current.question_type || '测试题'}</div>
            <h2>{current.question_text}</h2>
            {current.options?.length ? (
              <div className="options">
                {current.options.map((option, optionIndex) => {
                  const optionKey = typeof option === 'string'
                    ? String.fromCharCode(65 + optionIndex)
                    : option.key || String.fromCharCode(65 + optionIndex);
                  const optionText = typeof option === 'string' ? option : option.text || optionKey;
                  return (
                    <button
                      type="button"
                      key={optionKey}
                      className={answers[current.question_id] === optionKey ? 'selected' : ''}
                      onClick={() => setAnswers((prev) => ({ ...prev, [current.question_id]: optionKey }))}
                    >
                      <span>{optionKey}</span>
                      {optionText}
                      {answers[current.question_id] === optionKey && <Check size={18} />}
                    </button>
                  );
                })}
              </div>
            ) : (
              <textarea
                className="answer-textarea"
                value={answers[current.question_id] || ''}
                onChange={(event) => setAnswers((prev) => ({ ...prev, [current.question_id]: event.target.value }))}
                placeholder="请输入你的答案..."
              />
            )}
            {current.hint && <p className="question-hint">提示：{current.hint}</p>}
            <div className="test-nav">
              <button type="button" className="ghost" disabled={index === 0} onClick={() => setIndex(index - 1)}>上一题</button>
              {index < safeQuestions.length - 1 ? (
                <button type="button" className="primary" onClick={() => setIndex(index + 1)}>
                  下一题
                  <ChevronRight size={15} />
                </button>
              ) : (
                <button type="button" className="primary" disabled={!done} onClick={() => onComplete(answers)}>完成测试</button>
              )}
            </div>
          </main>
        </div>
      </section>
    </div>
  );
}

function AssessmentReviewModal({ review, onClose }: { review: AssessmentReview; onClose: () => void }) {
  return (
    <div className="modal-backdrop" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <section className="assessment-review-modal" role="dialog" aria-modal="true" aria-labelledby="assessment-review-title">
        <header>
          <div>
            <span>阶段测试回顾</span>
            <h2 id="assessment-review-title">{review.correct_count ?? 0} / {review.total_questions} 题正确</h2>
          </div>
          <button onClick={onClose} aria-label="关闭测试题目" title="关闭"><X size={18} /></button>
        </header>
        <div className="assessment-review-score">
          <strong>{pct(review.accuracy ?? 0)}%</strong><span>本阶段正确率</span>
        </div>
        <div className="assessment-review-list">
          {review.questions.map((question, index) => (
            <article className={`assessment-review-item ${question.is_correct ? 'is-correct' : 'is-incorrect'}`} key={question.question_id}>
              <div className="assessment-review-item-head">
                <span>第 {index + 1} 题 · {question.question_type || '测试题'}</span>
                <strong>{question.is_correct ? '回答正确' : '需要复习'}</strong>
              </div>
              <h3>{question.question_text}</h3>
              <div className="assessment-review-answers">
                <div><span>你的答案</span><b>{question.user_answer || '未作答'}</b></div>
                <div><span>正确答案</span><b>{question.correct_answer || '未提供'}</b></div>
              </div>
              {question.explanation && <p><span>解析</span>{question.explanation}</p>}
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}

function GoalCompletionModal({ goal, onClose }: { goal: string; onClose: () => void }) {
  const confirmButtonRef = useRef<HTMLButtonElement>(null);
  const displayGoal = goal.trim() || '当前学习目标';

  useEffect(() => {
    confirmButtonRef.current?.focus();
    const handleKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  return (
    <div className="stage-delete-modal-overlay goal-completion-overlay" role="dialog" aria-modal="true" aria-labelledby="goal-completion-title">
      <section className="goal-completion-modal">
        <div className="goal-completion-glow" aria-hidden />
        <div className="goal-completion-confetti" aria-hidden>
          {Array.from({ length: 12 }, (_, index) => <i key={index} />)}
        </div>
        <button type="button" className="goal-completion-close" onClick={onClose} aria-label="关闭学习目标完成弹窗">
          <X size={19} />
        </button>

        <div className="goal-completion-emblem" aria-hidden>
          <span className="goal-completion-emblem-ring" />
          <Award size={42} strokeWidth={1.9} />
          <Sparkles className="goal-completion-emblem-sparkle" size={22} />
        </div>

        <div className="goal-completion-copy">
          <span className="goal-completion-eyebrow"><Check size={14} /> LEARNING GOAL COMPLETE</span>
          <h2 id="goal-completion-title">“{displayGoal}”学习完毕</h2>
          <p>恭喜你，已经完成该学习目标下的全部知识点。本次学习路线正式结业。</p>
        </div>

        <div className="goal-completion-progress" aria-label="学习目标完成度百分之百">
          <div className="goal-completion-progress-heading">
            <span><Target size={16} />目标完成度</span>
            <strong>100%</strong>
          </div>
          <div className="goal-completion-progress-track"><span /></div>
          <div className="goal-completion-milestones">
            <span><Check size={14} />目标知识点全部掌握</span>
            <span><Check size={14} />学习画像已更新</span>
          </div>
        </div>

        <div className="goal-completion-note">
          <Sparkles size={16} />
          <span>关闭弹窗后仍可查看最后阶段总结、错题记录和画像变化。</span>
        </div>

        <button ref={confirmButtonRef} type="button" className="goal-completion-confirm" onClick={onClose}>
          查看最终阶段总结 <ChevronRight size={18} />
        </button>
      </section>
    </div>
  );
}

function ProjectSwitcherModal({
  projects,
  currentSessionId,
  loading,
  error,
  onClose,
  onSelect,
  onCreate
}: {
  projects: LearningProject[];
  currentSessionId: string | null;
  loading: boolean;
  error: string | null;
  onClose: () => void;
  onSelect: (sessionId: string) => void;
  onCreate: () => void;
}) {
  return (
    <div className="modal-backdrop" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <section className="project-switcher-modal" role="dialog" aria-modal="true" aria-labelledby="project-switcher-title">
        <header>
          <div>
            <span>学习项目</span>
            <h2 id="project-switcher-title">切换学习目标</h2>
          </div>
          <button className="project-modal-close" onClick={onClose} aria-label="关闭切换目标弹窗" title="关闭">
            <X size={18} />
          </button>
        </header>

        {error && <div className="project-switch-error">{error}</div>}

        <div className="project-list">
          {projects.length === 0 && (
            <div className="project-empty">
              <Target size={24} />
              <strong>还没有历史学习项目</strong>
              <span>创建目标并完成画像后，项目会保存在这里。</span>
            </div>
          )}
          {projects.map((project) => {
            const current = project.sessionId === currentSessionId;
            return (
              <button
                className={`project-option ${current ? 'current' : ''}`}
                key={project.sessionId}
                onClick={() => onSelect(project.sessionId)}
                disabled={loading || current}
              >
                <span className="project-option-icon"><Target size={17} /></span>
                <span className="project-option-copy">
                  <strong>{project.goal}</strong>
                  <small>阶段 {project.stageNum} · {project.stageName}</small>
                </span>
                <span className="project-option-state">{current ? '当前项目' : '切换'}</span>
              </button>
            );
          })}
        </div>

        <footer>
          <button className="new-project-button" onClick={onCreate} disabled={loading}>
            <Plus size={16} />
            创建新学习目标
          </button>
        </footer>
      </section>
    </div>
  );
}

function ProfileDetailModal({ project, onClose }: { project: LearningProject; onClose: () => void }) {
  const item = project.profile;
  const foundations = (item?.skill_tags || []).filter(Boolean);
  const gapNames = (item?.knowledge_gaps || []).map((gap) => gap.skill).filter(Boolean);
  const targetDailyHours = project.targetDailyHours || item?.daily_hours || 2;
  const currentStage = project.stages?.find((stage) => stage.stage_num === project.stageNum)
    || project.stages?.[project.stages.length - 1];

  return (
    <div className="modal-backdrop" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <section className="profile-detail-modal" role="dialog" aria-modal="true" aria-labelledby="profile-detail-title">
        <header>
          <div>
            <span>学习目标画像</span>
            <h2 id="profile-detail-title">{project.goal}</h2>
          </div>
          <button onClick={onClose} aria-label="关闭学习画像" title="关闭"><X size={18} /></button>
        </header>
        <div className="profile-detail-body">
          <section className="profile-overview-item profile-overview-tags foundations">
            <span className="profile-overview-icon"><BookOpen size={18} /></span>
            <div>
              <span className="profile-overview-label">已有基础</span>
              <div className="profile-overview-tag-list">
                {foundations.length
                  ? foundations.map((foundation) => <strong key={foundation}>{foundation}</strong>)
                  : <em>暂未记录</em>}
              </div>
            </div>
          </section>
          <section className="profile-overview-item profile-overview-tags gaps">
            <span className="profile-overview-icon"><Target size={18} /></span>
            <div>
              <span className="profile-overview-label">知识盲区</span>
              <div className="profile-overview-tag-list">
                {gapNames.length
                  ? gapNames.map((gap) => <strong key={gap}>{gap}</strong>)
                  : <em>暂未发现</em>}
              </div>
            </div>
          </section>
          <section className="profile-overview-item profile-overview-metric duration">
            <span className="profile-overview-icon"><Clock size={18} /></span>
            <div>
              <span className="profile-overview-label">学习时长</span>
              <strong>{targetDailyHours} 小时/阶段</strong>
            </div>
          </section>
          <section className="profile-overview-item profile-overview-metric accuracy">
            <span className="profile-overview-icon"><Gauge size={18} /></span>
            <div>
              <span className="profile-overview-label">正确率</span>
              <strong>
                {project.averageAccuracy == null ? '--' : pct(project.averageAccuracy)}
                {project.averageAccuracy != null && <small>%</small>}
              </strong>
              <p>{project.averageAccuracy == null ? '完成测试后生成' : '全部阶段平均'}</p>
            </div>
          </section>
        </div>
        {item && (
          <div className="profile-capability-panel">
            <KnowledgeNetwork
              profile={item}
              stage={currentStage}
              compact
              defaultMode="focus"
              showNodeDetails={false}
            />
          </div>
        )}
      </section>
    </div>
  );
}

function LearningPathModal({
  profile,
  stage,
  selectedSubtopicId,
  onSelectSubtopic,
  onClose
}: {
  profile: Profile;
  stage: StageInfo;
  selectedSubtopicId: string | null;
  onSelectSubtopic: (subtopicId: string) => void;
  onClose: () => void;
}) {
  const subtopics = resolveStageSubtopics(profile, stage);
  const prerequisites = Array.from(new Set(stage.prerequisites || []));

  return (
    <div className="modal-backdrop" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <section className="learning-path-modal" role="dialog" aria-modal="true" aria-labelledby="learning-path-title">
        <header>
          <div><span>当前阶段</span><h2 id="learning-path-title">阶段知识点</h2></div>
          <button onClick={onClose} aria-label="关闭阶段知识点" title="关闭"><X size={18} /></button>
        </header>
        <div className="learning-path-modal-body">
          <section className="stage-knowledge-panel">
            <div className="stage-knowledge-heading">
              <div><span>阶段 {stage.stage_num ?? 1}</span><h3>{stage.block_name || '当前学习阶段'}</h3></div>
              <strong>{subtopics.length} 个知识点</strong>
            </div>
            {prerequisites.length > 0 && <div className="stage-knowledge-prerequisites"><span>前置要求</span>{prerequisites.map((name) => <b key={name}>{name}</b>)}</div>}
            {subtopics.length > 0 ? (
              <div className="stage-knowledge-list" role="tablist" aria-label="阶段知识点">
                {subtopics.map((topic, index) => (
                  <div
                    className={`path-subtopic ${topicKey(topic) === selectedSubtopicId ? 'current' : ''}`}
                    key={topicKey(topic) || `${topic.subtopic_name}-${index}`}
                    role="tab"
                    tabIndex={0}
                    aria-selected={topicKey(topic) === selectedSubtopicId}
                    onClick={() => {
                      onSelectSubtopic(topicKey(topic));
                      onClose();
                    }}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter' || event.key === ' ') {
                        event.preventDefault();
                        onSelectSubtopic(topicKey(topic));
                        onClose();
                      }
                    }}
                  >
                    <i>{index + 1}</i>
                    <span>{topic.subtopic_name}</span>
                  </div>
                ))}
              </div>
            ) : <p className="stage-knowledge-empty">等待后端返回当前阶段知识点</p>}
          </section>
        </div>
      </section>
    </div>
  );
}

function LearningDashboardView({
  projects,
  currentSessionId,
  profile,
  stage,
  setView,
  onSelectProject,
  onDeleteProject,
  onUpdateTargetDuration,
  onLoadTargetSettings,
  projectsLoading,
  projectsError,
  onRetryProjects,
  loading,
  error
}: {
  projects: LearningProject[];
  currentSessionId: string | null;
  profile: Profile | null;
  stage: StageInfo;
  setView: (view: View) => void;
  onSelectProject: (sessionId: string) => void;
  onDeleteProject: (sessionId: string) => Promise<void>;
  onUpdateTargetDuration: (sessionId: string, totalHours: number, dailyHours: number) => Promise<void>;
  onLoadTargetSettings: (sessionId: string) => Promise<{ total: number; daily: number } | null>;
  projectsLoading: boolean;
  projectsError: string | null;
  onRetryProjects: () => void;
  loading: boolean;
  error: string | null;
}) {
  const [pendingDeleteProject, setPendingDeleteProject] = useState<LearningProject | null>(null);
  const [projectsExpanded, setProjectsExpanded] = useState(true);
  const [projectSearch, setProjectSearch] = useState('');
  const [editingTargetId, setEditingTargetId] = useState<string | null>(null);
  const [targetDurationDraft, setTargetDurationDraft] = useState({ total: '', daily: '' });
  const sortedProjects = sortProjectsForDashboard(projects, currentSessionId);
  const visibleProjects = filterProjectsForDashboard(sortedProjects, projectSearch);
  const currentProject = projects.find((project) => project.sessionId === currentSessionId);

  return (
    <div className="page-shell">
      <Topbar title="学习仪表盘" subtitle="学习画像、阶段进度和策略决策汇总" loading={loading} error={error} />
      <main className="all-profiles-page learning-dashboard-page">
        <section className={`profile-library-section ${projectsExpanded ? 'is-expanded' : 'is-collapsed'}`}>
          <header>
            <button
              type="button"
              className="profile-library-title-toggle"
              onClick={() => setProjectsExpanded((expanded) => !expanded)}
              aria-expanded={projectsExpanded}
              aria-controls="learning-goal-list"
            >
              <h2>全部学习目标</h2>
            </button>
            {projectsExpanded && (
              <label className="profile-project-search">
                <Search size={16} />
                <input
                  type="search"
                  value={projectSearch}
                  onChange={(event) => setProjectSearch(event.target.value)}
                  placeholder="搜索学习目标"
                  aria-label="搜索学习目标"
                />
              </label>
            )}
            <button
              type="button"
              className="profile-library-meta-toggle"
              onClick={() => setProjectsExpanded((expanded) => !expanded)}
              aria-expanded={projectsExpanded}
              aria-controls="learning-goal-list"
              aria-label={projectsExpanded ? '收起全部学习目标' : '展开全部学习目标'}
            >
              <span>{sortedProjects.length} 个目标</span>
              {projectsExpanded ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
            </button>
          </header>
          {projectsError && (
            <div className="profile-list-state error">
              <span>{projectsError}</span>
              <button type="button" onClick={onRetryProjects}>重新读取</button>
            </div>
          )}
          {projectsExpanded && <div className="profile-goal-list" id="learning-goal-list">
            {projectsLoading && sortedProjects.length === 0 && (
              <div className="profile-list-state loading"><Loader2 className="spin" size={18} />正在读取学习目标</div>
            )}
            {visibleProjects.map((project) => {
              const current = project.sessionId === currentSessionId;
              const stageTotal = (project.stages || []).reduce((sum, item) => sum + (item.estimated_hours || 0), 0);
              const targetTotal = project.targetTotalHours || stageTotal;
              const targetDaily = project.targetDailyHours || project.profile?.daily_hours || 2;
              const editingDuration = editingTargetId === project.sessionId;
              const draftTotal = Number(targetDurationDraft.total);
              const draftDaily = Number(targetDurationDraft.daily);
              const invalidTargetDuration = !Number.isFinite(draftTotal) || draftTotal < 0.5 || draftTotal > 1000
                || !Number.isFinite(draftDaily) || draftDaily < 0.5 || draftDaily > 24;
              return (
                <article
                  className={`${current ? 'current ' : ''}${editingDuration ? 'editing-duration' : ''}`.trim()}
                  key={project.sessionId}
                >
                  <span className="profile-goal-icon"><Target size={19} /></span>
                  <div className="profile-goal-copy">
                    <div className="profile-goal-title">
                      <h3>{project.goal}</h3>
                      {current && <b>当前项目</b>}
                    </div>
                    <p>已生成 {project.stages?.length || 0} 个阶段 · 当前第 {project.stageNum} 阶段</p>
                    <small>{project.stageName || '等待阶段规划'}</small>
                    <div className="goal-duration-summary">
                      <span><Clock size={12} />总计 {targetTotal || 0} 小时</span>
                      <span>每阶段学习时长 {targetDaily} 小时</span>
                      <span><CalendarDays size={12} />建立于 {formatProjectCreatedDate(project.createdAt ?? project.updatedAt)}</span>
                    </div>
                  </div>
                  <div className="profile-goal-actions">
                    <button
                      className="secondary edit-target-button"
                      type="button"
                      onClick={() => {
                        if (editingDuration) {
                          setEditingTargetId(null);
                          return;
                        }
                        void onLoadTargetSettings(project.sessionId).then((latest) => {
                          setTargetDurationDraft({
                            total: String((latest?.total ?? targetTotal) || 0.5),
                            daily: String(latest?.daily ?? targetDaily)
                          });
                          setEditingTargetId(project.sessionId);
                        });
                      }}
                      title="调整目标时长"
                    >
                      <Pencil size={15} />
                      调整时长
                    </button>
                    <button
                      className="primary"
                      onClick={() => onSelectProject(project.sessionId)}
                      disabled={loading || current}
                    >
                      {current ? <Check size={16} /> : <LayoutDashboard size={16} />}
                      {current ? '正在查看' : '查看仪表盘'}
                    </button>
                    <button
                      className="delete-target-button"
                      type="button"
                      onClick={() => setPendingDeleteProject(project)}
                      disabled={loading}
                      title="删除学习目标"
                      aria-label={`删除学习目标：${project.goal}`}
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>
                  {editingDuration && (
                    <div className="duration-editor target">
                      <DurationField
                        label="当前阶段目标总时长"
                        value={targetDurationDraft.total}
                        onChange={(total) => setTargetDurationDraft((draft) => ({ ...draft, total }))}
                        min={0.5}
                        max={1000}
                      />
                      <DurationField
                        label="每阶段学习时长"
                        value={targetDurationDraft.daily}
                        onChange={(daily) => setTargetDurationDraft((draft) => ({ ...draft, daily }))}
                        min={0.5}
                        max={24}
                      />
                      <div className="duration-editor-actions">
                        <button type="button" onClick={() => setEditingTargetId(null)} disabled={loading}>取消</button>
                        <button
                          type="button"
                          onClick={async () => {
                            const total = Number(targetDurationDraft.total);
                            const daily = Number(targetDurationDraft.daily);
                            if (!Number.isFinite(total) || !Number.isFinite(daily) || total < 0.5 || daily < 0.5) return;
                            try {
                              await onUpdateTargetDuration(project.sessionId, total, daily);
                              setEditingTargetId(null);
                            } catch {
                              // The parent displays the API error and keeps the editor open.
                            }
                          }}
                          disabled={loading || invalidTargetDuration}
                        >保存</button>
                      </div>
                    </div>
                  )}
                </article>
              );
            })}
            {!projectsLoading && !projectsError && sortedProjects.length === 0 && (
              <div className="profiles-empty">
                <UserRound size={28} />
                <strong>暂无学习画像</strong>
                <span>创建学习目标并完成诊断后，画像会显示在这里。</span>
              </div>
            )}
            {!projectsLoading && !projectsError && sortedProjects.length > 0 && visibleProjects.length === 0 && (
              <div className="profiles-empty search-empty">
                <Search size={24} />
                <strong>未找到匹配目标</strong>
                <span>请尝试其他关键词。</span>
              </div>
            )}
          </div>}
        </section>

        {profile ? (
          <DashboardView
            profile={profile}
            stage={stage}
            averageAccuracy={currentProject?.averageAccuracy ?? null}
            overallProgress={currentProject?.overallProgress ?? null}
            targetDailyHours={currentProject?.targetDailyHours ?? profile.daily_hours ?? 2}
            setView={setView}
            loading={loading}
            error={error}
            embedded
          />
        ) : sortedProjects.length > 0 ? (
          <section className="dashboard-selection-empty">
            <LayoutDashboard size={26} />
            <strong>选择一个学习目标查看仪表盘</strong>
            <span>画像对应的掌握度、阶段进度和学习策略将在这里显示。</span>
          </section>
        ) : null}
      </main>
      {pendingDeleteProject && (
        <div className="modal-backdrop" onMouseDown={(event) => event.target === event.currentTarget && setPendingDeleteProject(null)}>
          <section className="delete-goal-dialog" role="dialog" aria-modal="true" aria-labelledby="delete-goal-title">
            <span className="delete-goal-icon"><Trash2 size={20} /></span>
            <h2 id="delete-goal-title">删除学习目标？</h2>
            <p>“{pendingDeleteProject.goal}”及其学习记录将从目标列表中移除。</p>
            <div className="delete-goal-actions">
              <button type="button" onClick={() => setPendingDeleteProject(null)} disabled={loading}>取消</button>
              <button
                type="button"
                className="danger"
                disabled={loading}
                onClick={async () => {
                  try {
                    await onDeleteProject(pendingDeleteProject.sessionId);
                    setPendingDeleteProject(null);
                  } catch {
                    // The parent keeps the dialog open and displays the API error.
                  }
                }}
              >
                {loading ? <><Loader2 className="spin" size={14} />删除中</> : '确认删除'}
              </button>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}

function DashboardView({
  profile,
  stage,
  averageAccuracy,
  overallProgress,
  targetDailyHours,
  setView,
  loading,
  error,
  embedded = false
}: {
  profile: Profile;
  stage: StageInfo;
  averageAccuracy: number | null;
  overallProgress: number | null;
  targetDailyHours: number;
  setView: (view: View) => void;
  loading: boolean;
  error: string | null;
  embedded?: boolean;
}) {
  const gaps = profile.knowledge_gaps || [];
  const totalKnowledgeSubtopics = knowledgeGraphBlocks.reduce((total, block) => total + block.subtopics.length, 0);
  const [masteredSubtopics, setMasteredSubtopics] = useState<number | null>(null);
  const foundations = (profile.skill_tags || []).filter(Boolean);
  const gapNames = gaps.map((gap) => gap.skill).filter(Boolean);

  useEffect(() => {
    let active = true;
    apiClient.getKnowledgeNetwork()
      .then((network) => {
        if (active) setMasteredSubtopics(network.mastered_subtopics);
      })
      .catch(() => {
        if (active) setMasteredSubtopics(null);
      });
    return () => { active = false; };
  }, []);

  const content = (
    <>
        <div className="dashboard-head">
          <div>
            <span>{profile.goal || 'LoRA 微调与部署'}</span>
            <h2>学习进展总览</h2>
          </div>
          <button className="primary" onClick={() => setView('chat')}>
            <Play size={16} />
            继续学习
          </button>
        </div>

        <div className="dashboard-profile-grid">
          <div className="metric dashboard-profile-tags">
            <span className="metric-icon blue"><BookOpen size={20} /></span>
            <div>
              <small>已有基础</small>
              <div className="dashboard-profile-tag-list">
                {foundations.length
                  ? foundations.map((foundation) => <strong key={foundation}>{foundation}</strong>)
                  : <em>暂未记录</em>}
              </div>
            </div>
          </div>
          <div className="metric dashboard-profile-tags gaps">
            <span className="metric-icon orange"><Target size={20} /></span>
            <div>
              <small>知识盲区</small>
              <div className="dashboard-profile-tag-list">
                {gapNames.length
                  ? gapNames.map((gap) => <strong key={gap}>{gap}</strong>)
                  : <em>暂未发现</em>}
              </div>
            </div>
          </div>
          <div className="metric">
            <span className="metric-icon green"><Clock size={20} /></span>
            <div>
              <small>学习时长</small>
              <strong>{targetDailyHours}<em> 小时/阶段</em></strong>
              <p>当前目标的每阶段学习时长</p>
            </div>
          </div>
          <div className="metric">
            <span className="metric-icon blue"><Gauge size={22} /></span>
            <div>
              <small>平均正确率</small>
              <strong>{averageAccuracy == null ? '--' : pct(averageAccuracy)}{averageAccuracy != null && <em>%</em>}</strong>
              <p>{averageAccuracy == null ? '完成阶段测试后生成' : '所有阶段测试结果平均值'}</p>
            </div>
          </div>
          <div className="metric">
            <span className="metric-icon green"><Target size={22} /></span>
            <div>
              <small>当前进度</small>
              <strong>{overallProgress == null ? '--' : pct(overallProgress)}<em>%</em></strong>
              <p>已掌握知识点占比</p>
            </div>
          </div>
          <div className="metric">
            <span className="metric-icon orange"><Award size={22} /></span>
            <div><small>掌握知识点</small><strong>{masteredSubtopics ?? '--'}<em>/{totalKnowledgeSubtopics}</em></strong><p>来自知识网络状态</p></div>
          </div>
        </div>

        <div className="dashboard-grid">
          <section className="white-card">
            <div className="card-title">
              <div>
                <h3>知识网络</h3>
                <p>模块、知识点及前置关系</p>
              </div>
            </div>
            <KnowledgeNetwork profile={profile} stage={stage} defaultMode="overview" />
          </section>
        </div>
    </>
  );

  if (embedded) {
    return <section className="dashboard dashboard-overview-section">{content}</section>;
  }

  return (
    <div className="page-shell">
      <Topbar title="学习仪表盘" subtitle="阶段、掌握度和策略决策汇总" loading={loading} error={error} />
      <main className="dashboard">{content}</main>
    </div>
  );
}

function ResultView({
  profile,
  stage,
  plannedStage,
  existingNextStage,
  result,
  onPlanNext,
  onStartNext,
  onRestartCurrent,
  onContinueNext,
  onReturnToChat,
  onOpenAssessmentReview,
  hasAssessmentReview,
  readOnlyReview,
  loading,
  error
}: {
  profile: Profile;
  stage: StageInfo;
  plannedStage: StageInfo | null;
  existingNextStage: StageInfo | null;
  result: StageResult | null;
  onPlanNext: () => void;
  onStartNext: (nextStage?: StageInfo | null) => void;
  onRestartCurrent: () => void;
  onContinueNext: () => void;
  onReturnToChat: () => void;
  onOpenAssessmentReview: () => void;
  hasAssessmentReview: boolean;
  readOnlyReview: boolean;
  loading: boolean;
  error: string | null;
}) {
  const bypass = Boolean(result?.bypass_assessment);
  const hasAssessment = !bypass && typeof result?.accuracy === 'number';
  const accuracy = result?.accuracy;
  const gaps = profile.knowledge_gaps || [];
  const mastered = gaps.filter((gap) => gap.status === '已掌握' || (gap.progress ?? 0) >= 1);
  const needsWork = gaps.filter((gap) => gap.status !== '已掌握' && (gap.progress ?? 0) < 1);
  const canPlanNext = result?.can_plan_next_stage ?? result?.available_actions?.includes('plan_next_stage') ?? false;
  const needsReinforcement = hasAssessment && (accuracy ?? 0) < 0.7;
  const [networkZoom, setNetworkZoom] = useState(1.0);
  const [networkMode, setNetworkMode] = useState<'overview' | 'focus'>('overview');
  // A later stage may already be underway, not merely waiting in next_stage.
  // In either case this summary must never offer to generate a duplicate.
  const availableNextStage = plannedStage || existingNextStage;
  const actionMode = resolveStageResultActionMode({
    allMastered: Boolean(result?.all_mastered),
    historicalReview: readOnlyReview,
    needsReinforcement,
    hasNextStage: Boolean(availableNextStage)
  });
  const heroTitle = result?.all_mastered
    ? '学习目标已完成'
    : bypass
      ? '本阶段无测试题目，已直接完成'
      : (accuracy ?? 0) >= 0.7
        ? '很好，可以进入下一阶段'
        : '建议先巩固当前阶段';
  const heroSubtitle = readOnlyReview
    ? '这是已经完成的历史阶段，仅用于回顾总结、测试题目和画像变化。'
    : result?.all_mastered
      ? '全部目标知识点均已掌握，系统不会再生成新的学习阶段。'
      : bypass
        ? '系统已结合本阶段学习记录生成画像更新和后续建议。'
        : '系统已结合本阶段学习记录和测试结果生成反馈，并更新为画像 v{version}。'.replace('{version}', String(profile.version ?? 1));
  return (
    <div className="page-shell">
      <Topbar
        title="阶段结果"
        subtitle={readOnlyReview ? '历史阶段总结，仅供回顾' : result?.all_mastered ? '学习目标已完成，可查看最终总结与画像变化' : bypass ? '阶段完成（无测试题目）' : '阶段完成后展示画像更新和下一阶段策略'}
        loading={loading}
        error={error}
        onReturnToChat={onReturnToChat}
      />
      <main className="result-page">
        <section className="result-hero">
          {hasAssessment ? (
            <div className="score-ring" style={{ '--score': `${Math.round((accuracy ?? 0) * 100)}%` } as CSSProperties}>
              <span><strong>{pct(accuracy!)}</strong><small>正确率</small></span>
            </div>
          ) : (
            <div className="score-ring no-score" aria-hidden>
              <AlertTriangle size={28} />
              <small>无测试题目</small>
            </div>
          )}
          <div>
            <span className="success-label"><Check size={14} /> {readOnlyReview ? '历史回顾' : result?.all_mastered ? '目标完成' : '阶段完成'}</span>
            <h2>{heroTitle}</h2>
            <p>{heroSubtitle}</p>
          </div>
          {result?.all_mastered ? (
            <div className="strategy no-strategy">
              <span>学习目标</span>
              <strong>已全部掌握</strong>
              <small>Agent6 已终止后续阶段规划</small>
            </div>
          ) : bypass ? (
            <div className="strategy no-strategy">
              <span>阶段状态</span>
              <strong>直接完成</strong>
              <small>当前知识点未收录测试题</small>
            </div>
          ) : (
            <div className="strategy">
              <span>{readOnlyReview ? '历史阶段策略' : '下一阶段策略'}</span>
              <strong>{result?.next_strategy || ((accuracy ?? 0) >= 0.7 ? 'advance' : 'reinforce')}</strong>
              <small>Agent6 综合决策</small>
            </div>
          )}
        </section>
        <div className="feedback-summary-grid">
          <section className="white-card feedback-overview">
            <div className="section-title"><div><span>反馈总结</span><h3>本阶段学习情况</h3></div><b>{statusText[stage.status || 'completed'] || stage.status}</b></div>
            <div className="feedback-facts">
              <div><span>学习主题</span><strong>{stage.block_name || '当前阶段'}</strong></div>
              {hasAssessment && (
                <div><span>测试结果</span><strong>{pct(accuracy!)}% 正确率</strong></div>
              )}
              {!bypass && (
                <div><span>策略建议</span><strong>{result?.next_strategy || ((accuracy ?? 0) >= 0.7 ? 'advance' : 'reinforce')}</strong></div>
              )}
              {bypass && (
                <div><span>测试情况</span><strong>本阶段无测试题目</strong></div>
              )}
            </div>
            <div className="feedback-knowledge-columns">
              <div><span>已掌握知识点</span><p>{mastered.map((gap) => gap.skill).join('、') || '本阶段暂无新增完全掌握项'}</p></div>
              <div><span>仍需加强</span><p>{needsWork.map((gap) => gap.skill).join('、') || '当前目标知识点已全部掌握'}</p></div>
            </div>
            <div className="feedback-markdown"><ReactMarkdown remarkPlugins={[remarkGfm]}>{result?.stage_summary || '本阶段记录已完成。建议根据薄弱知识点继续学习。'}</ReactMarkdown></div>
            {hasAssessmentReview && (
              <button className="assessment-review-button" onClick={onOpenAssessmentReview}>
                <FileQuestion size={16} />
                查看测试题目
              </button>
            )}
          </section>
          <section className="white-card summary-knowledge-network-card">
            <div className="section-title">
              <div><span>画像变化</span><h3>知识网络更新</h3></div>
              <div className="section-title-actions">
                <div className="knowledge-network-zoom-controls" role="group" aria-label="知识点图放大缩小">
                  <button
                    type="button"
                    className="knowledge-zoom-button zoom-out"
                    onClick={() => setNetworkZoom((z) => Math.max(0.5, +(z - 0.1).toFixed(2)))}
                    disabled={networkZoom <= 0.5}
                    aria-label="缩小知识点图"
                    title={`缩小（当前 ${Math.round(networkZoom * 100)}%）`}
                  >
                    <Minus size={14} />
                  </button>
                  <span className="knowledge-zoom-indicator" aria-live="polite">{Math.round(networkZoom * 100)}%</span>
                  <button
                    type="button"
                    className="knowledge-zoom-button zoom-in"
                    onClick={() => setNetworkZoom((z) => Math.min(1.6, +(z + 0.1).toFixed(2)))}
                    disabled={networkZoom >= 1.6}
                    aria-label="放大知识点图"
                    title={`放大（当前 ${Math.round(networkZoom * 100)}%）`}
                  >
                    <Plus size={14} />
                  </button>
                </div>
                <button
                  type="button"
                  className="knowledge-locate-button"
                  onClick={() => setNetworkMode(networkMode === 'focus' ? 'overview' : 'focus')}
                  aria-pressed={networkMode === 'focus'}
                >
                  <LocateFixed size={15} />{networkMode === 'focus' ? '取消定位' : '定位'}
                </button>
                <b>v{profile.version ?? 1}</b>
              </div>
            </div>
            <KnowledgeNetwork
              profile={profile}
              stage={stage}
              compact
              externalZoom={networkZoom}
              onZoomChange={setNetworkZoom}
              externalMode={networkMode}
              onModeChange={setNetworkMode}
              showLocateControl={false}
            />
            <div className="updated-profile-facts">
              <div><span>信心指数</span><strong>{pct(profile.confidence_index ?? 0)}%</strong></div>
              <div><span>困惑标记</span><strong>{profile.confusion_markers?.join('、') || '暂无'}</strong></div>
            </div>
          </section>
          {actionMode !== 'review_only' && (
            <section className="white-card next-stage">
              <div className="stage-eyebrow">
                <span>{availableNextStage ? '下一阶段已就绪' : '下一阶段建议'}</span>
                <b>阶段 {availableNextStage?.stage_num ?? (stage.stage_num ?? 1) + 1}</b>
              </div>
              <h3>{availableNextStage?.block_name || '根据新画像按需生成'}</h3>
              <p>{availableNextStage?.gap_skill ? `下一阶段将继续处理「${availableNextStage.gap_skill}」。` : '生成动作只会创建下一个阶段，不会一次生成全部学习资料。'}</p>
              <div className="next-facts">
                <div><span>预计时长</span><strong>{availableNextStage?.estimated_hours ?? stage.estimated_hours ?? 1} 小时</strong></div>
                <div><span>学习方式</span><strong>{styleText[profile.learning_style || 'mixed']}</strong></div>
                <div><span>画像依据</span><strong className="green">v{profile.version ?? 1}</strong></div>
              </div>
              {actionMode === 'reinforce_and_start' || actionMode === 'reinforce_and_plan' ? (
                <div className="low-score-actions">
                  <button className="ghost wide" onClick={onRestartCurrent} disabled={loading}>
                    <Play size={17} />正确率低，建议重新开始学习该阶段
                  </button>
                  {actionMode === 'reinforce_and_start' ? (
                    <button className="primary wide" onClick={() => onStartNext(availableNextStage)} disabled={loading}>
                      <Play size={17} />开始下一阶段
                    </button>
                  ) : (
                    <button className="primary wide" onClick={onContinueNext} disabled={loading || !canPlanNext}>
                      <Sparkles size={17} />继续进行下一个阶段
                    </button>
                  )}
                </div>
              ) : actionMode === 'start_next' ? (
                <button className="primary wide" onClick={() => onStartNext(availableNextStage)} disabled={loading}><Play size={17} />开始下一阶段</button>
              ) : (
                <button className="primary wide" onClick={() => onPlanNext()} disabled={loading || !canPlanNext}><Sparkles size={17} />生成下一阶段</button>
              )}
            </section>
          )}
        </div>
      </main>
    </div>
  );
}

function SettingsView({
  profile,
  loading,
  error,
  onClear,
  onSave,
  foundationNames,
  onOpenFoundation
}: {
  profile: Profile;
  loading: boolean;
  error: string | null;
  onClear: () => void;
  onSave: (data: { grade: string; skill_tags: string[]; hardware: string }) => void;
  foundationNames: string[];
  onOpenFoundation: () => void;
}) {
  const [grade, setGrade] = useState(profile.grade || '');
  const [hardware, setHardware] = useState(profile.hardware || '');

  useEffect(() => {
    setHardware(profile.hardware || '');
    setGrade(profile.grade || '');
  }, [profile.hardware, profile.grade]);

  const hasProfileEdits = hardware.trim() !== (profile.hardware || '').trim()
    || grade.trim() !== (profile.grade || '').trim();

  return (
    <div className="page-shell">
      <Topbar title="设置" subtitle="用户信息与设备条件" loading={loading} error={error} />
      <main className="settings-page">
        <div className="settings-intro">
          <h2>用户基础与设备条件</h2>
        </div>
        {error && <div className="settings-inline-state error">{error}</div>}
        {(!profile.grade || !profile.skill_tags?.length) && (
          <div className="settings-inline-state empty">用户基础尚不完整，请补充身份背景和已有知识基础。</div>
        )}
        <section className="settings-card">
          <div className="setting-row">
            <div><strong>身份/学习背景</strong><p>用于生成符合你基础的学习内容</p></div>
            <input value={grade} onChange={(event) => setGrade(event.target.value)} placeholder="例如 在校大学生 / 有基础的开发者" />
          </div>
          <div className="setting-row">
            <div><strong>设备条件</strong><p>影响视频、代码与部署建议</p></div>
            <input value={hardware} onChange={(event) => setHardware(event.target.value)} placeholder="例如 RTX 4060 / MacBook / 无独显" />
          </div>
          <div className="setting-row settings-knowledge-row" aria-labelledby="settings-knowledge-title">
            <div className="settings-knowledge-heading">
              <h3 id="settings-knowledge-title">已有知识基础</h3>
            </div>
            <div className="settings-foundation-summary">
              {foundationNames.length > 0 ? (
                <div className={`foundation-summary-list ${foundationNames.length > 3 ? 'has-overflow' : ''}`} title={foundationNames.join('、')}>
                  {foundationNames.map((name) => <span key={name}>{name}</span>)}
                </div>
              ) : <span className="foundation-summary-empty">尚未确定基础</span>}
              <button type="button" className="foundation-trigger" onClick={onOpenFoundation}>重新确定基础</button>
            </div>
            <KnowledgeNetwork compact defaultMode="overview" showNodeDetails={false} showLocateControl={true} />
          </div>
          <div className="setting-row">
            <div><strong>保存用户基础</strong><p>调用用户级接口，不会修改任一目标的时长</p></div>
            <button
              className="primary settings-save-button"
              onClick={() => onSave({
                grade: grade.trim(),
                skill_tags: profile.skill_tags || [],
                hardware: hardware.trim()
              })}
              disabled={loading || !hasProfileEdits}
            >
              {loading ? <><Loader2 className="spin" size={14} />保存中</> : '保存设置'}
            </button>
          </div>
          <div className="setting-row">
            <div><strong>重新开始</strong><p>放弃当前学习目标并清除本地会话缓存</p></div>
            <button className="danger" onClick={onClear}>清除会话并重新开始</button>
          </div>
        </section>
      </main>
    </div>
  );
}

export default function App() {
  const genProgress = useGenerationProgress();
  const [view, setView] = useState<View>('chat');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [workflow, setWorkflow] = useState<WorkflowState>('onboarding');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [profile, setProfile] = useState<Profile | null>(null);
  const [storedUserProfile, setStoredUserProfile] = useState<Profile | null>(null);
  const [stage, setStage] = useState<StageInfo | null>(null);
  const [activeStageNum, setActiveStageNum] = useState<number | null>(null);
  const [learningPackage, setLearningPackage] = useState<LearningPackage | null>(null);
  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [input, setInput] = useState('');
  const [onboardingForm, setOnboardingForm] = useState<OnboardingFormData>(emptyOnboardingForm);
  const [followupQuestions, setFollowupQuestions] = useState<string[]>([]);
  const [highlightedFields, setHighlightedFields] = useState<OnboardingField[]>([]);
  const [timerSeconds, setTimerSeconds] = useState(0);
  const [timerRunning, setTimerRunning] = useState(false);
  const [dailyProgress, setDailyProgress] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [testOpen, setTestOpen] = useState(false);
  const [noQuestionsDialog, setNoQuestionsDialog] = useState<{
    open: boolean;
    goal: string;
    blockName: string;
  }>({ open: false, goal: '', blockName: '' });
  const [result, setResult] = useState<StageResult | null>(null);
  const [goalCompletionOpen, setGoalCompletionOpen] = useState(false);
  const [plannedStage, setPlannedStage] = useState<StageInfo | null>(null);
  const [assessmentState, setAssessmentState] = useState<AssessmentState>({ attemptId: null, questions: [] });
  const [assessmentReview, setAssessmentReview] = useState<AssessmentReview | null>(null);
  const [assessmentReviewOpen, setAssessmentReviewOpen] = useState(false);
  const [agentStatuses, setAgentStatuses] = useState<AgentStatusMap>(createAgentStatusMap);
  const [learningProjects, setLearningProjects] = useState<LearningProject[]>(() => {
    return readLearningProjects().filter((item) => item.sessionId !== DEMO_SESSION_ID);
  });
  const [stageConversations, setStageConversations] = useState<StoredStageConversation[]>(() => {
    const conversations = readStageConversations();
    return conversations.filter((item) => item.sessionId !== DEMO_SESSION_ID);
  });
  const [selectedConversationId, setSelectedConversationId] = useState('');
  const [projectSwitcherOpen, setProjectSwitcherOpen] = useState(false);
  const [materialPanelOpen, setMaterialPanelOpen] = useState(false);
  const [learningPathOpen, setLearningPathOpen] = useState(false);
  const [selectedSubtopicId, setSelectedSubtopicId] = useState<string | null>(null);
  const [foundationPickerContext, setFoundationPickerContext] = useState<'onboarding' | 'settings' | null>(null);
  const [foundationPickerBlocks, setFoundationPickerBlocks] = useState<KnowledgeGraphBlock[]>(knowledgeGraphBlocks);
  const [foundationSubtopicIds, setFoundationSubtopicIds] = useState<string[]>([]);
  const [foundationConfirmed, setFoundationConfirmed] = useState(false);
  const [foundationPickerLoading, setFoundationPickerLoading] = useState(false);
  const [profileModalOpen, setProfileModalOpen] = useState(false);
  const [newGoalReturnState, setNewGoalReturnState] = useState<WorkspaceSnapshot | null>(null);
  const [reuseUserProfile, setReuseUserProfile] = useState(false);
  const [checkingUserProfile, setCheckingUserProfile] = useState(() => !isUiPreviewMode);
  const [profileCheckError, setProfileCheckError] = useState<string | null>(null);
  const [projectsLoading, setProjectsLoading] = useState(!isUiPreviewMode);
  const [projectsError, setProjectsError] = useState<string | null>(null);
  const initialAutoRestoreRef = useRef({ tried: false });

  const beginAgentRun = (agentIds: AgentId[], reset = false) => {
    setAgentStatuses((current) => startAgents(current, agentIds, reset));
  };

  const finishAgentRun = (agentIds: AgentId[], succeeded: boolean) => {
    setAgentStatuses((current) => finishAgents(current, agentIds, succeeded));
  };

  const rememberProject = (
    nextSessionId: string,
    nextProfile: Profile,
    nextStage?: StageInfo | null,
    phase?: string
  ) => {
    rememberOwnedSessionId(nextSessionId);
    setLearningProjects((prev) => {
      const existing = prev.find((item) => item.sessionId === nextSessionId);
      // 导航是“学习阶段”视图，不是重试审计视图；同一编号只保留一条。
      const byStageNum = new Map<number, StageInfo>();
      for (const item of existing?.stages || []) byStageNum.set(item.stage_num ?? 1, item);
      const knownStages = [...byStageNum.values()];
      if (nextStage) {
        const stageNum = nextStage.stage_num ?? 1;
        const index = knownStages.findIndex((item) => (item.stage_num ?? 1) === stageNum);
        if (index >= 0) knownStages[index] = { ...knownStages[index], ...nextStage };
        else knownStages.push(nextStage);
      }
      const project: LearningProject = {
        sessionId: nextSessionId,
        goal: nextProfile.goal || '未命名学习目标',
        // A pending stage is a real plan returned by the backend. Keep it in
        // the summary so a planned next stage cannot look like stale onboarding.
        stageName: nextStage?.block_name || existing?.stageName || '等待阶段规划',
        stageNum: nextStage?.stage_num ?? existing?.stageNum ?? 1,
        phase: phase || existing?.phase,
        createdAt: existing?.createdAt ?? Date.now(),
        updatedAt: Date.now(),
        profile: nextProfile,
        stages: knownStages.sort((a, b) => (a.stage_num ?? 1) - (b.stage_num ?? 1)),
        targetTotalHours: existing?.targetTotalHours
          ?? knownStages.reduce((sum, item) => sum + (item.estimated_hours || 0), 0),
        targetDailyHours: existing?.targetDailyHours ?? nextProfile.daily_hours,
        averageAccuracy: existing?.averageAccuracy,
        overallProgress: existing?.overallProgress
      };
      return [project, ...prev.filter((item) => item.sessionId !== nextSessionId)].slice(0, 20);
    });
  };

  const rememberStageConversation = (nextSessionId: string, nextStage: StageInfo, nextProfile?: Profile | null) => {
    const stageNum = nextStage.stage_num ?? 1;
    const id = `${nextSessionId}:${nextStage.stage_id || stageNum}`;
    setStageConversations((prev) => {
      if (prev.some((item) => item.id === id)) return prev;
      return [{
        id,
        sessionId: nextSessionId,
        stage: nextStage,
        messages: createStageConversationMessages(nextStage, nextProfile),
        updatedAt: Date.now()
      }, ...prev].slice(0, 100);
    });
  };

  const rememberTargetSettings = (nextSessionId: string, settings?: Record<string, unknown>) => {
    if (!settings) return;
    setLearningProjects((projects) => projects.map((project) => project.sessionId === nextSessionId ? {
      ...project,
      targetTotalHours: typeof settings.target_total_hours === 'number'
        ? settings.target_total_hours
        : project.targetTotalHours,
      targetDailyHours: typeof settings.target_daily_hours === 'number'
        ? settings.target_daily_hours
        : project.targetDailyHours
    } : project));
  };

  const mergeDashboardStages = (
    nextSessionId: string,
    history: StageInfo[],
    currentStage?: StageInfo | null,
    averageAccuracy?: number | null,
    overallProgress?: number
  ) => {
    setLearningProjects((prev) => prev.map((project) => {
      if (project.sessionId !== nextSessionId) return project;
      const byStageKey = new Map<number, StageInfo>();
      const sourceStages = (history.length || currentStage)
        ? [...history, ...(currentStage ? [currentStage] : [])]
        : (project.stages || []);
      sourceStages.forEach((item) => {
        const stageNum = item.stage_num ?? 1;
        const key = stageNum;
        byStageKey.set(key, { ...(byStageKey.get(key) || {}), ...item, stage_num: stageNum });
      });
      return {
        ...project,
        stages: [...byStageKey.values()].sort((a, b) => (a.stage_num ?? 1) - (b.stage_num ?? 1)),
        averageAccuracy: averageAccuracy === undefined ? project.averageAccuracy : averageAccuracy,
        overallProgress: overallProgress === undefined ? project.overallProgress : overallProgress
      };
    }));
  };

  const loadDashboardStages = async (nextSessionId: string, currentStage?: StageInfo | null) => {
    try {
      const dashboard = await apiClient.getDashboard(nextSessionId);
      const history = Array.isArray(dashboard?.stage_history) ? dashboard.stage_history as StageInfo[] : [];
      mergeDashboardStages(
        nextSessionId,
        history,
        currentStage,
        dashboard.average_accuracy,
        typeof dashboard.overall_progress === 'number' ? dashboard.overall_progress : undefined
      );
    } catch {
      // Session restoration remains usable when the optional dashboard endpoint is unavailable.
    }
  };

  const refreshRemoteProjects = async () => {
    if (isUiPreviewMode) {
      setProjectsLoading(false);
      setProjectsError(null);
      return;
    }
    setProjectsLoading(true);
    setProjectsError(null);
    try {
      const response = await apiClient.listSessions();
      const sessionItems = (response.sessions || [])
        .filter((item) => item.status !== 'abandoned')
      // The compact session list only exposes the current stage. Hydrate the
      // durable history before rendering the navigation so completed stages
      // never disappear just because an old browser cache is incomplete.
      const dashboardResults = await Promise.allSettled(
        sessionItems.map((item) => apiClient.getDashboard(item.session_id))
      );
      const remoteProjects = sessionItems
        .map(sessionSummaryToProject)
        .filter((item): item is LearningProject => Boolean(item))
        .map((project, index) => {
          const dashboardResult = dashboardResults[index];
          if (dashboardResult?.status !== 'fulfilled') return project;
          const dashboard = dashboardResult.value;
          const history = Array.isArray(dashboard.stage_history)
            ? dashboard.stage_history as StageInfo[]
            : [];
          const current = dashboard.current_stage_summary as StageInfo | undefined;
          const stagesByKey = new Map<number, StageInfo>();
          [...history, ...(current ? [current] : []), ...(project.stages || [])].forEach((item) => {
            const stageNum = item.stage_num ?? 1;
            const key = stageNum;
            stagesByKey.set(key, { ...stagesByKey.get(key), ...item });
          });
          const stages = [...stagesByKey.values()]
            .sort((left, right) => (left.stage_num ?? 1) - (right.stage_num ?? 1));
          return {
            ...project,
            stages,
            averageAccuracy: typeof dashboard.average_accuracy === 'number'
              ? dashboard.average_accuracy
              : project.averageAccuracy,
            overallProgress: typeof dashboard.overall_progress === 'number'
              ? dashboard.overall_progress
              : project.overallProgress,
          };
        });
      setLearningProjects((prev) => {
        const localOnly = prev.filter((item) => isLocalSessionId(item.sessionId));
        const mergedRemote = remoteProjects.map((remote) => {
          const cached = prev.find((item) => item.sessionId === remote.sessionId);
          if (!cached) return remote;
          const remoteStages = remote.stages || [];
          return {
            ...cached,
            ...remote,
            // Dashboard history is authoritative after hydration. Do not
            // re-add stale retry snapshots from browser localStorage.
            stages: [...remoteStages]
              .sort((left, right) => (left.stage_num ?? 1) - (right.stage_num ?? 1)),
            stageName: remote.stageName !== '等待阶段规划' ? remote.stageName : cached.stageName,
            stageNum: remote.stageName !== '等待阶段规划' ? remote.stageNum : cached.stageNum,
          };
        });
        return [...mergedRemote, ...localOnly]
          .sort((a, b) => b.updatedAt - a.updatedAt)
          .slice(0, 20);
      });
    } catch (caught) {
      setProjectsError(`学习目标读取失败：${getErrorMessage(caught)}`);
      // Keep local cache if the session list endpoint is unavailable.
    } finally {
      setProjectsLoading(false);
    }
  };

  const checkUserFoundation = async () => {
    setCheckingUserProfile(true);
    setProfileCheckError(null);
    try {
      const userProfile = await apiClient.getUserProfile();
      setStoredUserProfile({
        grade: userProfile.grade,
        skill_tags: userProfile.skill_tags,
        foundation_confirmed: userProfile.foundation_confirmed,
        hardware: userProfile.hardware,
        learning_style: userProfile.learning_style,
        learning_rhythm: userProfile.learning_rhythm
      });
      const profileComplete = Boolean(
        userProfile.grade?.trim()
        && userProfile.hardware?.trim()
        && userProfile.foundation_confirmed
      );
      setReuseUserProfile(profileComplete);
      setFoundationConfirmed(Boolean(userProfile.foundation_confirmed));
      setOnboardingForm((current) => ({
        ...formDataFromProfile(userProfile),
        goal: current.goal,
        daily_hours: current.daily_hours,
        learning_style: current.learning_style
      }));
    } catch (caught) {
      const status = typeof caught === 'object' && caught && 'response' in caught
        ? (caught as { response?: { status?: number } }).response?.status
        : undefined;
      if (status === 404) {
        setReuseUserProfile(false);
        setFoundationConfirmed(false);
      } else {
        setProfileCheckError(getErrorMessage(caught));
      }
    } finally {
      setCheckingUserProfile(false);
    }
  };

  const restoreQaConversation = async (
    nextSessionId: string,
    nextStage: StageInfo,
    nextProfile?: Profile | null,
    replaceCurrentMessages = false
  ) => {
    try {
      const history = await apiClient.getQaHistory(nextSessionId, nextStage.stage_num);
      const qaMessages = qaItemsToMessages(history.items);
      const id = `${nextSessionId}:${nextStage.stage_num ?? 1}`;
      const restoredMessages = qaMessages.length
        ? qaMessages
        : createStageConversationMessages(nextStage, nextProfile);
      if (replaceCurrentMessages) {
        setMessages(restoredMessages);
        setSelectedConversationId(id);
      }
      setStageConversations((prev) => {
        const existing = prev.find((item) => item.id === id);
        const conversation: StoredStageConversation = {
          id,
          sessionId: nextSessionId,
          stage: nextStage,
          messages: qaMessages.length ? qaMessages : existing?.messages || restoredMessages,
          updatedAt: Date.now()
        };
        return [conversation, ...prev.filter((item) => item.id !== id)].slice(0, 100);
      });
    } catch {
      // Leave local conversation state intact when QA history is unavailable.
    }
  };

  const restoreDiagnosisState = async (nextSessionId: string) => {
    try {
      const history = await apiClient.getDiagnosisMessages(nextSessionId);
      const assistantMessages = (history.messages || []).filter((item) => item.role === 'assistant' && item.content).map((item) => item.content as string);
      if (assistantMessages.length) {
        setFollowupQuestions(assistantMessages.slice(-3));
        setHighlightedFields(inferHighlightedFields(assistantMessages));
      }
    } catch {
      // Keep onboarding usable even if diagnosis history cannot be recovered.
    }
  };

  useEffect(() => {
    localStorage.setItem(PROJECTS_STORAGE_KEY, JSON.stringify(learningProjects));
  }, [learningProjects]);

  useEffect(() => {
    localStorage.setItem(CONVERSATIONS_STORAGE_KEY, JSON.stringify(stageConversations));
  }, [stageConversations]);

  useEffect(() => {
    if (!sessionId || workflow !== 'learning') return;
    localStorage.setItem(LAST_LEARNING_LOCATION_KEY, JSON.stringify({
      sessionId,
      view,
      stageNum: activeStageNum ?? stage?.stage_num ?? null,
      subtopicId: selectedSubtopicId,
      materialPanelOpen,
    }));
  }, [activeStageNum, materialPanelOpen, selectedSubtopicId, sessionId, stage?.stage_num, view, workflow]);

  useEffect(() => {
    if (isUiPreviewMode) {
      setSessionId(DEMO_SESSION_ID);
      setProfile(fallbackProfile);
      setStage(fallbackStage);
      setActiveStageNum(fallbackStage.stage_num ?? 1);
      setLearningPackage(fallbackPackage);
      setMessages(demoMessages);
      setSelectedConversationId(`${DEMO_SESSION_ID}:${fallbackStage.stage_num ?? 1}`);
      setLearningProjects([createDemoProject()]);
      setStageConversations([createDemoConversation()]);
      setWorkflow('learning');
      setAgentStatuses(restoreAgentStatuses({ hasProfile: true, hasLearningPackage: true, phase: 'learning' }));
      setProjectsLoading(false);
      return;
    }
    void refreshRemoteProjects();
  }, []);

  useEffect(() => {
    if (isUiPreviewMode) return;
    // A stale or active session must not decide whether the stable user
    // profile is complete. Only the durable user profile controls the form.
    void checkUserFoundation();
  }, []);

  useEffect(() => {
    if (!sessionId || !stage || workflow !== 'learning' || view !== 'chat') return;
    const stageNum = stage.stage_num ?? 1;
    // A historical stage is rendered with its own messages while the active
    // session still belongs to a later stage. Do not let a render during that
    // switch overwrite the historical stage's persisted browser cache.
    if (activeStageNum !== null && stageNum !== activeStageNum) return;
    const id = `${sessionId}:${stageNum}`;
    setSelectedConversationId((selected) => selected || id);
    setStageConversations((prev) => {
      const conversation: StoredStageConversation = {
        id,
        sessionId,
        stage,
        messages,
        updatedAt: Date.now()
      };
      return [conversation, ...prev.filter((item) => item.id !== id)].slice(0, 100);
    });
  }, [activeStageNum, messages, sessionId, stage, workflow, view]);

  useEffect(() => {
    if (isUiPreviewMode) return;
    if (initialAutoRestoreRef.current.tried) return;
    if (projectsLoading) return;
    // 已经有 sessionId（可能是已保存或已切换）就不再自动导航。
    if (sessionId) {
      initialAutoRestoreRef.current.tried = true;
      return;
    }
    // localStorage 中有 savedSessionId：等后续 effect 负责恢复即可。
    if (localStorage.getItem(SESSION_STORAGE_KEY)) {
      initialAutoRestoreRef.current.tried = true;
      return;
    }
    // projects 都加载完了，但没有可用会话——才触发"自动进入最新学习目标最后阶段"
    if (!learningProjects.length) {
      initialAutoRestoreRef.current.tried = true;
      return;
    }
    initialAutoRestoreRef.current.tried = true;
    const sorted = sortProjectsForDashboard(learningProjects, null);
    const latest = sorted[0];
    if (!latest || isLocalSessionId(latest.sessionId)) return;
    void switchLearningProject(latest.sessionId, undefined, 'chat', true, true);
  }, [projectsLoading, learningProjects, sessionId]);

  const dailyHours = Math.max(0.5, Number(profile?.daily_hours ?? 2));
  const dailyTargetSeconds = Math.round(dailyHours * 3600);
  const dailyProgressKey = sessionId ? `daily-study:${sessionId}` : '';

  useEffect(() => {
    if (!dailyProgressKey) return;
    const today = new Date().toISOString().slice(0, 10);
    const stored = JSON.parse(localStorage.getItem(dailyProgressKey) || '{}') as { date?: string; seconds?: number };
    const seconds = stored.date === today ? Math.max(0, Number(stored.seconds) || 0) : 0;
    setTimerSeconds(Math.max(0, dailyTargetSeconds - seconds));
    setDailyProgress(Math.min(1, seconds / dailyTargetSeconds));
    setTimerRunning(false);
  }, [dailyProgressKey, dailyTargetSeconds]);

  useEffect(() => {
    if (!timerRunning) return;
    const intervalId = window.setInterval(() => {
      setTimerSeconds((prev) => {
        const today = new Date().toISOString().slice(0, 10);
        const stored = JSON.parse(localStorage.getItem(dailyProgressKey) || '{}') as { date?: string; seconds?: number };
        const seconds = (stored.date === today ? Number(stored.seconds) || 0 : 0) + 1;
        localStorage.setItem(dailyProgressKey, JSON.stringify({ date: today, seconds }));
        setDailyProgress(Math.min(1, seconds / dailyTargetSeconds));
        if (prev <= 1) {
          window.clearInterval(intervalId);
          setTimerRunning(false);
          setError('当前阶段倒计时结束，可以休息或进入阶段测试。');
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
    return () => window.clearInterval(intervalId);
  }, [dailyProgressKey, dailyTargetSeconds, timerRunning]);

  const toggleStageTimer = () => {
    setError(null);
    if (timerSeconds <= 0) {
      setTimerSeconds(dailyTargetSeconds);
      setTimerRunning(true);
      return;
    }
    setTimerRunning((prev) => !prev);
  };

  useEffect(() => {
    if (isUiPreviewMode) return;
    const savedSessionId = localStorage.getItem(SESSION_STORAGE_KEY);
    if (!savedSessionId) return;
    if (isLocalSessionId(savedSessionId)) {
      localStorage.removeItem(SESSION_STORAGE_KEY);
      setLearningProjects((projects) => projects.filter((project) => !isLocalSessionId(project.sessionId)));
      setStageConversations((conversations) => conversations.filter((conversation) => !isLocalSessionId(conversation.sessionId)));
      setWorkflow('onboarding');
      return;
    }
    // 恢复用户最后离开的项目、页面和知识点，而不是改变项目列表排序。
    let lastLocation: { sessionId?: string; view?: View; stageNum?: number; subtopicId?: string | null; materialPanelOpen?: boolean } = {};
    try {
      lastLocation = JSON.parse(localStorage.getItem(LAST_LEARNING_LOCATION_KEY) || '{}');
    } catch {
      lastLocation = {};
    }
    const restoreSameSession = lastLocation.sessionId === savedSessionId;
    void switchLearningProject(
      savedSessionId,
      restoreSameSession ? lastLocation.stageNum : undefined,
      restoreSameSession ? (lastLocation.view || 'chat') : 'chat',
      true,
      true,
    ).then(() => {
      if (!restoreSameSession) return;
      setSelectedSubtopicId(lastLocation.subtopicId || null);
      setMaterialPanelOpen(Boolean(lastLocation.materialPanelOpen));
    });
  }, []);

  const currentQuestions = useMemo(() => {
    const questions = assessmentState.questions.length
      ? assessmentState.questions
      : learningPackage?.questions?.length
        ? learningPackage.questions
        : [];
    return questions;
  }, [assessmentState.questions, learningPackage]);

  const inferHighlightedFields = (questions: string[]) => {
    const combined = questions.join(' ');
    const fields = new Set<OnboardingField>();
    if (/身份|背景|年级|职业|学生|工程师/.test(combined)) fields.add('grade');
    if (/基础|技能|掌握|会什么|知识/.test(combined)) fields.add('skill_tags');
    if (/目标|想学|希望|解决/.test(combined)) fields.add('goal');
    if (/多久|时长|小时|时间|投入/.test(combined)) fields.add('daily_hours');
    if (/方式|风格|理论|实操|实践|偏好/.test(combined)) fields.add('learning_style');
    if (/设备|显卡|硬件|GPU|电脑/.test(combined)) fields.add('hardware');
    return Array.from(fields);
  };

  const submitOnboarding = async () => {
    if (loading || workflow === 'diagnosing') return;
    setError(null);
    setFollowupQuestions([]);
    setHighlightedFields([]);
    setLoading(true);

    if (!reuseUserProfile) {
      try {
        const updated = await apiClient.updateUserProfile({
          grade: onboardingForm.grade.trim(),
          skill_tags: onboardingForm.skill_tags,
          foundation_confirmed: foundationConfirmed,
          hardware: onboardingForm.hardware.trim(),
          mastered_subtopic_ids: foundationSubtopicIds
        });
        setStoredUserProfile({
          grade: updated.grade,
          skill_tags: updated.skill_tags,
          foundation_confirmed: updated.foundation_confirmed,
          hardware: updated.hardware,
          learning_style: updated.learning_style,
          learning_rhythm: updated.learning_rhythm
        });
        setReuseUserProfile(true);
        setOnboardingForm((current) => ({
          ...current,
          grade: updated.grade || current.grade,
          skill_tags: updated.skill_tags || current.skill_tags,
          hardware: updated.hardware || current.hardware,
          goal: '',
          daily_hours: '',
          learning_style: ''
        }));
        setWorkflow('onboarding');
      } catch (caught) {
        setError(getErrorMessage(caught));
      } finally {
        setLoading(false);
      }
      return;
    }

    setWorkflow('diagnosing');
    beginAgentRun(['agent1'], true);

    try {
      // Persist a newly created session before its first protected request.
      // This keeps a failed first submission retryable and prevents requests
      // from racing ahead of local session initialization.
      const activeSessionId = sessionId || (await apiClient.createSession()).session_id;
      if (!sessionId) {
        setSessionId(activeSessionId);
        localStorage.setItem(SESSION_STORAGE_KEY, activeSessionId);
      }
      const response = await apiClient.submitFoundation(activeSessionId, {
        goal: onboardingForm.goal.trim(),
        mastered_subtopic_ids: foundationSubtopicIds,
        daily_hours: Number(onboardingForm.daily_hours),
        learning_style: onboardingForm.learning_style,
        reuse_user_profile: reuseUserProfile
      });
      setSessionId(activeSessionId);
      localStorage.setItem(SESSION_STORAGE_KEY, activeSessionId);

      if (response.status === 'needs_more_info') {
        const questions = response.followup_questions || [];
        const requestedFields = inferHighlightedFields(questions);
        const visibleFields = reuseUserProfile
          ? requestedFields.filter((field) => field === 'goal' || field === 'daily_hours' || field === 'learning_style')
          : requestedFields;
        setFollowupQuestions(questions);
        setHighlightedFields(visibleFields);
        if (response.profile) {
          const draftProfile = toProfile(response.profile);
          if (draftProfile) {
            const enrichedProfile = {
              ...draftProfile,
              grade: reuseUserProfile ? (draftProfile.grade || onboardingForm.grade.trim()) : onboardingForm.grade.trim(),
              hardware: reuseUserProfile ? (draftProfile.hardware || onboardingForm.hardware.trim()) : onboardingForm.hardware.trim(),
              foundation_confirmed: reuseUserProfile || foundationConfirmed
            };
            setProfile(enrichedProfile);
            setOnboardingForm((current) => ({ ...current, ...formDataFromProfile(enrichedProfile) }));
          }
        }
        setWorkflow('needs_more_info');
        finishAgentRun(['agent1'], true);
        return;
      }

      if (response.profile) {
        const nextProfile = {
          ...(toProfile(response.profile) || fallbackProfile),
          grade: reuseUserProfile ? (response.profile.grade || onboardingForm.grade.trim()) : onboardingForm.grade.trim(),
          hardware: reuseUserProfile ? (response.profile.hardware || onboardingForm.hardware.trim()) : onboardingForm.hardware.trim(),
          foundation_confirmed: reuseUserProfile || foundationConfirmed
        } as Profile;
        setProfile(nextProfile);
        setOnboardingForm(formDataFromProfile(nextProfile));
        if (!reuseUserProfile) {
          await apiClient.updateUserProfile({
            grade: nextProfile.grade,
            skill_tags: nextProfile.skill_tags || [],
            foundation_confirmed: foundationConfirmed,
            learning_style: nextProfile.learning_style,
            hardware: nextProfile.hardware,
            default_daily_hours: nextProfile.daily_hours
          });
        }
        setStage(toStageInfo(response.stage));
        rememberProject(activeSessionId, nextProfile, toStageInfo(response.stage), 'stage_ready');
      }
      void refreshRemoteProjects();
      finishAgentRun(['agent1'], true);
      setWorkflow('profile_ready');
    } catch (caught) {
      finishAgentRun(['agent1'], false);
      setError(getErrorMessage(caught));
      setWorkflow('onboarding');
    } finally {
      setLoading(false);
    }
  };

  const sendMessage = async (event: FormEvent) => {
    event.preventDefault();
    const text = input.trim();
    if (!text || loading) return;
    const selectedStage = stageConversations.find(
      (item) => item.id === selectedConversationId
    )?.stage;
    const questionStageNum = selectedStage?.stage_num ?? activeStageNum ?? stage?.stage_num;
    const appendStageMessage = (message: Message) => {
      if (questionStageNum !== activeStageNum && sessionId && questionStageNum) {
        const conversationId = `${sessionId}:${questionStageNum}`;
        setStageConversations((prev) => prev.map((item) => (
          item.id === conversationId
            ? { ...item, messages: [...item.messages, message], updatedAt: Date.now() }
            : item
        )));
        return;
      }
      setMessages((prev) => [...prev, message]);
    };
    setInput('');
    setError(null);
    appendStageMessage({ id: shortId(), role: 'user', content: text });
    if (isUiPreviewMode) {
      setMessages((prev) => [...prev, {
        id: shortId(),
        role: 'assistant',
        content: '（UI 预览模式：此演示环境不会调用真实后端接口。请将 frontend/.env.local 中 VITE_UI_PREVIEW 设为 false 并重启前端，以使用真实的多 Agent 服务。）',
        note: '预览模式提示'
      }]);
      return;
    }
    setLoading(true);
    const initialDiagnosis = !sessionId || !profile?.goal;
    const invokedAgents: AgentId[] = initialDiagnosis ? ['agent1'] : ['agent3', 'agent4'];
    beginAgentRun(invokedAgents, initialDiagnosis);

    try {
      if (initialDiagnosis) {
        const activeSessionId = sessionId || (await apiClient.createSession()).session_id;
        const response = await apiClient.submitDiagnosis(activeSessionId, text);
        setSessionId(activeSessionId);
        localStorage.setItem(SESSION_STORAGE_KEY, activeSessionId);
        if (response.status === 'needs_more_info') {
          setMessages((prev) => [...prev, {
            id: shortId(),
            role: 'assistant',
            content: `我还需要补充一点信息：\n${(response.followup_questions || []).join('\n')}`,
            note: '需要追问'
          }]);
        } else {
          if (response.profile_draft) {
            const nextProfile = { ...(toProfile(response.profile_draft) || fallbackProfile), confidence: response.confidence } as Profile;
            setProfile(nextProfile);
          }
          setMessages((prev) => [...prev, {
            id: shortId(),
            role: 'assistant',
            content: '画像已经建立，并生成了当前阶段。你可以点击「开始学习阶段」进入学习包。',
            note: '画像完成'
          }]);
        }
      } else {
        if (!sessionId) throw new Error('尚未创建后端会话');
        const response = await apiClient.askQuestion(sessionId, text, questionStageNum ?? undefined);
        appendStageMessage({
          id: shortId(),
          role: 'assistant',
          content: response.answer || '已收到问题，但后端没有返回回答内容。',
          note: response.review_summary?.technical_passed && response.review_summary?.quality_passed ? '审核通过' : '待复核',
          reviewSummary: response.review_summary,
        });
        finishAgentRun(['agent3'], Boolean(response.answer));
        finishAgentRun(['agent4'], Boolean(response.review_summary));
      }
      if (initialDiagnosis) finishAgentRun(['agent1'], true);
    } catch (caught) {
      finishAgentRun(invokedAgents, false);
      setError(getErrorMessage(caught));
      if (!initialDiagnosis) {
        appendStageMessage({
          id: shortId(),
          role: 'assistant',
          content: '本次答疑请求未完成，请检查后端服务后重试。',
          note: '请求失败'
        });
        return;
      }
      setSessionId(null);
      setProfile(null);
      setStage(null);
      setWorkflow('onboarding');
    } finally {
      setLoading(false);
    }
  };

  const startStage = async (stageToStart?: StageInfo | null) => {
    setError(null);
    if (workflow === 'learning' && (result?.all_mastered || isHistoricalStageReview(stage?.stage_num, activeStageNum))) {
      setError('该阶段总结仅用于回顾，不能从这里启动新的学习阶段');
      return;
    }
    if (!sessionId) {
      setError('尚未创建后端会话');
      return;
    }
    if (loading) return;
    setLoading(true);
    setWorkflow('starting_stage');

    const isFirst = workflow === 'profile_ready';
    beginAgentRun(['agent1', 'agent3', 'agent4'], true);

    // 用户可能画像确认后，当前 stage 已经有了（来自 submitFoundation 的返回），
    // 可以作为阶段①的 immediate 预览，弹窗一打开就看到，不用再等。
    // 下一阶段已由 plan-next 生成时必须直接使用它。否则低分用户的
    // force_advance 计划会被启动流程中的普通 plan-next 覆盖为当前阶段复习。
    let initialStage: GenStagePreview | ApiStage | null = stageToStart || plannedStage || null;
    if (isFirst) {
      initialStage = initialStage || stage || null;
    }

    await genProgress.run({
      sessionId,
      action: isFirst ? 'first' : 'next',
      initialStage,
      onStageReady: (readyStage) => {
        // 阶段①完成：把规划好的知识点先写入本地 stage，让用户在阶段②期间能看到
        const mapped: StageInfo = {
          stage_num: readyStage.stage_num,
          block_name: readyStage.block_name,
          subtopic_ids: readyStage.subtopic_ids,
          subtopic_names: readyStage.subtopic_names,
          subtopic_name: readyStage.subtopic_name,
          estimated_hours: readyStage.estimated_hours,
          prerequisites: readyStage.prerequisites,
          gap_skill: readyStage.gap_skill,
        };
        setStage(mapped);
        if (!isFirst) setPlannedStage(mapped);
        finishAgentRun(['agent1'], true);
      },
      onPackageReady: async ({ packageResponse, stageResponse, startResponse }) => {
        // eslint-disable-next-line no-console
        console.log('[startStage] API responses:', {
          'stage.subtopic_ids': stageResponse.stage?.subtopic_ids,
          'stage.subtopic_names': stageResponse.stage?.subtopic_names,
          'stage.block_id': stageResponse.stage?.block_id,
          'package.subtopic_ids': packageResponse.package?.subtopic_ids,
          'package.subtopic_names': packageResponse.package?.subtopic_names,
          'package.learning_resources_count': packageResponse.package?.learning_resources?.length,
          'package.article_resources': (packageResponse.package?.learning_resources || [])
            .filter((r: any) => r.resource_type === 'article').length,
          'package.resource_details': (packageResponse.package?.learning_resources || [])
            .map((r: any) => `${r.resource_type}:${r.subtopic_id || ''}:${r.url?.slice(0, 40) || ''}`),
        });
        const packageWithReview = {
          ...packageResponse.package,
          review_summary: startResponse?.review_summary || packageResponse.package?.review_summary,
        };
        setLearningPackage(toLearningPackage(packageWithReview));
        const generatedStage = toStageInfo(stageResponse.stage);
        if (!generatedStage) throw new Error('后端未返回当前阶段，无法启动学习包');
        setStage(generatedStage);
        setActiveStageNum(generatedStage.stage_num ?? 1);
        const generatedConversationId = `${sessionId}:${generatedStage.stage_num ?? 1}`;
        const existingConversation = stageConversations.find((item) => item.id === generatedConversationId);
        setMessages(existingConversation?.messages || createStageConversationMessages(generatedStage, profile));
        setSelectedConversationId(generatedConversationId);
        const latestProfile = generatedStage ? profile : profile;
        if (latestProfile && generatedStage) rememberProject(sessionId, latestProfile, generatedStage);
        rememberStageConversation(sessionId, generatedStage, latestProfile);
        finishAgentRun(['agent3'], Boolean(packageResponse.package));
        finishAgentRun(['agent4'], Boolean(packageResponse.package?.review_summary));
        setMessages((prev) => [...prev, {
          id: shortId(),
          role: 'assistant',
          content: startResponse?.review_summary?.iterations
            ? `学习包已生成，Agent4 已完成 ${startResponse.review_summary.iterations} 轮技术与教学审核。`
            : '学习包已生成，Agent4 已完成技术与教学审核。',
          note: '学习包'
        }]);
        setWorkflow('learning');
        setView('chat');
        setMaterialPanelOpen(true);
        setPlannedStage(null);
        setResult(null);
        setNewGoalReturnState(null);
        setAssessmentState({ attemptId: null, questions: [] });
        void refreshRemoteProjects();
      },
      onError: (caught) => {
        finishAgentRun(['agent1', 'agent3', 'agent4'], false);
        const msg = getErrorMessage(caught);
        setError(msg);
        setWorkflow(isFirst ? 'profile_ready' : workflow);
      },
      onFinally: () => {
        setLoading(false);
      },
    });
  };

  const completeStageWithPayload = async (
    options: { bypassAssessment?: boolean },
    answers?: Record<string, string>
  ) => {
    setLoading(true);
    setError(null);
    beginAgentRun(['agent6']);
    try {
      if (sessionId) {
        let submittedAttempt: AssessmentReview | null = null;
        if (!options.bypassAssessment) {
          const attemptId = assessmentState.attemptId || (await apiClient.createAssessmentAttempt(sessionId)).attempt_id;
          const attempt = await apiClient.getAssessmentAttempt(attemptId);
          if (attempt.status !== 'submitted') {
            if (answers) {
              for (const [question_id, user_answer] of Object.entries(answers)) {
                await apiClient.saveAssessmentAnswer(attemptId, question_id, { user_answer, time_spent: 0 });
              }
            }
            submittedAttempt = await apiClient.submitAssessmentAttempt(attemptId) as AssessmentReview;
          }
          if (!submittedAttempt) submittedAttempt = attempt as AssessmentReview;
        }
        const response = await apiClient.completeStage(sessionId, options);
        setResult(response);
        setGoalCompletionOpen(Boolean(response.all_mastered));
        if (response.updated_profile) setProfile(toProfile(response.updated_profile));
        const completedStage = response.state?.stage_info || stage;
        if (completedStage) {
          setStage(completedStage);
          setActiveStageNum(completedStage.stage_num ?? activeStageNum);
        }
        setPlannedStage(null);
        setAssessmentState({ attemptId: null, questions: [] });
        setAssessmentReview(submittedAttempt);
        rememberProject(sessionId, toProfile(response.updated_profile) || profile || fallbackProfile, completedStage, response.phase);
        void loadDashboardStages(sessionId, completedStage);
        void refreshRemoteProjects();
        setTestOpen(false);
        setView('result');
        finishAgentRun(['agent6'], true);
      } else {
        throw new Error('尚未创建后端会话');
      }
    } catch (caught) {
      finishAgentRun(['agent6'], false);
      setError(getErrorMessage(caught));
    } finally {
      setLoading(false);
    }
  };

  const completeStage = async (answers: Record<string, string>) => completeStageWithPayload({}, answers);

  const planNextStage = async (forceAdvance = false) => {
    if (
      !sessionId
      || loading
      || result?.all_mastered
      || isHistoricalStageReview(stage?.stage_num, activeStageNum)
    ) return;
    setLoading(true);
    setError(null);
    beginAgentRun(['agent1']);
    try {
      const response = await apiClient.planNextStage(sessionId, '', forceAdvance);
      if (response.all_mastered) {
        const completedProfile = toProfile(response.updated_profile) || profile || fallbackProfile;
        setProfile(completedProfile);
        setResult((current) => ({
          ...(current || {}),
          status: 'completed',
          can_plan_next_stage: false,
          all_mastered: true,
          session_status: response.session_status || 'completed',
          phase: response.phase || 'completed',
          available_actions: [],
          final_report: response.final_report,
        }));
        setGoalCompletionOpen(true);
        setPlannedStage(null);
        rememberProject(sessionId, completedProfile, stage, response.phase || 'completed');
        void loadDashboardStages(sessionId, stage);
        void refreshRemoteProjects();
        finishAgentRun(['agent1'], true);
        return;
      }

      const canStartNextStage =
        response.phase === 'next_stage_ready' ||
        response.available_actions?.includes('start_stage');

      if (!canStartNextStage) throw new Error('下一阶段规划请求未成功');

      let nextStage = response.next_stage;
      let nextProfile: Profile = toProfile(response.updated_profile) || profile || fallbackProfile;

      if (!nextStage) {
        const maxRetries = 12;
        const retryDelay = 1000;

        for (let i = 0; i < maxRetries; i += 1) {
          const session = await apiClient.getSession(sessionId);

          if (session.profile) {
            nextProfile = toProfile(session.profile) || nextProfile;
            setProfile(nextProfile);
          }

          if (session.next_stage) {
            nextStage = session.next_stage;
            break;
          }

          await new Promise((resolve) => window.setTimeout(resolve, retryDelay));
        }
      } else if (response.updated_profile) {
        const updatedProfile = toProfile(response.updated_profile);
        if (updatedProfile) setProfile(updatedProfile);
      }

      if (!nextStage) throw new Error('下一阶段仍在生成中，请稍后再试');

      setPlannedStage(nextStage);
      rememberProject(sessionId, nextProfile, nextStage, response.phase);
      rememberStageConversation(sessionId, nextStage, nextProfile);
      void refreshRemoteProjects();
      finishAgentRun(['agent1'], true);
    } catch (caught) {
      finishAgentRun(['agent1'], false);
      setError(getErrorMessage(caught));
    } finally {
      setLoading(false);
    }
  };

  const clearLocalSession = async () => {
    const targetSessionId = sessionId;
    const confirmed = window.confirm('将放弃全部学习目标并清除本地记录，确定重新开始吗？');
    if (!confirmed) return;

    // The session list is server-backed. Abandon every active remote target
    // before clearing local state, otherwise refreshRemoteProjects restores it.
    const remoteSessionIds = new Set<string>();
    try {
      const response = await apiClient.listSessions();
      response.sessions
        .filter((item) => item.status !== 'abandoned' && !isLocalSessionId(item.session_id))
        .forEach((item) => remoteSessionIds.add(item.session_id));
    } catch {
      // Local data can still be cleared when the server is unavailable.
    }
    readOwnedSessionIds().forEach((ownedSessionId) => {
      if (!isLocalSessionId(ownedSessionId)) remoteSessionIds.add(ownedSessionId);
    });
    if (targetSessionId && !isLocalSessionId(targetSessionId)) remoteSessionIds.add(targetSessionId);

    await Promise.allSettled([...remoteSessionIds].map((remoteSessionId) => apiClient.abandonSession(remoteSessionId)));

    localStorage.removeItem(SESSION_STORAGE_KEY);
    localStorage.removeItem(PROJECTS_STORAGE_KEY);
    localStorage.removeItem(CONVERSATIONS_STORAGE_KEY);
    localStorage.removeItem(OWNED_SESSION_IDS_STORAGE_KEY);
    setLearningProjects([]);
    setStageConversations([]);
    setSessionId(null);
    setProfile(null);
    setStage(null);
    setLearningPackage(null);
    setPlannedStage(null);
    setResult(null);
    setMessages(initialMessages);
    setSelectedConversationId('');
    setInput('');
    setReuseUserProfile(false);
    setOnboardingForm(emptyOnboardingForm);
    setFoundationSubtopicIds([]);
    setFoundationConfirmed(false);
    setFoundationPickerBlocks(knowledgeGraphBlocks);
    setFollowupQuestions([]);
    setHighlightedFields([]);
    setTimerSeconds(0);
    setTimerRunning(false);
    setMaterialPanelOpen(false);
    setLearningPathOpen(false);
    setProfileModalOpen(false);
    setAgentStatuses(createAgentStatusMap());
    setWorkflow('onboarding');
    setView('chat');
    setError(null);
    void checkUserFoundation();
    return;
  };

  const saveProfileSettings = async (data: { grade: string; skill_tags: string[]; hardware: string; mastered_subtopic_ids?: string[]; foundation_confirmed?: boolean }) => {
    setLoading(true);
    setError(null);
    try {
      const updated = await apiClient.updateUserProfile(data);
      const { mastered_subtopic_ids: _masteredSubtopicIds, ...profileData } = data;
      const profileBeforeGoalCreation = newGoalReturnState?.profile;
      const sourceProfile = profile || profileBeforeGoalCreation || storedUserProfile || fallbackProfile;
      const nextProfile = { ...sourceProfile, ...profileData, ...(toProfile(updated) || {}), daily_hours: sourceProfile.daily_hours };
      setStoredUserProfile({
        grade: nextProfile.grade,
        skill_tags: nextProfile.skill_tags,
        foundation_confirmed: nextProfile.foundation_confirmed,
        hardware: nextProfile.hardware,
        learning_style: nextProfile.learning_style,
        learning_rhythm: nextProfile.learning_rhythm
      });
      if (profile) {
        setProfile(nextProfile);
      } else if (newGoalReturnState) {
        setNewGoalReturnState((current) => current ? { ...current, profile: nextProfile } : current);
      }
      setOnboardingForm((current) => ({
        ...current,
        grade: nextProfile.grade || '',
        skill_tags: nextProfile.skill_tags || [],
        hardware: nextProfile.hardware || ''
      }));
      setLearningProjects((projects) => projects.map((project) => ({
        ...project,
        profile: project.profile ? { ...project.profile, ...profileData } : project.profile
      })));
      if (sessionId) rememberProject(sessionId, nextProfile, stage);
      void refreshRemoteProjects();
    } catch (caught) {
      setError(getErrorMessage(caught));
    } finally {
      setLoading(false);
    }
  };

  const updateTargetDuration = async (targetSessionId: string, totalHours: number, dailyHours: number) => {
    setLoading(true);
    setError(null);
    try {
      const updated = await apiClient.updateTargetSettings(targetSessionId, {
        target_total_hours: totalHours,
        target_daily_hours: dailyHours
      });
      const adjusted = new Map((updated.adjusted_stages || []).map((item) => [item.stage_id, item.estimated_hours]));
      setLearningProjects((projects) => projects.map((project) => {
        if (project.sessionId !== targetSessionId) return project;
        return {
          ...project,
          targetTotalHours: updated.target_total_hours,
          targetDailyHours: updated.target_daily_hours,
          profile: project.profile ? { ...project.profile, daily_hours: updated.target_daily_hours } : project.profile,
          stages: (project.stages || []).map((item) => (
            item.stage_id && adjusted.has(item.stage_id)
              ? { ...item, estimated_hours: adjusted.get(item.stage_id) }
              : item
          ))
        };
      }));
      if (targetSessionId === sessionId) {
        setProfile((current) => current ? { ...current, daily_hours: updated.target_daily_hours } : current);
        setStage((current) => current?.stage_id && adjusted.has(current.stage_id)
          ? { ...current, estimated_hours: adjusted.get(current.stage_id) }
          : current);
      }
    } catch (caught) {
      setError(getErrorMessage(caught));
      throw caught;
    } finally {
      setLoading(false);
    }
  };

  const loadTargetSettings = async (targetSessionId: string) => {
    try {
      const settings = await apiClient.getTargetSettings(targetSessionId);
      return { total: settings.target_total_hours, daily: settings.target_daily_hours };
    } catch (caught) {
      setError(getErrorMessage(caught));
      return null;
    }
  };

  const loadStageDuration = async (stageId: string) => {
    if (!sessionId) return null;
    try {
      const duration = await apiClient.getStageDuration(sessionId, stageId);
      return Number(duration.estimated_hours);
    } catch (caught) {
      setError(getErrorMessage(caught));
      return null;
    }
  };

  const updateStageDuration = async (stageId: string, hours: number) => {
    if (!sessionId) throw new Error('当前没有真实学习会话');
    setLoading(true);
    setError(null);
    try {
      const updated = await apiClient.updateStageDuration(sessionId, stageId, hours);
      const estimatedHours = Number(updated.estimated_hours ?? hours);
      setStage((current) => current?.stage_id === stageId ? { ...current, estimated_hours: estimatedHours } : current);
      setPlannedStage((current) => current?.stage_id === stageId ? { ...current, estimated_hours: estimatedHours } : current);
      setLearningProjects((projects) => projects.map((project) => project.sessionId === sessionId ? {
        ...project,
        stages: (project.stages || []).map((item) => item.stage_id === stageId
          ? { ...item, estimated_hours: estimatedHours }
          : item)
      } : project));
    } catch (caught) {
      setError(getErrorMessage(caught));
      throw caught;
    } finally {
      setLoading(false);
    }
  };

  const openTestModal = async () => {
    if (!sessionId) {
      setError('当前没有真实学习会话，无法开始测试。');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      setAssessmentReview(null);
      const attempt = await apiClient.createAssessmentAttempt(sessionId);
      const attemptDetail = await apiClient.getAssessmentAttempt(attempt.attempt_id);
      setAssessmentState({
        attemptId: attempt.attempt_id,
        questions: Array.isArray(attemptDetail.questions) ? attemptDetail.questions as TestQuestion[] : []
      });
      if (!attemptDetail.questions?.length) throw new Error('当前阶段没有可测试的数据库题目。');
      setTestOpen(true);
    } catch (caught) {
      setAssessmentState({ attemptId: null, questions: [] });
      // 结构化错误码：无测试题时不报错，改为弹窗询问"直接结束当前阶段"
      const code = getApiErrorCode(caught);
      if (code === 'NO_QUESTIONS_AVAILABLE') {
        const goal = profile?.goal || '当前学习目标';
        const blockName = stage?.block_name || '当前阶段';
        setNoQuestionsDialog({ open: true, goal, blockName });
      } else {
        setError(getErrorMessage(caught));
      }
    } finally {
      setLoading(false);
    }
  };

  const createLearningGoal = async () => {
    setProjectSwitcherOpen(false);
    setCheckingUserProfile(true);
    const loadedProfile = profile || learningProjects.find((project) => project.profile)?.profile;
    let reusableProfileForm = loadedProfile ? formDataFromProfile(loadedProfile) : emptyOnboardingForm;
    let hasReusableProfile = Boolean(
      loadedProfile?.grade?.trim()
      && loadedProfile?.hardware?.trim()
      && loadedProfile?.foundation_confirmed
    );
    let reusableFoundationConfirmed = Boolean(loadedProfile?.foundation_confirmed);
    try {
      const storedProfile = await apiClient.getUserProfile();
      setStoredUserProfile({
        grade: storedProfile.grade,
        skill_tags: storedProfile.skill_tags,
        foundation_confirmed: storedProfile.foundation_confirmed,
        hardware: storedProfile.hardware,
        learning_style: storedProfile.learning_style,
        learning_rhythm: storedProfile.learning_rhythm
      });
      const storedProfileComplete = Boolean(
        storedProfile.grade?.trim()
        && storedProfile.hardware?.trim()
        && storedProfile.foundation_confirmed
      );
      if (storedProfileComplete) {
        hasReusableProfile = true;
        reusableFoundationConfirmed = true;
        reusableProfileForm = formDataFromProfile(storedProfile);
      }
    } catch (caught) {
      // Keep an already loaded project usable if the optional refresh fails.
      setError(getErrorMessage(caught));
    } finally {
      setCheckingUserProfile(false);
    }
    setReuseUserProfile(hasReusableProfile);
    setFoundationConfirmed(reusableFoundationConfirmed);
    if (profile) {
      setNewGoalReturnState({
        sessionId,
        profile,
        stage,
        learningPackage,
        messages,
        selectedConversationId,
        input,
        result,
        plannedStage,
        timerSeconds,
        timerRunning,
        materialPanelOpen,
        agentStatuses
      });
    }
    setError(null);
    setSessionId(null);
    setProfile(null);
    setStage(null);
    setLearningPackage(null);
    setPlannedStage(null);
    setResult(null);
    setMessages(initialMessages);
    setSelectedConversationId('');
    setInput('');
    setOnboardingForm(hasReusableProfile ? {
      ...reusableProfileForm,
      goal: '',
      daily_hours: '',
      learning_style: ''
    } : emptyOnboardingForm);
    setFollowupQuestions([]);
    setHighlightedFields([]);
    setTimerSeconds(0);
    setTimerRunning(false);
    setMaterialPanelOpen(false);
    setLearningPathOpen(false);
    setProfileModalOpen(false);
    setAgentStatuses(createAgentStatusMap());
    setWorkflow('onboarding');
    setView('chat');
  };

  const cancelCreateLearningGoal = () => {
    if (!newGoalReturnState) return;
    const previous = newGoalReturnState;
    setSessionId(previous.sessionId);
    if (previous.sessionId) localStorage.setItem(SESSION_STORAGE_KEY, previous.sessionId);
    setProfile(previous.profile);
    setStage(previous.stage);
    setLearningPackage(previous.learningPackage);
    setMessages(previous.messages);
    setSelectedConversationId(previous.selectedConversationId);
    setInput(previous.input);
    setResult(previous.result);
    setPlannedStage(previous.plannedStage);
    setTimerSeconds(previous.timerSeconds);
    setTimerRunning(previous.timerRunning);
    setMaterialPanelOpen(previous.materialPanelOpen);
    setAgentStatuses(previous.agentStatuses);
    setOnboardingForm(formDataFromProfile(previous.profile));
    setFoundationConfirmed(Boolean(previous.profile.foundation_confirmed));
    setFollowupQuestions([]);
    setHighlightedFields([]);
    setError(null);
    setWorkflow('learning');
    setView('chat');
    setNewGoalReturnState(null);
  };

  const switchLearningProject = async (
    nextSessionId: string,
    requestedStageNum?: number,
    destination: View = 'chat',
    ignoreBusy = false,
    preferLastStage = false,
    requestedStageId?: string
  ) => {
    setGoalCompletionOpen(false);
    if (!ignoreBusy && (nextSessionId === sessionId || loading)) {
      const requestedConversationId = requestedStageNum === undefined
        ? ''
        : `${nextSessionId}:${requestedStageNum}`;
      if (nextSessionId === sessionId && requestedConversationId && selectedConversationId === requestedConversationId) {
        setSelectedConversationId(`${nextSessionId}:${requestedStageNum}`);
        return;
      }
      if (loading) return;
    }
    if (isLocalSessionId(nextSessionId)) {
      localStorage.removeItem(SESSION_STORAGE_KEY);
      setLearningProjects((projects) => projects.filter((project) => project.sessionId !== nextSessionId));
      setStageConversations((conversations) => conversations.filter((conversation) => conversation.sessionId !== nextSessionId));
      setSessionId(null);
      setProfile(null);
      setStage(null);
      setLearningPackage(null);
      setAgentStatuses(createAgentStatusMap());
      setWorkflow('onboarding');
      setView(destination);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const session = await apiClient.getSession(nextSessionId);
      if (!session?.profile) throw new Error('该学习项目尚未生成有效画像');
      const activeStage = resolveSessionStage(session);
      const hasLearningPackage = Boolean(session.learning_package);
      const nextProfile = toProfile(session.profile);
      if (!nextProfile) throw new Error('学习项目画像解析失败');

      // 如果 preferLastStage=true，则尝试读取 dashboard，用 stage_history 的最大 stage_num 作为目标阶段。
      let desiredStageNum = requestedStageNum;
      let fallbackStage = activeStage;
      if (preferLastStage && desiredStageNum === undefined) {
        try {
          const dashboard = await apiClient.getDashboard(nextSessionId);
          const history = Array.isArray(dashboard?.stage_history) ? dashboard.stage_history as StageInfo[] : [];
          const stageNumList = history.map((item) => item.stage_num ?? 0).filter((n) => n > 0);
          if (activeStage?.stage_num) stageNumList.push(activeStage.stage_num);
          if ((session.next_stage as { stage_num?: number } | null)?.stage_num) {
            stageNumList.push((session.next_stage as { stage_num: number }).stage_num);
          }
          if (stageNumList.length > 0) {
            const maxStage = Math.max(...stageNumList);
            desiredStageNum = maxStage;
            // 尝试从 history / next_stage / activeStage 里找到匹配的阶段 info 给 fallbackStage。
            const candidates: StageInfo[] = [];
            if (activeStage) candidates.push(activeStage);
            if (session.next_stage) candidates.push(session.next_stage as StageInfo);
            candidates.push(...history);
            const matched = candidates.find((c) => (c.stage_num ?? 0) === maxStage);
            if (matched) fallbackStage = matched;
            // 缓存 dashboard 阶段信息到 learningProjects
            mergeDashboardStages(
              nextSessionId,
              history,
              activeStage,
              dashboard.average_accuracy,
              typeof dashboard.overall_progress === 'number' ? dashboard.overall_progress : undefined
            );
          }
        } catch {
          // dashboard 失败就回退到 activeStage，不阻断切换
        }
      }

      setSessionId(nextSessionId);
      localStorage.setItem(SESSION_STORAGE_KEY, nextSessionId);
      setProfile(nextProfile);
      setOnboardingForm(formDataFromProfile(nextProfile));
      setFoundationConfirmed(Boolean(nextProfile.foundation_confirmed));
      setReuseUserProfile(Boolean(
        nextProfile.grade?.trim()
        && nextProfile.hardware?.trim()
        && nextProfile.foundation_confirmed
      ));
      const activeStageNum = activeStage?.stage_num ?? null;
      let displayStage = fallbackStage || activeStage;
      let displayPackage = toLearningPackage(session.learning_package);
      const displayStageNum = desiredStageNum ?? (displayStage?.stage_num ?? 1);
      if (displayStageNum !== activeStageNum || (requestedStageId && requestedStageId !== activeStage?.stage_id)) {
        const [stageResponse, packageResponse] = await Promise.allSettled([
          apiClient.getCurrentStage(nextSessionId, displayStageNum, requestedStageId),
          apiClient.getStagePackage(nextSessionId, displayStageNum, requestedStageId)
        ]);
        // An old stage can lack one runtime snapshot. Keep the dashboard's
        // stage metadata and any available package instead of abandoning the
        // whole learning target.
        if (stageResponse.status === 'fulfilled') {
          displayStage = toStageInfo(stageResponse.value.stage) || displayStage;
        }
        if (packageResponse.status === 'fulfilled') {
          displayPackage = toLearningPackage(packageResponse.value.package) || displayPackage;
        }
      }
      setActiveStageNum(activeStageNum);
      setStage(displayStage || null);
      setLearningPackage(displayPackage);
      setResult(null);
      setAssessmentReview(null);
      let displayStageHasResult = Boolean(
        displayStage && ['completed', 'retreat'].includes(displayStage.status || '')
      );
      try {
        const persisted = await apiClient.getStageResult(nextSessionId, displayStageNum, displayStage?.stage_id);
        if (persisted.stage_result) {
          displayStageHasResult = true;
          setResult(persisted.stage_result as StageResult);
        }
        if (persisted.assessment_attempt) setAssessmentReview(persisted.assessment_attempt as AssessmentReview);
      } catch {
        // Keep completed legacy stages accessible while an older backend
        // process has not yet loaded the persisted-result endpoint changes.
        if (displayStage && ['completed', 'retreat'].includes(displayStage.status || '')) {
          const bypass = displayStage.strategy_used === 'advance_due_to_no_questions';
          setResult({
            status: displayStage.status,
            accuracy: bypass ? undefined : displayStage.accuracy,
            bypass_assessment: bypass,
            stage_summary: `# 阶段学习总结\n\n你已完成“${displayStage.block_name || '当前学习主题'}”阶段。${bypass ? '本阶段没有匹配的测试题目，系统已按无题完成流程归档。' : '测试与学习记录已归档，可根据总结继续巩固。'}`,
            next_strategy: displayStage.strategy_used || 'review',
            can_plan_next_stage: false,
            available_actions: [],
          });
        }
      }
      setPlannedStage(session.next_stage || null);
      const conversationKey = displayStage?.stage_id || displayStageNum;
      const restoredConversation = stageConversations.find((item) => item.id === `${nextSessionId}:${conversationKey}`);
      setMessages(restoredConversation?.messages || initialMessages);
      setSelectedConversationId(`${nextSessionId}:${conversationKey}`);
      setAgentStatuses(restoreAgentStatuses({
        hasProfile: true,
        hasLearningPackage,
        hasStageResult: displayStageHasResult,
        phase: session.phase
      }));
      setWorkflow(isLearningPhase(session.phase) || hasLearningPackage ? 'learning' : 'profile_ready');
      setView(destination);
      rememberProject(nextSessionId, nextProfile, session.next_stage || fallbackStage || activeStage, session.phase);
      rememberTargetSettings(nextSessionId, session.target_settings);
      if (!preferLastStage) void loadDashboardStages(nextSessionId, fallbackStage || activeStage);
      if (displayStage) {
        void restoreQaConversation(nextSessionId, displayStage, nextProfile, true);
      } else if (activeStage) {
        void restoreQaConversation(nextSessionId, activeStage, nextProfile, true);
      }
      setMaterialPanelOpen(false);
      setProjectSwitcherOpen(false);
    } catch (caught) {
      if (isMissingSessionError(caught)) {
        // A cleared or replaced database can leave an old browser session ID
        // behind. Treat it as first use instead of surfacing a confusing 404.
        if (localStorage.getItem(SESSION_STORAGE_KEY) === nextSessionId) {
          localStorage.removeItem(SESSION_STORAGE_KEY);
        }
        forgetOwnedSessionId(nextSessionId);
        setLearningProjects((projects) => projects.filter((project) => project.sessionId !== nextSessionId));
        setStageConversations((conversations) => conversations.filter((conversation) => conversation.sessionId !== nextSessionId));
        if (sessionId === nextSessionId || !sessionId) {
          setSessionId(null);
          setProfile(null);
          setStage(null);
          setLearningPackage(null);
          setPlannedStage(null);
          setResult(null);
          setAssessmentReview(null);
          setAssessmentState({ attemptId: null, questions: [] });
          setMessages(initialMessages);
          setSelectedConversationId('');
          setActiveStageNum(null);
          setAgentStatuses(createAgentStatusMap());
          setWorkflow('onboarding');
          setView('chat');
        }
        setProjectSwitcherOpen(false);
        return;
      }
      setError(`无法切换项目：${getErrorMessage(caught)}`);
    } finally {
      setLoading(false);
    }
  };

  const openFoundationPicker = async (context: 'onboarding' | 'settings') => {
    if (context === 'settings') {
      setFoundationPickerBlocks(knowledgeGraphBlocks);
      try {
        const network = await apiClient.getKnowledgeNetwork();
        setFoundationSubtopicIds(network.blocks.flatMap((block) => block.subtopics)
          .filter((topic) => topic.status === 1 && topic.is_initial_foundation)
          .map((topic) => topic.subtopic_id));
      } catch (caught) {
        setError(`读取用户基础失败：${getErrorMessage(caught)}`);
        return;
      }
      setFoundationPickerContext(context);
      return;
    }
    setFoundationPickerBlocks(knowledgeGraphBlocks);
    setFoundationPickerContext(context);
  };

  const confirmFoundation = (selectedSubtopicIds: string[]) => {
    const selectedIds = new Set(selectedSubtopicIds);
    const skillTags = foundationPickerBlocks.flatMap((block) => {
      const selectedTopics = block.subtopics.filter((topic) => selectedIds.has(topic.subtopic_id));
      if (selectedTopics.length === block.subtopics.length) return [block.block_name];
      return [];
    });
    setFoundationSubtopicIds(selectedSubtopicIds);
    if (foundationPickerContext === 'onboarding') setFoundationConfirmed(true);
    setOnboardingForm((current) => ({ ...current, skill_tags: skillTags }));
    const settingsSourceProfile = storedUserProfile || profile || newGoalReturnState?.profile;
    if (foundationPickerContext === 'settings' && settingsSourceProfile) {
      void saveProfileSettings({
        grade: settingsSourceProfile.grade || '',
        skill_tags: skillTags,
        hardware: settingsSourceProfile.hardware || '',
        mastered_subtopic_ids: selectedSubtopicIds,
        foundation_confirmed: true
      });
    }
    setFoundationPickerContext(null);
  };

  const deleteLearningProject = async (targetSessionId: string) => {
    setLoading(true);
    setError(null);
    try {
      if (!isLocalSessionId(targetSessionId)) {
        // 物理删除：DELETE /sessions/{session_id}，级联清理 DB 所有关联快照
        await apiClient.deleteSession(targetSessionId);
      }
      forgetOwnedSessionId(targetSessionId);
      const remainingProjects = learningProjects.filter((project) => project.sessionId !== targetSessionId);
      setLearningProjects(remainingProjects);
      setStageConversations((conversations) => conversations.filter((item) => item.sessionId !== targetSessionId));

      if (targetSessionId !== sessionId) return;

      localStorage.removeItem(SESSION_STORAGE_KEY);
      const nextProject = sortProjectsForDashboard(remainingProjects, null)[0];
      if (nextProject && !isLocalSessionId(nextProject.sessionId)) {
        await switchLearningProject(nextProject.sessionId, undefined, 'learn', true);
        return;
      }

      setSessionId(null);
      setProfile(null);
      setStage(null);
      setLearningPackage(null);
      setPlannedStage(null);
      setResult(null);
      setAssessmentState({ attemptId: null, questions: [] });
      setMessages(initialMessages);
      setSelectedConversationId('');
      setAgentStatuses(createAgentStatusMap());
      setWorkflow('onboarding');
      setView('learn');
    } catch (caught) {
      setError(`删除学习目标失败：${getErrorMessage(caught)}`);
      throw caught;
    } finally {
      setLoading(false);
    }
  };

  const selectProjectStage = (nextSessionId: string, stageNum: number, stageId?: string) => {
    void switchLearningProject(nextSessionId, stageNum, 'chat', false, false, stageId);
  };

  const deleteLastLearningStage = async (targetSessionId: string) => {
    setLoading(true);
    setError(null);
    try {
      const result = await apiClient.deleteLastStage(targetSessionId);

      // 级联删除了整个学习目标：走与 deleteLearningProject 一致的收尾流程
      if (result?.session_deleted) {
        forgetOwnedSessionId(targetSessionId);
        const remainingProjects = learningProjects.filter((project) => project.sessionId !== targetSessionId);
        setLearningProjects(remainingProjects);
        setStageConversations((conversations) => conversations.filter((item) => item.sessionId !== targetSessionId));

        if (targetSessionId !== sessionId) {
          void refreshRemoteProjects();
          return;
        }

        localStorage.removeItem(SESSION_STORAGE_KEY);
        const nextProject = sortProjectsForDashboard(remainingProjects, null)[0];
        if (nextProject && !isLocalSessionId(nextProject.sessionId)) {
          await switchLearningProject(nextProject.sessionId, undefined, 'chat', true);
          return;
        }

        setSessionId(null);
        setProfile(null);
        setStage(null);
        setLearningPackage(null);
        setPlannedStage(null);
        setResult(null);
        setAssessmentState({ attemptId: null, questions: [] });
        setMessages(initialMessages);
        setSelectedConversationId('');
        setAgentStatuses(createAgentStatusMap());
        setWorkflow('onboarding');
        setView('chat');
        return;
      }

      // session 仍在：本地立即移除最后一个阶段，刷新列表
      setLearningProjects((projects) => projects.map((project) => {
        if (project.sessionId !== targetSessionId) return project;
        const sortedStages = [...(project.stages || [])].sort((a, b) => (a.stage_num ?? 1) - (b.stage_num ?? 1));
        return { ...project, stages: sortedStages.slice(0, -1) };
      }));

      // 若删的是当前会话，回退到删除后的最新阶段
      if (targetSessionId === sessionId) {
        try {
          const remainingStageNum = Number(result.details?.remaining_stage_num);
          // Reuse the normal stage switcher so package, conversation and result
          // state are restored from the preceding stage snapshot together.
          await switchLearningProject(
            targetSessionId,
            Number.isFinite(remainingStageNum) && remainingStageNum > 0 ? remainingStageNum : undefined,
            'chat',
            true,
          );
        } catch (caught) {
          console.warn('[deleteLastStage] failed to refresh current session snapshot', caught);
        }
      }
    } catch (caught) {
      setError(`删除学习阶段失败：${getErrorMessage(caught)}`);
      throw caught;
    } finally {
      setLoading(false);
    }
  };

  const openLearningPath = async () => {
    if (sessionId) {
      try {
        const latestSession = await apiClient.getSession(sessionId);
        const latestStage = resolveSessionStage(latestSession);
        if (latestStage) setStage(toStageInfo(latestStage));
      } catch (caught) {
        // Keep the locally cached stage available when refreshing the modal fails.
        console.warn('[learning path] failed to refresh current stage', caught);
      }
    }
    setLearningPathOpen(true);
  };

  const editProfile = () => {
    if (profile) {
      setOnboardingForm(formDataFromProfile(profile));
      setFoundationConfirmed(Boolean(profile.foundation_confirmed));
    }
    setError(null);
    setFollowupQuestions([]);
    setHighlightedFields([]);
    setWorkflow('onboarding');
    setView('chat');
  };

  const settingsProfile = useMemo(() => {
    // Settings is user-scoped. A target profile may omit hardware, so it must
    // never override the durable user profile with an empty value.
    const source = storedUserProfile || profile || newGoalReturnState?.profile;
    if (!source) return null;
    // Restored session profiles may omit the user-scoped hardware field.
    // Keep the settings view backed by the durable onboarding profile.
    return {
      ...source,
      hardware: source.hardware || onboardingForm.hardware
    };
  }, [profile, newGoalReturnState?.profile, storedUserProfile, onboardingForm.hardware]);
  const workspaceProfile = profile || fallbackProfile;
  const workspaceStage = stage || unavailableStage;
  const historicalStageReview = isHistoricalStageReview(workspaceStage.stage_num, activeStageNum);
  const onboardingFoundationNames = useMemo(
    () => onboardingForm.skill_tags.filter(Boolean),
    [onboardingForm.skill_tags]
  );
  const settingsFoundationNames = useMemo(
    () => (settingsProfile?.skill_tags || []).filter(Boolean),
    [settingsProfile?.skill_tags]
  );
  const stageSubtopics = useMemo(
    () => resolveStageSubtopics(workspaceProfile, workspaceStage),
    [workspaceProfile, workspaceStage]
  );
  const activeSubtopicId = stageSubtopics.some((topic) => topicKey(topic) === selectedSubtopicId)
    ? selectedSubtopicId
    : topicKey(stageSubtopics[0]) || null;
  const currentConversationId = sessionId && activeStageNum ? `${sessionId}:${activeStageNum}` : '';
  const selectedConversation = stageConversations.find((item) => item.id === selectedConversationId);
  const conversationStage = selectedConversation?.stage || workspaceStage;
  const conversationProject = learningProjects.find((project) => (
    project.sessionId === (selectedConversation?.sessionId || sessionId)
  ));
  const conversationGoalTitle = conversationProject?.goal || workspaceProfile.goal || '当前学习目标';
  const displayedMessages = selectedConversationId && selectedConversationId !== currentConversationId
    ? selectedConversation?.messages || []
    : messages;
  const historicalConversationSelected = Boolean(selectedConversationId && selectedConversationId !== currentConversationId);
  const selectedConversationStageNum = selectedConversation?.stage.stage_num ?? 0;
  const currentStageNum = stage?.stage_num ?? 1;
  const readOnlyConversationNotice = selectedConversationStageNum > currentStageNum
    ? '该阶段已规划，开始学习后可在此继续对话。'
    : '这是已结束阶段的对话记录，仅供查看。';

  return (
    <div className="app">
      <Sidebar
        view={view}
        setView={setView}
        agentStatuses={agentStatuses}
        collapsed={sidebarCollapsed}
        onToggleCollapsed={() => setSidebarCollapsed((prev) => !prev)}
      />
      {view === 'chat' && (!profile || workflow === 'onboarding' || workflow === 'diagnosing' || workflow === 'needs_more_info') && (
        <OnboardingPage
          formData={onboardingForm}
          onChange={(next) => {
            if (next.goal !== onboardingForm.goal && !reuseUserProfile) {
              setFoundationSubtopicIds([]);
              setFoundationConfirmed(false);
            }
            setOnboardingForm(next);
          }}
          onSubmit={submitOnboarding}
          onCancel={newGoalReturnState ? cancelCreateLearningGoal : undefined}
          loading={loading}
          mode={workflow === 'diagnosing' ? 'diagnosing' : workflow === 'needs_more_info' ? 'needs_more_info' : 'onboarding'}
          questions={followupQuestions}
          error={error}
          highlightedFields={highlightedFields}
          reuseUserProfile={reuseUserProfile}
          checkingUserProfile={checkingUserProfile}
          profileCheckError={profileCheckError}
          onRetryProfileCheck={() => void checkUserFoundation()}
          foundationNames={onboardingFoundationNames}
          foundationConfirmed={foundationConfirmed}
          onOpenFoundation={() => openFoundationPicker('onboarding')}
          foundationLoading={foundationPickerLoading}
        />
      )}
      {view === 'chat' && profile && (workflow === 'profile_ready' || workflow === 'starting_stage') && (
        <ProfileConfirmPage
          profile={profile}
          stage={workspaceStage}
          loading={loading}
          error={error}
          onEdit={editProfile}
          onStart={startStage}
          onCancel={newGoalReturnState ? cancelCreateLearningGoal : undefined}
          showUserProfile={!reuseUserProfile}
        />
      )}
      {profile && workflow === 'learning' && view === 'chat' && (
        <ChatView
          goalTitle={conversationGoalTitle}
          messages={displayedMessages}
          input={input}
          setInput={setInput}
          onSend={sendMessage}
          stage={workspaceStage}
          conversationStage={conversationStage}
          timerSeconds={timerSeconds}
          timerRunning={timerRunning}
          dailyProgress={dailyProgress}
          onToggleTimer={toggleStageTimer}
          onCreateGoal={createLearningGoal}
          onOpenLearningPath={() => void openLearningPath()}
          onOpenProfile={() => setProfileModalOpen(true)}
          learningPackage={learningPackage}
          stageSubtopics={stageSubtopics}
          selectedSubtopicId={activeSubtopicId}
          onSelectSubtopic={setSelectedSubtopicId}
          onTest={openTestModal}
          stageResultReady={Boolean(result)}
          onOpenResult={() => setView('result')}
          projects={learningProjects}
          currentSessionId={sessionId}
          selectedConversationId={selectedConversationId || currentConversationId}
          onSelectProjectStage={selectProjectStage}
          onStartStage={() => void startStage()}
          onDeleteLastStage={deleteLastLearningStage}
          readOnly={false}
          readOnlyNotice=""
          loading={loading}
          error={error}
        />
      )}
      {view === 'learn' && (
        <LearningDashboardView
          projects={learningProjects}
          currentSessionId={sessionId}
          profile={profile}
          stage={workspaceStage}
          setView={setView}
          onSelectProject={(nextSessionId) => void switchLearningProject(nextSessionId, undefined, 'learn')}
          onDeleteProject={deleteLearningProject}
          onUpdateTargetDuration={updateTargetDuration}
          onLoadTargetSettings={loadTargetSettings}
          projectsLoading={projectsLoading}
          projectsError={projectsError}
          onRetryProjects={() => void refreshRemoteProjects()}
          loading={loading}
          error={error}
        />
      )}
      {settingsProfile && view === 'settings' && <SettingsView profile={settingsProfile} loading={loading} error={error} onClear={clearLocalSession} onSave={saveProfileSettings} foundationNames={settingsFoundationNames} onOpenFoundation={() => void openFoundationPicker('settings')} />}
      <FoundationPickerModal
        open={foundationPickerContext !== null}
        blocks={foundationPickerBlocks}
        initialSelectedSubtopicIds={foundationSubtopicIds}
        onClose={() => setFoundationPickerContext(null)}
        onConfirm={confirmFoundation}
      />
      {profile && workflow === 'learning' && view === 'result' && (
        <ResultView
          profile={workspaceProfile}
          stage={workspaceStage}
          plannedStage={plannedStage}
          existingNextStage={learningProjects
            .find((project) => project.sessionId === sessionId)?.stages
            ?.filter((item) => (item.stage_num ?? 1) > (workspaceStage.stage_num ?? 1))
            .sort((left, right) => (left.stage_num ?? 1) - (right.stage_num ?? 1))[0] || null}
          result={result}
          onPlanNext={() => void planNextStage(false)}
          onStartNext={(nextStage) => void startStage(nextStage)}
          onRestartCurrent={() => void startStage()}
          onContinueNext={() => planNextStage(true)}
          onReturnToChat={() => {
            setView('chat');
            setMaterialPanelOpen(true);
          }}
          onOpenAssessmentReview={() => setAssessmentReviewOpen(true)}
          hasAssessmentReview={Boolean(assessmentReview?.questions?.length)}
          readOnlyReview={historicalStageReview}
          loading={loading}
          error={error}
        />
      )}
      {testOpen && <TestModal questions={currentQuestions} onClose={() => setTestOpen(false)} onComplete={completeStage} />}
      {assessmentReview && assessmentReviewOpen && <AssessmentReviewModal review={assessmentReview} onClose={() => setAssessmentReviewOpen(false)} />}
      {goalCompletionOpen && result?.all_mastered && (
        <GoalCompletionModal goal={workspaceProfile.goal || ''} onClose={() => setGoalCompletionOpen(false)} />
      )}
      {noQuestionsDialog.open && (
        <div className="stage-delete-modal-overlay" role="dialog" aria-modal="true">
          <div className="stage-delete-modal">
            <div className="stage-delete-modal-header">
              <AlertTriangle size={18} />
              <span>暂无测试题目</span>
            </div>
            <p className="stage-delete-modal-body">
              「{noQuestionsDialog.goal}」的
              <strong>{noQuestionsDialog.blockName}</strong> 暂无匹配的测试题目。
            </p>
            <p className="stage-delete-modal-hint">
              是否直接结束当前阶段并进入阶段总结？若有可规划的下一阶段，总结页会提供生成入口。
            </p>
            <div className="stage-delete-modal-actions">
              <button
                type="button"
                className="stage-delete-modal-cancel"
                onClick={() => setNoQuestionsDialog((d) => ({ ...d, open: false }))}
                disabled={loading}
              >
                取消
              </button>
              <button
                type="button"
                className="stage-delete-modal-confirm"
                onClick={() => {
                  setNoQuestionsDialog((d) => ({ ...d, open: false }));
                  void completeStageWithPayload({ bypassAssessment: true });
                }}
                disabled={loading}
              >
                {loading ? '处理中…' : '直接结束阶段'}
              </button>
            </div>
          </div>
        </div>
      )}
      {learningPathOpen && profile && stage && (
        <LearningPathModal
          profile={profile}
          stage={stage}
          selectedSubtopicId={activeSubtopicId}
          onSelectSubtopic={setSelectedSubtopicId}
          onClose={() => setLearningPathOpen(false)}
        />
      )}
      {profileModalOpen && profile && (
        <ProfileDetailModal
          project={{
            sessionId: sessionId || 'current',
            goal: profile.goal || '当前学习目标',
            stageName: stage?.block_name || '等待阶段规划',
            stageNum: stage?.stage_num ?? 1,
            updatedAt: Date.now(),
            profile,
            stages: stage ? [stage] : []
          }}
          onClose={() => setProfileModalOpen(false)}
        />
      )}
      {projectSwitcherOpen && (
        <ProjectSwitcherModal
          projects={learningProjects}
          currentSessionId={sessionId}
          loading={loading}
          error={error}
          onClose={() => setProjectSwitcherOpen(false)}
          onSelect={switchLearningProject}
          onCreate={createLearningGoal}
        />
      )}
    </div>
  );
}
