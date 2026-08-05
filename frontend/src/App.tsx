import { CSSProperties, FormEvent, PointerEvent as ReactPointerEvent, useEffect, useMemo, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import type { KeyboardEvent as ReactKeyboardEvent } from 'react';
import {
  Award,
  BookOpen,
  Bot,
  Brain,
  Check,
  ChevronLeft,
  ChevronRight,
  Circle,
  Clock,
  Code2,
  Copy,
  ExternalLink,
  FileQuestion,
  FileText,
  Folder,
  Gauge,
  Eye,
  LayoutDashboard,
  ListTree,
  Loader2,
  MessageCircle,
  PanelLeftClose,
  PanelLeftOpen,
  Pause,
  Play,
  Plus,
  RefreshCw,
  Send,
  Settings,
  Sparkles,
  SquarePen,
  Target,
  UserRound,
  Video,
  X
} from 'lucide-react';
import {
  apiClient,
  type ApiLearningPackage,
  type ApiProfile,
  type ApiStage,
  type SessionListItem
} from './api/client';
import { findKnowledgeGraphBlock } from './data/knowledgeGraph';
import OnboardingPage from './pages/OnboardingPage';
import ProfileConfirmPage from './pages/ProfileConfirmPage';
import {
  buildDiagnosticMessage,
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

type View = 'chat' | 'learn' | 'settings' | 'result';
type WorkflowState = 'onboarding' | 'diagnosing' | 'needs_more_info' | 'profile_ready' | 'starting_stage' | 'learning';
type Message = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  note?: string;
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
};
type LearningPackage = {
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
  updatedAt: number;
  profile?: Profile;
  stages?: StageInfo[];
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

const PROJECTS_STORAGE_KEY = 'learningProjects';
const CONVERSATIONS_STORAGE_KEY = 'stageConversations';
const OWNED_SESSION_IDS_STORAGE_KEY = 'ownedSessionIds';
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
  return {
    sessionId: DEMO_SESSION_ID,
    goal: fallbackProfile.goal || 'LoRA 微调与部署',
    stageName: fallbackStage.block_name || '第一阶段',
    stageNum: fallbackStage.stage_num ?? 1,
    updatedAt: Date.now(),
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
    title: '画像诊断',
    desc: '生成用户画像 / 规划学习阶段',
    icon: Brain,
    tone: 'blue'
  },
  {
    id: 'agent3',
    number: '3',
    title: '内容生成',
    desc: '生成学习资料',
    icon: BookOpen,
    tone: 'violet'
  },
  {
    id: 'agent4',
    number: '4',
    title: '双重审核',
    desc: '技术审核与质量评审',
    icon: Check,
    tone: 'orange'
  },
  {
    id: 'agent6',
    number: '6',
    title: '阶段决策',
    desc: '更新用户画像',
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
            const statusLabel = state === 'running'
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

function ProfilePanel({
  stage,
  projects,
  currentSessionId,
  selectedConversationId,
  onSelectProjectStage,
  onCreateGoal,
  loading,
  collapsed,
  onToggleCollapsed,
  timerSeconds,
  timerRunning,
  onToggleTimer
}: {
  stage: StageInfo;
  projects: LearningProject[];
  currentSessionId: string | null;
  selectedConversationId: string;
  onSelectProjectStage: (sessionId: string, stageNum: number) => void;
  onCreateGoal: () => void;
  loading: boolean;
  collapsed: boolean;
  onToggleCollapsed: () => void;
  timerSeconds: number;
  timerRunning: boolean;
  onToggleTimer: () => void;
}) {
  const [expandedGoals, setExpandedGoals] = useState<Record<string, boolean>>({});
  const stageDuration = Math.round((stage.estimated_hours ?? 1) * 3600);
  const timerLabel = timerRunning
    ? '暂停提醒'
    : timerSeconds <= 0
      ? '重新开始'
      : timerSeconds < stageDuration
        ? '继续计时'
        : '开始计时';
  const toggleGoal = (sessionId: string) => {
    setExpandedGoals((prev) => ({
      ...prev,
      [sessionId]: !(prev[sessionId] ?? sessionId === currentSessionId)
    }));
  };

  return (
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
      {!collapsed && <>
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
            <Clock size={15} />
            {stage.estimated_hours ?? 1} 小时
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
        {[...projects].sort((a, b) => b.updatedAt - a.updatedAt).map((project) => {
          const expanded = expandedGoals[project.sessionId] ?? project.sessionId === currentSessionId;
          const stages = [...(project.stages || [])].sort((a, b) => (a.stage_num ?? 1) - (b.stage_num ?? 1));
          return (
            <div className={`goal-nav-group ${project.sessionId === currentSessionId ? 'current' : ''}`} key={project.sessionId}>
              <button className="goal-nav-title" onClick={() => toggleGoal(project.sessionId)} aria-expanded={expanded}>
                <Folder size={17} />
                <span>{project.goal}</span>
              </button>
              {expanded && (
                <div className="goal-stage-list">
                  {stages.map((item) => {
                    const stageNum = item.stage_num ?? 1;
                    const conversationId = `${project.sessionId}:${stageNum}`;
                    const current = project.sessionId === currentSessionId && stageNum === (stage.stage_num ?? 1);
                    return (
                      <button
                        className={`${selectedConversationId === conversationId ? 'active' : ''} ${current ? 'current' : ''}`}
                        key={conversationId}
                        onClick={() => onSelectProjectStage(project.sessionId, stageNum)}
                      >
                        <span>
                          第 {stageNum} 阶段 · {item.block_name || '阶段对话'}
                        </span>
                        {current && <i aria-label="当前阶段" />}
                      </button>
                    );
                  })}
                  {stages.length === 0 && <p>完成画像后生成第一阶段</p>}
                </div>
              )}
            </div>
          );
        })}
      </section>
      </>}
    </aside>
  );
}

function getResourceUrl(resource: LearningResource) {
  if (resource.url) return resource.url;
  if (resource.content_ref?.startsWith('http')) return resource.content_ref;
  if (resource.locator?.startsWith('http')) return resource.locator;
  return '';
}

function uniqueResources(resources: LearningResource[]) {
  const seen = new Set<string>();
  return resources.filter((resource) => {
    const key = resource.resource_id || resource.source_id || getResourceUrl(resource) || resource.title;
    if (!key || seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function getEmbeddedVideoUrl(url: string) {
  const youtubeMatch = url.match(/(?:youtube\.com\/(?:watch\?v=|embed\/)|youtu\.be\/)([A-Za-z0-9_-]{6,})/);
  if (youtubeMatch) return `https://www.youtube.com/embed/${youtubeMatch[1]}`;
  const bilibiliMatch = url.match(/bilibili\.com\/video\/(BV[A-Za-z0-9]+)/i);
  if (bilibiliMatch) return `https://player.bilibili.com/player.html?bvid=${bilibiliMatch[1]}`;
  return '';
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
  return {
    ...pkg,
    tutorial: pkg.tutorial ?? undefined,
    code_examples: Array.isArray(pkg.code_examples) ? pkg.code_examples as LearningPackage['code_examples'] : undefined,
    learning_resources: (declaredResources.length ? declaredResources : recommendedVideos) as LearningResource[],
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
    phase: item.phase,
    updatedAt: item.updated_at ? Date.parse(item.updated_at) || Date.now() : Date.now(),
    profile: profile || undefined
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

function MaterialDocument({
  stage,
  learningPackage,
  compact = false
}: {
  stage: StageInfo;
  learningPackage: LearningPackage;
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
  const documentationResources = uniqueResources(
    learningResources.filter((resource) => resource.resource_type === 'documentation' && Boolean(resource.content_summary || getResourceUrl(resource)))
  );
  const codeResourceItems = uniqueResources(
    learningResources.filter((resource) => resource.resource_type === 'code' && Boolean(resource.content_summary || getResourceUrl(resource)))
  );
  const videos = uniqueResources(
    learningResources.filter((resource) => resource.resource_type === 'video' && Boolean(getResourceUrl(resource)))
  );
  const articles = uniqueResources(
    learningResources.filter((resource) => resource.resource_type === 'article' && Boolean(getResourceUrl(resource)))
  );
  const hasTextContent = Boolean(
    learningPackage.tutorial?.trim()
    || codeExamples.length
    || documentationResources.length
    || codeResourceItems.length
  );
  return (
    <article className={`material-document ${compact ? 'compact' : ''}`}>
      <div className="material-document-meta">
        <span>阶段 {stage.stage_num ?? 1}</span>
        <span><Clock size={13} /> {stage.estimated_hours ?? 1} 小时</span>
      </div>
      <h2>{stage.block_name || learningPackage.block_name || '未命名学习材料'}</h2>
      <section className="material-resource-section text-resources">
        <div className="material-section-heading"><BookOpen size={16} /><h3>学习文本</h3></div>
        {hasTextContent ? (
          <>
            {learningPackage.tutorial?.trim() && <div className="material-markdown"><ReactMarkdown>{learningPackage.tutorial}</ReactMarkdown></div>}
            {documentationResources.length > 0 && (
              <div className="material-article-list">
                {documentationResources.map((resource, index) => {
                  const url = getResourceUrl(resource);
                  return (
                    <article key={resource.resource_id || `${resource.title}-${index}`}>
                      <span>
                        <strong>{resource.title || `学习文本 ${index + 1}`}</strong>
                        {resource.content_summary && <small>{resource.content_summary}</small>}
                      </span>
                      {url && <a href={url} target="_blank" rel="noreferrer"><ExternalLink size={14} />查看原文</a>}
                    </article>
                  );
                })}
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
        ) : <p className="material-empty">无</p>}
      </section>
      <section className="material-resource-section article-resources">
        <div className="material-section-heading"><FileText size={16} /><h3>文章网址</h3></div>
        {articles.length > 0 ? (
          <div className="material-article-list">
            {articles.map((article, index) => {
              const url = getResourceUrl(article);
              return (
                <a href={url} target="_blank" rel="noreferrer" key={article.resource_id || article.source_id || `${url}-${index}`}>
                  <span><strong>{article.title || `参考文章 ${index + 1}`}</strong>{article.content_summary && <small>{article.content_summary}</small>}</span>
                  <ExternalLink size={15} />
                </a>
              );
            })}
          </div>
        ) : <p className="material-empty">无</p>}
      </section>
      <section className="material-resource-section video-resources">
        <div className="material-section-heading"><Video size={16} /><h3>视频网址</h3></div>
        {videos.length > 0 ? (
          <div className="material-video-list">
            {videos.map((video, index) => {
              const url = getResourceUrl(video);
              const embedUrl = getEmbeddedVideoUrl(url);
              const directVideo = /\.(mp4|webm|ogg)(?:\?|$)/i.test(url);
              return (
                <article key={video.resource_id || video.source_id || `${url}-${index}`}>
                  {embedUrl ? (
                    <iframe src={embedUrl} title={video.title || `阶段视频 ${index + 1}`} allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowFullScreen />
                  ) : directVideo ? (
                    <video controls preload="metadata" playsInline src={url} />
                  ) : null}
                  <div>
                    <strong>{video.title || `阶段视频 ${index + 1}`}</strong>
                    {video.content_summary && <p>{video.content_summary}</p>}
                    {url && <a href={url} target="_blank" rel="noreferrer"><ExternalLink size={13} />跳转播放</a>}
                  </div>
                </article>
              );
            })}
          </div>
        ) : <p className="material-empty">无</p>}
      </section>
    </article>
  );
}

function CurrentMaterialPanel({
  stage,
  learningPackage,
  onTest,
  stageResultReady,
  onOpenResult
}: {
  stage: StageInfo;
  learningPackage: LearningPackage | null;
  onTest: () => void;
  stageResultReady: boolean;
  onOpenResult: () => void;
}) {
  const content = learningPackage || fallbackPackage;
  return (
    <aside className="current-material-panel">
      <header>
        <div>
          <span>当前阶段学习包</span>
          <strong>{stage.block_name || content.block_name}</strong>
        </div>
      </header>
      <div className="current-material-scroll">
        <MaterialDocument stage={stage} learningPackage={content} compact />
      </div>
      <footer>
        <button onClick={stageResultReady ? onOpenResult : onTest}>
          {stageResultReady ? <ChevronRight size={16} /> : <FileQuestion size={16} />}
          {stageResultReady ? '去进行下一阶段' : '开始阶段测试'}
        </button>
      </footer>
    </aside>
  );
}

function ChatView({
  messages,
  input,
  setInput,
  onSend,
  stage,
  conversationStage,
  timerSeconds,
  timerRunning,
  onToggleTimer,
  onCreateGoal,
  onOpenLearningPath,
  onOpenProfile,
  learningPackage,
  onTest,
  stageResultReady,
  onOpenResult,
  projects,
  currentSessionId,
  selectedConversationId,
  onSelectProjectStage,
  readOnly,
  readOnlyNotice,
  loading,
  error
}: {
  messages: Message[];
  input: string;
  setInput: (input: string) => void;
  onSend: (event: FormEvent) => void;
  stage: StageInfo;
  conversationStage: StageInfo;
  timerSeconds: number;
  timerRunning: boolean;
  onToggleTimer: () => void;
  onCreateGoal: () => void;
  onOpenLearningPath: () => void;
  onOpenProfile: () => void;
  learningPackage: LearningPackage | null;
  onTest: () => void;
  stageResultReady: boolean;
  onOpenResult: () => void;
  projects: LearningProject[];
  currentSessionId: string | null;
  selectedConversationId: string;
  onSelectProjectStage: (sessionId: string, stageNum: number) => void;
  readOnly: boolean;
  readOnlyNotice: string;
  loading: boolean;
  error: string | null;
}) {
  const [contextPanelCollapsed, setContextPanelCollapsed] = useState(false);
  const [materialPanelWidth, setMaterialPanelWidth] = useState(600);

  const clampMaterialWidth = (width: number, layoutWidth: number) => {
    const contextWidth = contextPanelCollapsed ? 48 : 260;
    const maxMaterialWidth = Math.max(440, layoutWidth - contextWidth - 300 - 10);
    return Math.round(Math.max(440, Math.min(maxMaterialWidth, width)));
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

  const layoutStyle = {
    '--material-panel-width': `${materialPanelWidth}px`
  } as CSSProperties;

  return (
    <div className="page-shell">
      <Topbar
        title={conversationStage.block_name || `学习阶段 ${conversationStage.stage_num ?? 1}`}
        subtitle={conversationStage.subtopic_name || conversationStage.gap_skill || '阶段学习与实时答疑'}
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
          selectedConversationId={selectedConversationId}
          onSelectProjectStage={onSelectProjectStage}
          onCreateGoal={onCreateGoal}
          loading={loading}
          collapsed={contextPanelCollapsed}
          onToggleCollapsed={() => setContextPanelCollapsed((prev) => !prev)}
          timerSeconds={timerSeconds}
          timerRunning={timerRunning}
          onToggleTimer={onToggleTimer}
        />

        <CurrentMaterialPanel
          stage={stage}
          learningPackage={learningPackage}
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
          <div className="conversation">
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
                  <div className="message-bubble">{message.content}</div>
                  {message.role === 'assistant' && (
                    <div className="review-badge">
                      <Check size={13} />
                      技术审核 / 质量评审
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
                placeholder="针对当前阶段内容提问..."
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
  error
}: {
  stage: StageInfo;
  learningPackage: LearningPackage | null;
  onStart: () => void;
  onTest: () => void;
  loading: boolean;
  error: string | null;
}) {
  const [copied, setCopied] = useState<number | null>(null);
  const content = learningPackage || fallbackPackage;
  const codeExamples = content.code_examples || [];

  const copyCode = async (code: string, index: number) => {
    await navigator.clipboard?.writeText(code);
    setCopied(index);
    window.setTimeout(() => setCopied(null), 1200);
  };

  return (
    <div className="page-shell">
      <Topbar title="当前阶段" subtitle={`阶段 ${stage.stage_num ?? 1} · ${stage.block_name || content.block_name}`} loading={loading} error={error} />
      <main className="lesson">
        <div className="lesson-top">
          <div>
            <span className="lesson-kicker">阶段 {stage.stage_num ?? 1} · {statusText[stage.status || 'learning']}</span>
            <h2>{stage.block_name || content.block_name}</h2>
            <p>阅读学习包、运行代码示例，并在右侧状态里随时跟踪审核与掌握度变化。</p>
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
          <span style={{ width: content.tutorial ? '66%' : '18%' }} />
        </div>

        <div className="lesson-meta">
          <span>
            <Clock size={15} />
            预计 {stage.estimated_hours ?? 1} 小时
          </span>
          <span>
            <Target size={15} />
            {stage.gap_skill || '当前知识点'}
          </span>
          <span className="verified">
            <Check size={14} />
            双重审核展示
          </span>
        </div>

        <article className="lesson-article">
          <aside className="lesson-toc">
            <span>本节目录</span>
            <a href="#tutorial">教程</a>
            <a href="#code">代码示例</a>
            <a href="#quiz">阶段测试</a>
          </aside>
          <div className="lesson-content">
            <section id="tutorial">
              <span className="section-index">01</span>
              <h3>学习材料</h3>
              <div className="markdown-body">
                <ReactMarkdown>{content.tutorial || '点击「开始学习阶段」生成当前阶段学习包。'}</ReactMarkdown>
              </div>
            </section>

            <section id="code">
              <span className="section-index">02</span>
              <h3>代码示例</h3>
              {codeExamples.length === 0 && <p>当前学习包暂未返回代码示例。</p>}
              {codeExamples.map((example, index) => (
                <div className="code-block" key={`${example.title}-${index}`}>
                  <div className="code-head">
                    <span>
                      <Code2 size={14} />
                      {example.title || `代码示例 ${index + 1}`}
                    </span>
                    <button onClick={() => copyCode(example.code || '', index)}>
                      {copied === index ? <Check size={15} /> : <Copy size={15} />}
                      {copied === index ? '已复制' : '复制'}
                    </button>
                  </div>
                  <pre>
                    <code>{example.code}</code>
                  </pre>
                </div>
              ))}
            </section>

            <section id="quiz">
              <span className="section-index">03</span>
              <h3>阶段检查</h3>
              <p>完成测试后，系统会计算正确率，并调用阶段决策链路更新画像和下一阶段策略。</p>
              <button className="complete-reading" onClick={onTest}>
                <FileQuestion size={17} />
                开始阶段测试
              </button>
            </section>
          </div>
        </article>
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
  const current = questions[index];
  const done = questions.length > 0 && questions.every((question) => answers[question.question_id]);

  if (!current) return null;

  return (
    <div className="modal-backdrop">
      <section className="test-modal" role="dialog" aria-modal="true" aria-label="阶段测试">
        <header>
          <div>
            <span>阶段测试</span>
            <strong>{index + 1} / {questions.length}</strong>
          </div>
          <button onClick={onClose} aria-label="关闭测试">×</button>
        </header>
        <div className="test-body">
          <aside>
            <span>答题进度</span>
            <strong>{Object.keys(answers).length} / {questions.length}</strong>
            <div className="test-progress">
              <i style={{ width: `${questions.length ? (Object.keys(answers).length / questions.length) * 100 : 0}%` }} />
            </div>
            <div className="question-grid">
              {questions.map((question, questionIndex) => (
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
              {index < questions.length - 1 ? (
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
  const gaps = item?.knowledge_gaps || [];
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
          <div><span>身份/背景</span><strong>{item?.grade || '未提供'}</strong></div>
          <div><span>已有基础</span><strong>{item?.skill_tags?.join('、') || '暂无'}</strong></div>
          <div><span>知识盲区</span><strong>{item?.knowledge_gaps?.map((gap) => gap.skill).join('、') || '暂无'}</strong></div>
          <div><span>每日学习</span><strong>{item?.daily_hours ?? 2} 小时</strong></div>
          <div><span>学习方式</span><strong>{styleText[item?.learning_style || 'mixed'] || item?.learning_style}</strong></div>
          <div><span>设备条件</span><strong>{item?.hardware || '未提供'}</strong></div>
          <div><span>画像置信度</span><strong>{pct(item?.confidence ?? 0)}%</strong></div>
          <div><span>画像版本</span><strong>v{item?.version ?? 1}</strong></div>
        </div>
        {item && (
          <div className="profile-capability-panel">
            <div>
              <span>能力水平</span>
              <strong>{gaps.length ? `${gaps.filter((gap) => (gap.progress ?? 0) >= 1 || gap.status === '已掌握').length}/${gaps.length} 个知识点已掌握` : '等待学习记录'}</strong>
            </div>
            <Radar profile={item} />
          </div>
        )}
      </section>
    </div>
  );
}

function LearningPathModal({
  profile,
  stage,
  onClose
}: {
  profile: Profile;
  stage: StageInfo;
  onClose: () => void;
}) {
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
  const subtopics: KnowledgeSubtopic[] = (apiSubtopics.length ? apiSubtopics : selectedCatalogSubtopics)
    .map((topic) => ({ ...topic }))
    .sort((a, b) => (a.sequence_index ?? 0) - (b.sequence_index ?? 0));
  const currentTopicIndex = subtopics.findIndex((candidate) => (
      Boolean(stage.subtopic_id && candidate.subtopic_id === stage.subtopic_id)
      || Boolean(stage.subtopic_name && candidate.subtopic_name === stage.subtopic_name)
  ));
  const stageCompleted = ['completed', 'complete', '已完成'].includes(stage.status || '');
  const gapCompleted = currentGap?.status === '已掌握' || (currentGap?.progress ?? 0) >= 1;
  const inferredCompletedCount = Math.min(
    Math.max(subtopics.length - 1, 0),
    Math.floor((currentGap?.progress || 0) * subtopics.length)
  );
  const getSubtopicStatus = (topic: KnowledgeSubtopic, index: number): 'done' | 'current' | 'pending' => {
    if (stageCompleted || gapCompleted || topic.status === '已掌握' || (topic.progress ?? 0) >= 1) return 'done';
    if (currentTopicIndex >= 0) return index < currentTopicIndex ? 'done' : index === currentTopicIndex ? 'current' : 'pending';
    return index < inferredCompletedCount ? 'done' : index === inferredCompletedCount ? 'current' : 'pending';
  };
  const completedCount = subtopics.filter((topic, index) => getSubtopicStatus(topic, index) === 'done').length;
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
              <div><span>阶段 {stage.stage_num ?? 1}</span><h3>{stage.block_name || currentGraphBlock?.block_name || '当前学习阶段'}</h3></div>
              <strong>{completedCount}/{subtopics.length} 已完成</strong>
            </div>
            {prerequisites.length > 0 && <div className="stage-knowledge-prerequisites"><span>前置要求</span>{prerequisites.map((name) => <b key={name}>{name}</b>)}</div>}
            {subtopics.length > 0 ? (
              <div className="stage-knowledge-list">
                {subtopics.map((topic, index) => {
                  const topicStatus = getSubtopicStatus(topic, index);
                  return (
                    <div className={`path-subtopic ${topicStatus}`} key={topic.subtopic_id || `${topic.subtopic_name}-${index}`}>
                      <i>{topicStatus === 'done' ? <Check size={12} /> : topicStatus === 'current' ? <Play size={10} /> : index + 1}</i>
                      <span>{topic.subtopic_name}</span>
                      <small>{topicStatus === 'done' ? '已完成' : topicStatus === 'current' ? '当前' : '待学习'}</small>
                    </div>
                  );
                })}
              </div>
            ) : <p className="stage-knowledge-empty">等待后端返回当前阶段知识点</p>}
          </section>
          <footer className="path-legend">
            <span className="done">已完成</span><span className="current">当前知识点</span><span className="pending">待学习</span>
          </footer>
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
  result,
  setView,
  onSelectProject,
  loading,
  error
}: {
  projects: LearningProject[];
  currentSessionId: string | null;
  profile: Profile | null;
  stage: StageInfo;
  result: StageResult | null;
  setView: (view: View) => void;
  onSelectProject: (sessionId: string) => void;
  loading: boolean;
  error: string | null;
}) {
  const [selectedProject, setSelectedProject] = useState<LearningProject | null>(null);
  const sortedProjects = [...projects].sort((a, b) => b.updatedAt - a.updatedAt);

  return (
    <div className="page-shell">
      <Topbar title="学习仪表盘" subtitle="学习画像、阶段进度和策略决策汇总" loading={loading} error={error} />
      <main className="all-profiles-page learning-dashboard-page">
        <section className="profile-library-section">
          <header>
            <div>
              <span>学习画像</span>
              <h2>全部学习目标</h2>
            </div>
            <strong>{sortedProjects.length} 个目标</strong>
          </header>
          <div className="profile-goal-list">
            {sortedProjects.map((project) => {
              const current = project.sessionId === currentSessionId;
              return (
                <article className={current ? 'current' : ''} key={project.sessionId}>
                  <span className="profile-goal-icon"><Target size={19} /></span>
                  <div className="profile-goal-copy">
                    <div className="profile-goal-title">
                      <h3>{project.goal}</h3>
                      {current && <b>当前项目</b>}
                    </div>
                    <p>已生成 {project.stages?.length || 0} 个阶段 · 当前第 {project.stageNum} 阶段</p>
                    <small>{project.stageName || '等待阶段规划'}</small>
                  </div>
                  <div className="profile-goal-actions">
                    <button className="secondary" onClick={() => setSelectedProject(project)}>
                      <Eye size={16} />
                      查看画像
                    </button>
                    <button
                      className="primary"
                      onClick={() => onSelectProject(project.sessionId)}
                      disabled={loading || current}
                    >
                      {current ? <Check size={16} /> : <LayoutDashboard size={16} />}
                      {current ? '正在查看' : '查看仪表盘'}
                    </button>
                  </div>
                </article>
              );
            })}
            {sortedProjects.length === 0 && (
              <div className="profiles-empty">
                <UserRound size={28} />
                <strong>暂无学习画像</strong>
                <span>创建学习目标并完成诊断后，画像会显示在这里。</span>
              </div>
            )}
          </div>
        </section>

        {profile ? (
          <DashboardView
            profile={profile}
            stage={stage}
            result={result}
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
      {selectedProject && <ProfileDetailModal project={selectedProject} onClose={() => setSelectedProject(null)} />}
    </div>
  );
}

function Radar({ profile }: { profile: Profile }) {
  const gaps = profile.knowledge_gaps?.slice(0, 5) || fallbackProfile.knowledge_gaps!;
  const points = gaps.map((gap, index) => {
    const angle = (Math.PI * 2 * index) / gaps.length - Math.PI / 2;
    const radius = 64 * (gap.progress ?? 0.2);
    return `${120 + Math.cos(angle) * radius},${92 + Math.sin(angle) * radius}`;
  }).join(' ');

  return (
    <svg className="radar" viewBox="0 0 240 190" role="img" aria-label="能力掌握雷达图">
      <g>
        {[1, 0.75, 0.5, 0.25].map((scale) => (
          <polygon
            key={scale}
            points={`120,${92 - 64 * scale} ${120 + 61 * scale},${92 - 20 * scale} ${120 + 38 * scale},${92 + 52 * scale} ${120 - 38 * scale},${92 + 52 * scale} ${120 - 61 * scale},${92 - 20 * scale}`}
            className="radar-grid"
          />
        ))}
        <polygon points={points} className="radar-value" />
        {gaps.map((gap, index) => {
          const angle = (Math.PI * 2 * index) / gaps.length - Math.PI / 2;
          return (
            <text key={gap.skill} x={120 + Math.cos(angle) * 88} y={96 + Math.sin(angle) * 78} textAnchor="middle">
              {gap.skill.slice(0, 8)}
            </text>
          );
        })}
      </g>
    </svg>
  );
}

function DashboardView({
  profile,
  stage,
  result,
  setView,
  loading,
  error,
  embedded = false
}: {
  profile: Profile;
  stage: StageInfo;
  result: StageResult | null;
  setView: (view: View) => void;
  loading: boolean;
  error: string | null;
  embedded?: boolean;
}) {
  const gaps = profile.knowledge_gaps || [];
  const mastered = gaps.filter((gap) => gap.status === '已掌握').length;
  const accuracy = result?.accuracy ?? stage.accuracy ?? 0.87;

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

        <div className="metric-grid">
          <div className="metric">
            <span className="metric-icon blue"><Gauge size={22} /></span>
            <div><small>当前正确率</small><strong>{pct(accuracy)}<em>%</em></strong><p>阶段测试驱动更新</p></div>
          </div>
          <div className="metric">
            <span className="metric-icon violet"><Brain size={22} /></span>
            <div><small>信心指数</small><strong>{pct(profile.confidence_index ?? 0.76)}<em>/100</em></strong><p>结合问答与答题记录</p></div>
          </div>
          <div className="metric">
            <span className="metric-icon green"><Clock size={22} /></span>
            <div><small>每日计划</small><strong>{profile.daily_hours ?? 2}<em>小时</em></strong><p>{profile.learning_rhythm || 'medium'} 节奏</p></div>
          </div>
          <div className="metric">
            <span className="metric-icon orange"><Award size={22} /></span>
            <div><small>掌握知识点</small><strong>{mastered}<em>/{gaps.length || 4}</em></strong><p>来自知识盲区状态</p></div>
          </div>
        </div>

        <div className="dashboard-grid">
          <section className="white-card">
            <div className="card-title">
              <div>
                <h3>能力掌握分布</h3>
                <p>阶段结束后自动刷新</p>
              </div>
            </div>
            <Radar profile={profile} />
          </section>
          <section className="white-card">
            <div className="card-title">
              <div>
                <h3>阶段学习进度</h3>
                <p>当前阶段和下一阶段策略</p>
              </div>
              <span>实时调整</span>
            </div>
            <div className="learning-path">
              <div className="path-item done">
                <i><Check size={14} /></i>
                <div><span>阶段 0</span><strong>基础能力诊断</strong><small>画像可信度 {pct(profile.confidence ?? 0.7)}%</small></div>
                <b>完成</b>
              </div>
              <div className="path-item active">
                <i>{stage.stage_num ?? 1}</i>
                <div><span>当前阶段</span><strong>{stage.block_name || fallbackStage.block_name}</strong><small>预计 {stage.estimated_hours ?? 1} 小时</small></div>
                <b>{statusText[stage.status || 'learning']}</b>
              </div>
              <div className="path-item next">
                <i><ChevronRight size={13} /></i>
                <div><span>下一策略</span><strong>{result?.next_strategy || '等待测试结果'}</strong><small>{result?.stage_summary || '由 Agent6 根据正确率和交互记录决策'}</small></div>
                <b>待规划</b>
              </div>
            </div>
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
  result,
  onPlanNext,
  onStartNext,
  onReturnToChat,
  loading,
  error
}: {
  profile: Profile;
  stage: StageInfo;
  plannedStage: StageInfo | null;
  result: StageResult | null;
  onPlanNext: () => void;
  onStartNext: () => void;
  onReturnToChat: () => void;
  loading: boolean;
  error: string | null;
}) {
  const accuracy = result?.accuracy ?? 0.87;
  const gaps = profile.knowledge_gaps || [];
  const mastered = gaps.filter((gap) => gap.status === '已掌握' || (gap.progress ?? 0) >= 1);
  const needsWork = gaps.filter((gap) => gap.status !== '已掌握' && (gap.progress ?? 0) < 1);
  const canPlanNext = result?.can_plan_next_stage ?? result?.available_actions?.includes('plan_next_stage') ?? false;
  return (
    <div className="page-shell">
      <Topbar
        title="阶段结果"
        subtitle="阶段完成后展示画像更新和下一阶段策略"
        loading={loading}
        error={error}
        onReturnToChat={onReturnToChat}
      />
      <main className="result-page">
        <section className="result-hero">
          <div className="score-ring">
            <span><strong>{pct(accuracy)}</strong><small>正确率</small></span>
          </div>
          <div>
            <span className="success-label"><Check size={14} /> 阶段完成</span>
            <h2>{accuracy >= 0.8 ? '很好，可以进入下一阶段' : '建议先巩固当前阶段'}</h2>
            <p>系统已结合本阶段学习记录和测试结果生成反馈，并更新为画像 v{profile.version ?? 1}。</p>
          </div>
          <div className="strategy">
            <span>下一阶段策略</span>
            <strong>{result?.next_strategy || (accuracy >= 0.8 ? 'advance' : 'reinforce')}</strong>
            <small>Agent6 综合决策</small>
          </div>
        </section>
        <div className="feedback-summary-grid">
          <section className="white-card feedback-overview">
            <div className="section-title"><div><span>反馈总结</span><h3>本阶段学习情况</h3></div><b>{statusText[stage.status || 'completed'] || stage.status}</b></div>
            <div className="feedback-facts">
              <div><span>学习主题</span><strong>{stage.block_name || '当前阶段'}</strong></div>
              <div><span>测试结果</span><strong>{pct(accuracy)}% 正确率</strong></div>
              <div><span>策略建议</span><strong>{result?.next_strategy || (accuracy >= 0.8 ? 'advance' : 'reinforce')}</strong></div>
            </div>
            <div className="feedback-knowledge-columns">
              <div><span>已掌握知识点</span><p>{mastered.map((gap) => gap.skill).join('、') || '本阶段暂无新增完全掌握项'}</p></div>
              <div><span>仍需加强</span><p>{needsWork.map((gap) => gap.skill).join('、') || '当前目标知识点已全部掌握'}</p></div>
            </div>
            <div className="feedback-markdown"><ReactMarkdown>{result?.stage_summary || '本阶段记录已完成。建议根据测试结果巩固薄弱知识点，再继续后续学习。'}</ReactMarkdown></div>
          </section>
          <section className="white-card">
            <div className="section-title">
              <div><span>画像变化</span><h3>能力掌握度更新</h3></div>
              <b>v{profile.version ?? 1}</b>
            </div>
            <Radar profile={profile} />
            <div className="updated-profile-facts">
              <div><span>信心指数</span><strong>{pct(profile.confidence_index ?? 0)}%</strong></div>
              <div><span>困惑标记</span><strong>{profile.confusion_markers?.join('、') || '暂无'}</strong></div>
            </div>
          </section>
          <section className="white-card next-stage">
            <div className="stage-eyebrow">
              <span>{result?.all_mastered ? '学习目标' : plannedStage ? '下一阶段已就绪' : '下一阶段建议'}</span>
              <b>{result?.all_mastered ? '已完成' : `阶段 ${plannedStage?.stage_num ?? (stage.stage_num ?? 1) + 1}`}</b>
            </div>
            <h3>{result?.all_mastered ? profile.goal : plannedStage?.block_name || '根据新画像按需生成'}</h3>
            <p>{result?.all_mastered ? '全部目标知识点已掌握，本轮学习计划已经完成。' : plannedStage?.gap_skill ? `下一阶段将继续处理「${plannedStage.gap_skill}」。` : '生成动作只会创建下一个阶段，不会一次生成全部学习资料。'}</p>
            <div className="next-facts">
              <div><span>预计时长</span><strong>{plannedStage?.estimated_hours ?? stage.estimated_hours ?? 1} 小时</strong></div>
              <div><span>学习方式</span><strong>{styleText[profile.learning_style || 'mixed']}</strong></div>
              <div><span>画像依据</span><strong className="green">v{profile.version ?? 1}</strong></div>
            </div>
            {!result?.all_mastered && (plannedStage ? (
              <button className="primary wide" onClick={onStartNext} disabled={loading}><Play size={17} />开始下一阶段</button>
            ) : (
              <button className="primary wide" onClick={onPlanNext} disabled={loading || !canPlanNext}><Sparkles size={17} />生成下一阶段</button>
            ))}
          </section>
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
  onSave
}: {
  profile: Profile;
  loading: boolean;
  error: string | null;
  onClear: () => void;
  onSave: (data: { daily_hours?: number; learning_style?: string; hardware?: string }) => void;
}) {
  const [dailyHours, setDailyHours] = useState(String(profile.daily_hours ?? 2));
  const [learningStyle, setLearningStyle] = useState(profile.learning_style || 'mixed');
  const [hardware, setHardware] = useState(profile.hardware || '');

  useEffect(() => {
    setDailyHours(String(profile.daily_hours ?? 2));
    setLearningStyle(profile.learning_style || 'mixed');
    setHardware(profile.hardware || '');
  }, [profile.daily_hours, profile.learning_style, profile.hardware]);

  return (
    <div className="page-shell">
      <Topbar title="设置" subtitle="调整本地会话体验和学习偏好展示" loading={loading} error={error} />
      <main className="settings-page">
        <div className="settings-intro">
          <h2>学习偏好</h2>
          <p>这些设置会写回当前学习目标画像，用于后续阶段规划。</p>
        </div>
        <section className="settings-card">
          <div className="setting-row">
            <div><strong>每日学习时长</strong><p>用于规划阶段内容量</p></div>
            <select value={dailyHours} onChange={(event) => setDailyHours(event.target.value)}>
              <option value="1">1 小时</option>
              <option value="2">2 小时</option>
              <option value="3">3 小时</option>
            </select>
          </div>
          <div className="setting-row">
            <div><strong>学习风格</strong><p>影响教程理论、代码与练习比例</p></div>
            <div className="segmented">
              <button className={learningStyle === 'theory' ? 'active' : ''} onClick={() => setLearningStyle('theory')}>理论型</button>
              <button className={learningStyle === 'practice' ? 'active' : ''} onClick={() => setLearningStyle('practice')}>实操型</button>
              <button className={learningStyle === 'mixed' ? 'active' : ''} onClick={() => setLearningStyle('mixed')}>混合型</button>
            </div>
          </div>
          <div className="setting-row">
            <div><strong>设备条件</strong><p>影响视频、代码与部署建议</p></div>
            <input value={hardware} onChange={(event) => setHardware(event.target.value)} placeholder="例如 RTX 4060 / MacBook / 无独显" />
          </div>
          <div className="setting-row">
            <div><strong>保存画像偏好</strong><p>仅更新目标画像设置，不会重跑首次诊断</p></div>
            <button
              className="primary"
              onClick={() => onSave({ daily_hours: Number(dailyHours), learning_style: learningStyle, hardware })}
              disabled={loading}
            >
              保存设置
            </button>
          </div>
          <div className="setting-row">
            <div><strong>当前会话</strong><p>清除后会回到本地演示状态</p></div>
            <button className="danger" onClick={onClear}>清除本地会话</button>
          </div>
        </section>
      </main>
    </div>
  );
}

export default function App() {
  const [view, setView] = useState<View>('chat');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [workflow, setWorkflow] = useState<WorkflowState>('onboarding');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [profile, setProfile] = useState<Profile | null>(null);
  const [stage, setStage] = useState<StageInfo | null>(null);
  const [learningPackage, setLearningPackage] = useState<LearningPackage | null>(null);
  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [input, setInput] = useState('');
  const [onboardingForm, setOnboardingForm] = useState<OnboardingFormData>(emptyOnboardingForm);
  const [followupQuestions, setFollowupQuestions] = useState<string[]>([]);
  const [highlightedFields, setHighlightedFields] = useState<OnboardingField[]>([]);
  const [timerSeconds, setTimerSeconds] = useState(0);
  const [timerRunning, setTimerRunning] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [testOpen, setTestOpen] = useState(false);
  const [result, setResult] = useState<StageResult | null>(null);
  const [plannedStage, setPlannedStage] = useState<StageInfo | null>(null);
  const [assessmentState, setAssessmentState] = useState<AssessmentState>({ attemptId: null, questions: [] });
  const [agentStatuses, setAgentStatuses] = useState<AgentStatusMap>(createAgentStatusMap);
  const [learningProjects, setLearningProjects] = useState<LearningProject[]>(() => {
    const activeSessionId = localStorage.getItem('sessionId');
    const ownedSessionIds = readOwnedSessionIds();
    const projects = readLearningProjects().filter((item) =>
      isLocalSessionId(item.sessionId)
      || ownedSessionIds.has(item.sessionId)
      || item.sessionId === activeSessionId
    );
    return projects.filter((item) => item.sessionId !== DEMO_SESSION_ID);
  });
  const [stageConversations, setStageConversations] = useState<StoredStageConversation[]>(() => {
    const conversations = readStageConversations();
    return conversations.filter((item) => item.sessionId !== DEMO_SESSION_ID);
  });
  const [selectedConversationId, setSelectedConversationId] = useState('');
  const [projectSwitcherOpen, setProjectSwitcherOpen] = useState(false);
  const [materialPanelOpen, setMaterialPanelOpen] = useState(false);
  const [learningPathOpen, setLearningPathOpen] = useState(false);
  const [profileModalOpen, setProfileModalOpen] = useState(false);
  const [newGoalReturnState, setNewGoalReturnState] = useState<WorkspaceSnapshot | null>(null);
  const [reuseUserProfile, setReuseUserProfile] = useState(false);

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
      const knownStages = [...(existing?.stages || [])];
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
        updatedAt: Date.now(),
        profile: nextProfile,
        stages: knownStages.sort((a, b) => (a.stage_num ?? 1) - (b.stage_num ?? 1))
      };
      return [project, ...prev.filter((item) => item.sessionId !== nextSessionId)].slice(0, 20);
    });
  };

  const rememberStageConversation = (nextSessionId: string, nextStage: StageInfo, nextProfile?: Profile | null) => {
    const stageNum = nextStage.stage_num ?? 1;
    const id = `${nextSessionId}:${stageNum}`;
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

  const mergeDashboardStages = (nextSessionId: string, history: StageInfo[], currentStage?: StageInfo | null) => {
    setLearningProjects((prev) => prev.map((project) => {
      if (project.sessionId !== nextSessionId) return project;
      const byStageNum = new Map<number, StageInfo>();
      [...(project.stages || []), ...history, ...(currentStage ? [currentStage] : [])].forEach((item) => {
        const stageNum = item.stage_num ?? 1;
        byStageNum.set(stageNum, { ...(byStageNum.get(stageNum) || {}), ...item, stage_num: stageNum });
      });
      return {
        ...project,
        stages: [...byStageNum.values()].sort((a, b) => (a.stage_num ?? 1) - (b.stage_num ?? 1))
      };
    }));
  };

  const loadDashboardStages = async (nextSessionId: string, currentStage?: StageInfo | null) => {
    try {
      const dashboard = await apiClient.getDashboard(nextSessionId);
      const history = Array.isArray(dashboard?.stage_history) ? dashboard.stage_history as StageInfo[] : [];
      mergeDashboardStages(nextSessionId, history, currentStage);
    } catch {
      // Session restoration remains usable when the optional dashboard endpoint is unavailable.
    }
  };

  const refreshRemoteProjects = async () => {
    try {
      const response = await apiClient.listSessions();
      const ownedSessionIds = readOwnedSessionIds();
      const activeSessionId = localStorage.getItem('sessionId');
      if (activeSessionId && !isLocalSessionId(activeSessionId)) ownedSessionIds.add(activeSessionId);
      const remoteProjects = (response.sessions || [])
        .map(sessionSummaryToProject)
        .filter((item): item is LearningProject => Boolean(item && ownedSessionIds.has(item.sessionId)));
      setLearningProjects((prev) => {
        const localOnly = prev.filter((item) => isLocalSessionId(item.sessionId));
        const remoteBySession = new Map(remoteProjects.map((item) => [item.sessionId, item]));
        const mergedOwned = [...ownedSessionIds].map((ownedSessionId) => {
          const remote = remoteBySession.get(ownedSessionId);
          const cached = prev.find((item) => item.sessionId === ownedSessionId);
          if (!remote) return cached;
          if (!cached) return remote;
          return {
            ...cached,
            ...remote,
            // The list API is compact. Retain local history and a just-planned
            // stage until a full session snapshot has replaced it.
            stages: cached.stages,
            stageName: remote.stageName !== '等待阶段规划' ? remote.stageName : cached.stageName,
            stageNum: remote.stageName !== '等待阶段规划' ? remote.stageNum : cached.stageNum,
          };
        }).filter((item): item is LearningProject => Boolean(item));
        return [...mergedOwned, ...localOnly]
          .sort((a, b) => b.updatedAt - a.updatedAt)
          .slice(0, 20);
      });
    } catch {
      // Keep local cache if the session list endpoint is unavailable.
    }
  };

  const restoreQaConversation = async (
    nextSessionId: string,
    nextStage: StageInfo,
    nextProfile?: Profile | null,
    replaceCurrentMessages = false
  ) => {
    try {
      const history = await apiClient.getQaHistory(nextSessionId);
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
    void refreshRemoteProjects();
  }, []);

  useEffect(() => {
    if (!sessionId || !stage || workflow !== 'learning' || view !== 'chat') return;
    const stageNum = stage.stage_num ?? 1;
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
  }, [messages, sessionId, stage, workflow, view]);

  useEffect(() => {
    const estimatedSeconds = Math.round((stage?.estimated_hours ?? 1) * 3600);
    setTimerSeconds(estimatedSeconds);
    setTimerRunning(false);
  }, [stage?.stage_id, stage?.estimated_hours]);

  useEffect(() => {
    if (!timerRunning) return;
    const intervalId = window.setInterval(() => {
      setTimerSeconds((prev) => {
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
  }, [timerRunning]);

  const toggleStageTimer = () => {
    setError(null);
    if (timerSeconds <= 0) {
      setTimerSeconds(Math.round((stage?.estimated_hours ?? 1) * 3600));
      setTimerRunning(true);
      return;
    }
    setTimerRunning((prev) => !prev);
  };

  useEffect(() => {
    const savedSessionId = localStorage.getItem('sessionId');
    if (!savedSessionId) return;
    if (isLocalSessionId(savedSessionId)) {
      localStorage.removeItem('sessionId');
      setLearningProjects((projects) => projects.filter((project) => !isLocalSessionId(project.sessionId)));
      setStageConversations((conversations) => conversations.filter((conversation) => !isLocalSessionId(conversation.sessionId)));
      setWorkflow('onboarding');
      return;
    }
    setSessionId(savedSessionId);
    apiClient.getSession(savedSessionId)
      .then((session) => {
        const activeStage = resolveSessionStage(session);
        const hasLearningPackage = Boolean(session?.learning_package);
        const nextWorkflow = isLearningPhase(session.phase) || hasLearningPackage ? 'learning' : 'profile_ready';
        setAgentStatuses(restoreAgentStatuses({
          hasProfile: Boolean(session?.profile),
          hasLearningPackage,
          phase: session.phase
        }));
        if (session?.profile) {
          const nextProfile = toProfile(session.profile);
          if (!nextProfile) throw new Error('会话画像解析失败');
          setProfile(nextProfile);
          setOnboardingForm(formDataFromProfile(nextProfile));
          rememberProject(savedSessionId, nextProfile, session.next_stage || activeStage, session.phase);
          void loadDashboardStages(savedSessionId, activeStage);
          setWorkflow(nextWorkflow);
        } else {
          localStorage.removeItem('sessionId');
          setSessionId(null);
          void restoreDiagnosisState(savedSessionId);
          setWorkflow('onboarding');
        }
        if (activeStage) setStage(activeStage);
        if (session?.next_stage) setPlannedStage(session.next_stage);
        if (activeStage) setSelectedConversationId(`${savedSessionId}:${activeStage.stage_num ?? 1}`);
        if (hasLearningPackage) {
          setLearningPackage(toLearningPackage(session.learning_package));
          setView('chat');
          const conversationStage = activeStage || session.next_stage;
          if (conversationStage) {
            void restoreQaConversation(
              savedSessionId,
              conversationStage,
              session?.profile ? toProfile(session.profile) : profile,
              true
            );
          }
        }
      })
      .catch(() => {
        localStorage.removeItem('sessionId');
        setSessionId(null);
        setWorkflow('onboarding');
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

  const buildDemoProfile = (formData: OnboardingFormData): Profile => ({
    ...fallbackProfile,
    grade: formData.grade,
    skill_tags: formData.skill_tags,
    goal: formData.goal,
    daily_hours: Number(formData.daily_hours) || 2,
    learning_style: formData.learning_style,
    hardware: formData.hardware,
    confidence: 0.86
  });

  const submitOnboarding = async () => {
    if (loading || workflow === 'diagnosing') return;
    setError(null);
    setFollowupQuestions([]);
    setHighlightedFields([]);
    setLoading(true);
    setWorkflow('diagnosing');
    beginAgentRun(['agent1'], true);

    try {
      const activeSessionId = sessionId || (await apiClient.createSession()).session_id;
      const response = await apiClient.submitDiagnosis(
        activeSessionId,
        buildDiagnosticMessage(onboardingForm, reuseUserProfile)
      );
      setSessionId(activeSessionId);
      localStorage.setItem('sessionId', activeSessionId);

      if (response.status === 'needs_more_info') {
        const questions = response.followup_questions || [];
        setFollowupQuestions(questions);
        setHighlightedFields(response.missing_fields?.length ? response.missing_fields as OnboardingField[] : inferHighlightedFields(questions));
        if (response.profile_draft) {
          const draftProfile = toProfile(response.profile_draft);
          if (draftProfile) {
            setProfile(draftProfile);
            setOnboardingForm((current) => ({ ...current, ...formDataFromProfile(draftProfile) }));
          }
        }
        setWorkflow('needs_more_info');
        finishAgentRun(['agent1'], true);
        return;
      }

      if (response.profile_draft) {
        const nextProfile = { ...(toProfile(response.profile_draft) || fallbackProfile), confidence: response.confidence } as Profile;
        setProfile(nextProfile);
        setOnboardingForm(formDataFromProfile(nextProfile));
        rememberProject(activeSessionId, nextProfile, null);
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
    setInput('');
    setError(null);
    setMessages((prev) => [...prev, { id: shortId(), role: 'user', content: text }]);
    setLoading(true);
    const initialDiagnosis = !sessionId || !profile?.goal;
    const invokedAgents: AgentId[] = initialDiagnosis ? ['agent1'] : ['agent3', 'agent4'];
    beginAgentRun(invokedAgents, initialDiagnosis);

    try {
      if (initialDiagnosis) {
        const activeSessionId = sessionId || (await apiClient.createSession()).session_id;
        const response = await apiClient.submitDiagnosis(activeSessionId, text);
        setSessionId(activeSessionId);
        localStorage.setItem('sessionId', activeSessionId);
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
        const response = await apiClient.askQuestion(sessionId, text);
        setMessages((prev) => [...prev, {
          id: shortId(),
          role: 'assistant',
          content: response.answer || '已收到问题，但后端没有返回回答内容。',
          note: response.review_summary?.technical_passed && response.review_summary?.quality_passed ? '审核通过' : '待复核'
        }]);
        finishAgentRun(['agent3'], Boolean(response.answer));
        finishAgentRun(['agent4'], Boolean(response.review_summary));
      }
      if (initialDiagnosis) finishAgentRun(['agent1'], true);
    } catch (caught) {
      finishAgentRun(invokedAgents, false);
      setError(getErrorMessage(caught));
      if (!initialDiagnosis) {
        setMessages((prev) => [...prev, {
          id: shortId(),
          role: 'assistant',
          content: '本次答疑请求未完成，请检查后端服务后重试。',
          note: '请求失败'
        }]);
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

  const startStage = async () => {
    setError(null);
    setLoading(true);
    setWorkflow('starting_stage');
    beginAgentRun(['agent3', 'agent4']);
    try {
      if (!sessionId) throw new Error('尚未创建后端会话');
      let generatedStage = plannedStage || stage;
      if (workflow === 'profile_ready') {
        const confirmation = await apiClient.confirmProfile(sessionId);
        generatedStage = toStageInfo(confirmation.stage_info) || generatedStage;
      }
      await apiClient.startStage(sessionId);
      const [packageResponse, stageResponse] = await Promise.all([
        apiClient.getStagePackage(sessionId),
        apiClient.getCurrentStage(sessionId)
      ]);
      setLearningPackage(toLearningPackage(packageResponse.package));
      generatedStage = toStageInfo(stageResponse.stage) || generatedStage;
      if (!generatedStage) throw new Error('后端未返回当前阶段，无法启动学习包');
      setStage(generatedStage);
      const generatedConversationId = `${sessionId}:${generatedStage.stage_num ?? 1}`;
      const existingConversation = stageConversations.find((item) => item.id === generatedConversationId);
      setMessages(existingConversation?.messages || createStageConversationMessages(generatedStage, profile));
      setSelectedConversationId(generatedConversationId);
      if (profile) rememberProject(sessionId, profile, generatedStage);
      rememberStageConversation(sessionId, generatedStage, profile);
      finishAgentRun(['agent3'], Boolean(packageResponse.package));
      finishAgentRun(['agent4'], Boolean(packageResponse.package?.review_summary));
      setMessages((prev) => [...prev, {
        id: shortId(),
        role: 'assistant',
        content: `学习包已生成，审核结果：${packageResponse.package?.review_summary?.status || '已完成'}。`,
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
    } catch (caught) {
      finishAgentRun(['agent3', 'agent4'], false);
      setError(getErrorMessage(caught));
      setWorkflow('profile_ready');
    } finally {
      setLoading(false);
    }
  };

  const completeStage = async (answers: Record<string, string>) => {
    setLoading(true);
    setError(null);
    beginAgentRun(['agent6']);
    try {
      if (sessionId) {
        const attemptId = assessmentState.attemptId || (await apiClient.createAssessmentAttempt(sessionId)).attempt_id;
        const attempt = await apiClient.getAssessmentAttempt(attemptId);
        if (attempt.status !== 'submitted') {
          for (const [question_id, user_answer] of Object.entries(answers)) {
            await apiClient.saveAssessmentAnswer(attemptId, question_id, { user_answer, time_spent: 0 });
          }
          await apiClient.submitAssessmentAttempt(attemptId);
        }
        const response = await apiClient.completeStage(sessionId);
        setResult(response);
        if (response.updated_profile) setProfile(toProfile(response.updated_profile));
        const completedStage = response.state?.stage_info || stage;
        if (completedStage) setStage(completedStage);
        setPlannedStage(null);
        setAssessmentState({ attemptId: null, questions: [] });
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

  const planNextStage = async () => {
    if (!sessionId || loading) return;
    setLoading(true);
    setError(null);
    beginAgentRun(['agent1']);
    try {
      const response = await apiClient.planNextStage(sessionId);
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
    if (sessionId && !sessionId.startsWith('local-goal-') && sessionId !== DEMO_SESSION_ID) {
      try {
        await apiClient.abandonSession(sessionId);
      } catch {
        // Keep local reset usable even if abandon fails remotely.
      }
    }
    localStorage.removeItem('sessionId');
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
    setOnboardingForm(emptyOnboardingForm);
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
    void refreshRemoteProjects();
  };

  const saveProfileSettings = async (data: { daily_hours?: number; learning_style?: string; hardware?: string }) => {
    if (!sessionId || sessionId.startsWith('local-goal-')) {
      setProfile((current) => current ? { ...current, ...data } : current);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const updated = await apiClient.updateProfile(sessionId, data);
      const nextProfile = toProfile(updated) || { ...(profile || fallbackProfile), ...data };
      setProfile(nextProfile);
      setOnboardingForm(formDataFromProfile(nextProfile));
      rememberProject(sessionId, nextProfile, stage);
      void refreshRemoteProjects();
    } catch (caught) {
      setError(getErrorMessage(caught));
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
      setError(getErrorMessage(caught));
    } finally {
      setLoading(false);
    }
  };

  const createLearningGoal = () => {
    setProjectSwitcherOpen(false);
    const reusableProfile = profile || learningProjects.find((project) => project.profile)?.profile;
    setReuseUserProfile(Boolean(reusableProfile));
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
    setOnboardingForm(reusableProfile ? {
      ...formDataFromProfile(reusableProfile),
      goal: '',
      daily_hours: String(reusableProfile.daily_hours ?? 2)
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
    if (previous.sessionId) localStorage.setItem('sessionId', previous.sessionId);
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
    setFollowupQuestions([]);
    setHighlightedFields([]);
    setError(null);
    setWorkflow('learning');
    setView('chat');
    setNewGoalReturnState(null);
  };

  const switchLearningProject = async (nextSessionId: string, requestedStageNum?: number, destination: View = 'chat') => {
    if (nextSessionId === sessionId || loading) {
      if (nextSessionId === sessionId && requestedStageNum) {
        setSelectedConversationId(`${nextSessionId}:${requestedStageNum}`);
      }
      return;
    }
    if (isLocalSessionId(nextSessionId)) {
      localStorage.removeItem('sessionId');
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
      setSessionId(nextSessionId);
      localStorage.setItem('sessionId', nextSessionId);
      setProfile(nextProfile);
      setOnboardingForm(formDataFromProfile(nextProfile));
      setStage(activeStage);
      setLearningPackage(toLearningPackage(session.learning_package));
      setResult(null);
      setPlannedStage(session.next_stage || null);
      const currentStageNum = activeStage?.stage_num ?? 1;
      const restoredConversation = stageConversations.find((item) => item.id === `${nextSessionId}:${currentStageNum}`);
      setMessages(restoredConversation?.messages || initialMessages);
      setSelectedConversationId(`${nextSessionId}:${requestedStageNum ?? currentStageNum}`);
      setAgentStatuses(restoreAgentStatuses({
        hasProfile: true,
        hasLearningPackage,
        phase: session.phase
      }));
      setWorkflow(isLearningPhase(session.phase) || hasLearningPackage ? 'learning' : 'profile_ready');
      setView(destination);
      rememberProject(nextSessionId, nextProfile, session.next_stage || activeStage, session.phase);
      void loadDashboardStages(nextSessionId, activeStage);
      if (activeStage) void restoreQaConversation(nextSessionId, activeStage, nextProfile, true);
      setMaterialPanelOpen(false);
      setProjectSwitcherOpen(false);
    } catch (caught) {
      setError(`无法切换项目：${getErrorMessage(caught)}`);
    } finally {
      setLoading(false);
    }
  };

  const selectProjectStage = (nextSessionId: string, stageNum: number) => {
    void switchLearningProject(nextSessionId, stageNum);
  };

  const editProfile = () => {
    if (profile) setOnboardingForm(formDataFromProfile(profile));
    setReuseUserProfile(false);
    setError(null);
    setFollowupQuestions([]);
    setHighlightedFields([]);
    setWorkflow('onboarding');
    setView('chat');
  };

  const workspaceProfile = profile || fallbackProfile;
  const workspaceStage = stage || unavailableStage;
  const currentConversationId = sessionId && stage ? `${sessionId}:${stage.stage_num ?? 1}` : '';
  const selectedConversation = stageConversations.find((item) => item.id === selectedConversationId);
  const conversationStage = selectedConversation?.stage || workspaceStage;
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
      {view !== 'learn' && (!profile || workflow === 'onboarding' || workflow === 'diagnosing' || workflow === 'needs_more_info') && (
        <OnboardingPage
          formData={onboardingForm}
          onChange={setOnboardingForm}
          onSubmit={submitOnboarding}
          onCancel={newGoalReturnState ? cancelCreateLearningGoal : undefined}
          loading={loading}
          mode={workflow === 'diagnosing' ? 'diagnosing' : workflow === 'needs_more_info' ? 'needs_more_info' : 'onboarding'}
          questions={followupQuestions}
          error={error}
          highlightedFields={highlightedFields}
          reuseUserProfile={reuseUserProfile}
        />
      )}
      {view !== 'learn' && profile && (workflow === 'profile_ready' || workflow === 'starting_stage') && (
        <ProfileConfirmPage
          profile={profile}
          stage={workspaceStage}
          loading={loading}
          error={error}
          onEdit={editProfile}
          onStart={startStage}
          onCancel={newGoalReturnState ? cancelCreateLearningGoal : undefined}
        />
      )}
      {profile && workflow === 'learning' && view === 'chat' && (
        <ChatView
          messages={displayedMessages}
          input={input}
          setInput={setInput}
          onSend={sendMessage}
          stage={workspaceStage}
          conversationStage={conversationStage}
          timerSeconds={timerSeconds}
          timerRunning={timerRunning}
        onToggleTimer={toggleStageTimer}
        onCreateGoal={createLearningGoal}
        onOpenLearningPath={() => setLearningPathOpen(true)}
        onOpenProfile={() => setProfileModalOpen(true)}
        learningPackage={learningPackage}
        onTest={openTestModal}
          stageResultReady={Boolean(result)}
          onOpenResult={() => setView('result')}
          projects={learningProjects}
          currentSessionId={sessionId}
          selectedConversationId={selectedConversationId || currentConversationId}
          onSelectProjectStage={selectProjectStage}
          readOnly={historicalConversationSelected}
          readOnlyNotice={readOnlyConversationNotice}
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
          result={result}
          setView={setView}
          onSelectProject={(nextSessionId) => void switchLearningProject(nextSessionId, undefined, 'learn')}
          loading={loading}
          error={error}
        />
      )}
      {profile && workflow === 'learning' && view === 'settings' && <SettingsView profile={workspaceProfile} loading={loading} error={error} onClear={clearLocalSession} onSave={saveProfileSettings} />}
      {profile && workflow === 'learning' && view === 'result' && (
        <ResultView
          profile={workspaceProfile}
          stage={workspaceStage}
          plannedStage={plannedStage}
          result={result}
          onPlanNext={planNextStage}
          onStartNext={startStage}
          onReturnToChat={() => {
            setView('chat');
            setMaterialPanelOpen(true);
          }}
          loading={loading}
          error={error}
        />
      )}
      {testOpen && <TestModal questions={currentQuestions} onClose={() => setTestOpen(false)} onComplete={completeStage} />}
      {learningPathOpen && profile && stage && <LearningPathModal profile={profile} stage={stage} onClose={() => setLearningPathOpen(false)} />}
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
