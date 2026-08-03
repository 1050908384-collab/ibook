import { CSSProperties, FormEvent, PointerEvent as ReactPointerEvent, useEffect, useMemo, useState } from 'react';
import ReactMarkdown from 'react-markdown';
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
  FileQuestion,
  Folder,
  Gauge,
  Eye,
  LayoutDashboard,
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
  Target,
  UserRound,
  X
} from 'lucide-react';
import { apiClient } from './api/client';
import OnboardingPage from './pages/OnboardingPage';
import ProfileConfirmPage from './pages/ProfileConfirmPage';
import {
  buildDiagnosticMessage,
  emptyOnboardingForm,
  formDataFromProfile,
  OnboardingField,
  OnboardingFormData
} from './utils/buildDiagnosticMessage';

type View = 'chat' | 'learn' | 'dashboard' | 'settings' | 'result';
type WorkflowState = 'onboarding' | 'diagnosing' | 'needs_more_info' | 'profile_ready' | 'starting_stage' | 'learning';
type AgentId = 'agent1' | 'agent3' | 'agent4' | 'agent6';
type Message = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  note?: string;
};
type KnowledgeGap = {
  skill: string;
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
  options?: string[];
  correct_answer?: string;
  hint?: string;
};
type LearningPackage = {
  tutorial?: string;
  code_examples?: Array<{ title?: string; code?: string; language?: string }>;
  questions?: TestQuestion[];
  block_name?: string;
};
type StageResult = {
  status?: string;
  accuracy?: number;
  stage_summary?: string;
  next_strategy?: string;
  final_report?: Record<string, unknown>;
};
type LearningProject = {
  sessionId: string;
  goal: string;
  stageName: string;
  stageNum: number;
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

const PROJECTS_STORAGE_KEY = 'learningProjects';
const CONVERSATIONS_STORAGE_KEY = 'stageConversations';
const DEMO_SESSION_ID = 'demo-preview';

function canUseLocalPreviewDemo() {
  if (!import.meta.env.DEV) return false;
  const localHost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
  const onboardingRequested = new URLSearchParams(window.location.search).get('onboarding') === '1';
  return localHost && !onboardingRequested;
}

function shouldUseLocalPreviewDemo() {
  return canUseLocalPreviewDemo() && !localStorage.getItem('sessionId');
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
    { skill: '注意力机制', status: '学习中', progress: 0.68 },
    { skill: 'PEFT', status: '未掌握', progress: 0.34 },
    { skill: 'LoRA', status: '未开始', progress: 0.18 },
    { skill: '推理部署', status: '未开始', progress: 0.08 }
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
  block_name: 'Self-Attention 计算机制',
  estimated_hours: 1,
  gap_skill: '注意力机制',
  prerequisites: ['Python', 'PyTorch 张量基础'],
  status: 'learning'
};

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
  questions: [
    {
      question_id: 'Q-DEMO-1',
      question_type: '选择',
      question_text: '在缩放点积注意力中，除以 sqrt(d_k) 的主要目的是什么？',
      options: ['减少参数量', '避免点积过大导致 Softmax 梯度过小', '增加注意力头数量', '消除位置编码'],
      correct_answer: '避免点积过大导致 Softmax 梯度过小'
    },
    {
      question_id: 'Q-DEMO-2',
      question_type: '选择',
      question_text: '哪两个矩阵的点积决定注意力权重？',
      options: ['Q 与 K', 'K 与 V', 'Q 与 V', 'V 与输出'],
      correct_answer: 'Q 与 K'
    },
    {
      question_id: 'Q-DEMO-3',
      question_type: '选择',
      question_text: 'LoRA 通常可以注入 Transformer 的哪个位置？',
      options: ['数据加载器', 'Q、K、V 的线性映射层', '损失函数', 'Tokenizer 词表'],
      correct_answer: 'Q、K、V 的线性映射层'
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

function getErrorMessage(error: unknown) {
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
  activeAgent,
  completedAgents,
  collapsed,
  onToggleCollapsed
}: {
  view: View;
  setView: (view: View) => void;
  activeAgent: AgentId | null;
  completedAgents: AgentId[];
  collapsed: boolean;
  onToggleCollapsed: () => void;
}) {
  const nav = [
    { id: 'learn' as View, label: '所有学习画像', icon: UserRound },
    { id: 'chat' as View, label: '学习对话', icon: MessageCircle },
    { id: 'dashboard' as View, label: '学习仪表盘', icon: LayoutDashboard },
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
            const active = activeAgent === agent.id;
            const done = completedAgents.includes(agent.id);
            const state = active ? 'active' : done ? 'done' : 'pending';

            return (
              <div className={`agent-flow-card ${state}`} key={agent.id}>
                <span className={`agent-icon ${agent.tone}`}>
                  <Icon size={18} />
                </span>
                <div className="agent-copy">
                  <b>Agent{agent.number}</b>
                  <strong>{agent.title}</strong>
                  <small>{agent.desc}</small>
                </div>
                <span className={`agent-status ${state}`} aria-label={active ? '运行中' : done ? '已完成' : '待调用'}>
                  {active ? <Loader2 size={17} /> : done ? <Check size={15} /> : <Circle size={12} />}
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
  onCreateGoal,
  onOpenMaterials
}: {
  title: string;
  subtitle: string;
  loading: boolean;
  error: string | null;
  onCreateGoal?: () => void;
  onOpenMaterials?: () => void;
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
        {onCreateGoal && (
          <button className="switch-goal-button" onClick={onCreateGoal} disabled={loading}>
            <Plus size={15} />
            新建目标
          </button>
        )}
        {onOpenMaterials && (
          <button className="open-material-button" onClick={onOpenMaterials} disabled={loading}>
            <BookOpen size={15} />
            打开学习材料
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
        <p>{stage.gap_skill ? `围绕「${stage.gap_skill}」补齐关键知识点，并准备进入后续实操。` : '完成初始画像后，系统会自动生成第一阶段学习内容。'}</p>
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
  return (
    <article className={`material-document ${compact ? 'compact' : ''}`}>
      <div className="material-document-meta">
        <span>阶段 {stage.stage_num ?? 1}</span>
        <span><Clock size={13} /> {stage.estimated_hours ?? 1} 小时</span>
      </div>
      <h2>{stage.block_name || learningPackage.block_name || '未命名学习材料'}</h2>
      <div className="material-markdown">
        <ReactMarkdown>{learningPackage.tutorial || '当前学习包暂未返回教程内容。'}</ReactMarkdown>
      </div>
      {codeExamples.length > 0 && (
        <section className="material-code-section">
          <h3>代码示例</h3>
          {codeExamples.map((example, index) => (
            <div className="material-code" key={`${example.title}-${index}`}>
              <div><Code2 size={14} /> {example.title || `代码示例 ${index + 1}`}</div>
              <pre><code>{example.code}</code></pre>
            </div>
          ))}
        </section>
      )}
    </article>
  );
}

function CurrentMaterialPanel({
  stage,
  learningPackage,
  onClose,
  onTest,
  onResizeStart
}: {
  stage: StageInfo;
  learningPackage: LearningPackage | null;
  onClose: () => void;
  onTest: () => void;
  onResizeStart: (event: ReactPointerEvent<HTMLButtonElement>) => void;
}) {
  const content = learningPackage || fallbackPackage;
  return (
    <aside className="current-material-panel">
      <button
        className="material-resize-handle"
        onPointerDown={onResizeStart}
        aria-label="拖动调整学习资料宽度"
        title="拖动调整学习资料宽度"
      />
      <header>
        <div>
          <span>当前阶段学习包</span>
          <strong>{stage.block_name || content.block_name}</strong>
        </div>
        <button onClick={onClose} aria-label="关闭学习材料" title="关闭">
          <X size={17} />
        </button>
      </header>
      <div className="current-material-scroll">
        <MaterialDocument stage={stage} learningPackage={content} compact />
      </div>
      <footer>
        <button onClick={onTest}>
          <FileQuestion size={16} />
          开始阶段测试
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
  learningPackage,
  materialPanelOpen,
  onToggleMaterials,
  onTest,
  projects,
  currentSessionId,
  selectedConversationId,
  onSelectProjectStage,
  readOnly,
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
  learningPackage: LearningPackage | null;
  materialPanelOpen: boolean;
  onToggleMaterials: () => void;
  onTest: () => void;
  projects: LearningProject[];
  currentSessionId: string | null;
  selectedConversationId: string;
  onSelectProjectStage: (sessionId: string, stageNum: number) => void;
  readOnly: boolean;
  loading: boolean;
  error: string | null;
}) {
  const [contextPanelCollapsed, setContextPanelCollapsed] = useState(false);
  const [materialPanelWidth, setMaterialPanelWidth] = useState(380);

  const resizeMaterialPanel = (event: ReactPointerEvent<HTMLButtonElement>) => {
    event.preventDefault();
    const startX = event.clientX;
    const startWidth = materialPanelWidth;
    const maxWidth = Math.min(720, window.innerWidth * 0.65);

    const handlePointerMove = (moveEvent: PointerEvent) => {
      const nextWidth = Math.max(300, Math.min(maxWidth, startWidth + startX - moveEvent.clientX));
      setMaterialPanelWidth(Math.round(nextWidth));
    };
    const handlePointerUp = () => {
      document.body.classList.remove('resizing-material-panel');
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', handlePointerUp);
    };

    document.body.classList.add('resizing-material-panel');
    window.addEventListener('pointermove', handlePointerMove);
    window.addEventListener('pointerup', handlePointerUp, { once: true });
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
        onCreateGoal={onCreateGoal}
        onOpenMaterials={onToggleMaterials}
      />
      <div
        className={`chat-layout ${materialPanelOpen ? 'materials-open' : ''} ${contextPanelCollapsed ? 'context-collapsed' : ''}`}
        style={layoutStyle}
      >
        <ProfilePanel
          stage={stage}
          projects={projects}
          currentSessionId={currentSessionId}
          selectedConversationId={selectedConversationId}
          onSelectProjectStage={onSelectProjectStage}
          collapsed={contextPanelCollapsed}
          onToggleCollapsed={() => setContextPanelCollapsed((prev) => !prev)}
          timerSeconds={timerSeconds}
          timerRunning={timerRunning}
          onToggleTimer={onToggleTimer}
        />

        <main className="chat-main">
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
            <div className="historical-conversation-note">这是已结束阶段的对话记录，仅供查看。</div>
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
        {materialPanelOpen && (
          <CurrentMaterialPanel
            stage={stage}
            learningPackage={learningPackage}
            onClose={onToggleMaterials}
            onTest={onTest}
            onResizeStart={resizeMaterialPanel}
          />
        )}
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
  const current = questions[index] || fallbackPackage.questions![0];
  const done = questions.length > 0 && questions.every((question) => answers[question.question_id]);

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
                {current.options.map((option, optionIndex) => (
                  <button
                    key={option}
                    className={answers[current.question_id] === option ? 'selected' : ''}
                    onClick={() => setAnswers((prev) => ({ ...prev, [current.question_id]: option }))}
                  >
                    <span>{String.fromCharCode(65 + optionIndex)}</span>
                    {option}
                    {answers[current.question_id] === option && <Check size={18} />}
                  </button>
                ))}
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
              <button className="ghost" disabled={index === 0} onClick={() => setIndex(index - 1)}>上一题</button>
              {index < questions.length - 1 ? (
                <button className="primary" onClick={() => setIndex(index + 1)}>
                  下一题
                  <ChevronRight size={15} />
                </button>
              ) : (
                <button className="primary" disabled={!done} onClick={() => onComplete(answers)}>完成测试</button>
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
      </section>
    </div>
  );
}

function AllProfilesView({
  projects,
  loading,
  error
}: {
  projects: LearningProject[];
  loading: boolean;
  error: string | null;
}) {
  const [selectedProject, setSelectedProject] = useState<LearningProject | null>(null);
  const sortedProjects = [...projects].sort((a, b) => b.updatedAt - a.updatedAt);

  return (
    <div className="page-shell">
      <Topbar title="所有学习画像" subtitle="每个学习目标保留一份持续更新的独立画像" loading={loading} error={error} />
      <main className="all-profiles-page">
        <header>
          <div>
            <span>目标画像库</span>
            <h2>全部学习目标</h2>
          </div>
          <strong>{sortedProjects.length} 个目标</strong>
        </header>
        <div className="profile-goal-list">
          {sortedProjects.map((project) => (
            <article key={project.sessionId}>
              <span className="profile-goal-icon"><Target size={19} /></span>
              <div className="profile-goal-copy">
                <h3>{project.goal}</h3>
                <p>已生成 {project.stages?.length || 0} 个阶段 · 当前第 {project.stageNum} 阶段</p>
                <small>{project.stageName || '等待阶段规划'}</small>
              </div>
              <button onClick={() => setSelectedProject(project)}>
                <Eye size={16} />
                查看画像
              </button>
            </article>
          ))}
          {sortedProjects.length === 0 && (
            <div className="profiles-empty">
              <UserRound size={28} />
              <strong>暂无学习画像</strong>
              <span>创建学习目标并完成诊断后，画像会显示在这里。</span>
            </div>
          )}
        </div>
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
  error
}: {
  profile: Profile;
  stage: StageInfo;
  result: StageResult | null;
  setView: (view: View) => void;
  loading: boolean;
  error: string | null;
}) {
  const gaps = profile.knowledge_gaps || [];
  const mastered = gaps.filter((gap) => gap.status === '已掌握').length;
  const accuracy = result?.accuracy ?? stage.accuracy ?? 0.87;

  return (
    <div className="page-shell">
      <Topbar title="学习仪表盘" subtitle="画像、阶段、掌握度和策略决策汇总" loading={loading} error={error} />
      <main className="dashboard">
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

        <section className="dashboard-profile-card">
          <div className="dashboard-profile-head">
            <div>
              <span className="metric-icon blue"><UserRound size={20} /></span>
              <div>
                <small>学习画像</small>
                <h3>{profile.grade || '当前学习者'}</h3>
              </div>
            </div>
            <b>{pct(profile.confidence ?? 0.7)}% 可信</b>
          </div>
          <div className="dashboard-profile-details">
            <div>
              <span>已有基础</span>
              <strong>{profile.skill_tags?.join(' · ') || '等待诊断'}</strong>
            </div>
            <div>
              <span>知识盲区</span>
              <strong>{gaps.slice(0, 3).map((gap) => gap.skill).join(' · ') || '暂无'}</strong>
            </div>
            <div>
              <span>每日时长</span>
              <strong>{profile.daily_hours ?? 2} 小时</strong>
            </div>
            <div>
              <span>学习方式</span>
              <strong>{styleText[profile.learning_style || 'mixed'] || profile.learning_style}</strong>
            </div>
            <div>
              <span>设备条件</span>
              <strong>{profile.hardware || '未提供'}</strong>
            </div>
          </div>
        </section>

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
                <h3>动态学习路径</h3>
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
      </main>
    </div>
  );
}

function ResultView({
  profile,
  stage,
  result,
  setView,
  loading,
  error
}: {
  profile: Profile;
  stage: StageInfo;
  result: StageResult | null;
  setView: (view: View) => void;
  loading: boolean;
  error: string | null;
}) {
  const accuracy = result?.accuracy ?? 0.87;
  return (
    <div className="page-shell">
      <Topbar title="阶段结果" subtitle="阶段完成后展示画像更新和下一阶段策略" loading={loading} error={error} />
      <main className="result-page">
        <section className="result-hero">
          <div className="score-ring">
            <span><strong>{pct(accuracy)}</strong><small>正确率</small></span>
          </div>
          <div>
            <span className="success-label"><Check size={14} /> 阶段完成</span>
            <h2>{accuracy >= 0.8 ? '很好，可以进入下一阶段' : '建议先巩固当前阶段'}</h2>
            <p>{result?.stage_summary || '系统已记录本次阶段表现，并准备基于画像和正确率调整学习策略。'}</p>
          </div>
          <div className="strategy">
            <span>下一阶段策略</span>
            <strong>{result?.next_strategy || (accuracy >= 0.8 ? 'advance' : 'reinforce')}</strong>
            <small>Agent6 综合决策</small>
          </div>
        </section>
        <div className="result-grid">
          <section className="white-card">
            <div className="section-title">
              <div><span>画像变化</span><h3>能力掌握度更新</h3></div>
              <b>v{profile.version ?? 1}</b>
            </div>
            <Radar profile={profile} />
          </section>
          <section className="white-card next-stage">
            <div className="stage-eyebrow">
              <span>下一阶段</span>
              <b>阶段 {(stage.stage_num ?? 1) + 1}</b>
            </div>
            <h3>{stage.block_name || '等待后端规划下一阶段'}</h3>
            <p>继续沿着知识图谱依赖推进，优先处理仍未掌握的知识盲区。</p>
            <div className="next-facts">
              <div><span>预计时长</span><strong>{stage.estimated_hours ?? 1} 小时</strong></div>
              <div><span>学习方式</span><strong>{styleText[profile.learning_style || 'mixed']}</strong></div>
              <div><span>前置状态</span><strong className="green">已检查</strong></div>
            </div>
            <button className="primary wide" onClick={() => setView('chat')}>
              <Play size={17} />
              继续学习
            </button>
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
  onClear
}: {
  profile: Profile;
  loading: boolean;
  error: string | null;
  onClear: () => void;
}) {
  return (
    <div className="page-shell">
      <Topbar title="设置" subtitle="调整本地会话体验和学习偏好展示" loading={loading} error={error} />
      <main className="settings-page">
        <div className="settings-intro">
          <h2>学习偏好</h2>
          <p>初版先做前端展示，后续可把这些设置写入画像更新请求。</p>
        </div>
        <section className="settings-card">
          <div className="setting-row">
            <div><strong>每日学习时长</strong><p>用于规划阶段内容量</p></div>
            <select defaultValue={String(profile.daily_hours ?? 2)}>
              <option value="1">1 小时</option>
              <option value="2">2 小时</option>
              <option value="3">3 小时</option>
            </select>
          </div>
          <div className="setting-row">
            <div><strong>学习风格</strong><p>影响教程理论、代码与练习比例</p></div>
            <div className="segmented">
              <button>理论型</button>
              <button className="active">实操型</button>
              <button>混合型</button>
            </div>
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
  const localPreviewDemo = shouldUseLocalPreviewDemo();
  const [view, setView] = useState<View>('chat');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [workflow, setWorkflow] = useState<WorkflowState>(localPreviewDemo ? 'learning' : 'onboarding');
  const [sessionId, setSessionId] = useState<string | null>(localPreviewDemo ? DEMO_SESSION_ID : null);
  const [profile, setProfile] = useState<Profile | null>(localPreviewDemo ? fallbackProfile : null);
  const [stage, setStage] = useState<StageInfo | null>(localPreviewDemo ? fallbackStage : null);
  const [learningPackage, setLearningPackage] = useState<LearningPackage | null>(localPreviewDemo ? fallbackPackage : null);
  const [messages, setMessages] = useState<Message[]>(localPreviewDemo ? demoMessages : initialMessages);
  const [input, setInput] = useState('');
  const [onboardingForm, setOnboardingForm] = useState<OnboardingFormData>(localPreviewDemo ? formDataFromProfile(fallbackProfile) : emptyOnboardingForm);
  const [followupQuestions, setFollowupQuestions] = useState<string[]>([]);
  const [highlightedFields, setHighlightedFields] = useState<OnboardingField[]>([]);
  const [timerSeconds, setTimerSeconds] = useState(0);
  const [timerRunning, setTimerRunning] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [testOpen, setTestOpen] = useState(false);
  const [result, setResult] = useState<StageResult | null>(null);
  const [activeAgent, setActiveAgent] = useState<AgentId | null>(null);
  const [completedAgents, setCompletedAgents] = useState<AgentId[]>(localPreviewDemo ? ['agent1', 'agent3', 'agent4'] : []);
  const [learningProjects, setLearningProjects] = useState<LearningProject[]>(() => {
    const projects = readLearningProjects();
    if (!localPreviewDemo) return projects.filter((item) => item.sessionId !== DEMO_SESSION_ID);
    return [createDemoProject(), ...projects.filter((item) => item.sessionId !== DEMO_SESSION_ID)];
  });
  const [stageConversations, setStageConversations] = useState<StoredStageConversation[]>(() => {
    const conversations = readStageConversations();
    if (!localPreviewDemo) return conversations.filter((item) => item.sessionId !== DEMO_SESSION_ID);
    return [createDemoConversation(), ...conversations.filter((item) => item.sessionId !== DEMO_SESSION_ID)];
  });
  const [selectedConversationId, setSelectedConversationId] = useState(localPreviewDemo ? `${DEMO_SESSION_ID}:1` : '');
  const [projectSwitcherOpen, setProjectSwitcherOpen] = useState(false);
  const [materialPanelOpen, setMaterialPanelOpen] = useState(false);

  const completeAgent = (agentId: AgentId) => {
    setCompletedAgents((prev) => prev.includes(agentId) ? prev : [...prev, agentId]);
  };

  const rememberProject = (nextSessionId: string, nextProfile: Profile, nextStage?: StageInfo | null) => {
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
        stageName: nextStage?.block_name || existing?.stageName || '等待阶段规划',
        stageNum: nextStage?.stage_num ?? existing?.stageNum ?? 1,
        updatedAt: Date.now(),
        profile: nextProfile,
        stages: knownStages.sort((a, b) => (a.stage_num ?? 1) - (b.stage_num ?? 1))
      };
      return [project, ...prev.filter((item) => item.sessionId !== nextSessionId)].slice(0, 20);
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

  useEffect(() => {
    localStorage.setItem(PROJECTS_STORAGE_KEY, JSON.stringify(learningProjects));
  }, [learningProjects]);

  useEffect(() => {
    localStorage.setItem(CONVERSATIONS_STORAGE_KEY, JSON.stringify(stageConversations));
  }, [stageConversations]);

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
    setSessionId(savedSessionId);
    apiClient.getSession(savedSessionId)
      .then((session) => {
        if (session?.profile) {
          setProfile(session.profile);
          setOnboardingForm(formDataFromProfile(session.profile));
          rememberProject(savedSessionId, session.profile, session.stage_info);
          void loadDashboardStages(savedSessionId, session.stage_info);
          completeAgent('agent1');
          setWorkflow(session.learning_package ? 'learning' : 'profile_ready');
        } else {
          localStorage.removeItem('sessionId');
          setSessionId(null);
          setWorkflow('onboarding');
        }
        if (session?.stage_info) setStage(session.stage_info);
        if (session?.stage_info) setSelectedConversationId(`${savedSessionId}:${session.stage_info.stage_num ?? 1}`);
        if (session?.learning_package) {
          setLearningPackage(session.learning_package);
          completeAgent('agent3');
          completeAgent('agent4');
          setView('chat');
        }
      })
      .catch(() => {
        localStorage.removeItem('sessionId');
        setSessionId(null);
        setWorkflow('onboarding');
      });
  }, []);

  const currentQuestions = useMemo(() => {
    const questions = learningPackage?.questions?.length ? learningPackage.questions : fallbackPackage.questions || [];
    return questions;
  }, [learningPackage]);

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
    setActiveAgent('agent1');

    try {
      const response = await apiClient.chat({
        session_id: sessionId || undefined,
        message: buildDiagnosticMessage(onboardingForm)
      });
      setSessionId(response.session_id);
      localStorage.setItem('sessionId', response.session_id);

      if (response.response?.status === 'needs_more_info') {
        const questions = response.response.questions || [];
        setFollowupQuestions(questions);
        setHighlightedFields(inferHighlightedFields(questions));
        setWorkflow('needs_more_info');
        return;
      }

      if (response.response?.profile) {
        setProfile(response.response.profile);
        setOnboardingForm(formDataFromProfile(response.response.profile));
        rememberProject(response.session_id, response.response.profile, response.response.stage_info);
      }
      if (response.response?.stage_info) {
        setStage(response.response.stage_info);
        setSelectedConversationId(`${response.session_id}:${response.response.stage_info.stage_num ?? 1}`);
      }
      completeAgent('agent1');
      setWorkflow('profile_ready');
    } catch (caught) {
      setError(getErrorMessage(caught));
      setProfile(buildDemoProfile(onboardingForm));
      setStage({ ...fallbackStage, status: 'pending' });
      completeAgent('agent1');
      setWorkflow('profile_ready');
    } finally {
      setActiveAgent(null);
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
    setActiveAgent(initialDiagnosis ? 'agent1' : 'agent3');

    try {
      if (initialDiagnosis) {
        const response = await apiClient.chat({ session_id: sessionId || undefined, message: text });
        setSessionId(response.session_id);
        localStorage.setItem('sessionId', response.session_id);
        if (response.response?.status === 'needs_more_info') {
          setMessages((prev) => [...prev, {
            id: shortId(),
            role: 'assistant',
            content: `我还需要补充一点信息：\n${(response.response.questions || []).join('\n')}`,
            note: '需要追问'
          }]);
        } else {
          if (response.response?.profile) setProfile(response.response.profile);
          if (response.response?.stage_info) setStage(response.response.stage_info);
          completeAgent('agent1');
          setMessages((prev) => [...prev, {
            id: shortId(),
            role: 'assistant',
            content: '画像已经建立，并生成了当前阶段。你可以点击「开始学习阶段」进入学习包。',
            note: '画像完成'
          }]);
        }
      } else {
        if (!sessionId) throw new Error('尚未创建后端会话');
        const response = await apiClient.handleQuestion({ session_id: sessionId, message: text });
        completeAgent('agent3');
        completeAgent('agent4');
        setMessages((prev) => [...prev, {
          id: shortId(),
          role: 'assistant',
          content: response.response?.answer || '已收到问题，但后端没有返回回答内容。',
          note: response.response?.tech_pass && response.response?.quality_pass ? '审核通过' : '待复核'
        }]);
      }
    } catch (caught) {
      setError(getErrorMessage(caught));
      setProfile(fallbackProfile);
      setStage(fallbackStage);
      if (initialDiagnosis) {
        completeAgent('agent1');
      } else {
        completeAgent('agent3');
        completeAgent('agent4');
      }
      setMessages((prev) => [...prev, {
        id: shortId(),
        role: 'assistant',
        content: '后端当前没有响应，我先用本地演示数据生成一版学习工作台。等后端服务启动后，这里会自动接入真实画像、阶段和学习包。',
        note: '本地演示'
      }]);
    } finally {
      setActiveAgent(null);
      setLoading(false);
    }
  };

  const startStage = async () => {
    setError(null);
    setLoading(true);
    setWorkflow('starting_stage');
    setActiveAgent('agent3');
    try {
      if (!sessionId) throw new Error('尚未创建后端会话');
      const response = await apiClient.startStage({ session_id: sessionId });
      setLearningPackage(response.learning_package);
      const generatedStage = response.state?.stage_info || stage || fallbackStage;
      if (response.state?.stage_info) {
        setStage(response.state.stage_info);
        setSelectedConversationId(`${sessionId}:${response.state.stage_info.stage_num ?? 1}`);
        if (profile) rememberProject(sessionId, profile, response.state.stage_info);
      }
      completeAgent('agent3');
      completeAgent('agent4');
      setMessages((prev) => [...prev, {
        id: shortId(),
        role: 'assistant',
        content: `学习包已生成，审核轮次：${response.iterations ?? 1}，审核结果：${response.review_passed ? '通过' : '待复核'}。`,
        note: '学习包'
      }]);
      setWorkflow('learning');
      setView('chat');
      setMaterialPanelOpen(true);
    } catch (caught) {
      setError(getErrorMessage(caught));
      setLearningPackage(fallbackPackage);
      setStage({ ...(stage || fallbackStage), status: 'learning' });
      completeAgent('agent3');
      completeAgent('agent4');
      setWorkflow('learning');
      setView('chat');
      setMaterialPanelOpen(true);
    } finally {
      setActiveAgent(null);
      setLoading(false);
    }
  };

  const completeStage = async (answers: Record<string, string>) => {
    setTestOpen(false);
    setLoading(true);
    setError(null);
    setActiveAgent('agent6');
    try {
      if (sessionId) {
        for (const [question_id, user_answer] of Object.entries(answers)) {
          await apiClient.submitAnswer({ session_id: sessionId, question_id, user_answer });
        }
        const response = await apiClient.completeStage(sessionId);
        setResult(response);
        if (response.updated_profile) setProfile(response.updated_profile);
        if (response.next_stage) {
          setStage(response.next_stage);
          setSelectedConversationId(`${sessionId}:${response.next_stage.stage_num ?? 1}`);
          setMessages(initialMessages);
        }
        rememberProject(sessionId, response.updated_profile || profile || fallbackProfile, response.next_stage || stage);
        void loadDashboardStages(sessionId, response.next_stage || stage);
      } else {
        throw new Error('尚未创建后端会话');
      }
    } catch (caught) {
      setError(getErrorMessage(caught));
      setResult({ status: 'continue', accuracy: 0.87, next_strategy: 'advance', stage_summary: '本地演示结果：正确率较高，建议继续推进。' });
      setProfile({
        ...(profile || fallbackProfile),
        knowledge_gaps: fallbackProfile.knowledge_gaps?.map((gap, index) => index === 0 ? { ...gap, status: '已掌握', progress: 0.86 } : gap)
      });
    } finally {
      completeAgent('agent6');
      setActiveAgent(null);
      setLoading(false);
      setView('result');
    }
  };

  const clearLocalSession = () => {
    localStorage.removeItem('sessionId');
    setError(null);
    setSessionId(null);
    setProfile(null);
    setStage(null);
    setLearningPackage(null);
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
    setActiveAgent(null);
    setCompletedAgents([]);
    setWorkflow('onboarding');
    setView('chat');
  };

  const createLearningGoal = () => {
    setProjectSwitcherOpen(false);
    clearLocalSession();
  };

  const switchLearningProject = async (nextSessionId: string, requestedStageNum?: number) => {
    if (nextSessionId === sessionId || loading) {
      if (nextSessionId === sessionId && requestedStageNum) {
        setSelectedConversationId(`${nextSessionId}:${requestedStageNum}`);
      }
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const session = await apiClient.getSession(nextSessionId);
      if (!session?.profile) throw new Error('该学习项目尚未生成有效画像');
      setSessionId(nextSessionId);
      localStorage.setItem('sessionId', nextSessionId);
      setProfile(session.profile);
      setOnboardingForm(formDataFromProfile(session.profile));
      setStage(session.stage_info || null);
      setLearningPackage(session.learning_package || null);
      setResult(null);
      const currentStageNum = session.stage_info?.stage_num ?? 1;
      const restoredConversation = stageConversations.find((item) => item.id === `${nextSessionId}:${currentStageNum}`);
      setMessages(restoredConversation?.messages || initialMessages);
      setSelectedConversationId(`${nextSessionId}:${requestedStageNum ?? currentStageNum}`);
      setCompletedAgents(session.learning_package ? ['agent1', 'agent3', 'agent4'] : ['agent1']);
      setWorkflow(session.learning_package ? 'learning' : 'profile_ready');
      setView('chat');
      rememberProject(nextSessionId, session.profile, session.stage_info);
      void loadDashboardStages(nextSessionId, session.stage_info);
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
    setError(null);
    setFollowupQuestions([]);
    setHighlightedFields([]);
    setWorkflow('onboarding');
    setView('chat');
  };

  const workspaceProfile = profile || fallbackProfile;
  const workspaceStage = stage || fallbackStage;
  const currentConversationId = sessionId && stage ? `${sessionId}:${stage.stage_num ?? 1}` : '';
  const selectedConversation = stageConversations.find((item) => item.id === selectedConversationId);
  const conversationStage = selectedConversation?.stage || workspaceStage;
  const displayedMessages = selectedConversationId && selectedConversationId !== currentConversationId
    ? selectedConversation?.messages || []
    : messages;
  const historicalConversationSelected = Boolean(selectedConversationId && selectedConversationId !== currentConversationId);

  return (
    <div className="app">
      <Sidebar
        view={view}
        setView={setView}
        activeAgent={activeAgent}
        completedAgents={completedAgents}
        collapsed={sidebarCollapsed}
        onToggleCollapsed={() => setSidebarCollapsed((prev) => !prev)}
      />
      {view !== 'learn' && (!profile || workflow === 'onboarding' || workflow === 'diagnosing' || workflow === 'needs_more_info') && (
        <OnboardingPage
          formData={onboardingForm}
          onChange={setOnboardingForm}
          onSubmit={submitOnboarding}
          loading={loading}
          mode={workflow === 'diagnosing' ? 'diagnosing' : workflow === 'needs_more_info' ? 'needs_more_info' : 'onboarding'}
          questions={followupQuestions}
          error={error}
          highlightedFields={highlightedFields}
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
          learningPackage={learningPackage}
          materialPanelOpen={materialPanelOpen}
          onToggleMaterials={() => setMaterialPanelOpen((prev) => !prev)}
          onTest={() => setTestOpen(true)}
          projects={learningProjects}
          currentSessionId={sessionId}
          selectedConversationId={selectedConversationId || currentConversationId}
          onSelectProjectStage={selectProjectStage}
          readOnly={historicalConversationSelected}
          loading={loading}
          error={error}
        />
      )}
      {view === 'learn' && (
        <AllProfilesView
          projects={learningProjects}
          loading={loading}
          error={error}
        />
      )}
      {profile && workflow === 'learning' && view === 'dashboard' && (
        <DashboardView profile={workspaceProfile} stage={workspaceStage} result={result} setView={setView} loading={loading} error={error} />
      )}
      {profile && workflow === 'learning' && view === 'settings' && <SettingsView profile={workspaceProfile} loading={loading} error={error} onClear={clearLocalSession} />}
      {profile && workflow === 'learning' && view === 'result' && <ResultView profile={workspaceProfile} stage={workspaceStage} result={result} setView={setView} loading={loading} error={error} />}
      {testOpen && <TestModal questions={currentQuestions} onClose={() => setTestOpen(false)} onComplete={completeStage} />}
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
