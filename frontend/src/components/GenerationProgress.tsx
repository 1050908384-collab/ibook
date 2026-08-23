import React, { createContext, useCallback, useContext, useMemo, useState } from 'react';
import { Check, Loader2, Sparkles, X } from 'lucide-react';
import type { ApiStage, GenerationEvent } from '../api/client';
import { apiClient } from '../api/client';

// ============== Types ==============
export type GenPhase = 'idle' | 'stage' | 'package' | 'success' | 'error';

export type GenStagePreview = {
  stage_num?: number;
  block_name?: string;
  subtopic_ids?: string[];
  subtopic_names?: string[];
  subtopic_name?: string;
  estimated_hours?: number;
  prerequisites?: string[];
  gap_skill?: string;
};

type ReviewProgress = {
  iterations?: number;
  max_iterations?: number;
  status?: string;
  needs_human_review?: boolean;
  timeline?: Array<{
    round: number;
    technical_passed: boolean;
    quality_passed: boolean;
    issue_count: number;
  }>;
};

type GenState = {
  open: boolean;
  phase: GenPhase;
  /** 动作来源：首次画像确认后启动 / 下一阶段生成 */
  action: 'first' | 'next' | null;
  /** 阶段①完成后立刻赋值，后续 package 期间用户能看到知识点 */
  stage: GenStagePreview | null;
  /** 当前文案（"正在生成阶段中..." / "正在生成学习包中..."） */
  message: string;
  errorMsg: string | null;
  /** 进度条 0-100，细动画反馈 */
  progress: number;
  review: ReviewProgress | null;
  events: GenerationEvent[];
};

type RunParams = {
  sessionId: string;
  action: 'first' | 'next';
  /**
   * 仅 action="first" 时使用：若 confirmProfile 返回的 stage_info 已经存在，
   * 阶段①视为已经完成（画像确认已经帮用户规划了首阶段）。
   * 若外部在此之前已经拿到了 stage，可以直接传入跳过调用。
   */
  initialStage?: ApiStage | GenStagePreview | null;
  restartCurrent?: boolean;
  onStageReady?: (stage: GenStagePreview) => void | Promise<void>;
  onPackageReady?: (payload: { packageResponse: any; stageResponse: any; startResponse: any }) => void | Promise<void>;
  onError?: (error: unknown) => void;
  onFinally?: () => void;
  /** 下一步按钮点击后的 callback：关闭前允许外部刷新 dashboard / 切视图 */
  onSuccessClose?: () => void;
};

type GenProgressContextValue = {
  state: GenState;
  /** 触发两阶段生成流程（首次 or 下一阶段） */
  run: (p: RunParams) => Promise<void>;
  /** 关闭弹窗（仅在 success/error 时可关） */
  close: () => void;
  /** 外部控制：比如被取消/被其它错误打断 */
  failExternal: (message: string) => void;
};

// ============== Helpers ==============
function normalizeStage(src: ApiStage | GenStagePreview | null | undefined): GenStagePreview | null {
  if (!src) return null;
  const s = src as Record<string, unknown>;
  return {
    stage_num: typeof s.stage_num === 'number' ? s.stage_num : undefined,
    block_name: (s.block_name as string) || undefined,
    subtopic_ids: Array.isArray(s.subtopic_ids) ? (s.subtopic_ids as string[]) : undefined,
    subtopic_names: Array.isArray(s.subtopic_names) ? (s.subtopic_names as string[]) : undefined,
    subtopic_name: (s.subtopic_name as string) || undefined,
    estimated_hours: typeof s.estimated_hours === 'number' ? s.estimated_hours : undefined,
    prerequisites: Array.isArray(s.prerequisites) ? (s.prerequisites as string[]) : undefined,
    gap_skill: (s.gap_skill as string) || undefined,
  };
}

function describeGenerationEvent(event: GenerationEvent) {
  const labels: Record<string, { actor: string; text: string }> = {
    'start_stage:apply_planned_next_stage': { actor: '系统', text: '已切换到已规划的下一阶段' },
    'start_stage:start': { actor: '系统', text: '开始创建本阶段学习包' },
    'pipeline:phaseA_start': { actor: '协同调度', text: '个性化学习资源生成 Agent 与 专业知识与教学质量审核 Agent 开始并行处理知识块' },
    'pipeline:phaseA_done': { actor: '协同调度', text: '首轮生成与审核已汇总' },
    'pipeline:issue_package': { actor: '协同调度', text: '审核问题包已交给 个性化学习资源生成 Agent' },
    'pipeline:phaseC_round': { actor: '协同调度', text: '本轮修订与复核已完成' },
    'pipeline:converge_freeze': { actor: '协同调度', text: '已完成当前轮次审核汇总' },
    'pipeline:done': { actor: '系统', text: '学习包已完成生成与审核' },
    'agent3:block_generate_start': { actor: '个性化学习资源生成 Agent', text: '开始生成个性化内容块' },
    'agent3:block_generate_done': { actor: '个性化学习资源生成 Agent', text: '已完成内容块生成，提交 专业知识与教学质量审核 Agent 审核' },
    'agent4:first_review_start': { actor: '专业知识与教学质量审核 Agent', text: '开始技术正确性与教学质量审核' },
    'agent4:first_review_done': { actor: '专业知识与教学质量审核 Agent', text: '已完成首轮审核' },
    'agent4:first_review_failed': { actor: '专业知识与教学质量审核 Agent', text: '本块审核异常，已记录为待处理问题' },
    'agent3:review_reply_start': { actor: '个性化学习资源生成 Agent', text: '正在回应审核意见' },
    'agent3:review_reply_done': { actor: '个性化学习资源生成 Agent', text: '已提交审核意见回应' },
    'agent4:verdict_start': { actor: '专业知识与教学质量审核 Agent', text: '正在裁决 个性化学习资源生成 Agent 的回应' },
    'agent4:verdict_done': { actor: '专业知识与教学质量审核 Agent', text: '已完成审核裁决' },
    'agent3:revision_start': { actor: '个性化学习资源生成 Agent', text: '正在按裁决修订内容' },
    'agent3:revision_done': { actor: '个性化学习资源生成 Agent', text: '已完成修订，提交复核' },
    'agent4:recheck_start': { actor: '专业知识与教学质量审核 Agent', text: '正在复核修订结果' },
    'agent4:recheck_done': { actor: '专业知识与教学质量审核 Agent', text: '已完成修订复核' },
  };
  const mapped = labels[event.event];
  // Runtime logs may contain internal counts (issues/replies/decisions). Keep
  // those in the backend trace for auditability, but do not expose them in
  // the learner-facing modal or competition recording.
  const publicDetail = (event.detail || '')
    .replace(/(?:^|[, ]+)issues=\d+/gi, '')
    .replace(/(?:^|[, ]+)replies=\d+/gi, '')
    .replace(/(?:^|[, ]+)decisions=\d+/gi, '')
    .replace(/\s{2,}/g, ' ')
    .replace(/^[, ]+|[, ]+$/g, '')
    .trim();
  const detail = publicDetail ? ` (${publicDetail})` : '';
  return mapped
    ? { ...mapped, detail }
    : { actor: '协同调度', text: event.event, detail };
}

// ============== Context ==============
const GenProgressContext = createContext<GenProgressContextValue | null>(null);

export function GenerationProgressProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<GenState>({
    open: false,
    phase: 'idle',
    action: null,
    stage: null,
    message: '',
    errorMsg: null,
    progress: 0,
  review: null,
  events: [],
});

  const close = useCallback(() => {
    setState((s) => {
      // 只允许 success / error / idle 时关
      if (s.phase === 'stage' || s.phase === 'package') return s;
      return { ...s, open: false };
    });
  }, []);

  const failExternal = useCallback((message: string) => {
    setState((s) => ({
      ...s,
      phase: 'error',
      errorMsg: message || '生成失败',
      message: message || '生成失败，请重试',
      progress: s.progress,
    }));
  }, []);

  const run = useCallback(async (p: RunParams) => {
    const {
      sessionId, action, initialStage, restartCurrent, onStageReady, onPackageReady, onError, onFinally, onSuccessClose,
    } = p;

    // 打开弹窗，进入阶段①
    setState({
      open: true,
      phase: 'stage',
      action,
      stage: normalizeStage(initialStage) || null,
      message: action === 'first' ? '正在规划学习阶段，请稍等…' : '正在生成下一阶段，请稍等…',
      errorMsg: null,
      progress: 15,
      review: null,
      events: [],
    });

    let stagePreview: GenStagePreview | null = normalizeStage(initialStage) || null;

    try {
      // ========= 阶段 ①：生成阶段结构 =========
      if (action === 'first') {
        // 首次：必须调用 confirmProfile 设置 profile_confirmed=True，
        // 否则后端 start_stage 会返回 409（submit_foundation 不设此字段）。
        // 即使 initialStage 已有（来自 submitFoundation），也要调用；
        // confirmProfile 检测到 stage_info 已存在时不会重新生成。
        const confirmation = await apiClient.confirmProfile(sessionId);
        const confirmedStage = normalizeStage(confirmation?.stage_info);
        if (confirmedStage) stagePreview = confirmedStage;
        if (!stagePreview) {
          throw new Error('画像确认后未返回首阶段规划，请重试');
        }
      } else if (!stagePreview) {
        // 下一阶段：planNextStage
        const response = await apiClient.planNextStage(sessionId);
        stagePreview = normalizeStage(response?.next_stage);
        // 兼容：有的后端会异步落库，没拿到就轮询 getSession
        if (!stagePreview) {
          const maxRetries = 12;
          for (let i = 0; i < maxRetries; i += 1) {
            // eslint-disable-next-line no-await-in-loop
            await new Promise((r) => window.setTimeout(r, 900));
            // eslint-disable-next-line no-await-in-loop
            const session = await apiClient.getSession(sessionId);
            stagePreview = normalizeStage(session?.next_stage);
            if (stagePreview) break;
          }
        }
        if (!stagePreview) {
          throw new Error('下一阶段仍在生成中，请稍后再试');
        }
      }

      // 阶段①完成：写入 stage，切换到 package —— 用户现在能看到知识点了
      setState((s) => ({
        ...s,
        phase: 'package',
        stage: stagePreview!,
        message: '正在生成学习包中，请稍等…',
        progress: 52,
      }));
      if (onStageReady) await Promise.resolve(onStageReady(stagePreview!));

      // ========= 阶段 ②：生成学习包 =========
      let eventCursor = 0;
      let stopPolling = false;
      const appendEvents = (events: GenerationEvent[]) => {
        if (!events.length) return;
        setState((current) => {
          const known = new Set(current.events.map((item) => item.id));
          const merged = [...current.events, ...events.filter((item) => !known.has(item.id))].slice(-30);
          return { ...current, events: merged };
        });
      };
      const pollEvents = async () => {
        while (!stopPolling) {
          try {
            const eventResponse = await apiClient.getGenerationEvents(sessionId, eventCursor);
            eventCursor = Math.max(eventCursor, eventResponse.latest_id || 0);
            appendEvents(eventResponse.events || []);
          } catch {
            // The generation request remains authoritative; a transient polling
            // failure must never interrupt learning-package generation.
          }
          if (!stopPolling) await new Promise((resolve) => window.setTimeout(resolve, 700));
        }
      };

      const startRequest = apiClient.startStage(sessionId, { restartCurrent });
      const pollingTask = pollEvents();
      let startResponse: any;
      try {
        startResponse = await startRequest;
      } finally {
        stopPolling = true;
        await pollingTask;
      }
      // Fetch any final events emitted between the last polling cycle and the
      // completed start response, then preserve them in the success modal.
      try {
        const eventResponse = await apiClient.getGenerationEvents(sessionId, eventCursor);
        appendEvents(eventResponse.events || []);
      } catch {
        // Completion already succeeded; the live log is supplementary evidence.
      }
      const [packageResponse, stageResponse] = await Promise.all([
        apiClient.getStagePackage(sessionId),
        apiClient.getCurrentStage(sessionId),
      ]);

      setState((s) => ({
        ...s,
        phase: 'success',
        message: '生成完成，即将进入学习…',
        progress: 100,
        stage: normalizeStage(stageResponse?.stage) || s.stage,
        review: startResponse?.review_summary
          ? { ...startResponse.review_summary, timeline: startResponse.review_timeline || [] }
          : null,
      }));

      if (onPackageReady) {
        await Promise.resolve(onPackageReady({ packageResponse, stageResponse, startResponse }));
      }

      // 审核时间线是本块多 Agent 协同的可见证据，成功后由用户主动进入学习。
      // 保留回调字段以兼容现有调用方，但不再自动关闭弹窗。
      void onSuccessClose;
    } catch (caught) {
      const msg =
        (caught instanceof Error ? caught.message : '')
        || (typeof (caught as any)?.response?.data?.detail === 'string' ? (caught as any).response.data.detail : '')
        || (action === 'first' ? '生成学习阶段失败' : '生成下一阶段失败');
      setState((s) => ({
        ...s,
        phase: 'error',
        errorMsg: msg,
        message: action === 'first'
          ? '生成学习阶段失败，请重试'
          : (s.phase === 'package' ? '学习包生成失败，请重试' : '阶段规划失败，请重试'),
        progress: s.progress,
      }));
      if (onError) onError(caught);
    } finally {
      if (onFinally) onFinally();
    }
  }, []);

  const value = useMemo<GenProgressContextValue>(
    () => ({ state, run, close, failExternal }),
    [state, run, close, failExternal],
  );

  return (
    <GenProgressContext.Provider value={value}>
      {children}
      <GenerationProgressModal
        state={state}
        onClose={close}
        onRetry={() => {
          // 当前这个 Retry 是给 error 场景提供一个"手动重跑"入口：
          // 简单做法——直接关闭由用户在界面上再点按钮。更复杂的 retry 会保留上一次参数，
          // 但 action 参数是通过 run 传进来的，这里没法拿到。因此 error 场景推荐用"关闭+重试"路径。
          // 这里只把错误提示清掉，交给用户再点一次按钮，避免误重复。
          close();
        }}
      />
    </GenProgressContext.Provider>
  );
}

export function useGenerationProgress() {
  const ctx = useContext(GenProgressContext);
  if (!ctx) throw new Error('useGenerationProgress 必须在 GenerationProgressProvider 内使用');
  return ctx;
}

// ============== Modal ==============
type StepStatus = 'done' | 'active' | 'pending' | 'error';

function stepStatusFromPhase(phase: GenPhase): { stage: StepStatus; pkg: StepStatus } {
  switch (phase) {
    case 'idle': return { stage: 'pending', pkg: 'pending' };
    case 'stage': return { stage: 'active', pkg: 'pending' };
    case 'package': return { stage: 'done', pkg: 'active' };
    case 'success': return { stage: 'done', pkg: 'done' };
    case 'error': return { stage: 'done', pkg: 'error' }; // 失败通常出在 package 或 stage 前半段，展示为主路径
    default: return { stage: 'pending', pkg: 'pending' };
  }
}

function GenerationProgressModal({
  state,
  onClose,
  onRetry,
}: {
  state: GenState;
  onClose: () => void;
  onRetry: () => void;
}) {
  const status = stepStatusFromPhase(state.phase);
  const closable = state.phase === 'success' || state.phase === 'error' || state.phase === 'idle';
  if (!state.open) return null;

  const subtopicNames = (state.stage?.subtopic_names && state.stage.subtopic_names.length > 0)
    ? state.stage.subtopic_names
    : (state.stage?.subtopic_name ? [state.stage.subtopic_name] : []);

  return (
    <div
      className="modal-overlay generation-progress-overlay"
      onClick={closable ? onClose : undefined}
      role="dialog"
      aria-modal="true"
      aria-labelledby="gen-progress-title"
    >
      <div
        className="modal-card generation-progress-card"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="gen-progress-header">
          <div id="gen-progress-title" className="gen-progress-title-row">
            <Sparkles size={16} className="gen-progress-title-icon" />
            <strong>{state.action === 'first' ? '生成学习阶段' : '生成下一阶段'}</strong>
          </div>
          <button
            type="button"
            className="gen-progress-close"
            onClick={onClose}
            disabled={!closable}
            aria-label="关闭"
          >
            <X size={16} />
          </button>
        </header>

        <div className="gen-progress-scroll">
        {/* ===== 两步指示器 ===== */}
        <ol className="gen-progress-stepper">
          <li className={`gen-step is-${status.stage}`}>
            <span className="gen-step-circle">
              {status.stage === 'done' ? <Check size={14} /> : status.stage === 'active' ? <Loader2 size={14} className="spin" /> : <span>1</span>}
            </span>
            <span className="gen-step-label">
              <strong>生成阶段</strong>
              <em>规划知识点与阶段结构</em>
            </span>
          </li>
          <li className={`gen-step-line is-${status.stage === 'pending' ? 'pending' : 'done'}`} />
          <li className={`gen-step is-${status.pkg}`}>
            <span className="gen-step-circle">
              {status.pkg === 'done' ? <Check size={14} /> : status.pkg === 'active' ? <Loader2 size={14} className="spin" /> : <span>2</span>}
            </span>
            <span className="gen-step-label">
              <strong>生成学习包</strong>
              <em>{state.review?.iterations
                ? `专业知识与教学质量审核 Agent 实际完成第 ${state.review.iterations} 轮审核`
                : '教程、示例与资源整理'}</em>
            </span>
          </li>
        </ol>

        {/* ===== 当前阶段信息：阶段①结束后才显示 ===== */}
        {state.stage && (
          <section
            className={`gen-stage-preview ${state.phase === 'package' || state.phase === 'success' || state.phase === 'error' ? 'is-visible' : ''}`}
            aria-live="polite"
          >
            <div className="gen-stage-preview-head">
              <div>
                <span className="gen-stage-kicker">阶段 {state.stage.stage_num ?? '—'}</span>
                <h3>{state.stage.block_name || state.stage.gap_skill || '阶段预览'}</h3>
                {state.stage.estimated_hours && (
                  <span className="gen-stage-hours">预计 {state.stage.estimated_hours} 小时/阶段</span>
                )}
              </div>
            </div>

            {subtopicNames.length > 0 ? (
              <ul className="gen-stage-subtopics">
                {subtopicNames.map((name, i) => (
                  <li key={`${name}-${i}`} style={{ animationDelay: `${50 + i * 40}ms` }}>
                    <span className="gen-subtopic-dot" /> {name}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="gen-stage-subtopics-hint">阶段结构已就绪，正在整理具体知识点…</p>
            )}
          </section>
        )}

        {/* ===== 状态文案 + 进度条 ===== */}
        <div className="gen-progress-status-row">
          <p className="gen-progress-message">
            {state.phase === 'error' ? state.message : state.message}
          </p>
          <span className="gen-progress-percent">{Math.min(100, Math.max(0, Math.round(state.progress)))}%</span>
        </div>
        <div className="gen-progress-bar" aria-hidden>
          <div
            className="gen-progress-bar-fill"
            style={{ width: `${Math.min(100, Math.max(0, state.progress))}%` }}
          />
        </div>

        {state.events.length > 0 && (
          <section className="gen-live-events" aria-label="多智能体实时协同日志" aria-live="polite">
            <div className="gen-live-events-title">
              <strong>多智能体实时协同</strong>
              <span>仅保留最近 {state.events.length} 条</span>
            </div>
            <div className="gen-live-events-list">
              {state.events.map((event) => {
                const described = describeGenerationEvent(event);
                const actorClass = described.actor === '个性化学习资源生成 Agent'
                  ? 'agent3'
                  : described.actor === '专业知识与教学质量审核 Agent'
                    ? 'agent4'
                    : 'system';
                return (
                  <div className={`gen-live-event ${actorClass}`} key={event.id}>
                    <span>{described.actor}</span>
                    <p>{described.text}<small>{described.detail}</small></p>
                  </div>
                );
              })}
            </div>
          </section>
        )}

        {state.phase === 'error' && state.errorMsg && (
          <div className="gen-progress-error" role="alert">
            {state.errorMsg}
          </div>
        )}

        {state.phase === 'success' && state.review?.iterations && (
          <section className="gen-review-timeline" aria-label="本块学习包的协同审核过程">
            <div className="gen-review-timeline-title">
              <strong>个性化学习资源生成 Agent-专业知识与教学质量审核 Agent 协同审核</strong>
              <span>本块学习包</span>
            </div>
            <div className="gen-review-timeline-events">
              <div className="gen-review-event agent3">
                <span>个性化学习资源生成 Agent</span>
                <p>已生成个性化教程、代码示例与测试题。</p>
              </div>
              {(state.review.timeline || []).map((round) => (
                <div className="gen-review-round" key={round.round}>
                  <div className="gen-review-event agent4">
                    <span>专业知识与教学质量审核 Agent 第 {round.round} 轮审核</span>
                    <p>已完成技术正确性与教学质量审核{round.issue_count ? '，发现优化项并进入下一轮复审。' : '，本轮通过。'}</p>
                  </div>
                  {round.issue_count > 0 && (
                    <div className="gen-review-event agent3 revision">
                      <span>个性化学习资源生成 Agent 修订</span>
                      <p>已根据审核意见修订学习包，提交下一轮复审。</p>
                    </div>
                  )}
                </div>
              ))}
            </div>
            <p className="gen-progress-review-result">
              已完成 {state.review.iterations} 轮技术与教学协同审核，学习包已进入可用状态。
            </p>
          </section>
        )}
        </div>

        {/* ===== 底部按钮 ===== */}
        <footer className="gen-progress-footer">
          {state.phase === 'success' && (
            <button type="button" className="primary wide" onClick={onClose}>
              <Check size={16} /> 进入学习
            </button>
          )}
          {state.phase === 'error' && (
            <button type="button" className="primary wide" onClick={onRetry}>
              关闭后重试
            </button>
          )}
          {(state.phase === 'stage' || state.phase === 'package') && (
            <button type="button" className="ghost wide" disabled>
              <Loader2 size={15} className="spin" /> 请勿关闭，正在生成中…
            </button>
          )}
        </footer>
      </div>
    </div>
  );
}
