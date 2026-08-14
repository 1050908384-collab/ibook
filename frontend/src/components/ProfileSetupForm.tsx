import { FormEvent, useMemo } from 'react';
import { AlertCircle, Loader2 } from 'lucide-react';
import {
  getFormCompleteness,
  isOnboardingFormComplete,
  isValidDailyHours,
  learningStyleText,
  OnboardingField,
  OnboardingFormData
} from '../utils/buildDiagnosticMessage';

type ProfileSetupFormProps = {
  value: OnboardingFormData;
  onChange: (value: OnboardingFormData) => void;
  onSubmit: () => void;
  loading: boolean;
  mode: 'onboarding' | 'needs_more_info';
  questions: string[];
  error: string | null;
  highlightedFields: OnboardingField[];
  targetOnly?: boolean;
  profileOnly?: boolean;
  foundationNames: string[];
  foundationConfirmed: boolean;
  onOpenFoundation: () => void;
  foundationLoading?: boolean;
};

const fieldLabels: Record<OnboardingField, string> = {
  grade: '身份/学习背景',
  skill_tags: '已有知识基础',
  goal: '学习目标',
  daily_hours: '每日学习时长',
  learning_style: '学习风格',
  hardware: '设备条件'
};

const styleOptions = [
  { value: 'theory', title: '理论型', desc: '更偏原理、概念和推导' },
  { value: 'practice', title: '实操型', desc: '更偏代码、实验和部署' },
  { value: 'mixed', title: '混合型', desc: '理论和实践均衡推进' }
] as const;

export default function ProfileSetupForm({
  value,
  onChange,
  onSubmit,
  loading,
  mode,
  questions,
  error,
  highlightedFields,
  targetOnly = false,
  profileOnly = false,
  foundationNames,
  foundationConfirmed,
  onOpenFoundation,
  foundationLoading = false
}: ProfileSetupFormProps) {
  const completeness = useMemo(() => getFormCompleteness(value, targetOnly, foundationConfirmed, profileOnly), [value, targetOnly, foundationConfirmed, profileOnly]);
  const canSubmit = isOnboardingFormComplete(value, targetOnly, foundationConfirmed, profileOnly) && !loading;

  const update = <K extends OnboardingField>(field: K, fieldValue: OnboardingFormData[K]) => {
    onChange({ ...value, [field]: fieldValue });
  };

  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (canSubmit) onSubmit();
  };

  const isHighlighted = (field: OnboardingField) => highlightedFields.includes(field);

  return (
    <form className="profile-setup-form" onSubmit={submit}>
      <div className="form-status-row">
        <div>
          <span>信息完整度</span>
          <strong>{Math.round(completeness * 100)}%</strong>
        </div>
        <div className="completeness-track" aria-label="信息完整度">
          <i style={{ width: `${Math.round(completeness * 100)}%` }} />
        </div>
      </div>

      {mode === 'needs_more_info' && (
        <div className="form-alert">
          <AlertCircle size={18} />
          <div>
            <strong>还需要补充部分信息</strong>
            <p>{questions.length ? questions.join(' ') : '请检查高亮字段后重新生成画像。'}</p>
          </div>
        </div>
      )}

      {error && (
        <div className="form-alert error">
          <AlertCircle size={18} />
          <div>
            <strong>接口请求失败</strong>
            <p>{error}</p>
          </div>
        </div>
      )}

      {profileOnly && <section className={`form-section ${isHighlighted('grade') ? 'needs-attention' : ''}`}>
        <label htmlFor="grade">{fieldLabels.grade} *</label>
        <p>告诉 Agent 你现在的身份或学习背景，用来判断适合的起点。</p>
        <input
          id="grade"
          value={value.grade}
          onChange={(event) => update('grade', event.target.value)}
          placeholder="例如：研二、算法工程师、后端开发者"
        />
      </section>}

      {!profileOnly && <section className={`form-section ${isHighlighted('goal') ? 'needs-attention' : ''}`}>
        <label htmlFor="goal">{fieldLabels.goal} *</label>
        <p>描述你希望达成的真实结果，Agent 会据此推断知识盲区和第一阶段。</p>
        <textarea
          id="goal"
          value={value.goal}
          onChange={(event) => update('goal', event.target.value)}
          rows={4}
          placeholder="例如：掌握 LoRA 微调并能独立部署"
        />
      </section>}

      {profileOnly && <section className={`form-section foundation-summary-section ${isHighlighted('skill_tags') ? 'needs-attention' : ''}`}>
        <div className="foundation-summary-heading">
          <div>
            <label>确定基础</label>
            <p>选择你已经完整掌握的知识板块，系统会据此规划更合适的起点。</p>
          </div>
          <button type="button" className="foundation-trigger" onClick={onOpenFoundation} disabled={foundationLoading}>
            {foundationLoading ? <><Loader2 className="spin" size={14} />读取知识目录</> : foundationConfirmed ? '重新确定' : '明确我的基础'}
          </button>
        </div>
        <div className="foundation-summary-content">
          {foundationNames.length > 0 ? (
            <div className={`foundation-summary-list ${foundationNames.length > 3 ? 'has-overflow' : ''}`} title={foundationNames.join('、')}>
              {foundationNames.map((name) => <span key={name}>{name}</span>)}
            </div>
          ) : <span className={foundationConfirmed ? "foundation-summary-list" : "foundation-summary-empty"}>{foundationConfirmed ? "已确认：暂无完整掌握的板块" : "尚未填写，请明确你的基础"}</span>}
        </div>
      </section>}

      {!profileOnly && <div className="form-two-col">
        <section className={`form-section ${isHighlighted('daily_hours') ? 'needs-attention' : ''}`}>
          <label htmlFor="daily-hours">{fieldLabels.daily_hours} *</label>
          <p>合法范围为 0.5 到 8 小时。</p>
          <input
            id="daily-hours"
            type="number"
            min="0.5"
            max="8"
            step="0.5"
            value={value.daily_hours}
            onChange={(event) => update('daily_hours', event.target.value)}
            placeholder="例如：2"
          />
          {value.daily_hours && !isValidDailyHours(value.daily_hours) && <small className="field-error">请输入 0.5 到 8 之间的数值。</small>}
        </section>

      </div>}

      {profileOnly && <section className={`form-section ${isHighlighted('hardware') ? 'needs-attention' : ''}`}>
        <label htmlFor="hardware">{fieldLabels.hardware} *</label>
        <p>作为长期画像保存，用于适配部署和实验建议。</p>
        <input
          id="hardware"
          value={value.hardware}
          onChange={(event) => update('hardware', event.target.value)}
          placeholder="例如：RTX 4060、MacBook、无独立显卡"
        />
      </section>}

      {!profileOnly && <section className={`form-section ${isHighlighted('learning_style') ? 'needs-attention' : ''}`}>
        <label>{fieldLabels.learning_style} *</label>
        <p>选择你更舒服的学习推进方式。</p>
        <div className="style-segments">
          {styleOptions.map((option) => (
            <button
              key={option.value}
              type="button"
              className={value.learning_style === option.value ? 'active' : ''}
              onClick={() => update('learning_style', option.value)}
            >
              <strong>{learningStyleText[option.value]}</strong>
              <span>{option.desc}</span>
            </button>
          ))}
        </div>
      </section>}

      <button className="primary onboarding-submit" disabled={!canSubmit}>
        {loading ? <Loader2 className="spin" size={17} /> : null}
        {loading ? (profileOnly ? '正在保存画像' : '正在生成学习计划') : (profileOnly ? '保存画像，下一步' : '生成学习计划')}
      </button>
    </form>
  );
}
