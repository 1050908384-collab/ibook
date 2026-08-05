import { FormEvent, KeyboardEvent, useMemo, useState } from 'react';
import { AlertCircle, Loader2, Plus, X } from 'lucide-react';
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
};

const fieldLabels: Record<OnboardingField, string> = {
  grade: '身份/学习背景',
  skill_tags: '已有知识基础',
  goal: '学习目标',
  daily_hours: '每日学习时长',
  learning_style: '学习方式',
  hardware: '设备条件'
};

const styleOptions = [
  { value: 'theory', title: '理论型', desc: '更偏原理、概念和推导' },
  { value: 'practice', title: '实操型', desc: '更偏代码、实验和部署' },
  { value: 'mixed', title: '混合型', desc: '理论和实践均衡推进' }
] as const;

function splitTags(raw: string) {
  return raw
    .split(/[、,，;；\n]/)
    .map((item) => item.trim())
    .filter(Boolean);
}

export default function ProfileSetupForm({
  value,
  onChange,
  onSubmit,
  loading,
  mode,
  questions,
  error,
  highlightedFields,
  targetOnly = false
}: ProfileSetupFormProps) {
  const [tagDraft, setTagDraft] = useState('');
  const completeness = useMemo(() => getFormCompleteness(value, targetOnly), [value, targetOnly]);
  const canSubmit = isOnboardingFormComplete(value, targetOnly) && !loading;

  const update = <K extends OnboardingField>(field: K, fieldValue: OnboardingFormData[K]) => {
    onChange({ ...value, [field]: fieldValue });
  };

  const addTags = (raw: string) => {
    const tags = splitTags(raw).filter((tag) => !value.skill_tags.includes(tag));
    if (!tags.length) return;
    update('skill_tags', [...value.skill_tags, ...tags]);
    setTagDraft('');
  };

  const handleTagKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key !== 'Enter' && event.key !== ',') return;
    event.preventDefault();
    addTags(tagDraft);
  };

  const removeTag = (tag: string) => {
    update('skill_tags', value.skill_tags.filter((item) => item !== tag));
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

      {!targetOnly && <section className={`form-section ${isHighlighted('grade') ? 'needs-attention' : ''}`}>
        <label htmlFor="grade">{fieldLabels.grade} *</label>
        <p>告诉 Agent 你现在的身份或学习背景，用来判断适合的起点。</p>
        <input
          id="grade"
          value={value.grade}
          onChange={(event) => update('grade', event.target.value)}
          placeholder="例如：研二、算法工程师、后端开发者"
        />
      </section>}

      {!targetOnly && <section className={`form-section ${isHighlighted('skill_tags') ? 'needs-attention' : ''}`}>
        <label htmlFor="skill-tags">{fieldLabels.skill_tags} *</label>
        <p>输入你已经掌握或接触过的知识，可用逗号分隔，也可以逐个添加。</p>
        <div className="tag-input-wrap">
          <div className="setup-tags">
            {value.skill_tags.map((tag) => (
              <span key={tag}>
                {tag}
                <button type="button" onClick={() => removeTag(tag)} aria-label={`移除 ${tag}`}>
                  <X size={12} />
                </button>
              </span>
            ))}
          </div>
          <div className="tag-entry">
            <input
              id="skill-tags"
              value={tagDraft}
              onChange={(event) => setTagDraft(event.target.value)}
              onKeyDown={handleTagKeyDown}
              onBlur={() => addTags(tagDraft)}
              placeholder="例如：Python、PyTorch、Transformer"
            />
            <button type="button" onClick={() => addTags(tagDraft)} aria-label="添加标签">
              <Plus size={16} />
            </button>
          </div>
        </div>
      </section>}

      <section className={`form-section ${isHighlighted('goal') ? 'needs-attention' : ''}`}>
        <label htmlFor="goal">{fieldLabels.goal} *</label>
        <p>描述你希望达成的真实结果，Agent 会据此推断知识盲区和第一阶段。</p>
        <textarea
          id="goal"
          value={value.goal}
          onChange={(event) => update('goal', event.target.value)}
          rows={4}
          placeholder="例如：掌握 LoRA 微调并能独立部署"
        />
      </section>

      <div className="form-two-col">
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

        {!targetOnly && <section className={`form-section ${isHighlighted('hardware') ? 'needs-attention' : ''}`}>
          <label htmlFor="hardware">{fieldLabels.hardware}</label>
          <p>选填。设备条件会帮助 Agent 适配部署和实验建议。</p>
          <input
            id="hardware"
            value={value.hardware}
            onChange={(event) => update('hardware', event.target.value)}
            placeholder="例如：RTX 4060、MacBook、无独立显卡"
          />
        </section>}
      </div>

      {!targetOnly && <section className={`form-section ${isHighlighted('learning_style') ? 'needs-attention' : ''}`}>
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
        {loading ? '正在生成学习画像' : '生成学习画像'}
      </button>
    </form>
  );
}
