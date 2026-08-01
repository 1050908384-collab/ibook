import { Brain, CheckCircle2, FileText, Sparkles } from 'lucide-react';
import ProfileSetupForm from '../components/ProfileSetupForm';
import { OnboardingField, OnboardingFormData } from '../utils/buildDiagnosticMessage';

type OnboardingPageProps = {
  formData: OnboardingFormData;
  onChange: (value: OnboardingFormData) => void;
  onSubmit: () => void;
  loading: boolean;
  mode: 'onboarding' | 'diagnosing' | 'needs_more_info';
  questions: string[];
  error: string | null;
  highlightedFields: OnboardingField[];
};

export default function OnboardingPage({
  formData,
  onChange,
  onSubmit,
  loading,
  mode,
  questions,
  error,
  highlightedFields
}: OnboardingPageProps) {
  return (
    <div className="page-shell onboarding-shell">
      <main className="onboarding-page">
        <div className="onboarding-steps" aria-label="创建学习目标步骤">
          <span className="active"><FileText size={15} />填写信息</span>
          <span className={mode === 'diagnosing' ? 'active' : ''}><Brain size={15} />生成画像</span>
          <span><CheckCircle2 size={15} />开始学习</span>
        </div>

        <section className="onboarding-hero">
          <div className="hero-mark">
            <Sparkles size={22} />
          </div>
          <div>
            <span>创建学习目标</span>
            <h1>先把你的学习背景交给 Agent</h1>
            <p>系统会根据这些结构化信息调用现有 Agent1 诊断流程，再由 Agent2 规划当前一个学习阶段。</p>
          </div>
        </section>

        <ProfileSetupForm
          value={formData}
          onChange={onChange}
          onSubmit={onSubmit}
          loading={loading || mode === 'diagnosing'}
          mode={mode === 'needs_more_info' ? 'needs_more_info' : 'onboarding'}
          questions={questions}
          error={error}
          highlightedFields={highlightedFields}
        />
      </main>
    </div>
  );
}
