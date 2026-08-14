import { ArrowLeft, Brain, CheckCircle2, FileText, Loader2, Sparkles } from 'lucide-react';
import ProfileSetupForm from '../components/ProfileSetupForm';
import { OnboardingField, OnboardingFormData } from '../utils/buildDiagnosticMessage';

type OnboardingPageProps = {
  formData: OnboardingFormData;
  onChange: (value: OnboardingFormData) => void;
  onSubmit: () => void;
  onCancel?: () => void;
  loading: boolean;
  mode: 'onboarding' | 'diagnosing' | 'needs_more_info';
  questions: string[];
  error: string | null;
  highlightedFields: OnboardingField[];
  reuseUserProfile: boolean;
  checkingUserProfile?: boolean;
  profileCheckError?: string | null;
  onRetryProfileCheck?: () => void;
  foundationNames: string[];
  foundationConfirmed: boolean;
  onOpenFoundation: () => void;
  foundationLoading?: boolean;
};

export default function OnboardingPage({
  formData,
  onChange,
  onSubmit,
  onCancel,
  loading,
  mode,
  questions,
  error,
  highlightedFields,
  reuseUserProfile,
  checkingUserProfile = false,
  profileCheckError = null,
  onRetryProfileCheck,
  foundationNames,
  foundationConfirmed,
  onOpenFoundation,
  foundationLoading = false
}: OnboardingPageProps) {
  if (checkingUserProfile) {
    return (
      <div className="page-shell onboarding-shell">
        <main className="onboarding-page onboarding-checking-profile">
          <Loader2 className="spin" size={24} />
          <strong>正在读取用户基础</strong>
          <span>确认是否需要填写首次画像信息。</span>
        </main>
      </div>
    );
  }
  if (profileCheckError) {
    return (
      <div className="page-shell onboarding-shell">
        <main className="onboarding-page onboarding-checking-profile error">
          <strong>用户基础读取失败</strong>
          <span>{profileCheckError}</span>
          <button className="primary" type="button" onClick={onRetryProfileCheck}>重新读取</button>
        </main>
      </div>
    );
  }
  return (
    <div className="page-shell onboarding-shell">
      <main className="onboarding-page">
        <div className="onboarding-toolbar">
          <div className="onboarding-steps" aria-label={reuseUserProfile ? '创建学习目标步骤' : '首次画像设置步骤'}>
            <span className="active"><FileText size={15} />{reuseUserProfile ? '填写目标' : '建立画像'}</span>
            <span className={mode === 'diagnosing' ? 'active' : ''}><Brain size={15} />{reuseUserProfile ? '生成计划' : '创建目标'}</span>
            <span><CheckCircle2 size={15} />开始学习</span>
          </div>
          {onCancel && (
            <button className="cancel-goal-creation" type="button" onClick={onCancel} disabled={loading}>
              <ArrowLeft size={16} />
              返回原学习目标
            </button>
          )}
        </div>

        <section className="onboarding-hero">
          <div className="hero-mark">
            <Sparkles size={22} />
          </div>
          <div>
            <span>{reuseUserProfile ? '创建学习目标' : '首次使用'}</span>
            <h1>{reuseUserProfile ? '设定这次要完成的学习目标' : '先把你的学习背景交给 Agent'}</h1>
            <p>{reuseUserProfile
              ? '已复用你的用户基础和知识点掌握状态，请填写本次目标、学习时间和学习方式。'
              : '这里只建立你的长期用户画像，不设置学习目标。保存后再单独创建第一个学习目标。'}</p>
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
          targetOnly={reuseUserProfile}
          profileOnly={!reuseUserProfile}
          foundationNames={foundationNames}
          foundationConfirmed={foundationConfirmed}
          onOpenFoundation={onOpenFoundation}
          foundationLoading={foundationLoading}
        />
      </main>
    </div>
  );
}
