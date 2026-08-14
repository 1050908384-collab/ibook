import { ArrowLeft, Clock, Play, ShieldCheck, Target, UserRound } from 'lucide-react';
import { learningStyleText } from '../utils/buildDiagnosticMessage';

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
  foundation_confirmed?: boolean;
  knowledge_gaps?: KnowledgeGap[];
  goal?: string;
  daily_hours?: number;
  learning_style?: string;
  hardware?: string;
  confidence?: number;
};

type StageInfo = {
  stage_num?: number;
  block_name?: string;
  subtopic_name?: string;
  gap_skill?: string;
  estimated_hours?: number;
  prerequisites?: string[];
};

type ProfileConfirmPageProps = {
  profile: Profile;
  stage: StageInfo;
  loading: boolean;
  error: string | null;
  onEdit: () => void;
  onStart: () => void;
  onCancel?: () => void;
  showUserProfile?: boolean;
};

function pct(value = 0) {
  return Math.round(Math.max(0, Math.min(1, value)) * 100);
}

function displayStyle(style?: string) {
  if (style === 'theory' || style === 'practice' || style === 'mixed') {
    return learningStyleText[style];
  }
  return style || '混合型';
}

export default function ProfileConfirmPage({ profile, stage, loading, error, onEdit, onStart, onCancel, showUserProfile = true }: ProfileConfirmPageProps) {
  const gaps = profile.knowledge_gaps || [];
  return (
    <div className="page-shell confirm-shell">
      <main className="profile-confirm-page">
        <div className="confirm-head">
          <div>
            <span>{showUserProfile ? '画像确认' : '阶段确认'}</span>
            <h1>{showUserProfile ? '请确认这份学习画像' : '请确认本次学习目标与阶段'}</h1>
            <p>{showUserProfile ? '用户基础只在首次创建时填写，确认后将用于后续学习目标。' : '已复用首次填写的用户基础，这里只确认本次目标和第一阶段。'}</p>
          </div>
          <div className="confidence-card">
            <ShieldCheck size={22} />
            <span>画像置信度</span>
            <strong>{pct(profile.confidence ?? 0)}%</strong>
          </div>
        </div>

        {error && <div className="confirm-error">{error}</div>}

        <div className="confirm-grid">
          {showUserProfile && <section className="white-card confirm-card profile-summary">
            <div className="card-title">
              <div>
                <h3>学习画像</h3>
                <p>来自 Agent1 诊断结果</p>
              </div>
              <UserRound size={20} />
            </div>
            <dl className="confirm-list">
              <div><dt>身份/学习背景</dt><dd>{profile.grade || '未提供'}</dd></div>
              <div><dt>已有基础</dt><dd>{profile.skill_tags?.length ? profile.skill_tags.join('、') : profile.foundation_confirmed ? '已确认：暂无基础' : '未填写'}</dd></div>
              <div><dt>知识盲区</dt><dd>{gaps.map((gap) => gap.skill).join('、') || '等待推断'}</dd></div>
              <div><dt>学习目标</dt><dd>{profile.goal || '未提供'}</dd></div>
              <div><dt>每日学习时长</dt><dd>{profile.daily_hours ?? 2} 小时</dd></div>
              <div><dt>学习方式</dt><dd>{displayStyle(profile.learning_style)}</dd></div>
              <div><dt>设备条件</dt><dd>{profile.hardware || '未提供'}</dd></div>
            </dl>
          </section>}
        </div>

        <section className="current-stage-panel">
          <div>
            <span className="lesson-kicker">阶段 {stage.stage_num ?? 1}</span>
            <h2>{stage.block_name || '等待阶段规划'}</h2>
            {!showUserProfile && <p>学习目标：{profile.goal || '未提供'}</p>}
            <p>{stage.subtopic_name || stage.gap_skill || '当前子知识点待后端返回'}</p>
          </div>
          <div className="stage-facts">
            <span><Clock size={15} />预计 {stage.estimated_hours ?? 1} 小时</span>
            <span><Target size={15} />前置要求：{stage.prerequisites?.join('、') || '暂无'}</span>
          </div>
        </section>

        <div className="confirm-actions">
          {onCancel && (
            <button className="ghost" onClick={onCancel} disabled={loading}>
              <ArrowLeft size={16} />
              取消新建
            </button>
          )}
          <button className="ghost" onClick={onEdit} disabled={loading}>
            <ArrowLeft size={16} />
            {showUserProfile ? '修改信息' : '修改本次目标'}
          </button>
          <button className="primary" onClick={onStart} disabled={loading}>
            <Play size={17} />
            {loading ? '正在开始阶段' : `开始学习阶段 ${stage.stage_num ?? 1}`}
          </button>
        </div>
      </main>
    </div>
  );
}
