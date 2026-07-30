import { ArrowLeft, Clock, Play, ShieldCheck, Target, UserRound } from 'lucide-react';
import RadarChart from '../components/RadarChart';
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

export default function ProfileConfirmPage({ profile, stage, loading, error, onEdit, onStart }: ProfileConfirmPageProps) {
  const gaps = profile.knowledge_gaps || [];
  const radarData = {
    labels: gaps.length ? gaps.slice(0, 5).map((gap) => gap.skill) : ['基础', '目标', '实践', '部署', '复盘'],
    values: gaps.length ? gaps.slice(0, 5).map((gap) => Math.round((gap.progress ?? 0.2) * 100)) : [55, 45, 35, 30, 50]
  };

  return (
    <div className="page-shell confirm-shell">
      <main className="profile-confirm-page">
        <div className="confirm-head">
          <div>
            <span>画像确认</span>
            <h1>请确认这份学习画像</h1>
            <p>确认后会进入 Agent2 生成的当前阶段，学习页仍沿用现有流程。</p>
          </div>
          <div className="confidence-card">
            <ShieldCheck size={22} />
            <span>画像置信度</span>
            <strong>{pct(profile.confidence ?? 0)}%</strong>
          </div>
        </div>

        {error && <div className="confirm-error">{error}</div>}

        <div className="confirm-grid">
          <section className="white-card confirm-card profile-summary">
            <div className="card-title">
              <div>
                <h3>学习画像</h3>
                <p>来自 Agent1 诊断结果</p>
              </div>
              <UserRound size={20} />
            </div>
            <dl className="confirm-list">
              <div><dt>身份/学习背景</dt><dd>{profile.grade || '未提供'}</dd></div>
              <div><dt>已有基础</dt><dd>{profile.skill_tags?.join('、') || '未识别'}</dd></div>
              <div><dt>知识盲区</dt><dd>{gaps.map((gap) => gap.skill).join('、') || '等待推断'}</dd></div>
              <div><dt>学习目标</dt><dd>{profile.goal || '未提供'}</dd></div>
              <div><dt>每日学习时长</dt><dd>{profile.daily_hours ?? 2} 小时</dd></div>
              <div><dt>学习方式</dt><dd>{displayStyle(profile.learning_style)}</dd></div>
              <div><dt>设备条件</dt><dd>{profile.hardware || '未提供'}</dd></div>
            </dl>
          </section>

          <section className="white-card confirm-card">
            <div className="card-title">
              <div>
                <h3>能力雷达图</h3>
                <p>根据待补齐知识点生成</p>
              </div>
            </div>
            <RadarChart data={radarData} />
          </section>
        </div>

        <section className="current-stage-panel">
          <div>
            <span className="lesson-kicker">阶段 {stage.stage_num ?? 1}</span>
            <h2>{stage.block_name || '等待阶段规划'}</h2>
            <p>{stage.subtopic_name || stage.gap_skill || '当前子知识点待后端返回'}</p>
          </div>
          <div className="stage-facts">
            <span><Clock size={15} />预计 {stage.estimated_hours ?? 1} 小时</span>
            <span><Target size={15} />前置要求：{stage.prerequisites?.join('、') || '暂无'}</span>
          </div>
        </section>

        <div className="confirm-actions">
          <button className="ghost" onClick={onEdit} disabled={loading}>
            <ArrowLeft size={16} />
            修改信息
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
