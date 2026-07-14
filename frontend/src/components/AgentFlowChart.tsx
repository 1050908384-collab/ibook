import React from 'react';
import { Brain, FileText, CheckCircle, XCircle, TrendingUp, Target } from 'lucide-react';

const AgentFlowChart: React.FC = () => {
  const agents = [
    { id: 1, name: '学情诊断', icon: Brain, color: 'blue', description: '生成用户画像' },
    { id: 2, name: '目录生成', icon: FileText, color: 'green', description: '规划学习阶段' },
    { id: 3, name: '内容生成', icon: Target, color: 'purple', description: '生成学习资料' },
    { id: 4, name: '技术审核', icon: CheckCircle, color: 'orange', description: '检查技术正确性' },
    { id: 5, name: '质量评审', icon: XCircle, color: 'red', description: '评估教学质量' },
    { id: 6, name: '阶段决策', icon: TrendingUp, color: 'cyan', description: '更新用户画像' },
  ];

  return (
    <div className="agent-flow-chart">
      <h3>Agent协同流程</h3>
      <div className="flow-container">
        {agents.map((agent, index) => (
          <div key={agent.id} className="agent-node">
            <div className={`agent-icon ${agent.color}`}>
              <agent.icon size={24} />
            </div>
            <div className="agent-info">
              <p className="agent-name">Agent{agent.id}</p>
              <p className="agent-title">{agent.name}</p>
              <p className="agent-desc">{agent.description}</p>
            </div>
            {index < agents.length - 1 && (
              <div className="flow-arrow">→</div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default AgentFlowChart;
