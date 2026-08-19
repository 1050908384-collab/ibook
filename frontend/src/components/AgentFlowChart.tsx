import React from 'react';
import { Brain, CheckCircle, TrendingUp, Target } from 'lucide-react';

const AgentFlowChart: React.FC = () => {
  const agents = [
    { id: 1, name: '学情诊断与规划', icon: Brain, color: 'blue', description: '生成画像并规划知识点阶段' },
    { id: 3, name: '个性化内容生成', icon: Target, color: 'purple', description: '生成教程、代码与测试题' },
    { id: 4, name: '技术与教学双审', icon: CheckCircle, color: 'orange', description: '核验技术正确性与学习适配度' },
    { id: 6, name: '学情决策与推进', icon: TrendingUp, color: 'cyan', description: '更新画像并确定后续策略' },
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
