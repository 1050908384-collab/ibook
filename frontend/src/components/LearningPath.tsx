import React from 'react';
import { ChevronRight, Play, CheckCircle, Circle } from 'lucide-react';

interface Stage {
  id: string;
  name: string;
  status: string;
  estimated_hours: number;
}

interface LearningPathProps {
  stages: Stage[];
  currentStage: number;
}

const LearningPath: React.FC<LearningPathProps> = ({ stages, currentStage }) => {
  return (
    <div className="learning-path">
      <h3>学习路径</h3>
      <div className="path-container">
        {stages.map((stage, index) => (
          <div key={stage.id} className={`stage-item ${index === currentStage ? 'current' : ''}`}>
            <div className="stage-indicator">
              {stage.status === 'completed' ? (
                <CheckCircle size={20} />
              ) : index === currentStage ? (
                <Play size={20} />
              ) : (
                <Circle size={20} />
              )}
            </div>
            <div className="stage-info">
              <p className="stage-number">阶段 {index + 1}</p>
              <p className="stage-name">{stage.name}</p>
              <p className="stage-duration">预计 {stage.estimated_hours} 小时</p>
            </div>
            {index < stages.length - 1 && (
              <div className="stage-connector">
                <ChevronRight size={24} />
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default LearningPath;
