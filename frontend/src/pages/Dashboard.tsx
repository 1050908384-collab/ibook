import React, { useState, useEffect } from 'react';
import { Radar, Doughnut } from 'react-chartjs-2';
import { Chart as ChartJS, RadialLinearScale, PointElement, LineElement, Filler, Tooltip, Legend, ArcElement } from 'chart.js';
import { TrendingUp, Target, Clock, Award } from 'lucide-react';
import { apiClient } from '../api/client';

ChartJS.register(RadialLinearScale, PointElement, LineElement, Filler, Tooltip, Legend, ArcElement);

interface KnowledgeGap {
  skill: string;
  status: string;
  progress: number;
}

interface ProgressData {
  accuracy: number;
  knowledge_gaps: KnowledgeGap[];
  learning_rhythm: string;
  confidence_index: number;
}

const Dashboard: React.FC = () => {
  const [progressData, setProgressData] = useState<ProgressData | null>(null);

  useEffect(() => {
    const sessionId = localStorage.getItem('sessionId');
    if (sessionId) {
      fetchProgress(sessionId);
    }
  }, []);

  const fetchProgress = async (sessionId: string) => {
    try {
      const data = await apiClient.getProgress(sessionId);
      setProgressData(data);
    } catch (error) {
      console.error('Error fetching progress:', error);
    }
  };

  if (!progressData) {
    return <div className="dashboard">加载中...</div>;
  }

  const radarData = {
    labels: progressData.knowledge_gaps.map(g => g.skill),
    datasets: [{
      label: '掌握程度',
      data: progressData.knowledge_gaps.map(g => g.progress * 100),
      backgroundColor: 'rgba(54, 162, 235, 0.2)',
      borderColor: 'rgba(54, 162, 235, 1)',
      borderWidth: 2,
      pointBackgroundColor: 'rgba(54, 162, 235, 1)',
      pointBorderColor: '#fff',
      pointHoverBackgroundColor: '#fff',
      pointHoverBorderColor: 'rgba(54, 162, 235, 1)'
    }]
  };

  const doughnutData = {
    labels: ['已掌握', '部分掌握', '未开始'],
    datasets: [{
      data: [
        progressData.knowledge_gaps.filter(g => g.status === '已掌握').length,
        progressData.knowledge_gaps.filter(g => g.status === '部分掌握').length,
        progressData.knowledge_gaps.filter(g => g.status === '未开始').length
      ],
      backgroundColor: ['#10B981', '#F59E0B', '#EF4444'],
      borderWidth: 0
    }]
  };

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <h1>学习进度</h1>
        <p>追踪你的学习旅程</p>
      </div>
      
      <div className="stats-grid">
        <div className="stat-card">
          <TrendingUp className="stat-icon" />
          <div className="stat-content">
            <p className="stat-label">阶段正确率</p>
            <p className="stat-value">{(progressData.accuracy * 100).toFixed(0)}%</p>
          </div>
        </div>
        
        <div className="stat-card">
          <Target className="stat-icon" />
          <div className="stat-content">
            <p className="stat-label">信心指数</p>
            <p className="stat-value">{(progressData.confidence_index * 100).toFixed(0)}%</p>
          </div>
        </div>
        
        <div className="stat-card">
          <Clock className="stat-icon" />
          <div className="stat-content">
            <p className="stat-label">学习节奏</p>
            <p className="stat-value">{progressData.learning_rhythm}</p>
          </div>
        </div>
        
        <div className="stat-card">
          <Award className="stat-icon" />
          <div className="stat-content">
            <p className="stat-label">已掌握知识点</p>
            <p className="stat-value">{progressData.knowledge_gaps.filter(g => g.status === '已掌握').length}/{progressData.knowledge_gaps.length}</p>
          </div>
        </div>
      </div>
      
      <div className="charts-grid">
        <div className="chart-card">
          <h3>知识盲区雷达图</h3>
          <Radar data={radarData} />
        </div>
        
        <div className="chart-card">
          <h3>掌握状态分布</h3>
          <Doughnut data={doughnutData} />
        </div>
      </div>
      
      <div className="gaps-list">
        <h3>知识盲区详情</h3>
        <div className="gaps-table">
          {progressData.knowledge_gaps.map((gap, index) => (
            <div key={index} className="gap-row">
              <span className="gap-skill">{gap.skill}</span>
              <span className={`gap-status ${gap.status}`}>{gap.status}</span>
              <div className="gap-progress">
                <div className="progress-bar" style={{ width: `${gap.progress * 100}%` }}></div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
