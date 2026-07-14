import React from 'react';
import { User, Target, Clock, Palette, Award } from 'lucide-react';

interface ProfileCardProps {
  profile: {
    skill_tags: string[];
    goal: string;
    daily_hours: number;
    learning_style: string;
    confidence: number;
  };
}

const ProfileCard: React.FC<ProfileCardProps> = ({ profile }) => {
  return (
    <div className="profile-card">
      <div className="card-header">
        <User size={28} />
        <h3>用户画像</h3>
      </div>
      
      <div className="card-content">
        <div className="profile-item">
          <Award size={18} />
          <div className="item-content">
            <p className="item-label">技能标签</p>
            <div className="tags">
              {profile.skill_tags.map((tag, index) => (
                <span key={index} className="tag">{tag}</span>
              ))}
            </div>
          </div>
        </div>
        
        <div className="profile-item">
          <Target size={18} />
          <div className="item-content">
            <p className="item-label">学习目标</p>
            <p className="item-value">{profile.goal}</p>
          </div>
        </div>
        
        <div className="profile-item">
          <Clock size={18} />
          <div className="item-content">
            <p className="item-label">每日学习时长</p>
            <p className="item-value">{profile.daily_hours}小时</p>
          </div>
        </div>
        
        <div className="profile-item">
          <Palette size={18} />
          <div className="item-content">
            <p className="item-label">学习风格</p>
            <p className="item-value">
              {profile.learning_style === 'theory' ? '理论型' : 
               profile.learning_style === 'practice' ? '实操型' : '混合型'}
            </p>
          </div>
        </div>
        
        <div className="profile-item confidence">
          <div className="item-content">
            <p className="item-label">画像置信度</p>
            <div className="confidence-bar">
              <div className="confidence-fill" style={{ width: `${profile.confidence * 100}%` }}></div>
            </div>
            <p className="confidence-value">{(profile.confidence * 100).toFixed(0)}%</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProfileCard;
