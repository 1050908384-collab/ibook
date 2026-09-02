import React, { useState } from 'react';
import { Settings, User, Clock, Palette, Save, RotateCcw } from 'lucide-react';

interface UserSettings {
  dailyHours: number;
  learningStyle: string;
  preferredModel: string;
}

const SettingsPage: React.FC = () => {
  const [settings, setSettings] = useState<UserSettings>({
    dailyHours: 2,
    learningStyle: 'mixed',
    preferredModel: 'deepseek-r1:7b'
  });

  const handleChange = (key: keyof UserSettings, value: string | number) => {
    setSettings(prev => ({ ...prev, [key]: value }));
  };

  const handleSave = () => {
    localStorage.setItem('userSettings', JSON.stringify(settings));
    alert('设置已保存');
  };

  const handleReset = () => {
    setSettings({
      dailyHours: 2,
      learningStyle: 'mixed',
      preferredModel: 'deepseek-r1:7b'
    });
    localStorage.removeItem('userSettings');
  };

  return (
    <div className="settings-page">
      <div className="settings-header">
        <Settings size={24} />
        <h1>设置</h1>
      </div>
      
      <div className="settings-content">
        <div className="settings-section">
          <div className="section-header">
            <User size={20} />
            <h2>学习偏好</h2>
          </div>
          
          <div className="setting-item">
            <label>每阶段学习时长</label>
            <div className="setting-control">
              <Clock size={16} />
              <input
                type="number"
                value={settings.dailyHours}
                onChange={(e) => handleChange('dailyHours', parseFloat(e.target.value))}
                min={0.5}
                max={8}
                step={0.5}
              />
              <span>小时</span>
            </div>
          </div>
          
          <div className="setting-item">
            <label>学习风格</label>
            <div className="setting-control">
              <Palette size={16} />
              <select
                value={settings.learningStyle}
                onChange={(e) => handleChange('learningStyle', e.target.value)}
              >
                <option value="theory">理论型</option>
                <option value="practice">实操型</option>
                <option value="mixed">混合型</option>
              </select>
            </div>
          </div>
        </div>
        
        <div className="settings-section">
          <div className="section-header">
            <Settings size={20} />
            <h2>模型设置</h2>
          </div>
          
          <div className="setting-item">
            <label>首选模型</label>
            <div className="setting-control">
              <select
                value={settings.preferredModel}
                onChange={(e) => handleChange('preferredModel', e.target.value)}
              >
                <option value="deepseek-r1:7b">DeepSeek-R1 (7B)</option>
                <option value="deepseek-r1:1.5b">DeepSeek-R1 (1.5B)</option>
                <option value="qwen2.5:7b">Qwen2.5 (7B)</option>
              </select>
            </div>
          </div>
        </div>
        
        <div className="settings-actions">
          <button onClick={handleSave} className="save-btn">
            <Save size={16} />
            保存设置
          </button>
          <button onClick={handleReset} className="reset-btn">
            <RotateCcw size={16} />
            重置
          </button>
        </div>
      </div>
    </div>
  );
};

export default SettingsPage;
