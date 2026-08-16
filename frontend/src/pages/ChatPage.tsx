import React, { useState, useEffect } from 'react';
import { MessageCircle, Send, BookOpen, FileQuestion, CheckCircle, XCircle } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { apiClient } from '../api/client';

interface Message {
  id: string;
  content: string;
  isUser: boolean;
  techPass?: boolean;
  qualityPass?: boolean;
  timestamp: Date;
}

interface LearningPackage {
  tutorial: string;
  code_examples: any[];
  questions: any[];
}

const ChatPage: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [learningPackage, setLearningPackage] = useState<LearningPackage | null>(null);
  const [isTyping, setIsTyping] = useState(false);

  useEffect(() => {
    const savedSession = localStorage.getItem('sessionId');
    if (savedSession) {
      setSessionId(savedSession);
    }
  }, []);

  const handleSend = async () => {
    if (!inputValue.trim()) return;

    const newMessage: Message = {
      id: Date.now().toString(),
      content: inputValue,
      isUser: true,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, newMessage]);
    setInputValue('');
    setIsTyping(true);

    try {
      let response;
      if (!sessionId) {
        const created = await apiClient.createSession();
        response = await apiClient.submitDiagnosis(created.session_id, inputValue);
        setSessionId(created.session_id);
        localStorage.setItem('sessionId', created.session_id);
      } else {
        response = await apiClient.askQuestion(sessionId, inputValue);
      }

      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: response.answer || response.assistant_message || JSON.stringify(response),
        isUser: false,
        techPass: response.review_summary?.technical_passed,
        qualityPass: response.review_summary?.quality_passed,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
    } finally {
      setIsTyping(false);
    }
  };

  const handleStartStage = async () => {
    if (!sessionId) return;
    setIsTyping(true);
    try {
      await apiClient.startStage(sessionId);
      const response = await apiClient.getStagePackage(sessionId);
      setLearningPackage(response.package);
      setMessages(prev => [...prev, {
        id: Date.now().toString(),
        content: '学习包已生成，请开始学习！',
        isUser: false,
        timestamp: new Date()
      }]);
    } catch (error) {
      console.error('Error starting stage:', error);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <div className="chat-page">
      <div className="chat-container">
        <div className="chat-messages">
          {messages.map(msg => (
            <div key={msg.id} className={`message ${msg.isUser ? 'user' : 'bot'}`}>
              <div className="message-content">{msg.content}</div>
              {!msg.isUser && (msg.techPass !== undefined) && (
                <div className="message-verification">
                  {msg.techPass && <CheckCircle className="pass" />}
                  {!msg.techPass && <XCircle className="fail" />}
                  <span>技术审核</span>
                  {msg.qualityPass && <CheckCircle className="pass" />}
                  {!msg.qualityPass && <XCircle className="fail" />}
                  <span>质量评审</span>
                </div>
              )}
            </div>
          ))}
          {isTyping && (
            <div className="message bot">
              <div className="typing-indicator">正在思考...</div>
            </div>
          )}
        </div>
        
        <div className="chat-input">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSend()}
            placeholder="输入你的问题或描述学习背景..."
          />
          <button onClick={handleSend} disabled={isTyping}>
            <Send size={20} />
          </button>
        </div>
      </div>
      
      <div className="sidebar">
        <div className="sidebar-header">
          <MessageCircle size={24} />
          <h2>学习助手</h2>
        </div>
        
        <button onClick={handleStartStage} className="action-button">
          <BookOpen size={20} />
          开始学习阶段
        </button>
        
        <button className="action-button">
          <FileQuestion size={20} />
          查看测试题
        </button>
        
        {learningPackage && (
          <div className="learning-package">
            <h3>学习资料</h3>
            <div className="package-content material-markdown">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  pre({ children }) {
                    return <div className="material-code-block">{children}</div>;
                  },
                  code({ className, children, ...props }) {
                    const isInline = !className;
                    if (isInline) {
                      return <code className="material-inline-code" {...props}>{children}</code>;
                    }
                    return <code className={className} {...props}>{children}</code>;
                  },
                }}
              >
                {learningPackage.tutorial}
              </ReactMarkdown>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatPage;
