import React, { useState } from 'react';
import { FileQuestion, Send, CheckCircle, XCircle, Lightbulb } from 'lucide-react';

interface Question {
  question_id: string;
  question_type: string;
  question_text: string;
  options?: string[];
  hint?: string;
}

interface TestQuestionProps {
  questions: Question[];
  onSubmit: (questionId: string, answer: string) => void;
}

const TestQuestion: React.FC<TestQuestionProps> = ({ questions, onSubmit }) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [userAnswer, setUserAnswer] = useState('');
  const [submitted, setSubmitted] = useState(false);
  const [showHint, setShowHint] = useState(false);

  const currentQuestion = questions[currentIndex];

  const handleSubmit = () => {
    if (userAnswer.trim()) {
      onSubmit(currentQuestion.question_id, userAnswer);
      setSubmitted(true);
    }
  };

  const handleNext = () => {
    if (currentIndex < questions.length - 1) {
      setCurrentIndex(prev => prev + 1);
      setUserAnswer('');
      setSubmitted(false);
      setShowHint(false);
    }
  };

  return (
    <div className="test-question">
      <div className="question-header">
        <FileQuestion size={24} />
        <h3>测试题 {currentIndex + 1}/{questions.length}</h3>
      </div>
      
      <div className="question-content">
        <p className="question-type">{currentQuestion.question_type}</p>
        <p className="question-text">{currentQuestion.question_text}</p>
        
        {currentQuestion.options && currentQuestion.question_type === '选择' && (
          <div className="options">
            {currentQuestion.options.map((option, index) => (
              <label key={index} className={`option ${userAnswer === option ? 'selected' : ''}`}>
                <input
                  type="radio"
                  name="answer"
                  value={option}
                  checked={userAnswer === option}
                  onChange={(e) => setUserAnswer(e.target.value)}
                  disabled={submitted}
                />
                <span>{option}</span>
              </label>
            ))}
          </div>
        )}
        
        {currentQuestion.question_type !== '选择' && (
          <textarea
            value={userAnswer}
            onChange={(e) => setUserAnswer(e.target.value)}
            placeholder="请输入你的答案..."
            disabled={submitted}
          />
        )}
        
        {currentQuestion.hint && (
          <button className="hint-btn" onClick={() => setShowHint(!showHint)}>
            <Lightbulb size={16} />
            {showHint ? '隐藏提示' : '显示提示'}
          </button>
        )}
        
        {showHint && currentQuestion.hint && (
          <div className="hint">{currentQuestion.hint}</div>
        )}
      </div>
      
      <div className="question-actions">
        {!submitted ? (
          <button onClick={handleSubmit} disabled={!userAnswer.trim()} className="submit-btn">
            <Send size={16} />
            提交答案
          </button>
        ) : (
          <button onClick={handleNext} className="next-btn" disabled={currentIndex === questions.length - 1}>
            {currentIndex === questions.length - 1 ? '完成测试' : '下一题'}
          </button>
        )}
      </div>
      
      {submitted && (
        <div className="submit-feedback">
          <CheckCircle size={24} />
          <p>答案已提交</p>
        </div>
      )}
    </div>
  );
};

export default TestQuestion;
