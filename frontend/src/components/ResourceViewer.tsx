import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { FileText, Code, Copy, Check } from 'lucide-react';

interface ResourceViewerProps {
  tutorial: string;
  codeExamples: any[];
}

const ResourceViewer: React.FC<ResourceViewerProps> = ({ tutorial, codeExamples }) => {
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);

  const handleCopy = (code: string, index: number) => {
    navigator.clipboard.writeText(code);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  return (
    <div className="resource-viewer">
      <div className="viewer-section">
        <div className="section-header">
          <FileText size={20} />
          <h3>教程</h3>
        </div>
        <div className="markdown-content">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{tutorial}</ReactMarkdown>
        </div>
      </div>
      
      <div className="viewer-section">
        <div className="section-header">
          <Code size={20} />
          <h3>代码示例</h3>
        </div>
        <div className="code-examples">
          {codeExamples.map((example, index) => (
            <div key={index} className="code-block">
              <div className="code-header">
                <span>{example.title || `代码示例 ${index + 1}`}</span>
                <button onClick={() => handleCopy(example.code, index)}>
                  {copiedIndex === index ? <Check size={16} /> : <Copy size={16} />}
                </button>
              </div>
              <pre className="code-content"><code>{example.code}</code></pre>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

function useState<T>(initialState: T): [T, React.Dispatch<React.SetStateAction<T>>] {
  return React.useState(initialState);
}

export default ResourceViewer;
