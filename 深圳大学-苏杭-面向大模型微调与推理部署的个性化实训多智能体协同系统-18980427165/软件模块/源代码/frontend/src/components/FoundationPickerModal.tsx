import { useEffect, useMemo, useState } from 'react';
import { Check, ChevronLeft, ChevronRight, Circle, X } from 'lucide-react';
import type { KnowledgeGraphBlock } from '../data/knowledgeGraph';

type FoundationPickerModalProps = {
  open: boolean;
  blocks: KnowledgeGraphBlock[];
  initialSelectedSubtopicIds: string[];
  onClose: () => void;
  onConfirm: (selectedSubtopicIds: string[]) => void;
};

function fullBlockIds(blocks: KnowledgeGraphBlock[], selectedSubtopicIds: Set<string>) {
  return blocks
    .filter((block) => block.subtopics.length > 0 && block.subtopics.every((topic) => selectedSubtopicIds.has(topic.subtopic_id)))
    .map((block) => block.block_id);
}
export default function FoundationPickerModal({
  open,
  blocks,
  initialSelectedSubtopicIds,
  onClose,
  onConfirm
}: FoundationPickerModalProps) {
  const [step, setStep] = useState<1 | 2>(1);
  const [selectedBlockIds, setSelectedBlockIds] = useState<Set<string>>(new Set());
  const [selectedSubtopicIds, setSelectedSubtopicIds] = useState<Set<string>>(new Set());

  useEffect(() => {
    if (!open) return;
    const initialSubtopics = new Set(initialSelectedSubtopicIds);
    const initialBlocks = new Set(
      blocks.filter((block) => block.subtopics.some((topic) => initialSubtopics.has(topic.subtopic_id)))
        .map((block) => block.block_id)
    );
    setStep(1);
    setSelectedBlockIds(initialBlocks);
    setSelectedSubtopicIds(initialSubtopics);
  }, [open, blocks, initialSelectedSubtopicIds]);

  const selectedBlocks = useMemo(
    () => blocks.filter((block) => selectedBlockIds.has(block.block_id)),
    [blocks, selectedBlockIds]
  );
  const completedBlockIds = useMemo(
    () => new Set(fullBlockIds(blocks, selectedSubtopicIds)),
    [blocks, selectedSubtopicIds]
  );

  if (!open) return null;

  const toggleBlock = (block: KnowledgeGraphBlock) => {
    setSelectedBlockIds((current) => {
      const next = new Set(current);
      if (next.has(block.block_id)) {
        next.delete(block.block_id);
        setSelectedSubtopicIds((topics) => {
          const nextTopics = new Set(topics);
          block.subtopics.forEach((topic) => nextTopics.delete(topic.subtopic_id));
          return nextTopics;
        });
      } else {
        next.add(block.block_id);
      }
      return next;
    });
  };

  const toggleSubtopic = (subtopicId: string) => {
    setSelectedSubtopicIds((current) => {
      const next = new Set(current);
      if (next.has(subtopicId)) next.delete(subtopicId);
      else next.add(subtopicId);
      return next;
    });
  };

  const confirm = () => onConfirm([...selectedSubtopicIds]);
  const chooseNoFoundation = () => onConfirm([]);

  return (
    <div className="modal-backdrop foundation-picker-backdrop" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <section className="foundation-picker-modal" role="dialog" aria-modal="true" aria-labelledby="foundation-picker-title">
        <header>
          <div>
            <span>确定基础 · 第 {step} 步 / 2</span>
            <h2 id="foundation-picker-title">{step === 1 ? '选择知识板块' : '选择你已经掌握的知识点'}</h2>
          </div>
          <button type="button" onClick={onClose} aria-label="关闭基础选择" title="关闭"><X size={18} /></button>
        </header>

        <div className="foundation-picker-body">
          {step === 1 ? (
            <>
              <p className="foundation-picker-intro">从完整的 15 个知识板块中，先选择你接触或学习过的板块。</p>
              <div className="foundation-block-grid">
                {blocks.map((block) => {
                  const selected = selectedBlockIds.has(block.block_id);
                  return (
                    <button
                      type="button"
                      className={`foundation-block-option ${selected ? 'selected' : ''}`}
                      aria-pressed={selected}
                      key={block.block_id}
                      onClick={() => toggleBlock(block)}
                    >
                      <span className="foundation-option-icon">{selected ? <Check size={15} /> : <Circle size={15} />}</span>
                      <span><strong>{block.block_name}</strong><small>{block.subtopics.length} 个知识点</small></span>
                    </button>
                  );
                })}
              </div>
            </>
          ) : (
            <>
              <p className="foundation-picker-intro">只勾选你确实掌握的知识点。只有一个板块下的知识点全部选中，它才会计入已有基础。</p>
              <div className="foundation-topic-groups">
                {selectedBlocks.map((block) => (
                  <section className="foundation-topic-group" key={block.block_id}>
                    <div className="foundation-topic-heading">
                      <div><strong>{block.block_name}</strong><small>{block.subtopics.length} 个知识点</small></div>
                      {completedBlockIds.has(block.block_id) && <span><Check size={13} />已完整掌握</span>}
                    </div>
                    <div className="foundation-topic-list">
                      {block.subtopics.map((topic) => {
                        const selected = selectedSubtopicIds.has(topic.subtopic_id);
                        return (
                          <label className={`foundation-topic-option ${selected ? 'selected' : ''}`} key={topic.subtopic_id}>
                            <input type="checkbox" checked={selected} onChange={() => toggleSubtopic(topic.subtopic_id)} />
                            <span>{selected ? <Check size={14} /> : <Circle size={14} />}</span>
                            <strong>{topic.subtopic_name}</strong>
                          </label>
                        );
                      })}
                    </div>
                  </section>
                ))}
                {!selectedBlocks.length && <div className="foundation-empty-state">请先返回上一步选择知识板块。</div>}
              </div>
            </>
          )}
        </div>

        <footer className="foundation-picker-footer">
          {step === 1 ? (
            <>
              <button type="button" className="foundation-secondary-action" onClick={chooseNoFoundation}>暂时没有已掌握的基础</button>
              <button type="button" className="primary" onClick={() => setStep(2)} disabled={!selectedBlockIds.size}>进入知识点选择 <ChevronRight size={16} /></button>
            </>
          ) : (
            <>
              <button type="button" className="foundation-secondary-action" onClick={() => setStep(1)}><ChevronLeft size={16} />返回板块选择</button>
              <button type="button" className="primary" onClick={confirm}>确定基础 <Check size={16} /></button>
            </>
          )}
        </footer>
      </section>
    </div>
  );
}
