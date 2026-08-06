import { useEffect, useMemo, useRef, useState } from 'react';
import { Check, CircleAlert, LocateFixed, Target } from 'lucide-react';
import {
  apiClient,
  type KnowledgeNetworkSubtopicStatus
} from '../api/client';
import {
  knowledgeGraphBlocks,
  type KnowledgeGraphSubtopic
} from '../data/knowledgeGraph';
import {
  buildKnowledgeNetworkLayout,
  KNOWLEDGE_NETWORK_CANVAS
} from '../utils/knowledgeNetworkLayout';

type KnowledgeGapLike = {
  target_subtopic_ids?: string[];
};

type KnowledgeNetworkProps = {
  profile?: { knowledge_gaps?: KnowledgeGapLike[] } | null;
  stage?: { subtopic_id?: string; subtopic_name?: string } | null;
  compact?: boolean;
};

type DisplayStatus = 'mastered' | 'learning' | 'pending';

const positionForBlock = (index: number) => ({
  x: (index % 5) * KNOWLEDGE_NETWORK_CANVAS.cellWidth,
  y: Math.floor(index / 5) * KNOWLEDGE_NETWORK_CANVAS.cellHeight
});

function statusLabel(status: DisplayStatus) {
  if (status === 'mastered') return '已掌握';
  if (status === 'learning') return '正在学习';
  return '未学习';
}

export default function KnowledgeNetwork({ profile, stage, compact = false }: KnowledgeNetworkProps) {
  const [statuses, setStatuses] = useState<KnowledgeNetworkSubtopicStatus[]>([]);
  const [loadFailed, setLoadFailed] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const viewportRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let active = true;
    setLoadFailed(false);
    apiClient.getKnowledgeNetwork()
      .then((response) => {
        if (!active) return;
        setStatuses(response.blocks.flatMap((block) => block.subtopics || []));
      })
      .catch(() => {
        if (!active) return;
        setStatuses([]);
        setLoadFailed(true);
      });
    return () => { active = false; };
  }, []);

  const statusById = useMemo(
    () => new Map(statuses.map((item) => [item.subtopic_id, item])),
    [statuses]
  );
  const targetIds = useMemo(
    () => new Set((profile?.knowledge_gaps || []).flatMap((gap) => gap.target_subtopic_ids || [])),
    [profile]
  );
  const layout = useMemo(() => buildKnowledgeNetworkLayout(knowledgeGraphBlocks), []);
  const layoutById = useMemo(() => new Map(layout.map((item) => [item.block_id, item])), [layout]);
  const allSubtopics = useMemo(
    () => knowledgeGraphBlocks.flatMap((block) => block.subtopics.map((subtopic) => ({ block, subtopic }))),
    []
  );
  const currentId = stage?.subtopic_id
    || allSubtopics.find((item) => item.subtopic.subtopic_name === stage?.subtopic_name)?.subtopic.subtopic_id
    || [...targetIds][0];
  const currentEntry = currentId
    ? allSubtopics.find((item) => item.subtopic.subtopic_id === currentId)
    : undefined;
  const selected = selectedId
    ? allSubtopics.find((item) => item.subtopic.subtopic_id === selectedId)
    : undefined;
  const selectedStatus = selected ? statusById.get(selected.subtopic.subtopic_id) : undefined;

  const getDisplayStatus = (subtopic: KnowledgeGraphSubtopic): DisplayStatus => {
    if (subtopic.subtopic_id === currentId) return 'learning';
    if (statusById.get(subtopic.subtopic_id)?.status === 1) return 'mastered';
    return 'pending';
  };

  const focusCurrentSubtopic = (behavior: ScrollBehavior = 'smooth') => {
    const viewport = viewportRef.current;
    const node = currentId
      ? viewport?.querySelector<HTMLElement>(`[data-subtopic-id="${currentId}"]`)
      : null;
    if (!viewport || !node) return;
    const viewportRect = viewport.getBoundingClientRect();
    const nodeRect = node.getBoundingClientRect();
    viewport.scrollTo({
      left: viewport.scrollLeft + nodeRect.left - viewportRect.left - viewport.clientWidth / 2 + nodeRect.width / 2,
      top: viewport.scrollTop + nodeRect.top - viewportRect.top - viewport.clientHeight / 2 + nodeRect.height / 2,
      behavior
    });
    setSelectedId(currentId || null);
  };

  useEffect(() => {
    if (!currentId) return;
    const frame = window.requestAnimationFrame(() => focusCurrentSubtopic('auto'));
    return () => window.cancelAnimationFrame(frame);
  }, [currentId]);

  return (
    <section className={`knowledge-network${compact ? ' compact' : ''}`} aria-label="知识网络">
      <div className="knowledge-network-toolbar">
        <div className="knowledge-network-legend" aria-label="图例">
          <span className="mastered"><i />已掌握</span>
          <span className="learning"><i />正在学习</span>
          <span className="pending"><i />未掌握 / 未学习</span>
          <span className="foundation"><i />首次填写基础</span>
          <span className="target"><i />当前目标</span>
        </div>
        <strong>15 个模块 · 73 个知识点</strong>
      </div>

      {currentEntry && (
        <div className="knowledge-current-focus">
          <span><LocateFixed size={15} />当前学习</span>
          <strong>{currentEntry.block.block_name}</strong>
          <p>{currentEntry.subtopic.subtopic_name}</p>
          <button type="button" onClick={() => focusCurrentSubtopic()} title="定位当前知识点">
            <LocateFixed size={15} />
            <span>定位</span>
          </button>
        </div>
      )}

      {loadFailed && (
        <p className="knowledge-network-warning"><CircleAlert size={14} />知识状态暂时无法读取，当前仅展示知识结构。</p>
      )}

      <div className="knowledge-network-viewport" ref={viewportRef}>
        <div className="knowledge-network-canvas" style={{ width: KNOWLEDGE_NETWORK_CANVAS.width, height: KNOWLEDGE_NETWORK_CANVAS.height }}>
          <svg className="knowledge-block-links" viewBox={`0 0 ${KNOWLEDGE_NETWORK_CANVAS.width} ${KNOWLEDGE_NETWORK_CANVAS.height}`} aria-hidden="true">
            <defs>
              <marker id="knowledge-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
                <path d="M0,0 L7,3.5 L0,7 Z" />
              </marker>
            </defs>
            {knowledgeGraphBlocks.flatMap((block) => {
              const target = layoutById.get(block.block_id);
              if (!target) return [];
              return block.prerequisites.map((sourceId) => {
                const source = layoutById.get(sourceId);
                if (!source) return null;
                const startX = source.x + 108;
                const startY = source.y + 156;
                const endX = target.x + 108;
                const endY = target.y + 12;
                const midY = (startY + endY) / 2;
                return (
                  <path
                    key={`${sourceId}-${block.block_id}`}
                    d={`M ${startX} ${startY} C ${startX} ${midY}, ${endX} ${midY}, ${endX} ${endY}`}
                    markerEnd="url(#knowledge-arrow)"
                  />
                );
              });
            })}
          </svg>

          {knowledgeGraphBlocks.map((block, blockIndex) => {
            const position = positionForBlock(blockIndex);
            const masteredCount = block.subtopics.filter((item) => statusById.get(item.subtopic_id)?.status === 1).length;
            return (
              <section
                className="knowledge-module"
                key={block.block_id}
                style={{ left: position.x + 8, top: position.y + 14 }}
                aria-labelledby={`knowledge-module-${block.block_id}`}
              >
                <header>
                  <span>{block.block_id}</span>
                  <strong id={`knowledge-module-${block.block_id}`}>{block.block_name}</strong>
                  <small>{masteredCount}/{block.subtopics.length}</small>
                </header>
                <div className="knowledge-subtopic-line">
                  {block.subtopics.map((subtopic, index) => {
                    const apiStatus = statusById.get(subtopic.subtopic_id);
                    const displayStatus = getDisplayStatus(subtopic);
                    const initial = Boolean(apiStatus?.is_initial_foundation);
                    const target = targetIds.has(subtopic.subtopic_id);
                    const classes = [displayStatus, initial ? 'initial-foundation' : '', target ? 'current-target' : ''].filter(Boolean).join(' ');
                    const tooltip = `${subtopic.subtopic_name} · ${statusLabel(displayStatus)}${initial ? ' · 首次填写基础' : ''}${target ? ' · 当前目标' : ''}`;
                    return (
                      <span className="knowledge-node-slot" key={subtopic.subtopic_id}>
                        {index > 0 && <i className="knowledge-subtopic-edge" />}
                        <button
                          type="button"
                          className={`knowledge-subtopic-node ${classes}`}
                          aria-label={tooltip}
                          data-tooltip={tooltip}
                          data-subtopic-id={subtopic.subtopic_id}
                          aria-pressed={selectedId === subtopic.subtopic_id}
                          onClick={() => setSelectedId(subtopic.subtopic_id)}
                        >
                          {displayStatus === 'mastered' && <Check size={10} strokeWidth={3} />}
                        </button>
                      </span>
                    );
                  })}
                </div>
                <p>{block.subtopics.length} 个知识点</p>
              </section>
            );
          })}
        </div>
      </div>

      {selected && (
        <div className="knowledge-node-detail" role="status">
          <span className={`knowledge-detail-state ${getDisplayStatus(selected.subtopic)}`}>
            {statusLabel(getDisplayStatus(selected.subtopic))}
          </span>
          <div>
            <small>{selected.block.block_id} · {selected.subtopic.subtopic_id}</small>
            <strong>{selected.subtopic.subtopic_name}</strong>
            <p>
              掌握度 {Math.round((selectedStatus?.mastery_score || 0) * 100)}%
              {' · '}尝试 {selectedStatus?.attempt_count || 0} 次
              {' · '}前置知识点 {selected.subtopic.prerequisites.join('、') || '无'}
            </p>
          </div>
          <div className="knowledge-detail-tags">
            {selectedStatus?.is_initial_foundation && <span><Check size={12} />首次基础</span>}
            {targetIds.has(selected.subtopic.subtopic_id) && <span><Target size={12} />当前目标</span>}
          </div>
        </div>
      )}
    </section>
  );
}
