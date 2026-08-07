import { useEffect, useMemo, useRef, useState } from 'react';
import { Check, CircleAlert, Loader2, LocateFixed, Map as MapIcon, Target } from 'lucide-react';
import { apiClient, isUiPreviewMode, type KnowledgeNetworkSubtopicStatus } from '../api/client';
import { knowledgeGraphBlocks, type KnowledgeGraphSubtopic } from '../data/knowledgeGraph';
import { buildKnowledgeNetworkLayout, type KnowledgeRouteEdge } from '../utils/knowledgeNetworkLayout';

type KnowledgeGapLike = { target_subtopic_ids?: string[] };
type NetworkMode = 'overview' | 'focus';

type KnowledgeNetworkProps = {
  profile?: { knowledge_gaps?: KnowledgeGapLike[] } | null;
  stage?: { subtopic_id?: string; subtopic_name?: string } | null;
  compact?: boolean;
  defaultMode?: NetworkMode;
  showNodeDetails?: boolean;
};

type DisplayStatus = 'mastered' | 'learning' | 'pending';

function statusLabel(status: DisplayStatus) {
  if (status === 'mastered') return '已掌握';
  if (status === 'learning') return '正在学习';
  return '未学习';
}

function shortLabel(label: string) {
  return label.length > 10 ? `${label.slice(0, 9)}…` : label;
}

function edgePath(edge: KnowledgeRouteEdge, positions: Map<string, { x: number; y: number; block_id: string }>) {
  const source = positions.get(edge.source_id);
  const target = positions.get(edge.target_id);
  if (!source || !target) return '';
  if (source.block_id === target.block_id) {
    const bend = Math.max(18, Math.abs(target.x - source.x) * 0.35);
    return `M ${source.x} ${source.y} C ${source.x + bend} ${source.y}, ${target.x - bend} ${target.y}, ${target.x} ${target.y}`;
  }
  const span = Math.max(55, (target.x - source.x) * 0.46);
  return `M ${source.x} ${source.y} C ${source.x + span} ${source.y}, ${target.x - span} ${target.y}, ${target.x} ${target.y}`;
}

function collectAncestors(ids: Iterable<string>, reverseEdges: Map<string, string[]>) {
  const found = new Set<string>();
  const queue = [...ids];
  while (queue.length) {
    const id = queue.shift()!;
    if (found.has(id)) continue;
    found.add(id);
    queue.push(...(reverseEdges.get(id) || []));
  }
  return found;
}

export default function KnowledgeNetwork({
  profile,
  stage,
  compact = false,
  defaultMode = 'overview',
  showNodeDetails = true
}: KnowledgeNetworkProps) {
  const [statuses, setStatuses] = useState<KnowledgeNetworkSubtopicStatus[]>([]);
  const [loading, setLoading] = useState(!isUiPreviewMode);
  const [loadFailed, setLoadFailed] = useState(false);
  const [reloadKey, setReloadKey] = useState(0);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [mode, setMode] = useState<NetworkMode>(defaultMode);
  const viewportRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (isUiPreviewMode) {
      setLoading(false);
      setLoadFailed(false);
      return;
    }
    let active = true;
    setLoading(true);
    setLoadFailed(false);
    apiClient.getKnowledgeNetwork()
      .then((response) => {
        if (!active) return;
        setStatuses(response.blocks.flatMap((block) => block.subtopics || []));
        setLoading(false);
      })
      .catch(() => {
        if (!active) return;
        setStatuses([]);
        setLoadFailed(true);
        setLoading(false);
      });
    return () => { active = false; };
  }, [reloadKey]);

  const layout = useMemo(() => buildKnowledgeNetworkLayout(knowledgeGraphBlocks), []);
  const positionById = useMemo(() => new Map(layout.nodes.map((node) => [node.subtopic_id, node])), [layout]);
  const blockAreaById = useMemo(() => new Map(layout.blocks.map((block) => [block.block_id, block])), [layout]);
  const statusById = useMemo(() => new Map(statuses.map((item) => [item.subtopic_id, item])), [statuses]);
  const targetIds = useMemo(
    () => new Set((profile?.knowledge_gaps || []).flatMap((gap) => gap.target_subtopic_ids || [])),
    [profile]
  );
  const allSubtopics = useMemo(
    () => knowledgeGraphBlocks.flatMap((block) => block.subtopics.map((subtopic) => ({ block, subtopic }))),
    []
  );
  const currentId = stage?.subtopic_id
    || allSubtopics.find((item) => item.subtopic.subtopic_name === stage?.subtopic_name)?.subtopic.subtopic_id
    || [...targetIds][0];
  const currentEntry = currentId ? allSubtopics.find((item) => item.subtopic.subtopic_id === currentId) : undefined;
  const selected = selectedId ? allSubtopics.find((item) => item.subtopic.subtopic_id === selectedId) : undefined;
  const selectedStatus = selected ? statusById.get(selected.subtopic.subtopic_id) : undefined;

  const reverseEdges = useMemo(() => {
    const map = new Map<string, string[]>();
    layout.edges.forEach((edge) => map.set(edge.target_id, [...(map.get(edge.target_id) || []), edge.source_id]));
    return map;
  }, [layout]);
  const forwardEdges = useMemo(() => {
    const map = new Map<string, string[]>();
    layout.edges.forEach((edge) => map.set(edge.source_id, [...(map.get(edge.source_id) || []), edge.target_id]));
    return map;
  }, [layout]);
  const focusIds = useMemo(() => {
    const relevant = collectAncestors(currentId ? [currentId] : [], reverseEdges);
    collectAncestors(targetIds, reverseEdges).forEach((id) => relevant.add(id));
    targetIds.forEach((id) => relevant.add(id));
    if (currentId) (forwardEdges.get(currentId) || []).forEach((id) => relevant.add(id));
    return relevant;
  }, [currentId, forwardEdges, reverseEdges, targetIds]);

  const getDisplayStatus = (subtopic: KnowledgeGraphSubtopic): DisplayStatus => {
    if (subtopic.subtopic_id === currentId) return 'learning';
    const status = statusById.get(subtopic.subtopic_id)?.status;
    if (status === 1) return 'mastered';
    if (status === 0) return 'learning';
    return 'pending';
  };

  const centerCurrentSubtopic = (behavior: ScrollBehavior = 'smooth') => {
    const viewport = viewportRef.current;
    const node = currentId
      ? viewport?.querySelector<SVGGraphicsElement>(`[data-subtopic-id="${currentId}"]`)
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
    if (mode !== 'focus' || !currentId) return;
    const frame = window.requestAnimationFrame(() => centerCurrentSubtopic('auto'));
    return () => window.cancelAnimationFrame(frame);
  }, [currentId, mode]);

  const selectMode = (nextMode: NetworkMode) => {
    setMode(nextMode);
    if (nextMode === 'focus') window.requestAnimationFrame(() => centerCurrentSubtopic());
  };

  return (
    <section className={`knowledge-network ${mode}${compact ? ' compact' : ''}`} aria-label="知识网络">
      <div className="knowledge-network-toolbar">
        <div className="knowledge-network-legend" aria-label="图例">
          <span className="mastered"><i />已掌握</span>
          <span className="learning"><i />正在学习</span>
          <span className="pending"><i />未掌握 / 未学习</span>
          <span className="foundation"><i />首次填写基础</span>
          <span className="target"><i />当前目标</span>
        </div>
        <div className="knowledge-network-actions" aria-label="图谱显示模式">
          <button type="button" className={mode === 'overview' ? 'active' : ''} onClick={() => selectMode('overview')}>
            <MapIcon size={14} />查看完整图谱
          </button>
          <button type="button" className={mode === 'focus' ? 'active' : ''} onClick={() => selectMode('focus')} disabled={!currentId}>
            <LocateFixed size={14} />聚焦当前位置
          </button>
        </div>
      </div>

      {currentEntry && (
        <div className="knowledge-current-focus">
          <span><LocateFixed size={15} />当前学习</span>
          <strong>{currentEntry.block.block_name}</strong>
          <p>{currentEntry.subtopic.subtopic_name}</p>
          <button type="button" onClick={() => selectMode('focus')} title="定位当前知识点">
            <LocateFixed size={15} /><span>定位</span>
          </button>
        </div>
      )}

      {loading && <p className="knowledge-network-status"><Loader2 className="spin" size={14} />正在读取知识点状态</p>}
      {!loading && !loadFailed && statuses.length === 0 && (
        <p className="knowledge-network-status empty">暂无知识点状态，当前仅展示知识结构。</p>
      )}
      {loadFailed && (
        <div className="knowledge-network-warning">
          <span><CircleAlert size={14} />知识状态读取失败，当前仅展示知识结构。</span>
          <button type="button" onClick={() => setReloadKey((value) => value + 1)}>重新读取</button>
        </div>
      )}

      <div className="knowledge-network-viewport" ref={viewportRef}>
        <svg
          className="knowledge-route-map"
          viewBox={`0 0 ${layout.width} ${layout.height}`}
          style={{ width: layout.width, height: layout.height }}
          role="img"
          aria-label="15 个模块、73 个知识点及其前置关系组成的知识航线图"
        >
          <g className="knowledge-regions">
            {knowledgeGraphBlocks.map((block, index) => {
              const area = blockAreaById.get(block.block_id)!;
              const relevant = block.subtopics.some((subtopic) => focusIds.has(subtopic.subtopic_id));
              const masteredCount = block.subtopics.filter((item) => statusById.get(item.subtopic_id)?.status === 1).length;
              return (
                <g className={`knowledge-region tone-${index % 4}${mode === 'focus' && !relevant ? ' is-muted' : ''}`} key={block.block_id}>
                  <title>{block.block_name}</title>
                  <rect x={area.x} y={area.y} width={area.width} height={area.height} rx="24" />
                  <text className="knowledge-region-code" x={area.x + 20} y={area.y + 27}>{block.block_id}</text>
                  <text className="knowledge-region-title" x={area.x + 20} y={area.y + 48}>{shortLabel(block.block_name)}</text>
                  <text className="knowledge-region-count" x={area.x + area.width - 18} y={area.y + 28} textAnchor="end">
                    {masteredCount}/{block.subtopics.length}
                  </text>
                </g>
              );
            })}
          </g>

          <g className="knowledge-route-edges" aria-hidden="true">
            {layout.edges.map((edge) => {
              const relevant = focusIds.has(edge.source_id) && focusIds.has(edge.target_id);
              return (
                <path
                  key={`${edge.source_id}-${edge.target_id}`}
                  className={`${edge.kind}${mode === 'focus' && !relevant ? ' is-muted' : ''}`}
                  d={edgePath(edge, positionById)}
                />
              );
            })}
          </g>

          <g className="knowledge-route-nodes">
            {allSubtopics.map(({ subtopic }) => {
              const position = positionById.get(subtopic.subtopic_id)!;
              const apiStatus = statusById.get(subtopic.subtopic_id);
              const displayStatus = getDisplayStatus(subtopic);
              const initial = Boolean(apiStatus?.is_initial_foundation);
              const target = targetIds.has(subtopic.subtopic_id);
              const current = subtopic.subtopic_id === currentId;
              const relevant = focusIds.has(subtopic.subtopic_id);
              const classes = [
                'knowledge-route-node', displayStatus,
                initial ? 'initial-foundation' : '', target ? 'current-target' : '',
                current ? 'is-current' : '', mode === 'focus' && !relevant ? 'is-muted' : ''
              ].filter(Boolean).join(' ');
              const tooltip = `${subtopic.subtopic_name} · ${statusLabel(displayStatus)}${initial ? ' · 首次填写基础' : ''}${target ? ' · 当前目标' : ''}`;
              return (
                <g
                  className={classes}
                  key={subtopic.subtopic_id}
                  data-subtopic-id={subtopic.subtopic_id}
                  role="button"
                  tabIndex={0}
                  aria-label={tooltip}
                  aria-pressed={selectedId === subtopic.subtopic_id}
                  onClick={() => setSelectedId(subtopic.subtopic_id)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' || event.key === ' ') setSelectedId(subtopic.subtopic_id);
                  }}
                >
                  <title>{tooltip}</title>
                  {current && <circle className="knowledge-current-pulse" cx={position.x} cy={position.y} r="25" />}
                  {current && <circle className="knowledge-current-ring" cx={position.x} cy={position.y} r="16" />}
                  {initial && <circle className="knowledge-foundation-ring" cx={position.x} cy={position.y} r="12" />}
                  {target && <circle className="knowledge-target-ring" cx={position.x} cy={position.y} r="9.5" />}
                  <circle className="knowledge-station" cx={position.x} cy={position.y} r={current ? 9 : 6} />
                  {displayStatus === 'mastered' && <Check x={position.x - 4} y={position.y - 4} size={8} strokeWidth={3} />}
                  <text
                    className="knowledge-station-label"
                    x={position.x}
                    y={position.y - (current ? 20 : 14)}
                    textAnchor="middle"
                  >
                    {shortLabel(subtopic.subtopic_name)}
                  </text>
                  {current && (
                    <>
                      <line className="knowledge-current-leader" x1={position.x} y1={position.y + 11} x2={position.x} y2={position.y + 20} />
                      <g className="knowledge-you-are-here" transform={`translate(${position.x - 32} ${position.y + 20})`}>
                        <rect width="64" height="22" rx="11" />
                        <text x="32" y="15" textAnchor="middle">你在这里</text>
                      </g>
                    </>
                  )}
                </g>
              );
            })}
          </g>
        </svg>
      </div>

      {showNodeDetails && selected && (
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
