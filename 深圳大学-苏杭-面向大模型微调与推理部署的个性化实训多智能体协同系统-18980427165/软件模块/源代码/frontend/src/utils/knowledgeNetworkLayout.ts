import type { KnowledgeGraphBlock } from '../data/knowledgeGraph';

export type KnowledgeRouteNode = {
  subtopic_id: string;
  block_id: string;
  x: number;
  y: number;
  row: 0 | 1;
};

export type KnowledgeRouteBlock = {
  block_id: string;
  layer: number;
  x: number;
  y: number;
  width: number;
  height: number;
};

export type KnowledgeRouteEdge = {
  source_id: string;
  target_id: string;
  kind: 'subtopic' | 'block-fallback';
};

export type KnowledgeRouteLayout = {
  width: number;
  height: number;
  blocks: KnowledgeRouteBlock[];
  nodes: KnowledgeRouteNode[];
  edges: KnowledgeRouteEdge[];
};

const BLOCK_WIDTH = 320;
const BLOCK_HEIGHT = 194;
const LAYER_GAP = 92;
const LANE_GAP = 34;
const CANVAS_PADDING = 44;

function blockDependencies(blocks: KnowledgeGraphBlock[]) {
  const blockBySubtopic = new Map(
    blocks.flatMap((block) => block.subtopics.map((subtopic) => [subtopic.subtopic_id, block.block_id] as const))
  );
  const dependencies = new Map<string, Set<string>>(
    blocks.map((block) => [block.block_id, new Set(block.prerequisites)])
  );

  for (const block of blocks) {
    for (const subtopic of block.subtopics) {
      for (const prerequisite of subtopic.prerequisites) {
        const sourceBlock = blockBySubtopic.get(prerequisite);
        if (sourceBlock && sourceBlock !== block.block_id) dependencies.get(block.block_id)?.add(sourceBlock);
      }
    }
  }
  return dependencies;
}

function resolveLayers(blocks: KnowledgeGraphBlock[], dependencies: Map<string, Set<string>>) {
  const depthById = new Map<string, number>();
  const visiting = new Set<string>();
  const resolve = (blockId: string): number => {
    const known = depthById.get(blockId);
    if (known !== undefined) return known;
    if (visiting.has(blockId)) return 0;
    visiting.add(blockId);
    const sources = [...(dependencies.get(blockId) || [])];
    const depth = sources.length ? Math.max(...sources.map((source) => resolve(source))) + 1 : 0;
    visiting.delete(blockId);
    depthById.set(blockId, depth);
    return depth;
  };
  blocks.forEach((block) => resolve(block.block_id));
  return depthById;
}

function orderLayers(blocks: KnowledgeGraphBlock[], depthById: Map<string, number>, dependencies: Map<string, Set<string>>) {
  const layers = new Map<number, KnowledgeGraphBlock[]>();
  blocks.forEach((block) => {
    const depth = depthById.get(block.block_id) || 0;
    layers.set(depth, [...(layers.get(depth) || []), block]);
  });

  const orderById = new Map<string, number>();
  [...layers.keys()].sort((a, b) => a - b).forEach((depth) => {
    const ordered = (layers.get(depth) || []).sort((a, b) => {
      const averageSourceOrder = (block: KnowledgeGraphBlock) => {
        const sourceOrders = [...(dependencies.get(block.block_id) || [])]
          .map((source) => orderById.get(source))
          .filter((value): value is number => value !== undefined);
        return sourceOrders.length
          ? sourceOrders.reduce((sum, value) => sum + value, 0) / sourceOrders.length
          : Number.MAX_SAFE_INTEGER;
      };
      return averageSourceOrder(a) - averageSourceOrder(b) || a.block_id.localeCompare(b.block_id);
    });
    ordered.forEach((block, index) => orderById.set(block.block_id, index));
    layers.set(depth, ordered);
  });
  return layers;
}

function nodePositions(block: KnowledgeGraphBlock, area: KnowledgeRouteBlock): KnowledgeRouteNode[] {
  const count = block.subtopics.length;
  const columns = Math.max(1, Math.ceil(count / 2));
  const left = area.x + 31;
  const right = area.x + area.width - 31;
  const columnGap = columns === 1 ? 0 : (right - left) / (columns - 1);

  return block.subtopics.map((subtopic, index) => {
    const secondRow = index >= columns;
    const rowIndex = secondRow ? index - columns : index;
    const column = secondRow ? columns - 1 - rowIndex : rowIndex;
    return {
      subtopic_id: subtopic.subtopic_id,
      block_id: block.block_id,
      x: columns === 1 ? (left + right) / 2 : left + column * columnGap,
      y: area.y + (secondRow ? 146 : 88),
      row: secondRow ? 1 : 0
    };
  });
}

export function buildKnowledgeNetworkLayout(blocks: KnowledgeGraphBlock[]): KnowledgeRouteLayout {
  const dependencies = blockDependencies(blocks);
  const depthById = resolveLayers(blocks, dependencies);
  const layers = orderLayers(blocks, depthById, dependencies);
  const maxLayer = Math.max(0, ...layers.keys());
  const maxLaneCount = Math.max(1, ...[...layers.values()].map((items) => items.length));
  const width = CANVAS_PADDING * 2 + (maxLayer + 1) * BLOCK_WIDTH + maxLayer * LAYER_GAP;
  const height = CANVAS_PADDING * 2 + maxLaneCount * BLOCK_HEIGHT + (maxLaneCount - 1) * LANE_GAP;
  const blocksLayout: KnowledgeRouteBlock[] = [];

  [...layers.entries()].sort(([a], [b]) => a - b).forEach(([layer, layerBlocks]) => {
    const layerHeight = layerBlocks.length * BLOCK_HEIGHT + Math.max(0, layerBlocks.length - 1) * LANE_GAP;
    const startY = (height - layerHeight) / 2;
    layerBlocks.forEach((block, lane) => {
      blocksLayout.push({
        block_id: block.block_id,
        layer,
        x: CANVAS_PADDING + layer * (BLOCK_WIDTH + LAYER_GAP),
        y: startY + lane * (BLOCK_HEIGHT + LANE_GAP),
        width: BLOCK_WIDTH,
        height: BLOCK_HEIGHT
      });
    });
  });

  const areaById = new Map(blocksLayout.map((block) => [block.block_id, block]));
  const nodes = blocks.flatMap((block) => nodePositions(block, areaById.get(block.block_id)!));
  const blockBySubtopic = new Map(nodes.map((node) => [node.subtopic_id, node.block_id]));
  const edges: KnowledgeRouteEdge[] = [];
  const explicitBlockPairs = new Set<string>();

  for (const block of blocks) {
    for (const subtopic of block.subtopics) {
      for (const prerequisite of subtopic.prerequisites) {
        if (!blockBySubtopic.has(prerequisite)) continue;
        edges.push({ source_id: prerequisite, target_id: subtopic.subtopic_id, kind: 'subtopic' });
        const sourceBlock = blockBySubtopic.get(prerequisite);
        if (sourceBlock !== block.block_id) explicitBlockPairs.add(`${sourceBlock}:${block.block_id}`);
      }
    }
  }

  for (const block of blocks) {
    for (const prerequisiteBlockId of block.prerequisites) {
      if (explicitBlockPairs.has(`${prerequisiteBlockId}:${block.block_id}`)) continue;
      const sourceBlock = blocks.find((candidate) => candidate.block_id === prerequisiteBlockId);
      const source = sourceBlock?.subtopics[sourceBlock.subtopics.length - 1];
      const target = block.subtopics[0];
      if (source && target) {
        edges.push({ source_id: source.subtopic_id, target_id: target.subtopic_id, kind: 'block-fallback' });
      }
    }
  }

  return { width, height, blocks: blocksLayout, nodes, edges };
}
