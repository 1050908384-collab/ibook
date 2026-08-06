export const KNOWLEDGE_NETWORK_CANVAS = {
  width: 1150,
  height: 620,
  cellWidth: 230,
  cellHeight: 190
} as const;

export function buildKnowledgeNetworkLayout(blocks: Array<{ block_id: string }>) {
  return blocks.map((block, index) => ({
    block_id: block.block_id,
    x: (index % 5) * KNOWLEDGE_NETWORK_CANVAS.cellWidth,
    y: Math.floor(index / 5) * KNOWLEDGE_NETWORK_CANVAS.cellHeight
  }));
}
