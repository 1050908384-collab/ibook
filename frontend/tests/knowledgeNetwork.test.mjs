import assert from 'node:assert/strict';
import test from 'node:test';
import { knowledgeGraphBlocks } from '../node_modules/.cache/agent-flow-tests/data/knowledgeGraph.js';
import { buildKnowledgeNetworkLayout } from '../node_modules/.cache/agent-flow-tests/utils/knowledgeNetworkLayout.js';

test('知识网络包含 15 个模块和 73 个知识点', () => {
  assert.equal(knowledgeGraphBlocks.length, 15);
  assert.equal(knowledgeGraphBlocks.flatMap((block) => block.subtopics).length, 73);
});

test('知识图谱关系数量与一致性审查报告保持一致', () => {
  const blockBySubtopicId = new Map(
    knowledgeGraphBlocks.flatMap((block) =>
      block.subtopics.map((subtopic) => [subtopic.subtopic_id, block.block_id])
    )
  );
  const blockPrerequisiteCount = knowledgeGraphBlocks.reduce(
    (count, block) => count + block.prerequisites.length,
    0
  );
  const subtopicPrerequisites = knowledgeGraphBlocks.flatMap((block) =>
    block.subtopics.flatMap((subtopic) =>
      subtopic.prerequisites.map((prerequisite) => ({
        prerequisite,
        targetBlockId: block.block_id
      }))
    )
  );
  const crossBlockPrerequisiteCount = subtopicPrerequisites.filter(
    ({ prerequisite, targetBlockId }) => blockBySubtopicId.get(prerequisite) !== targetBlockId
  ).length;

  assert.equal(blockPrerequisiteCount, 21);
  assert.equal(subtopicPrerequisites.length, 87);
  assert.equal(crossBlockPrerequisiteCount, 30);
  assert.deepEqual(
    knowledgeGraphBlocks.find((block) => block.block_id === 'KB-02')?.prerequisites,
    []
  );
});

test('每个知识点前置关系都指向现有知识点', () => {
  const ids = new Set(knowledgeGraphBlocks.flatMap((block) => block.subtopics.map((item) => item.subtopic_id)));
  for (const block of knowledgeGraphBlocks) {
    for (const subtopic of block.subtopics) {
      for (const prerequisite of subtopic.prerequisites) {
        assert.ok(ids.has(prerequisite), `${subtopic.subtopic_id} 的前置知识点 ${prerequisite} 不存在`);
      }
    }
  }
});

test('知识航线布局稳定且 73 个站点没有坐标重叠', () => {
  const first = buildKnowledgeNetworkLayout(knowledgeGraphBlocks);
  const second = buildKnowledgeNetworkLayout(knowledgeGraphBlocks);
  assert.deepEqual(first, second);
  assert.equal(first.blocks.length, 15);
  assert.equal(first.nodes.length, 73);
  assert.equal(new Set(first.nodes.map(({ x, y }) => `${x}:${y}`)).size, 73);
});

test('每条知识点前置关系都生成航线', () => {
  const layout = buildKnowledgeNetworkLayout(knowledgeGraphBlocks);
  const edgeKeys = new Set(layout.edges.map((edge) => `${edge.source_id}:${edge.target_id}`));
  for (const block of knowledgeGraphBlocks) {
    for (const subtopic of block.subtopics) {
      for (const prerequisite of subtopic.prerequisites) {
        assert.ok(edgeKeys.has(`${prerequisite}:${subtopic.subtopic_id}`));
      }
    }
  }
});

test('跨模块航线按学习先后从左向右布局', () => {
  const layout = buildKnowledgeNetworkLayout(knowledgeGraphBlocks);
  const nodeById = new Map(layout.nodes.map((node) => [node.subtopic_id, node]));
  const blockById = new Map(layout.blocks.map((block) => [block.block_id, block]));
  for (const edge of layout.edges) {
    const source = nodeById.get(edge.source_id);
    const target = nodeById.get(edge.target_id);
    if (source.block_id === target.block_id) continue;
    assert.ok(blockById.get(source.block_id).x < blockById.get(target.block_id).x);
  }
});
