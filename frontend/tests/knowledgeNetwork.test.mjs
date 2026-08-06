import assert from 'node:assert/strict';
import test from 'node:test';
import { knowledgeGraphBlocks } from '../node_modules/.cache/agent-flow-tests/data/knowledgeGraph.js';
import { buildKnowledgeNetworkLayout } from '../node_modules/.cache/agent-flow-tests/utils/knowledgeNetworkLayout.js';

test('知识网络包含 15 个模块和 73 个知识点', () => {
  assert.equal(knowledgeGraphBlocks.length, 15);
  assert.equal(knowledgeGraphBlocks.flatMap((block) => block.subtopics).length, 73);
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

test('知识网络布局稳定且没有坐标重叠', () => {
  const first = buildKnowledgeNetworkLayout(knowledgeGraphBlocks);
  const second = buildKnowledgeNetworkLayout(knowledgeGraphBlocks);
  assert.deepEqual(first, second);
  assert.equal(new Set(first.map(({ x, y }) => `${x}:${y}`)).size, 15);
});
