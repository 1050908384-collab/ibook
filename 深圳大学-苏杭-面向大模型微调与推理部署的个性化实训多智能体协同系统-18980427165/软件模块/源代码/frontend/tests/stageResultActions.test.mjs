import test from 'node:test';
import assert from 'node:assert/strict';
import {
  isHistoricalStageReview,
  resolveStageResultActionMode
} from '../node_modules/.cache/agent-flow-tests/utils/stageResultActions.js';

test('当前阶段完成后可生成或开始下一阶段', () => {
  assert.equal(resolveStageResultActionMode({
    allMastered: false,
    historicalReview: false,
    needsReinforcement: false,
    hasNextStage: false
  }), 'plan_next');
  assert.equal(resolveStageResultActionMode({
    allMastered: false,
    historicalReview: false,
    needsReinforcement: false,
    hasNextStage: true
  }), 'start_next');
});

test('低正确率阶段保留重新学习并允许继续下一阶段', () => {
  assert.equal(resolveStageResultActionMode({
    allMastered: false,
    historicalReview: false,
    needsReinforcement: true,
    hasNextStage: false
  }), 'reinforce_and_plan');
  assert.equal(resolveStageResultActionMode({
    allMastered: false,
    historicalReview: false,
    needsReinforcement: true,
    hasNextStage: true
  }), 'reinforce_and_start');
});

test('目标全部掌握和历史阶段都只能回顾', () => {
  assert.equal(resolveStageResultActionMode({
    allMastered: true,
    historicalReview: false,
    needsReinforcement: false,
    hasNextStage: false
  }), 'review_only');
  assert.equal(resolveStageResultActionMode({
    allMastered: false,
    historicalReview: true,
    needsReinforcement: true,
    hasNextStage: true
  }), 'review_only');
});

test('只有早于活动阶段的阶段属于历史回顾', () => {
  assert.equal(isHistoricalStageReview(2, 3), true);
  assert.equal(isHistoricalStageReview(3, 3), false);
  assert.equal(isHistoricalStageReview(4, 3), false);
  assert.equal(isHistoricalStageReview(undefined, 3), false);
});