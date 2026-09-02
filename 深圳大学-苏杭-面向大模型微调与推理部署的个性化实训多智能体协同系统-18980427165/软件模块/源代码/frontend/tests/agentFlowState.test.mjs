import assert from 'node:assert/strict';
import test from 'node:test';

import {
  createAgentStatusMap,
  finishAgents,
  restoreAgentStatuses,
  startAgents
} from '../node_modules/.cache/agent-flow-tests/utils/agentFlowState.js';

test('a new flow starts with every agent pending', () => {
  assert.deepEqual(createAgentStatusMap(), {
    agent1: 'pending',
    agent3: 'pending',
    agent4: 'pending',
    agent6: 'pending'
  });
});

test('only invoked agents run and successful agents complete', () => {
  const running = startAgents(createAgentStatusMap(), ['agent3', 'agent4']);
  assert.equal(running.agent1, 'pending');
  assert.equal(running.agent3, 'running');
  assert.equal(running.agent4, 'running');
  assert.equal(running.agent6, 'pending');

  const completed = finishAgents(running, ['agent3', 'agent4'], true);
  assert.equal(completed.agent3, 'completed');
  assert.equal(completed.agent4, 'completed');
});

test('failed agents do not receive completion checks', () => {
  const running = startAgents(createAgentStatusMap(), ['agent1']);
  const failed = finishAgents(running, ['agent1'], false);
  assert.equal(failed.agent1, 'failed');
});

test('agents sharing one request complete independently from their own evidence', () => {
  const running = startAgents(createAgentStatusMap(), ['agent3', 'agent4']);
  const contentFinished = finishAgents(running, ['agent3'], true);
  const reviewFailed = finishAgents(contentFinished, ['agent4'], false);
  assert.equal(reviewFailed.agent3, 'completed');
  assert.equal(reviewFailed.agent4, 'failed');
});

test('starting a new stage clears every previous-stage status', () => {
  const historical = createAgentStatusMap(['agent1', 'agent3', 'agent4', 'agent6']);
  const nextFlow = startAgents(historical, ['agent1'], true);
  assert.deepEqual(nextFlow, {
    agent1: 'running',
    agent3: 'pending',
    agent4: 'pending',
    agent6: 'pending'
  });
});

test('session restore never restores a running state', () => {
  const restored = restoreAgentStatuses({
    hasProfile: true,
    hasLearningPackage: true,
    phase: 'learning'
  });
  assert.deepEqual(restored, {
    agent1: 'completed',
    agent3: 'completed',
    agent4: 'completed',
    agent6: 'pending'
  });
  assert.equal(Object.values(restored).includes('running'), false);
});
test('学情评估与动态决策 Agent follows the displayed stage result instead of the session phase', () => {
  const historicalSummary = restoreAgentStatuses({
    hasProfile: true,
    hasLearningPackage: true,
    hasStageResult: true,
    phase: 'learning'
  });
  assert.equal(historicalSummary.agent6, 'completed');

  const newStage = restoreAgentStatuses({
    hasProfile: true,
    hasLearningPackage: true,
    hasStageResult: false,
    phase: 'next_stage_ready'
  });
  assert.equal(newStage.agent6, 'pending');
});
