import assert from 'node:assert/strict';
import test from 'node:test';

import {
  buildDiagnosticMessage,
  getFormCompleteness,
  isOnboardingFormComplete
} from '../node_modules/.cache/agent-flow-tests/utils/buildDiagnosticMessage.js';

const targetOnlyForm = {
  grade: '',
  skill_tags: [],
  goal: '完成一个 RAG 项目',
  daily_hours: '2',
  learning_style: 'mixed',
  hardware: ''
};

test('subsequent goals require goal, duration and learning style', () => {
  assert.equal(getFormCompleteness(targetOnlyForm, true), 1);
  assert.equal(isOnboardingFormComplete(targetOnlyForm, true), true);
  assert.equal(isOnboardingFormComplete({ ...targetOnlyForm, learning_style: '' }, true), false);
  assert.equal(isOnboardingFormComplete(targetOnlyForm, false), false);
});

test('a new goal starts empty and cannot appear partially complete', () => {
  const blankTargetForm = {
    ...targetOnlyForm,
    goal: '',
    daily_hours: '',
    learning_style: ''
  };
  assert.equal(getFormCompleteness(blankTargetForm, true), 0);
  assert.equal(isOnboardingFormComplete(blankTargetForm, true), false);
});

test('first-time profile requires only stable profile fields and confirmed foundation', () => {
  const profileOnlyForm = {
    ...targetOnlyForm,
    grade: '后端工程师',
    hardware: 'RTX 4060',
    goal: '',
    daily_hours: ''
  };
  assert.equal(getFormCompleteness(profileOnlyForm, false, true, true), 1);
  assert.equal(isOnboardingFormComplete(profileOnlyForm, false, true, true), true);
  assert.equal(isOnboardingFormComplete(profileOnlyForm, false, false, true), false);
});

test('target-only diagnosis sends learning style but not stable profile fields', () => {
  const message = buildDiagnosticMessage(targetOnlyForm, true);
  assert.match(message, /学习目标/);
  assert.match(message, /每天可以学习/);
  assert.match(message, /学习方式：混合型/);
  assert.doesNotMatch(message, /身份背景|已有基础|设备条件/);
});

test('first goal diagnosis includes the full learner profile', () => {
  const message = buildDiagnosticMessage({
    ...targetOnlyForm,
    grade: '后端工程师',
    skill_tags: ['Python'],
    hardware: 'RTX 4060'
  });
  assert.match(message, /身份背景：后端工程师/);
  assert.match(message, /已有基础：Python/);
  assert.match(message, /设备条件：RTX 4060/);
});
