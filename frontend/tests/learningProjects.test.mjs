import assert from 'node:assert/strict';
import test from 'node:test';

import {
  filterProjectsForDashboard,
  formatProjectCreatedDate,
  sortProjectsForDashboard
} from '../node_modules/.cache/agent-flow-tests/utils/learningProjects.js';

test('当前目标始终排在第一位，其余目标按建立日期由近到远排序', () => {
  const projects = [
    { sessionId: 'old', createdAt: 100, updatedAt: 500 },
    { sessionId: 'current', createdAt: 50, updatedAt: 50 },
    { sessionId: 'new', createdAt: 300, updatedAt: 300 }
  ];

  const result = sortProjectsForDashboard(projects, 'current');
  assert.deepEqual(result.map((project) => project.sessionId), ['current', 'new', 'old']);
  assert.notEqual(result, projects);
});

test('目标缺少建立日期时使用更新时间排序', () => {
  const result = sortProjectsForDashboard([
    { sessionId: 'older', updatedAt: 100 },
    { sessionId: 'newer', updatedAt: 200 }
  ], null);

  assert.deepEqual(result.map((project) => project.sessionId), ['newer', 'older']);
});

test('建立日期格式稳定，缺失日期显示明确空状态', () => {
  assert.equal(formatProjectCreatedDate(Date.UTC(2026, 7, 8)), '2026/08/08');
  assert.equal(formatProjectCreatedDate(undefined), '日期未知');
});

test('学习目标搜索同时匹配目标名称和阶段名称', () => {
  const projects = [
    { sessionId: 'lora', goal: '掌握 LoRA 微调', stageName: '注意力机制', updatedAt: 10 },
    { sessionId: 'python', goal: '学习 Python', stageName: '函数基础', updatedAt: 20 }
  ];

  assert.deepEqual(filterProjectsForDashboard(projects, 'lora').map((item) => item.sessionId), ['lora']);
  assert.deepEqual(filterProjectsForDashboard(projects, '函数').map((item) => item.sessionId), ['python']);
  assert.equal(filterProjectsForDashboard(projects, '不存在').length, 0);
});
