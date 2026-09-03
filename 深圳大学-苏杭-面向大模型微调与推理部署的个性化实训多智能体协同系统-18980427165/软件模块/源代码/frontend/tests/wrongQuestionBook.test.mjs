import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const appSource = fs.readFileSync(path.join(here, '../src/App.tsx'), 'utf8');
const clientSource = fs.readFileSync(path.join(here, '../src/api/client.ts'), 'utf8');

test('错题本入口和视图已接入前端导航', () => {
  assert.match(appSource, /id: 'wrongbook' as View, label: '错题本'/);
  assert.match(appSource, /function WrongQuestionBookView\(/);
  assert.match(appSource, /view === 'wrongbook'/);
});

test('错题本展示目标、阶段、答案和解析信息', () => {
  for (const label of ['选择需要复盘的目标', '测试正确率', '道错题', '你的答案', '正确答案', '解析']) {
    assert.match(appSource, new RegExp(label));
  }
  assert.match(appSource, /stage\.questions\.map/);
});

test('错题本调用按会话查询错题接口', () => {
  assert.match(clientSource, /getWrongQuestions: async \(sessionId: string\)/);
  assert.match(clientSource, /sessions\/\$\{sessionId\}\/wrong-questions/);
});
