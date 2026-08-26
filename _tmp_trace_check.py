# -*- coding: utf-8 -*-
import json
from pathlib import Path

ROOT = Path('d:/code/ai_agent-trainer')
trace_path = ROOT / '测试数据与评测材料/02_50组个性化主评测/raw_outputs/case_001/stage_1/review_trace.json'
with open(trace_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

hist = data.get('review_history', [])
print(f'轮次数: {len(hist)}')
for r in hist:
    if isinstance(r, dict):
        print(f'  iter={r.get("iteration")}, keys={list(r.keys())}')
        for k in ['technical', 'quality']:
            sub = r.get(k)
            if isinstance(sub, dict):
                errs = sub.get('errors', [])
                perrs = sub.get('personalization_errors', [])
                print(f'    {k}.errors={len(errs)}, personalization_errors={len(perrs)}, passed={sub.get("passed")}')

# 统计全部50个case第1轮的errors
print('\n=== 全局统计: 第1轮初稿 errors ===')
all_first_round_errors = 0
all_last_round_errors = 0
single_round_cases = 0
multi_round_cases = 0
for case_dir in sorted((ROOT / '测试数据与评测材料/02_50组个性化主评测/raw_outputs').iterdir()):
    if not case_dir.is_dir() or not case_dir.name.startswith('case_'):
        continue
    for stage in ['stage_1', 'stage_2']:
        tp = case_dir / stage / 'review_trace.json'
        if not tp.exists():
            continue
        with open(tp, 'r', encoding='utf-8') as f:
            tr = json.load(f)
        h = tr.get('review_history', [])
        if not h:
            continue
        if len(h) == 1:
            single_round_cases += 1
        else:
            multi_round_cases += 1
        # 第1轮errors
        first = h[0]
        for k in ['technical', 'quality']:
            sub = first.get(k, {})
            all_first_round_errors += len(sub.get('errors', []))
        # 末轮errors
        last = h[-1]
        for k in ['technical', 'quality']:
            sub = last.get(k, {})
            all_last_round_errors += len(sub.get('errors', []))

print(f'单轮 case 数: {single_round_cases}')
print(f'多轮 case 数: {multi_round_cases}')
print(f'第1轮 errors 总数(分母): {all_first_round_errors}')
print(f'末轮 errors 总数: {all_last_round_errors}')
