# AI Agent Trainer - 澶欰gent涓€у寲杈呭绯荤粺

涓€涓熀浜庡Agent鍗忓悓鐨凙I涓€у寲瀛︿範杈呭绯荤粺锛屼笓娉ㄤ簬澶фā鍨嬪井璋冧笌鎺ㄧ悊閮ㄧ讲棰嗗煙鐨勬暀瀛︺€?

## 鏍稿績鐗规€?

- **涓€у寲鐢诲儚**锛氶€氳繃瀵硅瘽寮忎氦浜掔敓鎴愮敤鎴峰涔犵敾鍍?
- **闃舵寮忓涔?*锛氭寜鐭ヨ瘑鏉垮潡渚濊禆銆佹澘鍧椾笅瀛愮煡璇嗙偣椤哄簭閫愰樁娈垫帹閫佸涔犲唴瀹癸紝鍔ㄦ€佽皟鏁村涔犺矾寰?
- **瀹炴椂绛旂枒**锛氬涔犺繃绋嬩腑闅忔椂鎻愰棶锛岀郴缁熷嵆鏃跺洖绛?
- **鍙岄噸瀹℃牳**锛氭妧鏈鏍?璐ㄩ噺璇勫锛岀‘淇濆涔犲唴瀹瑰噯纭€у拰鏁欏鏁堟灉
- **绛栫暐璋冩暣**锛氭牴鎹祴璇曠粨鏋滆嚜鍔ㄨ皟鏁翠笅涓€闃舵瀛︿範绛栫暐
- **鍙鍖栨姤鍛?*锛氬疄鏃跺睍绀哄涔犺繘搴﹀拰鑳藉姏鍙樺寲

## 鎶€鏈爤

- **鍚庣**锛欶astAPI, Python 3.12+
- **鏁版嵁搴?*锛歋QLite, Chroma Vector DB, NetworkX
- **LLM**锛歄llama鏈湴妯″瀷 (deepseek-r1 / qwen2.5)
- **鍓嶇**锛歊eact + TypeScript

鍚庣榛樿浣跨敤 `EXECUTION_MODE=hybrid`锛氱姸鎬佺鐞嗐€侀樁娈佃鍒掑拰绛旈鍒ゅ畾淇濇寔纭畾鎬э紝鏁欑▼涓庨棶绛旂敱 Ollama 鐢熸垚銆傝缃负 `deterministic` 鍚庯紝鏃犻渶鍚姩 Ollama 涔熷彲浠ヨ窇閫氬畬鏁撮棴鐜€?

## 椤圭洰缁撴瀯

```
project-root/
鈹溾攢鈹€ backend/          # 鍚庣鏈嶅姟
鈹溾攢鈹€ frontend/         # 鍓嶇鐣岄潰
鈹溾攢鈹€ data/             # 鏁版嵁瀛樺偍
鈹?  鈹溾攢鈹€ seeds/        # 缁撴瀯鍖栫瀛愭暟鎹紙5涓狫SON鏂囦欢锛?
鈹?  鈹溾攢鈹€ sqlite_data/  # 涓籗QLite鏁版嵁搴?project.db
鈹?  鈹溾攢鈹€ chroma_db/    # FAQ/鐭ヨ瘑妫€绱㈠悜閲忓簱
鈹?  鈹溾攢鈹€ graph_data/   # 鐭ヨ瘑渚濊禆鍥捐氨缁撴灉灞傦紙current/runtime锛?
鈹?  鈹斺攢鈹€ test_data/    # 娴嬭瘯鏍蜂緥鏁版嵁
鈹溾攢鈹€ scripts/          # 宸ュ叿鑴氭湰
鈹斺攢鈹€ docs/             # 椤圭洰鏂囨。
```

## 蹇€熷紑濮?

### 鐜瑕佹眰

- Python 3.12+
- Ollama (鏈湴妯″瀷鎺ㄧ悊)
- Node.js 18+ (鍓嶇)

### 瀹夎渚濊禆

```bash
pip install -r requirements.txt
```

### 閰嶇疆鐜鍙橀噺

```bash
cp .env.example .env
# 缂栬緫 .env 鏂囦欢
```

### 鍒濆鍖栨暟鎹?

```bash
python scripts/init_db.py
python scripts/insert_knowledge_data.py
python scripts/build_knowledge_graph.py
python scripts/init_faq_vectors.py
```

濡傛灉褰撳墠涓嶄緷璧?FAQ 妫€绱紝鍙互鍏堣烦杩?`python scripts/init_faq_vectors.py`銆?

褰撳墠鏁版嵁瀵煎叆鑴氭湰鍙鍙?`data/seeds/` 涓嬬殑 5 涓粨鏋勫寲鏂囦欢锛?

- `knowledge_blocks.json`
- `knowledge_subtopics.json`
- `knowledge_docs.json`
- `faq_items.json`
- `test_questions.json`

## 鏁版嵁搴撶害瀹?

- 褰撳墠鍞竴涓?SQLite 搴撲负 `data/sqlite_data/project.db`
- `project.db` 鏄繍琛屾椂鍞竴鐪熺浉婧愶紝鍖呭惈鐭ヨ瘑鏉垮潡銆佸瓙鐭ヨ瘑鐐广€佺煡璇嗘枃妗ｃ€丗AQ銆侀搴撱€佷細璇濅笌鐘舵€佸揩鐓?
- 鍘嗗彶搴?`data/knowledge.db` 涓嶅啀鍙備笌杩愯閾捐矾锛屼粎淇濈暀澶囦唤褰掓。鐢ㄩ€?
- 鎵€鏈夊悗绔唬鐮併€佸垵濮嬪寲鑴氭湰銆丗AQ 鍚戦噺鍖栬剼鏈粺涓€璇诲彇 `DATABASE_URL`
- 鎵€鏈?Chroma 鏁版嵁缁熶竴璇诲彇 `CHROMA_PATH`
- 鍥捐氨杩愯鏃舵枃浠剁粺涓€璇诲彇 `GRAPH_PATH`锛屽綋鍓嶉粯璁ゆ寚鍚?`data/graph_data/runtime/knowledge_graph.gpickle`
- `data/seeds/knowledge_blocks.json` 涓?`data/seeds/knowledge_subtopics.json` 浠嶇劧鏄浘璋辩殑鏉冨▉婧愭暟鎹?
- `data/graph_data/current/` 瀛樻斁鍙鏌ョ殑鍥捐氨缁撴灉锛歚nodes.jsonl`銆乣edges.jsonl`銆乣manifest.json` 鍙婃淳鐢熺粨鏋?
- `data/graph_data/runtime/` 瀛樻斁杩愯鏃跺浘缂撳瓨 `knowledge_graph.gpickle`
- Agent1 褰撳墠閫氳繃 `graph_data` 鏌ヨ鏉垮潡鍓嶇疆渚濊禆
- Agent2 褰撳墠鎸夋澘鍧椾緷璧栧拰鏉垮潡鍐?`subtopics.sequence_index` 缁勭粐闃舵杈撳叆锛屽苟杩斿洖鏉垮潡涓嬬殑鐭ヨ瘑鐐瑰垪琛?
- Agent3 褰撳墠鎸?`knowledge_docs` 妫€绱㈡暀绋嬩笌绛旂枒鍐呭锛沗knowledge_docs.subtopic_id` 鐢ㄤ簬灏嗘枃妗ｇ簿纭叧鑱斿埌瀛愮煡璇嗙偣
- Agent6 褰撳墠娑堣垂鐢诲儚涓殑 `knowledge_gaps.prerequisites` 鍋氱瓥鐣ュ垽鏂紱璇ュ瓧娈甸€氬父鐢卞墠缃浘璋辨煡璇㈢粨鏋滃～鍏?

## FAQ 闂瓟閾捐矾
- Agent3 浠呭湪鐢ㄦ埛鎻愰棶鏃惰皟鐢?FAQ 妫€绱紝涓嶅湪鏁欑▼鐢熸垚闃舵璋冪敤
- FAQ 妫€绱娇鐢ㄢ€滃畬鏁撮棶棰樼洿鏌ュ悜閲忓簱鈥濇柟妗堬紝涓嶅厛璋冪敤 LLM 鎻愬彇鍏抽敭璇?
- FAQ 妫€绱㈣繑鍥炲€欓€夊悗锛屽彧鏈夋渶楂樼浉浼煎害杈惧埌闃堝€兼椂鎵嶇洿鎺ヨ繑鍥?FAQ 鏍囧噯绛旀
- FAQ 鏈懡涓垨鏈€楂樼浉浼煎害浣庝簬闃堝€兼椂锛岀洿鎺ュ洖閫€鍒扮煡璇嗗簱妫€绱笌 LLM 鏅€氱瓟鐤?
- FAQ 鍚戦噺鏁版嵁鏉ユ簮浜?`faq_items`锛岄€氳繃 `python scripts/init_faq_vectors.py` 鍐欏叆 Chroma 鐨?`faq_collection`

## PowerShell 璋冭瘯娴佺▼

```powershell
# 1. 鍒涘缓 Session
$session = Invoke-RestMethod -Method POST -Uri http://localhost:8500/api/session/
$sessionId = $session.session_id
$sessionId

# 2. 鍒濆鍖栫敤鎴风敾鍍?+ 鐢熸垚褰撳墠闃舵
$body = "{`"session_id`": `"$sessionId`", `"message`": `"鎴戞兂瀛︿範澶фā鍨嬪井璋冿紝鎴戞湁Python鍜孡inux鍩虹锛屾瘡澶╁彲浠ュ涔?灏忔椂锛屽亸瀹炴搷鍨媊"}"
$bytes = [System.Text.Encoding]::UTF8.GetBytes($body)
Invoke-RestMethod -Method POST -Uri http://localhost:8500/api/chat/ -Body $bytes -ContentType "application/json; charset=utf-8"

# 3. 鍚姩褰撳墠闃舵锛岀敓鎴愭暀绋嬪苟瑙﹀彂 Agent4/5 瀹℃牳
$body = "{`"session_id`": `"$sessionId`"}"
$bytes = [System.Text.Encoding]::UTF8.GetBytes($body)
Invoke-RestMethod -Method POST -Uri http://localhost:8500/api/stage/start -Body $bytes -ContentType "application/json; charset=utf-8"

# 4. 鍦ㄥ綋鍓嶉樁娈靛唴鎻愰棶
$body = "{`"session_id`": `"$sessionId`", `"message`": `"浠€涔堟槸LoRA寰皟锛焋"}"
$bytes = [System.Text.Encoding]::UTF8.GetBytes($body)
Invoke-RestMethod -Method POST -Uri http://localhost:8500/api/chat/question -Body $bytes -ContentType "application/json; charset=utf-8"

# 5. 鏌ョ湅褰撳墠 Session 鐘舵€?
Invoke-RestMethod -Method GET -Uri "http://localhost:8500/api/session/$sessionId"

# 6. 濡傛灉瑕佹彁浜ら鐩瓟妗?
$questionId = "浣犵殑棰樼洰ID"
$body = "{`"session_id`": `"$sessionId`", `"question_id`": `"$questionId`", `"user_answer`": `"LoRA`"}"
$bytes = [System.Text.Encoding]::UTF8.GetBytes($body)
Invoke-RestMethod -Method POST -Uri http://localhost:8500/api/stage/submit-answer -Body $bytes -ContentType "application/json; charset=utf-8"

# 7. 瀹屾垚褰撳墠闃舵锛岃Е鍙?Agent6 鍐崇瓥
Invoke-RestMethod -Method POST -Uri "http://localhost:8500/api/stage/complete/$sessionId"
```

### 鍚姩鏈嶅姟

```bash
cd project-root
pip install -r requirements.txt
python scripts/init_db.py
python scripts/insert_knowledge_data.py
python scripts/build_knowledge_graph.py
python scripts/init_faq_vectors.py
cd backend
python main.py
```

### 杩愯纭畾鎬ч棴鐜祴璇?

```bash
python -m unittest discover -s tests -v
```

## Agent架构

| Agent | 职责 | 核心功能 |
|-------|------|----------|
| Agent1 | 学情诊断 | 生成用户画像 JSON，并查询知识图谱前置依赖 |
| Agent2 | 路径规划 | 基于 `subtopic_status` 与知识点前置依赖选择下一学习知识点 |
| Agent3 | 内容生成 | 生成学习包、回答问题 |
| Agent4 | 审核代理 | 统一承担技术审核与质量评审 |
| Agent6 | 阶段决策 | 更新画像、选择策略，并驱动下一阶段或重复当前知识点 |

## 学习流程

1. 用户首次输入 → Agent1 生成画像
2. Agent2 根据知识点级 DAG 与 `subtopic_status` 生成下一阶段
3. Agent3 生成学习包 → Agent4 执行统一审核
4. 用户学习 + 实时答疑
5. 完成测试 → Agent6 更新画像
6. 若当前知识点通过，则 Agent2 选择下一个可学习知识点；未通过则重复当前知识点
7. 循环直到所有知识盲区掌握


## DAG + mastery state锛?026-07-30澧炶ˉ锛?

- 褰撳墠闃舵閫夎矾宸蹭粠鈥滃己绾挎€ф澘鍧楁帹杩涒€濆垏鍒扳€滅煡璇嗙偣绾х姸鎬佹帹杩涒€?
- 杩愯鏃舵柊澧?`subtopic_status` 琛紝浣滀负 Agent2 鐨勭湡瀹炵姸鎬佹簮
- 鐘舵€佽涔夛細`-1` 鏈帉鎻★紝`0` 瀛︿範涓紝`1` 宸叉帉鎻?
- Agent2 姣忔鍙仛涓夋锛?
  1. 鎵惧嚭鎵€鏈?`status = -1` 鐨勭煡璇嗙偣
  2. 杩囨护鍑哄墠缃煡璇嗙偣閮藉凡 `= 1` 鐨勫€欓€夐」
  3. 浠庡€欓€夐」涓€変竴涓渶閫傚悎褰撳墠鐢ㄦ埛鐨勭煡璇嗙偣
- `knowledge_gaps` 缁х画淇濈暀锛屼絾鐜板湪鏇撮€傚悎浣滀负鏉垮潡绾ф眹鎬昏鍥?
- 褰撳墠鐗堟湰涓嶅寘鍚洖閫€绛栫暐锛涢樁娈靛け璐ユ椂淇濇寔褰撳墠鐭ヨ瘑鐐?`0`锛岄噸澶嶅綋鍓嶇煡璇嗙偣
