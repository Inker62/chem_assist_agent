# ChemAssist 测试体系详解

> 本文档面向开发者，逐项解析静态 mock 测试（`tests/`）和动态 API 测试（`eval_agent.py`）的设计思路、覆盖范围与运行方式。

---

## 一、测试体系概览

```
                    ChemAssist 测试体系
                    ┌───────┴───────┐
              静态 Mock 测试      动态 API 测试
              (tests/ 56 项)      (eval_agent.py 11 用例)
                    │                    │
              ┌─────┼─────┐          真实 DeepSeek API
            工具层 节点层 集成层      真实 ChemSpider/CrossRef
                    │
              完全离线，5 秒跑完      需要 API Key，~2 分钟
```

| 维度 | 静态 Mock 测试 | 动态 API 测试 |
|------|---------------|--------------|
| 目的 | 验证代码逻辑正确性 | 验证真实场景端到端表现 |
| 依赖 | 无（Fake + MagicMock） | DeepSeek + ChemSpider + CrossRef API |
| 速度 | ~5 秒（56 项） | ~138 秒（11 用例） |
| 何时运行 | 每次改代码后 | 功能发布前 / 怀疑回归时 |
| 可重复性 | 100% 确定性 | 依赖 API 可用性 |

---

## 二、测试基础设施：`tests/conftest.py`

这是整个测试体系的基石。它解决了三个致命问题：

### 问题 1：源码模块级 `load_dotenv()` 会读取真实 `.env`

5 个生产文件在模块导入时调用 `load_dotenv()`。如果不处理，pytest 收集测试时就会触发。

**解决方案**：conftest.py 模块顶部先设置 `os.environ` 假值，`load_dotenv(override=False)` 不会覆盖已存在的变量。

```python
os.environ['DEEPSEEK_API_KEY'] = 'test-deepseek-key'  # 假值
os.environ['RSC_API_KEY'] = 'test-rsc-key'
```

### 问题 2：`tools/chem_memory.py` 导入时创建 ChromaDB + 下载模型

```python
# chem_memory.py 模块级代码
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # 会下载 80MB 模型！
client = chromadb.PersistentClient(path=PERSIST_DIR)  # 会写磁盘！
```

**解决方案**：在 `sys.modules` 中注入 Fake 模块，让 Python 以为 `chromadb` 和 `sentence_transformers` 已经安装好了。

```python
sys.modules['chromadb'] = _mock_chroma      # MagicMock 包装的 FakeCollection
sys.modules['sentence_transformers'] = _mock_st  # MagicMock 包装的 FakeEmbedder
```

任何 `import chromadb` 都会拿到我们的 MagicMock，不会触发真实的文件 I/O。

### Fake 组件一览

| Fake 类 | 模拟对象 | 用途 |
|---------|---------|------|
| `FakeEmbedding` | numpy 数组 | 提供 `.tolist()` 方法，让 `embedder.encode().tolist()` 不报错 |
| `FakeEmbedder` | SentenceTransformer | 基于 MD5 哈希产生 48 维确定性向量（真实是 384 维） |
| `FakeCollection` | ChromaDB Collection | 纯内存字典实现 `count/get/query/upsert/delete` |

### 共享 Fixtures

| Fixture | 用途 |
|---------|------|
| `reset_memory` | 每个测试前清空 FakeCollection，确保测试隔离 |
| `sample_compound_json` | 模拟 ChemSpider 返回的 Aspirin 数据（id=2157） |
| `sample_crossref_response` | 模拟 CrossRef 返回的文献列表 |

---

## 三、工具层测试（28 项）

### `tests/test_chem_memory.py`（13 项）

测试化学记忆库（ChromaDB 向量存储）的增删查功能。

```
TestFakeCollection (4 项)
├── test_collection_starts_empty          → 新集合 count() == 0
├── test_get_returns_nothing_when_empty   → 空集合 get() 返回空列表
├── test_query_returns_empty_when_empty   → 空集合 query() 返回 [[]]
└── test_delete_nonexistent_does_not_raise → 删除不存在的 ID 不抛异常

TestAddToMemory (3 项)
├── test_add_new_compound_with_csid       → 有 ChemSpider ID 时键为 "csid_2157"
├── test_add_new_compound_without_csid    → 无 ID 时键回退到小写名称
└── test_add_duplicate_updates_existing   → 重复写入触发 upsert，count 不变

TestSearchMemory (2 项)
├── test_returns_none_when_collection_empty → 空库搜索返回 None
└── test_returns_json_string_when_found     → 命中后返回完整 JSON 字符串

TestClearMemory (1 项)
└── test_clear_removes_all_entries        → 清空后 count == 0

TestChemicalMemoryTool (3 项)
├── test_run_returns_json_when_found      → _run("aspirin") 返回含 id=2157 的 JSON
├── test_run_returns_not_found_when_missing → _run("unknown") 返回 "Not Found"
└── test_tool_metadata_is_correct         → name/description/args_schema 正确
```

**设计要点**：测试不关心向量是否真的"相似"——FakeCollection 直接返回所有条目，简化了语义搜索的验证。

### `tests/test_chemspider_search.py`（10 项）

测试 ChemSpider API 工具的两步法查询流程。

```
TestSearchCompoundByName (5 项)
├── test_success_returns_compound_data    → 模拟 4 步 HTTP 调用完整成功
├── test_empty_results_returns_none       → API 返回空结果列表
├── test_query_id_missing_returns_none    → POST 响应缺少 queryId
├── test_status_timeout_returns_none      → 状态轮询 15 次始终不是 "Complete"
└── test_http_error_propagates            → HTTP 异常正确抛出

TestChemSpiderTool (5 项)
├── test_run_success                      → _run 返回 JSON，且调用了 add_to_memory
├── test_run_not_found                    → 未找到时返回含 "未找到" 的提示
├── test_run_http_error                   → HTTP 错误时返回含 "API 请求失败" 的信息
├── test_run_unexpected_error             → 意外异常时返回含 "未预期的错误" 的信息
└── test_tool_metadata_is_correct         → name/description/args_schema 正确
```

**设计要点**：
- `time.sleep(3)` 必须 mock：否则 `test_status_timeout_returns_none` 会等 45 秒
- `requests.post` 和 `requests.get` 分开 mock：因为两步法调用了不同的 HTTP 方法
- `add_to_memory` 被 mock：防止写 FakeCollection 产生副作用

### `tests/test_literature_search.py`（7 项）

测试 CrossRef 文献检索工具的请求参数和响应处理。

```
TestCrossrefSearchTool (7 项)
├── test_run_returns_literature_list      → 正常返回含 title/doi/year 的 JSON 列表
├── test_run_empty_results_returns_empty_list → 无结果时返回 []
├── test_run_respects_max_results_parameter → max_results=10 传递到 API
├── test_run_network_error_returns_error_string → ConnectionError 返回友好提示
├── test_run_http_error_returns_error_string → HTTPError 返回友好提示
├── test_run_missing_fields_uses_defaults → 空字段用 "N/A" 填充（此测试发现了 2 个真实 Bug）
└── test_tool_metadata_is_correct         → name/description/args_schema 正确
```

**发现的生产 Bug**：
- `item.get("title", ["N/A"])[0]` — 当 title 存在但为 `[]` 时 IndexError
- `item.get("DOI", "N/A")` — 当 DOI 存在但为 `None` 时返回 None 而非 "N/A"

---

## 四、多智能体图测试（24 项）

### `tests/test_multiagent.py`（18 项）

这是测试体系的核心——验证 LangGraph 图的每个节点和路由逻辑。

#### TestCleanMessages（3 项）

```
test_returns_empty_for_empty_list         → [] → []
test_keeps_human_and_plain_ai             → HumanMessage 保留，AIMessage(tool_calls) 过滤
test_mixed_bag_filters_toolmessage        → ToolMessage 被过滤
```

验证消息清洗逻辑：只保留 HumanMessage 和不带 tool_calls 的 AIMessage。

#### TestSupervisorNode（8 项）

```
test_decides_chem_agent                   → {"next": "chem_agent"} → 路由到 chem_agent
test_decides_literature_agent             → {"next": "literature_agent"} → 路由到 literature_agent
test_decides_both_returns_route_key       → {"next": "both"} → 返回 both_active=True
test_decides_finish_calls_summary_llm     → {"next": "finish"} → 调用 summary_llm 直接回复
test_max_rounds_forces_end                → round_count=5 → 强制结束
test_invalid_json_returns_error           → 非 JSON 输出 → 错误处理
test_unknown_decision_returns_error       → {"next": "invalid"} → 无法理解
test_finish_llm_error_returns_fallback    → summary_llm 抛异常 → 回退错误消息
```

**Mock 策略**：替换整个 `decision_llm` 和 `summary_llm` 模块变量为 MagicMock，控制 `invoke()` 的返回值。

**关键陷阱**：ChatOpenAI 是 Pydantic frozen model，不能用 `mocker.patch.object(llm, 'invoke', ...)`。必须用 `mocker.patch('agents.multiagent.decision_llm', mock_instance)` 替换整个模块级实例。

#### TestChemAgentNode（2 项）/ TestLiteratureAgentNode（2 项）

```
test_with_tool_calls_routes_to_chem_tools  → AIMessage(tool_calls=[...]) → next="chem_tools"
test_without_tool_calls_routes_to_summary  → AIMessage(content="...") → next="summary"
```

验证专家节点的工具循环路由：有 tool_call 就去执行工具，没有再返回 summary。

#### TestSummaryNode（2 项）

```
test_success_returns_final_message        → summary_llm 正常返回 → [AIMessage]
test_llm_error_returns_fallback           → summary_llm 抛异常 → "抱歉，处理时遇到错误"
```

#### TestBuildWorkflow（1 项）

```
test_has_all_required_nodes               → 图包含全部 6 个节点
```

确保图构建没有遗漏节点。

### `tests/test_multiagent_integration.py`（6 项）

全图端到端测试——用 `invoke()` 完整执行图，验证条件边路由。

```
test_chem_agent_route_to_end              → supervisor → chem_agent → summary → END
test_chem_agent_with_tool_loop            → supervisor → chem_agent → chem_tools → chem_agent → summary → END
test_literature_agent_route_to_end        → supervisor → literature_agent → summary → END
test_finish_directly                      → supervisor(finish) → END（验证专家未被调用）
test_max_rounds_guard_stops_execution     → round_count=5 → 超时错误
test_both_parallel_dispatch               → supervisor → chem → lit（顺序链）
```

**Mock 层级**：

| 层级 | Mock 对象 | 方式 |
|------|----------|------|
| LLM | `decision_llm`, `summary_llm`, `chem_model`, `literature_model` | `mocker.patch('agents.multiagent.XXX', MagicMock(…))` |
| ToolNode | `chem_tool_node` | 替换为 `fake_tool_node()` 函数（返回 ToolMessage） |
| 工具 | ChemSpiderTool, ChemicalMemoryTool | 不执行——由 FakeCollection 拦截 |

**为什么 ToolNode 不能用 MagicMock？** ToolNode 是 `RunnableCallable`，图执行时会调用 `tool_node(state)`。MagicMock 被调用返回新 MagicMock，LangGraph 报 `InvalidUpdateError: Expected dict, got MagicMock`。必须用普通函数。

---

## 五、单 Agent 测试（2 项）

### `tests/test_agent.py`

```
test_initialize_agent_returns_compiled_graph → 返回可 invoke 的编译后对象
test_agent_tools_configured                  → 配置了 ChemicalMemoryTool 和 ChemSpiderTool
```

验证 `initialize_agent()` 正确组装了 LangChain Agent。

---

## 六、动态 API 测试：`eval_agent.py`

### 设计理念

与 mock 测试互补——mock 测试验证"代码写对了"，eval 验证"在真实世界里能用"。

### 测试用例矩阵（11 个）

```
基础化学查询
├── basic-aspirin         查询 aspirin 的 SMILES/分子量
├── basic-caffeine-cn     中文查询 caffeine 结构
└── basic-cache-hit       第二次查询 aspirin（验证记忆缓存命中）

文献检索
├── lit-AIE               查询 AIE 荧光分子文献（≥5 篇）
└── lit-RAFT              查询 RAFT 聚合相关文献

组合查询
└── combined-chem-lit     查询 MMA 化学结构 + 聚合文献（both 顺序链）

边界测试
├── edge-nonexistent      查询不存在的化合物 xyzzy12345
└── edge-chitchat         闲聊 "今天天气怎么样？"（应走 finish 路径）

单 Agent 模式
├── single-acetaminophen  查询对乙酰氨基酚
└── single-graphene-lit   单 Agent 文献检索
```

### 评估指标

| 指标 | 含义 | 判定方式 |
|------|------|---------|
| `expect_tools` | 必须调用了这些工具 | 检查消息中 tool_calls 的 name |
| `expect_keywords` | 输出必须包含这些关键词 | 大小写不敏感的 `in` 匹配 |
| `expect_no_keywords` | 输出不能包含这些词 | 如 "Error"、"抱歉" 表示异常 |

### 运行方式

```bash
# 全部用例（单 + 多 Agent）
.venv/Scripts/python eval_agent.py

# 仅多 Agent
.venv/Scripts/python eval_agent.py --mode multi

# 仅单 Agent
.venv/Scripts/python eval_agent.py --mode single

# 临时查询
.venv/Scripts/python eval_agent.py --query "查询咖啡因的化学结构"
```

### 上次评估结果（2026-05-20）

```
Total: 8 | Passed: 8 | Failed: 0
Time: 138.0s total | 17.3s avg
Pass rate: 8/8 (100%)
```

---

## 七、测试运行速查

```bash
# 全部 mock 测试（56 项，~5 秒）
.venv/Scripts/python -m pytest tests/ -v

# 单个文件
.venv/Scripts/python -m pytest tests/test_multiagent.py -v

# 带覆盖率（需 pip install pytest-cov）
.venv/Scripts/python -m pytest tests/ -v --cov=. --cov-report=term-missing

# 全部动态测试（~2 分钟，需要 API Key）
.venv/Scripts/python eval_agent.py

# 单个动态测试
.venv/Scripts/python eval_agent.py --mode multi --query "aspirin"
```

---

## 八、踩坑汇总

| 问题 | 原因 | 解决 |
|------|------|------|
| `mocker.patch.object(llm, 'invoke')` 报 AttributeError | ChatOpenAI 是 Pydantic frozen model | 改用 `mocker.patch('module.llm_var', mock_instance)` |
| `langgraph.END == '__end__'` 而非 `"END"` | LangGraph 内部常量 | 断言用 `from langgraph.graph import END` |
| ToolNode mock 报 `InvalidUpdateError` | MagicMock 被调用返回 MagicMock 而非 dict | 替换为普通函数 |
| `send` 并行 fan-out 与 `stream(mode="custom")` 冲突 | LangGraph 1.1.10 的 custom stream 不支持 Send 返回值 | 改用 `stream_mode="updates"` |
| 循环导入 `nodes.py ↔ multiagent.py` | 互相 import | nodes.py 用 `_get_mam()` 延迟导入 |
| `replace_all` 误伤 `from config import RSC_BASE_URL` | 文本替换不够精确 | 慎用 replace_all 处理 import 语句 |
