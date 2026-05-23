# ChemAssist 架构文档

> 2026-05-23 | 版本 2.0 | 93 项单元测试

## 一、项目概述

ChemAssist 是一个 AI 驱动的化学信息学平台，前端使用 Streamlit，后端基于 LangGraph 多智能体协同架构，LLM 由 DeepSeek V4 驱动。

**核心能力**：化学物质查询 / 文献检索 / 分子结构绘制与标注 / RDKit 分子计算 / 个人知识库 / 流式对话

---

## 二、目录结构

```
ChemAssist/
├── app.py                          # Streamlit 前端主入口
├── config.py                       # 集中配置（API Key/URL/DB 路径）
├── requirements.txt                # Python 依赖
├── eval_agent.py                   # 真实 API 评测脚本（11 用例）
│
├── agents/                         # === Agent 核心模块 ===
│   ├── multiagent.py               # 多 Agent 总入口——LLM/ToolNode 实例
│   ├── state.py                    # MultiAgentState 类型定义
│   ├── nodes.py                    # 6 个图节点函数
│   └── graph.py                    # StateGraph 构建 + 条件路由
│   └── agent.py                    # 单 Agent（旧版——已落后，待移除）
│
├── tools/                          # === 工具注册中心 ===
│   ├── __init__.py                 # Registry: CHEM_TOOLS / LITERATURE_TOOLS
│   ├── chem_memory.py              # 知识库（ChromaDB 语义检索 + 生命周期）
│   ├── chemspider_search.py        # RSC ChemSpider API（名称→化合物数据）
│   ├── pubchem_search.py           # PubChem REST API（SMILES→名称/同义词）
│   ├── chem_calc.py                # RDKit 分子计算 + 结构标注
│   └── literature_search.py        # CrossRef 文献检索
│
├── tests/                          # === 测试（93 项 / 5 秒） ===
│   ├── conftest.py                 # Fake 注入体系
│   ├── test_agent.py               # 2 项
│   ├── test_chem_calc.py           # 11 项
│   ├── test_chem_memory.py         # 20 项
│   ├── test_chemspider_search.py   # 10 项
│   ├── test_image_display.py       # 4 项
│   ├── test_literature_search.py   # 7 项
│   ├── test_multiagent.py          # 19 项
│   ├── test_multiagent_integration.py  # 6 项
│   ├── test_pubchem_search.py      # 9 项
│   └── test_streaming.py           # 5 项
│
├── docs/                           # === 文档 ===
│   ├── architecture.md             # 本文档
│   ├── dev-log-2026-05-20.md       # RDKit 集成
│   ├── dev-log-2026-05-22.md       # 流式输出修复
│   ├── testing-guide.md            # 测试体系详解
│   └── testing-log-2026-05-18.md   # 离线测试搭建
│
└── local_data/                     # 本地持久化（.gitignore）
    ├── checkpoints.db              # LangGraph 会话状态
    ├── model_cache/                # sentence-transformers 模型
    ├── chemical_memory_db/         # ChromaDB 知识库
    └── mol_images/                 # RDKit 2D 结构图
```

---

## 三、系统架构

```
┌──────────────────────────────────────────────────────────────────┐
│                     app.py (Streamlit 前端)                       │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────┐   │
│  │ 对话历史     │  │ Ketcher 绘图 │  │ 侧栏: 知识库浏览器    │   │
│  │ + 流式渲染   │  │ → SMILES     │  │ + 模式/管理/导出       │   │
│  └──────┬───────┘  └──────┬───────┘  └───────────────────────┘   │
│         │                 │                                      │
│         └─────────┬───────┘                                      │
│                   │ HumanMessage(content=prompt)                 │
└───────────────────┼──────────────────────────────────────────────┘
                    │
┌───────────────────┼──────────────────────────────────────────────┐
│                   ▼            multiagent.py                     │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                  LangGraph StateGraph                        │  │
│  │                                                             │  │
│  │   ┌──────────┐                                              │  │
│  │   │supervisor│──→ chem_agent ──→ (tools?) ──→ summary      │  │
│  │   │  _node   │──→ lit_agent  ──→ (tools?) ──→ summary      │  │
│  │   │          │──→ both ──→ chem ──→ lit ──→ summary         │  │
│  │   │          │──→ finish ──→ summary                        │  │
│  │   └──────────┘                                              │  │
│  │        │                                                    │  │
│  │   ┌────┴──────────────────────────────────────┐            │  │
│  │   │ chem_tool_node         │ lit_tool_node    │            │  │
│  │   │ ─────────────────────  │ ──────────────── │            │  │
│  │   │ ChemicalMemory (cache) │ CrossrefSearch   │            │  │
│  │   │ ChemSpiderSearch (API) │                  │            │  │
│  │   │ PubChemSearch (SMILES) │                  │            │  │
│  │   │ ChemCalc (RDKit)       │                  │            │  │
│  │   │ StructureAnnotator     │                  │            │  │
│  │   └────────────────────────┴──────────────────┘            │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  LLM: DeepSeek V4 Flash (via ChatOpenAI adapter)                 │
│  ┌─────────────┬──────────────┬──────────────────────────┐       │
│  │ decision_llm│ summary_llm   │ chem_model │ lit_model   │       │
│  │ (json_obj)  │ (streaming)   │ (bind_tools)             │       │
│  └─────────────┴──────────────┴──────────────────────────┘       │
└──────────────────────────────────────────────────────────────────┘
```

---

## 四、核心调用链路

### 4.1 请求生命周期

```
用户输入 → app.py session_state.append(user_msg)
         → agent.stream({messages}, stream_mode=["updates","messages"])
         → StateGraph.compile() 遍历节点
         → _route_supervisor(state["next"]) 决定路由
         → supervisor_node: DeepSeek 输出 {"next": "chem_agent"|"literature_agent"|"both"|"finish"}
         → expert nodes: LLM 决定 tool_calls → ToolNode 执行 → 循环
         → summary_node: 整理专家结果 → _stream_summary() → token 流
         → app.py: stream mode "messages" 逐 token → response_container.markdown()
```

### 4.2 化学查询时的 Tool 调用链

```
ChemicalMemory.search(query)          ← 第1步：查本地知识库
  ├── 命中 (score ≥ 0.4) → 返回缓存数据
  └── 未命中
       └→ ChemSpiderSearch.run(name)  ← 第2步：名称检索
       └→ PubChemSearch.run(smiles)   ← 第2步备选：SMILES→名称
          └→ ChemCalc.run(smiles)     ← 第3步：RDKit 描述符 + 2D图
          └→ StructureAnnotator       ← 第4步：官能团/手性/骨架
```

### 4.3 流式输出机制

```
DeepSeek API (streaming=True)
  → ChatOpenAI.invoke() 内部走流式
  → LangGraph stream_mode="messages" 捕获每个 chunk
  → app.py for item in agent.stream():
      mode=="updates"  → 节点进度标签
      mode=="messages" → chunk.content → full_response += token
  → st.markdown(full_response + "▌") 逐 token 刷新
```

---

## 五、关键设计决策

### 5.1 both 路径：顺序链而非并行

LangGraph 的 `Send` 并行 fan-out 导致消息交错，DeepSeek API 报 400 错误。改为 `both_active` 标志驱动的顺序链：`supervisor → chem → lit → summary`，化学专家的结果作为文献专家的上下文。

### 5.2 循环导入解决

`nodes.py` ↔ `multiagent.py` 双向依赖。通过 `_get_mam()` 延迟导入：仅在节点函数执行时 import。

### 5.3 Tool Registry

不用 MCP（对本地 Python 工具过重），采用轻量集中注册：

```
tools/__init__.py → CHEM_TOOLS = [ChemicalMemory(), ChemSpider(), ...]
agents/multiagent.py → chem_tools = CHEM_TOOLS
```

新增工具只需在 `tools/__init__.py` 加一行 import + 一行实例化。

### 5.4 知识库生命周期

- 每条记录含 `created_at` / `last_accessed` / `access_count`
- 检索返回 top-3 结果 + 余弦相似度分数（默认阈值 0.4）
- LRU 淘汰上限 500 条
- 前端可浏览、筛选、单条删除

### 5.5 测试 Fake 注入

`conftest.py` 用 `sys.modules` 替换 `chromadb` 和 `sentence_transformers`：
- `FakeEmbedder`：MD5 哈希 → 48 维向量
- `FakeCollection`：纯内存字典 + 真实余弦相似度计算
- 全部 93 项测试离线完成，5 秒跑完

---

## 六、关键数据结构

### 6.1 MultiAgentState

```python
class MultiAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: Annotated[str | None, _last_write]
    round_count: int          # 防死循环，≥5 强制结束
    both_active: bool         # both 顺序链标志
```

### 6.2 工具返回示例

**ChemSpider：**
```json
{"id": 2157, "commonName": "Aspirin", "SMILES": "CC(=O)Oc1ccccc1C(=O)O",
 "MolecularWeight": "180.16", "MolecularFormula": "C9H8O4"}
```

**PubChem：**
```json
{"CID": 2244, "Title": "Aspirin", "IUPACName": "2-acetyloxybenzoic acid",
 "Synonyms": ["Aspirin", "Acetylsalicylic acid", "阿司匹林"]}
```

**StructureAnnotator：**
```json
{"smiles": "C[C@H](O)C(=O)O",
 "chiral_centers": [{"atom_index": 1, "stereo": "S", "symbol": "C"}],
 "functional_groups": [{"name": "羧酸", "count": 1}, {"name": "醇羟基", "count": 1}],
 "reactive_sites": [{"atom_index": 3, "type": "亲电位点"}]}
```

---

## 七、外部依赖

| 服务 | 用途 | API Key |
|------|------|---------|
| DeepSeek API | LLM 推理 | ✅ 需要 |
| RSC ChemSpider | 名称→化合物 | ✅ 需要 |
| PubChem REST | SMILES→名称/性质 | ❌ 免费 |
| CrossRef | 文献检索 | ❌ 免费 |
| Ketcher (CDN) | 前端分子绘图 | ❌ 开源 |

| 核心 Python 库 | 版本 |
|---------------|------|
| langchain / langchain-openai | ≥1.0.0 |
| langgraph | 1.1.10 |
| chromadb | ≥0.5.0 |
| sentence-transformers | ≥3.0.0 |
| streamlit / streamlit-ketcher | ≥1.30.0 |
| rdkit | 2026.3.2 |

---

## 八、已知限制与未来方向

| 限制 | 说明 | 优先级 |
|------|------|--------|
| 单 Agent 模式未同步工具 | agents/agent.py 缺少 PubChem / StructureAnnotator | P1（建议移除） |
| 无 MSDS 安全数据 | PubChem GHS/LCSS 接口可用但未接入 | P2 |
| 无类药性评估 | Lipinski / QED / TPSA | P2 |
| 无反应预测 | 需要专门数据库和模型 | P3 |
| 无 NMR 预测 | 需要大规模谱图数据 + 深度学习 | P3 |
