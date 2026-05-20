# 🧪 ChemAssist

**ChemAssist** 是一个 AI 驱动的化学信息学助手，支持自然语言查询化学物质的结构信息（SMILES、分子量、InChIKey、LogP 等）以及学术文献检索。项目基于 LangChain + LangGraph，提供单 Agent 与多 Agent 协同两种运行模式。

## ✨ 核心特性

- **自然语言交互**：输入中英文化学物质名称或文献关键词即可查询
- **双模式架构**：
  - **单 Agent 模式**：轻量级，直接调用工具完成查询
  - **多 Agent 协同模式**：Supervisor 调度化学专家和文献专家，自动路由任务
- **多数据源**：
  - **RSC ChemSpider API**：化学物质结构数据（SMILES、分子量、InChIKey 等）
  - **RDKit ChemCalc**：分子描述符计算（LogP、原子数、环数、氢键供/受体）+ 2D 结构图
  - **CrossRef API**：学术文献检索（标题、作者、期刊、DOI、引用次数）
- **长期记忆**：ChromaDB 向量数据库，以 ChemSpider ID 为唯一键，自动缓存查询结果
- **会话持久化**：SQLite checkpoint（SqliteSaver），应用重启后对话历史不丢失
- **流式进度提示**：Streamlit 实时显示当前执行节点（主管决策中 / 化学专家查询中 / 生成总结中 ...）
- **前端诊断面板**：可折叠查看 Agent 内部消息流与工具调用过程
- **离线测试套件**：56 项 mock 测试，5 秒跑完，无需网络或 API Key
- **真实 API 评估**：`eval_agent.py` 11 个测试用例，覆盖 5 个维度

## 🛠 技术栈

| 层次 | 技术 |
|------|------|
| LLM | DeepSeek V4 Flash（OpenAI 兼容 API） |
| Agent 编排 | LangGraph 1.1.10（StateGraph + 条件边 + SqliteSaver） |
| Agent 框架 | LangChain 1.x |
| 前端 | Streamlit 1.30+ + 自定义 CSS |
| 向量记忆 | ChromaDB 0.5 + SentenceTransformer (all-MiniLM-L6-v2) |
| 化学计算 | RSC ChemSpider API + RDKit |
| 文献检索 | CrossRef REST API |
| 测试 | pytest 9.0 + pytest-mock（56 项，完全离线） |

## 📂 项目结构

```
ChemAssist/
├── agents/                       # Agent 编排层
│   ├── __init__.py               # 包标记
│   ├── state.py                  # MultiAgentState — 共享状态定义
│   ├── nodes.py                  # 6 个图节点 + 消息清洗/配对函数
│   ├── graph.py                  # StateGraph 构建 + 条件边 + 初始化
│   ├── multiagent.py             # LLM/ToolNode 实例 + 向后兼容 re-export
│   └── agent.py                  # 单 Agent（LangChain create_agent）
│
├── tools/                        # 外部能力
│   ├── chemspider_search.py      # ChemSpider API（两步法查询）
│   ├── chem_calc.py              # RDKit 分子描述符 + 2D 结构图
│   ├── chem_memory.py            # ChromaDB 长期记忆（csid 去重）
│   └── literature_search.py      # CrossRef 文献检索
│
├── tests/                        # 测试（56 项，离线 mock）
│   ├── conftest.py               # Fake ChromaDB/Embedder 注入框架
│   ├── test_agent.py             # 2 项
│   ├── test_chem_memory.py       # 13 项
│   ├── test_chemspider_search.py # 10 项
│   ├── test_literature_search.py # 7 项
│   ├── test_multiagent.py        # 18 项（节点 + 路由）
│   └── test_multiagent_integration.py # 6 项（全图集成）
│
├── app.py                        # Streamlit 前端（中文 UI + CSS 美化）
├── config.py                     # 集中配置（env / LLM / API / DB）
├── eval_agent.py                 # 真实 API 评估脚本（11 用例，5 维度）
├── main.py                       # 旧版 CLI 入口（早期原型）
│
├── docs/
│   ├── dev-log-2026-05-20.md     # 开发日志
│   └── testing-log-2026-05-18.md # 测试日志
│
├── .env.example                  # 环境变量模板
├── requirements.txt              # Python 依赖
└── local_data/                   # 运行时数据（ChromaDB + checkpoints + 模型缓存）
```

## 🔀 多 Agent 图架构

```
START → supervisor
         │
         ├─ chem_agent ⇄ chem_tools（工具循环）
         │    └─ 完成后 → summary
         │             → literature_agent（both 顺序链：chem → lit）
         │
         ├─ literature_agent ⇄ literature_tools（工具循环）
         │    └─ 完成后 → summary
         │
         └─ finish → 直接回复

         summary → END
```

**路由规则**：
- Supervisor 用 JSON 强制输出决策 4 路之一：`chem_agent` / `literature_agent` / `both` / `finish`
- 最大 5 轮循环保护，防止无限重试
- `both` 路径为顺序链——先 chem 后 lit，避免并行 fan-out 导致 DeepSeek API 消息格式交错
- `_ensure_tool_pairs()` 在节点调用 LLM 前清理破坏的消息配对

## ⚙️ 快速开始

### 1. 克隆仓库
```bash
git clone https://github.com/Inker62/chem_assist_agent.git
cd chem_assist_agent
```

### 2. 创建虚拟环境并安装依赖
```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows
pip install -r requirements.txt
pip install rdkit              # ChemCalc 工具依赖
```

### 3. 配置 API 密钥
```bash
cp .env.example .env
# 编辑 .env，填入你的 DEEPSEEK_API_KEY、RSC_API_KEY、PUBMED_EMAIL
```

### 4. 启动应用
```bash
streamlit run app.py           # 浏览器打开 http://localhost:8501
```

### 5. 运行测试
```bash
.venv/Scripts/python -m pytest tests/ -v   # 56 项，~5 秒
```

### 6. 运行真实评估
```bash
.venv/Scripts/python eval_agent.py              # 全部 11 用例
.venv/Scripts/python eval_agent.py --mode multi # 仅多 Agent
.venv/Scripts/python eval_agent.py --query "aspirin"   # 单条查询
```

## 🧪 使用说明

- **标准模式**：直接输入化学物质名称，获取化学信息
- **多 Agent 协同模式**：侧边栏选择后，主管根据问题自动调用化学/文献专家
- **文献检索**：输入 `查询XXX相关文献` 触发文献检索流程
- **组合查询**：输入 `查询MMA的化学结构及相关文献` 触发 both 顺序链
- **记忆库管理**：侧边栏可查看记忆库条目数，支持一键清空
- **对话导出**：可下载对话记录为 Markdown 文件
- **消息流诊断**：每条回复下可展开查看 Agent 内部消息和工具调用轨迹

## 🚧 后续规划

- [x] 模块化 agents/ 拆分（state / nodes / graph / multiagent）
- [x] 集中配置管理（config.py）
- [x] RDKit 分子性质计算工具
- [x] 会话持久化（SqliteSaver）
- [x] 流式进度提示
- [x] Both 路径顺序链（修复 DeepSeek 消息交错）
- [x] 离线单元测试（56 项）
- [x] 前端中文化 + CSS 美化
- [x] 真实 API 评估脚本
- [ ] PubMed 文献检索恢复
- [ ] 前端 2D 结构图展示
- [ ] Token 级别流式输出
- [ ] 部署到 Streamlit Cloud
