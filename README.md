# 🧪 ChemAssist

**ChemAssist** 是一个 AI 驱动的化学信息学助手，支持自然语言查询化学物质的结构信息、分子标注、学术文献检索，以及手绘分子结构的智能分析。项目基于 LangChain + LangGraph，采用多 Agent 协同架构。

## ✨ 核心特性

- **🎨 分子结构绘制**：集成 Ketcher 编辑器，用户可直接在界面手绘分子结构，系统自动将 SMILES 提交给 Agent 分析
- **🔬 智能化学查询**：输入中英文名称或 SMILES，自动查询化合物信息（SMILES、分子量、InChIKey、LogP 等）
- **🏷️ 分子结构标注**：自动识别手性中心（R/S）、E/Z 异构、22 种官能团、Murcko 核心骨架、反应活性位点
- **📚 文献检索**：CrossRef API 检索学术文献，支持化学+文献组合查询
- **📖 个人知识库**：每次查询自动积累，支持语义检索（相似度评分）、浏览、筛选、单条删除，LRU 自动淘汰
- **🖼️ 2D 结构图**：RDKit 自动生成分子 2D 结构图并展示
- **⚡ Token 级流式输出**：回复逐字呈现，实时显示 Agent 执行进度
- **🔗 多数据源整合**：ChemSpider（名称检索）+ PubChem（SMILES 检索）+ RDKit（分子计算）+ CrossRef（文献）
- **💬 会话持久化**：SQLite checkpoint，应用重启后对话历史不丢失
- **🧪 完整测试体系**：93 项离线单元测试 + 11 用例真实 API 评估

## 🛠 技术栈

| 层次 | 技术 |
|------|------|
| LLM | DeepSeek V4 Flash（OpenAI 兼容 API） |
| Agent 编排 | LangGraph 1.1.10（StateGraph + 条件边 + SqliteSaver） |
| Agent 框架 | LangChain 1.x |
| 前端 | Streamlit 1.30+ + streamlit-ketcher + 自定义 CSS |
| 向量知识库 | ChromaDB 0.5 + SentenceTransformer (all-MiniLM-L6-v2) |
| 化学计算 | RSC ChemSpider API + PubChem REST API + RDKit |
| 分子绘图 | Ketcher（EPAM 开源） |
| 文献检索 | CrossRef REST API |
| 测试 | pytest 9.0（93 项，完全离线，5 秒跑完） |

## 📂 项目结构

```
ChemAssist/
├── agents/                          # Agent 编排层
│   ├── multiagent.py                # LLM/ToolNode 实例 + Tool Registry 导入
│   ├── state.py                     # MultiAgentState — 共享状态定义
│   ├── nodes.py                     # 6 个图节点（supervisor / expert / summary）
│   ├── graph.py                     # StateGraph 构建 + 条件路由
│   └── agent.py                     # 单 Agent（旧版——待移除）
│
├── tools/                           # 工具注册中心
│   ├── __init__.py                  # Tool Registry — CHEM_TOOLS / LITERATURE_TOOLS
│   ├── chem_memory.py               # 知识库（ChromaDB 语义检索 + 生命周期管理）
│   ├── chemspider_search.py         # ChemSpider API（名称→化合物数据）
│   ├── pubchem_search.py            # PubChem REST API（SMILES→名称/同义词）
│   ├── chem_calc.py                 # RDKit 分子计算 + StructureAnnotator 结构标注
│   └── literature_search.py         # CrossRef 文献检索
│
├── tests/                           # 测试（93 项）
│   ├── conftest.py                  # Fake 注入体系（ChromaDB + Embedder + 余弦相似度）
│   ├── test_agent.py                # 2 项
│   ├── test_chem_calc.py            # 11 项（描述符 + 结构标注）
│   ├── test_chem_memory.py          # 20 项（知识库 CRUD + 检索 + 阈值）
│   ├── test_chemspider_search.py    # 10 项
│   ├── test_image_display.py        # 4 项
│   ├── test_literature_search.py    # 7 项
│   ├── test_multiagent.py           # 19 项（节点 + 路由）
│   ├── test_multiagent_integration.py  # 6 项（全图集成）
│   ├── test_pubchem_search.py       # 9 项
│   └── test_streaming.py            # 5 项
│
├── app.py                           # Streamlit 前端（中文 UI + Ketcher + 知识库浏览器）
├── config.py                        # 集中配置（env / LLM / API / DB）
├── eval_agent.py                    # 真实 API 评估脚本（11 用例，5 维度）
│
├── docs/
│   ├── architecture.md              # 完整架构文档
│   ├── dev-log-2026-05-20.md        # RDKit 集成日志
│   ├── dev-log-2026-05-22.md        # 流式输出修复日志
│   ├── dev-log-2026-05-23.md        # 最新开发日志
│   ├── testing-guide.md             # 测试体系详解
│   └── testing-log-2026-05-18.md    # 离线测试搭建日志
│
├── .env.example                     # 环境变量模板
├── requirements.txt                 # Python 依赖
└── local_data/                      # 运行时数据（gitignore）
    ├── checkpoints.db               # 会话持久化
    ├── model_cache/                 # 模型缓存
    ├── chemical_memory_db/          # ChromaDB 知识库
    └── mol_images/                  # RDKit 2D 结构图
```

## 🔀 多 Agent 图架构

```
START → supervisor
         │
         ├─ chem_agent ⇄ chem_tools（工具循环）
         │    └─ ChemicalMemory → ChemSpiderSearch → PubChemSearch → ChemCalc → StructureAnnotator
         │    └─ 完成后 → summary
         │             → literature_agent（both 顺序链：chem → lit）
         │
         ├─ literature_agent ⇄ literature_tools（工具循环）
         │    └─ CrossrefSearch
         │    └─ 完成后 → summary
         │
         └─ finish → 直接回复（闲聊/已完成任务）

         summary → END（token 级流式输出）
```

**路由规则**：
- Supervisor 用 JSON 强制输出决策 4 路之一：`chem_agent` / `literature_agent` / `both` / `finish`
- 最大 5 轮循环保护，防止无限重试
- `both` 路径为顺序链——先 chem 后 lit，避免并行 fan-out 导致 DeepSeek API 消息格式交错
- `_ensure_tool_pairs()` 在节点调用 LLM 前清理破坏的消息配对
- mol2D/mol3D 坐标数据三层过滤（存储完整 → LLM 剥离 → UI 替换），避免上下文污染

## ⚙️ 快速开始

### 1. 克隆仓库
```bash
git clone https://github.com/Inker62/chem_assist_agent.git
cd chem_assist_agent
```

### 2. 安装依赖
```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
.venv\Scripts\activate             # Windows
pip install -r requirements.txt
```

### 3. 配置 API 密钥
```bash
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY、RSC_API_KEY
```

### 4. 启动应用
```bash
streamlit run app.py               # 浏览器打开 http://localhost:8501
```

### 5. 运行测试
```bash
pytest tests/ -v                   # 93 项，~5 秒，完全离线
```

### 6. 真实 API 评估
```bash
python eval_agent.py               # 全部 11 用例
python eval_agent.py --mode multi  # 仅多 Agent
python eval_agent.py --query "aspirin"  # 单条查询
```

## 🧪 使用说明

- **分子绘制**：展开 "🎨 分子结构绘制" 面板，手绘分子后点击 Apply，再在输入框输入问题
- **化学查询**：直接输入中英文化学物质名称、俗称或 IUPAC 名
- **SMILES 查询**：粘贴 SMILES 结构式，或通过 Ketcher 绘制后自动转换
- **结构分析**：查询后自动标注官能团、手性中心、核心骨架
- **文献检索**：输入 `查询XXX相关文献` 触发文献检索
- **组合查询**：输入 `查询MMA的化学结构及相关文献` 触发 both 顺序链
- **知识库管理**：侧边栏可浏览所有缓存条目、筛选、单条删除
- **对话导出**：可下载对话记录为 Markdown 文件

## 🚧 后续规划

- [x] 模块化 agents/ 拆分（state / nodes / graph / multiagent）
- [x] 集中配置管理（config.py）
- [x] RDKit 分子性质计算工具（ChemCalc）
- [x] 分子结构标注（StructureAnnotator）
- [x] 会话持久化（SqliteSaver）
- [x] Token 级流式输出
- [x] Both 路径顺序链（修复 DeepSeek 消息交错）
- [x] 前端 2D 结构图展示
- [x] 离线单元测试（93 项）
- [x] 前端中文化 + CSS 美化
- [x] 真实 API 评估脚本
- [x] Ketcher 分子结构绘制
- [x] PubChem SMILES 检索
- [x] Tool Registry 统一注册中心
- [x] 个人知识库浏览器
- [x] mol2D/mol3D 上下文污染过滤
- [ ] MSDS 安全数据检索（PubChem GHS/LCSS）
- [ ] QED / Lipinski 类药性评估
- [ ] 移除单 Agent 模式
- [ ] 评测数据集生成
- [ ] 部署到 Streamlit Cloud
