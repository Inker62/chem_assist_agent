# 🧪 ChemAssist

**ChemAssist** 是一个由 AI 驱动的化学信息学助手，支持自然语言查询化学物质的结构信息（SMILES、分子量、InChIKey 等）以及学术文献检索。项目基于 LangChain 和 LangGraph，提供单 Agent 与多 Agent 协同两种运行模式。

## ✨ 核心特性
- **自然语言交互**：输入中英文化学物质名称或文献关键词即可查询。
- **双模式架构**：
  - **单 Agent 模式**：轻量级，直接调用工具完成查询。
  - **多 Agent 协同模式**：由 Supervisor 调度化学专家和文献专家，自动路由任务。
- **权威数据源**：
  - **RSC ChemSpider API**：获取化学物质结构数据。
  - **CrossRef API**：检索学术文献（标题、作者、DOI、引用次数等）。
- **长期记忆**：基于 Chroma 向量数据库，自动缓存查询结果，避免重复请求。
- **前端诊断面板**：可折叠查看 Agent 内部的消息流与工具调用过程。
- **工程化实践**：模块化目录结构，环境变量管理，安全忽略敏感文件。

## 🛠️ 技术栈
- **语言 & 框架**：Python 3.10+，Streamlit
- **AI 核心**：LangChain 1.x, LangGraph, DeepSeek API
- **化学数据**：RSC ChemSpider, RDKit (规划中)
- **文献数据**：CrossRef REST API
- **记忆库**：Chroma + SentenceTransformers
- **对话记忆**：LangGraph InMemorySaver

## 📂 项目结构
``` markdown
ChemAssist/
├── agents/
│   ├── agent.py              # 单 Agent 初始化
│   └── multiagent.py         # 多 Agent 协同图定义
├── tools/
│   ├── chemspider_search.py  # ChemSpider API 工具
│   ├── chem_memory.py        # Chroma 长期记忆工具
│   └── literature_search.py  # CrossRef 文献检索工具
├── app.py                    # Streamlit 前端主入口
├── main.py                   # 命令行入口（早期版本）
├── requirements.txt          # 项目依赖
├── .env.example              # 环境变量模板
├── .gitignore                # 忽略规则
└── README.md                 # 本文件
```

## ⚙️ 快速开始
1. **克隆仓库**
   ```bash
   git clone https://github.com/Inker62/chem_assist_agent.git
   cd chem_assist_agent
2. **创建虚拟环境并安装依赖**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    pip install -r requirements.txt
3. **配置 API 密钥**
    - 复制 .env.example 为 .env
    - 填入你的 DEEPSEEK_API_KEY, RSC_API_KEY, PUBMED_EMAIL 等
4. **启动应用**
   ```bash
   streamlit run app.py
   
🧪 使用说明
- 标准模式：直接输入化学物质名称或结构，获取化学信息。
- 多 Agent 协同模式：在侧边栏选择该模式，主管会根据问题自动调用化学专家或文献专家。
- 记忆库管理：侧边栏可查看记忆库物质数，支持一键清空。
- 对话导出：可下载对话记录为 Markdown 文件。

🚧 后续规划
- 恢复 PubMed 文献检索（需解决 SSL 问题）
- 增加 RDKit 分子性质计算工具
- 实现流式输出（适配多 Agent 模式）
- 前端展示 Agent 思考链（工具调用过程）
- 支持并行调用化学与文献专家（both 分支）
- 部署到 Streamlit Cloud 或 Hugging Face Spaces
