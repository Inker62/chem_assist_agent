# 🧪 ChemAssist

ChemAssist 是一个由 AI 驱动的化学信息学助手，可以查询化学物质的 SMILES、分子量、InChIKey 等权威数据。

## ⚙️ 技术栈
*   **Agent 框架**: LangChain (ReAct / Tool Calling)
*   **大语言模型**: DeepSeek-V4-flash
*   **数据来源**: RSC ChemSpider API
*   **前端交互**: Streamlit

## ✨ 核心特性
*   **自然语言查询**: 输入物质的中英文名、商品名、IUPAC名即可查询。
*   **权威数据源**: 实时调用 ChemSpider 官方数据库。
*   **可解释性**: 前端可折叠查看 Agent 的工具调用思考链和 API 返回的原始 JSON 数据。
*   **工程化**: 模块化设计 (Agent / Tools / UI)，含 `.gitignore`、`.env.example`，开箱即用。

## 🚀 快速开始
1.  Clone 仓库
2.  安装依赖：`pip install -r requirements.txt`
3.  配置 API 密钥：将 `.env.example` 复制为 `.env` 并填入你的 Key
4.  启动：`streamlit run app.py`
### 备注：text_api.py是测试用debug脚本，非必要组件

### 🚧 正在开发
- 多智能体协同架构（`multiagent.py`），基于 LangGraph，待测试通过后合入主分支

## 外部API获取链接：
- RSC ChemSpider:  'https://developer.rsc.org/'
- PubMed: 'https://account.ncbi.nlm.nih.gov/'