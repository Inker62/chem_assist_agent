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


## 各依赖库作用：
langchain	Agent 框架（create_openai_tools_agent, AgentExecutor）
langchain-openai	LangChain 与 OpenAI 兼容模型的连接器（我们的 DeepSeek 通过它调用）
langchain-community	LangChain 社区工具集（部分工具依赖此包）
openai	OpenAI SDK（ChatOpenAI 底层依赖，实际发给 DeepSeek）
pydantic	数据模型定义（BaseTool 的输入 schema ChemSpiderInput）
python-dotenv	从 .env 文件加载环境变量
requests	调用 RSC ChemSpider API
streamlit	前端交互界面

## 备注：text_api.py是测试用debug脚本，非必要组件