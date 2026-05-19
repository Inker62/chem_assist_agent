import streamlit as st
import re
import json
import os
import uuid
import config
from agents.agent import initialize_agent
from agents.multiagent import initialize_multiagent
from tools.chem_memory import collection, clear_memory
from langchain_core.messages import HumanMessage, AIMessage

import warnings
import logging

logging.basicConfig(level=logging.WARNING, format="%(name)s [%(levelname)s] %(message)s")

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langgraph")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("langgraph").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="ChemAssist 化学智能助手",
    page_icon="🧪",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ==================== 自定义样式 ====================
st.markdown("""
<style>
    /* 主标题 */
    .main-header {
        background: linear-gradient(135deg, #1a5276 0%, #2e86c1 50%, #3498db 100%);
        padding: 1.8rem 2rem;
        border-radius: 12px;
        margin-bottom: 0.5rem;
        text-align: center;
        color: white;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    .main-header p {
        margin: 0.3rem 0 0 0;
        opacity: 0.85;
        font-size: 0.9rem;
    }

    /* 侧边栏 */
    section[data-testid="stSidebar"] .stMarkdown h2 {
        color: #2e86c1;
        font-size: 1.1rem;
        border-bottom: 2px solid #2e86c1;
        padding-bottom: 0.3rem;
        margin-top: 1.5rem;
    }
    section[data-testid="stSidebar"] .stMarkdown h2:first-child {
        margin-top: 0;
    }

    /* 按钮 */
    .stButton button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }

    /* 指标卡片 */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #ebf5fb 0%, #d6eaf8 100%);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #aed6f1;
    }

    /* 聊天输入 */
    [data-testid="stChatInput"] textarea {
        border-radius: 12px !important;
        border: 2px solid #d6eaf8 !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #2e86c1 !important;
        box-shadow: 0 0 0 2px rgba(46,134,193,0.2) !important;
    }

    /* 进度提示 */
    .stAlert {
        border-radius: 8px;
    }

    /* 展开面板 */
    .streamlit-expanderHeader {
        font-size: 0.85rem;
        color: #7f8c8d;
    }

    /* 响应式 */
    @media (max-width: 768px) {
        .main-header h1 { font-size: 1.4rem; }
    }
</style>
""", unsafe_allow_html=True)

# ==================== 头部 ====================
st.markdown("""
<div class="main-header">
    <h1>🧪 ChemAssist 化学智能助手</h1>
    <p>DeepSeek V4 &times; RSC ChemSpider &times; CrossRef &mdash; AI 驱动的化学信息学平台</p>
</div>
""", unsafe_allow_html=True)

# ==================== Agent 缓存 ====================
@st.cache_resource
def load_agent():
    return initialize_agent()

@st.cache_resource
def load_multiagent():
    return initialize_multiagent()

# ==================== 侧边栏 ====================
with st.sidebar:
    st.markdown("## 📖 使用说明")
    st.markdown("""
    - 输入任意化学物质名称（中英文商品名、IUPAC 名、俗称均可）
    - 示例：`aspirin`、`咖啡因`、`苯甲酸`、`acetaminophen`
    - 也可输入文献检索关键词，如 `AIE荧光分子合成`
    - 多 Agent 模式会自动判断查询类型
    """)

    if st.session_state.get("messages"):
        md = ""
        for msg in st.session_state.messages:
            role = "**👤 用户**" if msg["role"] == "user" else "**🤖 ChemAssist**"
            md += f"{role}\n\n{msg['content']}\n\n---\n\n"
        st.download_button(
            label="📥 导出对话记录 (Markdown)",
            data=md,
            file_name="ChemAssist_对话记录.md",
            mime="text/markdown",
            use_container_width=True,
        )

    st.markdown("## ⚙️ 运行模式")
    mode_config = {
        "single_agent": "🔹 标准模式 — 单 Agent 化学/文献查询",
        "multi_agent": "🔹 协同模式 — 多 Agent 并行检索 + 文献",
    }

    selected_mode = st.selectbox(
        "选择工作模式",
        options=list(mode_config.keys()),
        format_func=lambda x: mode_config[x],
        key="agent_mode",
    )

    agent_loaders = {
        "single_agent": load_agent,
        "multi_agent": load_multiagent,
    }

    agent = agent_loaders.get(selected_mode, load_agent)()

    st.markdown("## 📊 系统状态")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("记忆库条目", collection.count())
    with col2:
        st.metric("当前模式", "单Agent" if selected_mode == "single_agent" else "多Agent")

    st.caption(f"模型：DeepSeek-V4-flash | 工具：ChemSpider + CrossRef + ChemCalc (RDKit)")

    st.markdown("## 🛠️ 管理")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄 清空对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with c2:
        if st.button("🗑️ 清空记忆库", use_container_width=True):
            clear_memory()
            st.success("记忆库已清空")
            st.rerun()

# ==================== 初始化消息历史 ====================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 你好！我是 **ChemAssist**，你的化学信息学助手。\n\n"
                "我可以帮你：\n"
                "- 🔬 查询化学物质的结构信息（SMILES、分子量、InChIKey 等）\n"
                "- 📚 检索学术文献（通过 CrossRef）\n"
                "- 🤖 多 Agent 协同模式下自动判断查询类型\n\n"
                "请在下方输入框开始提问吧！"
            ),
        }
    ]

# ==================== 渲染历史消息 ====================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "raw_json" in msg:
            with st.expander("🔍 查看原始化学数据 (JSON)"):
                st.json(msg["raw_json"])

# ==================== 用户输入 ====================
prompt = st.chat_input("请输入化学物质名称或文献检索关键词...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        progress_placeholder = st.empty()
        response_container = st.empty()

        full_response = ""
        all_messages = []
        raw_json = None

        progress_labels = {
            "supervisor": "🧠 主管决策中...",
            "chem_agent": "⚗️ 化学专家查询中...",
            "chem_tools": "🔧 化学工具调用中...",
            "literature_agent": "📚 文献专家检索中...",
            "literature_tools": "🔧 文献工具调用中...",
            "summary": "📝 生成总结中...",
        }

        try:
            for item in agent.stream(
                {"messages": [HumanMessage(content=prompt)]},
                config={"configurable": {"thread_id": str(uuid.uuid4())}},
                stream_mode="updates",
            ):
                if isinstance(item, dict):
                    for node_name, node_output in item.items():
                        status = progress_labels.get(node_name, f"⏳ {node_name}")
                        progress_placeholder.info(status)
                        if isinstance(node_output, dict) and "messages" in node_output:
                            all_messages.extend(node_output["messages"])

            progress_placeholder.empty()

            # 提取最终回复
            if not full_response:
                for msg in reversed(all_messages):
                    if isinstance(msg, AIMessage) and msg.content:
                        full_response = msg.content
                        break
            if not full_response:
                full_response = "⚠️ 未生成有效回复，请重试。"

            # 提取内嵌 JSON
            json_match = re.search(r'```json\s*({.*?})\s*```', full_response, re.DOTALL)
            if json_match:
                try:
                    raw_json = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass

            response_container.markdown(full_response)

            # 消息流诊断面板
            with st.expander("🧠 消息流诊断"):
                st.caption(f"共 {len(all_messages)} 条消息")
                for i, msg in enumerate(all_messages):
                    msg_type = type(msg).__name__
                    preview = getattr(msg, 'content', str(msg))[:200]
                    st.text(f"{i+1}. [{msg_type}] {preview}")

        except Exception as e:
            full_response = f"⚠️ 查询出错：{str(e)}"
            st.error(full_response)
            all_messages = []
            progress_placeholder.empty()

    assistant_msg = {"role": "assistant", "content": full_response}
    if raw_json:
        assistant_msg["raw_json"] = raw_json
    st.session_state.messages.append(assistant_msg)
