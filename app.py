import streamlit as st
import re
import json
import os
import uuid
from dotenv import load_dotenv
from agents.agent import initialize_agent
from agents.multiagent import initialize_multiagent
from tools.chem_memory import collection, clear_memory
from langchain_core.messages import HumanMessage, AIMessage

import warnings
import logging

# ---------- 静音无关日志 ----------
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

st.set_page_config(page_title="ChemAssist App", page_icon="🧪", layout="centered")

# ---------- 缓存 Agent 实例 ----------
@st.cache_resource
def load_agent():
    load_dotenv()
    return initialize_agent()

@st.cache_resource
def load_multiagent():
    load_dotenv()
    return initialize_multiagent()

st.title("🧪 ChemAssist App")
st.caption("Powered by DeepSeek & RSC ChemSpider & CrossRef· 化学助手智能体")

# ---------- 侧边栏 ----------
with st.sidebar:
    st.markdown("## ℹ️ 使用说明")
    st.markdown("""
        - 输入**任意化学物质名称**（中英文商品名、IUPAC 名、俗称均可）
        - 示例：`aspirin`, `caffeine`, `苯甲酸`, `acetaminophen`
        - 我会调用 ChemSpider 数据库返回 **SMILES、分子量、InChIKey** 等信息。
        - 当切换为多Agent协同模式时，我会根据你的输入自动判断化学物质检索/文献检索
        """)

    if st.session_state.get("messages"):
        md = ""
        for msg in st.session_state.messages:
            role = "**🧑 User**" if msg["role"] == "user" else "**🤖 ChemAssist**"
            md += f"{role}\n\n{msg['content']}\n\n---\n\n"
        st.download_button(
            label="📥 导出对话记录 (Markdown)",
            data=md,
            file_name="ChemAssist_conversation.md",
            mime="text/markdown"
        )

    st.markdown("## ⚙ 运行模式选择")
    mode_config = {
        "single_agent": "标准模式，调用单Agent完成化学物质/文献查询",
        "multi_agent": "多Agent协同模式，调用MultiAgent完成化学物质检索及关联文献检索功能"
    }

    selected_mode = st.selectbox(
        "选择ChemAssist的工作模式",
        options=list(mode_config.keys()),
        format_func=lambda x: mode_config[x],
        key="agent_mode"
    )

    agent_loaders = {
        "single_agent": load_agent,
        "multi_agent": load_multiagent,
    }

    agent = agent_loaders.get(selected_mode, load_agent)()

    st.caption(f"当前模式：{mode_config[selected_mode]}")

    st.divider()
    st.metric("🧠 记忆库存储化学物质数量：", collection.count())
    st.markdown("**当前工具**: ChemSpiderTool, ChemicalMemoryTool")
    st.markdown("**核心模型**: DeepSeek-V4-flash")
    if st.button("🔄 清空对话"):
        st.session_state.messages = []

    if st.button("🗑️ 清空记忆库"):
        clear_memory()
        st.success("记忆库已清空")
        st.rerun()

# ---------- 初始化消息历史 ----------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "我是 ChemAssist，您的化学信息学助手。请输入化学物质名称，我将为您检索其 SMILES、分子量等信息。"}
    ]

# ---------- 渲染历史消息 ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "raw_json" in msg:
            with st.expander("🔍 查看原始化学数据 (JSON)"):
                st.json(msg["raw_json"])

# ---------- 用户输入 ----------
prompt = st.chat_input("请输入化学物质名称或文献检索关键词...", key="main_chat_input")

if prompt:
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("正在查询中..."):
            try:
                # 同步调用 Agent
                response = agent.invoke(
                    {"messages": [HumanMessage(content=prompt)]},
                    config={"configurable": {"thread_id": str(uuid.uuid4())}}
                )
                messages_list = response.get("messages", [])

                # 提取最终回复（最后一条 AIMessage）
                output = None
                for msg in reversed(messages_list):
                    if isinstance(msg, AIMessage) and msg.content:
                        output = msg.content
                        break
                if not output:
                    output = "⚠️ 未生成有效回复。"

                # 提取原始 JSON
                raw_json = None
                json_match = re.search(r'```json\s*({.*?})\s*```', output, re.DOTALL)
                if json_match:
                    try:
                        raw_json = json.loads(json_match.group(1))
                    except:
                        pass

                # 显示最终回复
                st.markdown(output)

                # 消息流诊断（可折叠）
                with st.expander("🧠 消息流诊断"):
                    for i, msg in enumerate(messages_list):
                        msg_type = type(msg).__name__
                        content_preview = getattr(msg, 'content', str(msg))[:200]
                        st.text(f"{i+1}. [{msg_type}] {content_preview}")

            except Exception as e:
                output = f"⚠️ 查询出错: {str(e)}"
                st.error(output)
                messages_list = []
                raw_json = None

    # 保存到历史
    assistant_msg = {"role": "assistant", "content": output}
    if raw_json:
        assistant_msg["raw_json"] = raw_json
    st.session_state.messages.append(assistant_msg)