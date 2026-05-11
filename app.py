import streamlit as st
import re #正则表示库
import json
from dotenv import load_dotenv
from agent import initialize_agent

from tools.chem_memory import collection, clear_memory

import warnings
import logging
import os

# 1. 屏蔽 HuggingFace Hub 和 Transformers 的内部扫描警告
# 设置环境变量，直接禁用进度条和冗长日志
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# 这一行是关键：将 Transformers 的日志等级设为 ERROR，不再输出 __path__ 等警告
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# 2. 屏蔽 Python 层级的警告
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langgraph")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", module="transformers")

# 3. 全局降低日志等级
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("langgraph").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
# 直接关闭 transformers 库的日志输出
logging.getLogger("transformers").setLevel(logging.ERROR)

st.set_page_config(
    page_title="ChemAssist App",
    page_icon="🧪",
    layout="centered",
)


@st.cache_resource
def load_agent():
    load_dotenv()
    return initialize_agent()

agent = load_agent()

st.title("🧪 ChemAssist App")
st.caption("Powered by DeepSeek & RSC ChemSpider · 化学助手智能体")

with st.sidebar:
    st.markdown("## ℹ️ 使用说明")
    st.markdown("""
        - 输入**任意化学物质名称**（中英文商品名、IUPAC 名、俗称均可）
        - 示例：`aspirin`, `caffeine`, `苯甲酸`, `acetaminophen`
        - 我会调用 ChemSpider 数据库返回 **SMILES、分子量、InChIKey** 等信息。
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

    st.divider()
    st.metric("🧠 记忆库存储化学物质数量：", collection.count())
    st.markdown("**当前工具**: ChemSpiderTool")
    st.markdown("**核心模型**: DeepSeek-V4-flash")
    if st.button("🔄 清空对话"):
        st.session_state.messages = []

    if st.button("🗑️ 清空记忆库"):
        clear_memory()
        st.success("记忆库已清空")
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "我是 ChemAssist，您的化学信息学助手。请输入化学物质名称，我将为您检索其 SMILES、分子量等信息。"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "raw_json" in msg:
            with st.expander("🔍 查看原始化学数据 (JSON)"):
                st.json(msg["raw_json"])

prompt = st.chat_input("请输入化学物质名称...",key="main_chat_input")

if prompt:
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 调用 Agent（会返回工具调用过程）
    with st.chat_message("assistant"):
        with st.spinner("正在查询 ChemSpider 数据库..."):
            try:
                response = agent.invoke({ "messages": [{"role": "user", "content": prompt}]},config={"configurable": {"thread_id": "chemassist-session-1"}})
                output = response["messages"][-1].content
                intermediate_steps = response.get("intermediate_steps", [])
                thoughts = []
                for step in intermediate_steps:
                    # step 是 (AgentAction, output) 的元组
                    action, tool_output = step
                    thoughts.append({
                        "tool": action.tool,
                        "input": action.tool_input,
                        "result": tool_output[:500] + "..." if len(tool_output) > 500 else tool_output
                    })
                if thoughts:
                    with st.expander("🧠 查看 Agent 思考链 (工具调用过程)"):
                        for idx, thought in enumerate(thoughts, 1):
                            st.markdown(f"**第 {idx} 步: 调用工具 `{thought['tool']}`**")
                            st.markdown(f"* 输入参数: `{thought['input']}`")
                            st.markdown(f"* 工具返回 (前 500 字符):\n```json\n{thought['result']}\n```")
                json_match = re.search(r'```json\s*({.*?})\s*```', output, re.DOTALL)
                raw_json = None
                if json_match:
                    try:
                        raw_json = json.loads(json_match.group(1))
                    except:
                        pass

                # 如果输出中包含 JSON，用 expander 展示原始数据
                if raw_json:
                    with st.expander("🔍 查看原始化学数据 (JSON)"):
                        st.json(raw_json)

                # 显示最终文本回答（可能包含提取后的部分，但保留原样）
                st.markdown(output)
                
            except Exception as e:
                output = f"⚠️ 查询出错: {str(e)}"
                st.error(output)
                raw_json = None

            assistant_msg = {"role": "assistant", "content": output}
            if raw_json:
                assistant_msg["raw_json"] = raw_json
            st.session_state.messages.append(assistant_msg)