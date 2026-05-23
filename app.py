import streamlit as st
import re
import json
import os
import uuid
import config
from agents.agent import initialize_agent
from agents.multiagent import initialize_multiagent
from tools.chem_memory import collection, clear_memory, list_entries, delete_entry, get_stats
from tools.chem_calc import extract_mol_images
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from streamlit_ketcher import st_ketcher


def _strip_molfile(text: str) -> str:
    """移除 Molfile V2000/V3000 坐标块（数百行原子坐标对用户无意义）

    Molfile 特征：以 'V2000' 或 'V3000' 开头行，以 'M  END' 结尾
    """
    # 匹配从 V2000/V3000 行到 M  END 的所有内容（含前后换行）
    text = re.sub(
        r'\n\s+\d+\s+\d+\s+[\s\d]*V[23]000.*?M\s+END\s*\n',
        '\n[2D/3D 结构坐标数据已存入知识库]\n',
        text, flags=re.DOTALL
    )
    # 清理残留的 PUBCHEM_BONDANNOTATIONS 等注释块
    text = re.sub(
        r'\n> <PUBCHEM_\w+>.*?\n\n\$\$\$\$\n',
        '', text, flags=re.DOTALL
    )
    return text

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

st.set_page_config(
    page_title="ChemAssist 化学智能助手",
    page_icon="🧪",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
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
    .stButton button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #ebf5fb 0%, #d6eaf8 100%);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #aed6f1;
    }
    [data-testid="stChatInput"] textarea {
        border-radius: 12px !important;
        border: 2px solid #d6eaf8 !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #2e86c1 !important;
        box-shadow: 0 0 0 2px rgba(46,134,193,0.2) !important;
    }
    .stAlert {
        border-radius: 8px;
    }
    .streamlit-expanderHeader {
        font-size: 0.85rem;
        color: #7f8c8d;
    }
    /* Ketcher 绘图面板样式 */
    [data-testid="stExpander"] details summary {
        border: 2px dashed #aed6f1;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        background: linear-gradient(135deg, #f0f8ff 0%, #e8f4fd 100%);
        transition: all 0.2s;
    }
    [data-testid="stExpander"] details summary:hover {
        border-color: #2e86c1;
        background: linear-gradient(135deg, #d6eaf8 0%, #c5e0f5 100%);
    }
    /* Ketcher iframe 容器 */
    iframe[title="streamlit_ketcher\\.streamlit_ketcher"] {
        border: 1px solid #d6eaf8;
        border-radius: 8px;
    }
    @media (max-width: 768px) {
        .main-header h1 { font-size: 1.4rem; }
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🧪 ChemAssist 化学智能助手</h1>
    <p>DeepSeek V4 &times; RSC ChemSpider &times; CrossRef &mdash; AI 驱动的化学信息学平台</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_agent():
    return initialize_agent()


@st.cache_resource
def load_multiagent():
    return initialize_multiagent()


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
        "multi_agent": "🔹 协同模式 — 多 Agent 并行检索 + 文献",
        "single_agent": "🔹 标准模式 — 单 Agent 化学/文献查询",
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
    stats = get_stats()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("知识库条目", stats["total"])
    with col2:
        st.metric("累计命中", stats.get("total_accesses", 0))
    with col3:
        st.metric("容量使用", f"{stats.get('capacity_pct', 0)}%")

    st.caption(f"模型：DeepSeek-V4-flash | 工具：ChemSpider + CrossRef + RDKit")

    # ---- 知识库浏览器 ----
    st.markdown("## 📚 个人知识库")
    entries = list_entries()
    if entries:
        kb_filter = st.text_input("🔍 筛选知识库", key="kb_filter", placeholder="输入名称或 SMILES 筛选...")
        filtered = entries
        if kb_filter:
            filt_lower = kb_filter.lower()
            filtered = [
                e for e in entries
                if filt_lower in e["commonName"].lower()
                or filt_lower in e["query"].lower()
                or filt_lower in str(e.get("smiles", "")).lower()
            ]
        st.caption(f"共 {len(entries)} 条，显示 {len(filtered)} 条")
        for entry in filtered[:20]:  # 最多显示 20 条，避免侧边栏过长
            with st.container():
                c1, c2 = st.columns([5, 1])
                with c1:
                    name = entry["commonName"] or entry["query"] or entry["id"]
                    smiles = entry.get("smiles", "")
                    created = entry.get("created_at", "")[:10]
                    st.markdown(f"**{name}**  ")
                    if smiles:
                        st.caption(f"`{smiles}`")
                    st.caption(f"📅 {created} · 🔍 {entry['access_count']}次")
                with c2:
                    if st.button("🗑️", key=f"del_{entry['id']}", help=f"删除 {name}"):
                        delete_entry(entry["id"])
                        st.rerun()
    else:
        st.caption("知识库为空，查询化学物质后会自动积累。")

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

# ==================== Ketcher 分子结构绘制 ====================
with st.expander("🎨 分子结构绘制（点击展开，绘制分子后点击 Apply 确认）", expanded=False):
    ketcher_smiles = st_ketcher(
        value=st.session_state.get("ketcher_smiles", ""),
        height=400,
        key="ketcher_editor"
    )
    if ketcher_smiles:
        st.session_state["ketcher_smiles"] = ketcher_smiles
        st.info(f"✅ 当前分子 SMILES: `{ketcher_smiles}`")
    else:
        st.session_state["ketcher_smiles"] = ""
    st.caption(
        "1. 在画布上绘制分子结构 → 2. 点击 **Apply** 确认 → "
        "3. 在下方输入框中输入问题（如\"识别官能团\"、\"计算分子量\"等）→ 4. 回车发送"
    )

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 你好！我是 **ChemAssist**，你的化学信息学助手。\n\n"
                "我可以帮你：\n"
                "- 🔬 查询化学物质的结构信息（SMILES、分子量、InChIKey 等）\n"
                "- 📚 检索学术文献（通过 CrossRef）\n"
                "- 🖼️ 生成分子 2D 结构图（通过 RDKit ChemCalc）\n"
                "- 🎨 手绘分子结构并智能分析（点击上方展开绘图面板）\n"
                "- 🤖 多 Agent 协同模式下自动判断查询类型\n\n"
                "请在下方输入框开始提问，或先在上方绘制分子结构！"
            ),
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "raw_json" in msg:
            with st.expander("🔍 查看原始化学数据 (JSON)"):
                st.json(msg["raw_json"])

prompt = st.chat_input("输入问题，或先在上方绘制分子结构...")

if prompt:
    ketcher_smiles = st.session_state.get("ketcher_smiles", "")
    if ketcher_smiles:
        agent_prompt = f"SMILES: {ketcher_smiles}\n\n用户问题: {prompt}"
        display_msg = f"🧬 分子: `{ketcher_smiles}`\n\n{prompt}"
    else:
        agent_prompt = prompt
        display_msg = prompt

    st.session_state.messages.append({"role": "user", "content": display_msg})
    with st.chat_message("user"):
        st.markdown(display_msg)

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
                {"messages": [HumanMessage(content=agent_prompt)]},
                config={"configurable": {"thread_id": str(uuid.uuid4())}},
                stream_mode=["updates", "messages"]
            ):
                if isinstance(item, tuple):
                    if len(item) == 3:
                        namespace, mode, data = item
                        node_name = namespace[0] if isinstance(namespace, tuple) else str(namespace)
                    else:
                        mode, data = item
                        node_name = ""

                    if mode == "updates":
                        if node_name:
                            status = progress_labels.get(node_name, node_name)
                            progress_placeholder.info(status)
                        if isinstance(data, dict):
                            for node_output in data.values():
                                if isinstance(node_output, dict) and "messages" in node_output:
                                    all_messages.extend(node_output["messages"])

                    elif mode == "messages":
                        if isinstance(data, tuple) and len(data) >= 1:
                            chunk = data[0]
                            token = getattr(chunk, 'content', '') or ''
                            if token:
                                full_response += token
                                # 实时过滤 Molfile 坐标数据，避免满屏 V2000 糊脸
                                response_container.markdown(
                                    _strip_molfile(full_response) + "▌"
                                )

                elif isinstance(item, dict):
                    for node_name, node_output in item.items():
                        status = progress_labels.get(node_name, node_name)
                        progress_placeholder.info(status)
                        if isinstance(node_output, dict) and "messages" in node_output:
                            all_messages.extend(node_output["messages"])

            progress_placeholder.empty()

            if not full_response:
                for msg in reversed(all_messages):
                    if isinstance(msg, AIMessage) and msg.content:
                        full_response = msg.content
                        break
            if not full_response:
                full_response = "No valid response generated."

            json_match = re.search(r'```json\s*({.*?})\s*```', full_response, re.DOTALL)
            if json_match:
                try:
                    raw_json = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass

            response_container.markdown(_strip_molfile(full_response))

            # 展示 RDKit 生成的 2D 结构图
            mol_images = extract_mol_images(all_messages)
            if mol_images:
                st.divider()
                for img_path, smiles in mol_images:
                    caption = f"2D Structure: {smiles}" if smiles else "2D Structure"
                    st.image(img_path, caption=caption, use_container_width=True)

            with st.expander("🧠 消息流诊断"):
                for i, msg in enumerate(all_messages):
                    msg_type = type(msg).__name__
                    preview = getattr(msg, 'content', str(msg))[:200]
                    st.text(f"{i+1}. [{msg_type}] {preview}")

        except Exception as e:
            full_response = f"Error: {str(e)}"
            st.error(full_response)
            all_messages = []
            progress_placeholder.empty()

    assistant_msg = {"role": "assistant", "content": full_response}
    if raw_json:
        assistant_msg["raw_json"] = raw_json
    st.session_state.messages.append(assistant_msg)
