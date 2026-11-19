# -*- coding: utf-8 -*-
"""Web 界面（Streamlit）
运行：uv run streamlit run app.py
"""
import shutil

import streamlit as st

import config
from ingest import build_index
from qa import CampusRAG, REFUSE_REPLY

st.set_page_config(page_title="校园知识问答助手", page_icon="🎓", layout="centered")


@st.cache_resource
def load_rag() -> CampusRAG:
    return CampusRAG()


# ---------------- 侧边栏 ----------------
with st.sidebar:
    st.title("⚙️ 系统设置")

    if not config.DEEPSEEK_API_KEY or "xxxx" in config.DEEPSEEK_API_KEY:
        st.error("请先在项目根目录的 .env 文件中填写 DEEPSEEK_API_KEY")

    st.divider()
    st.subheader("📚 知识库管理")
    uploaded = st.file_uploader(
        "上传校园文档（PDF / Word / TXT，可多选）",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )
    if uploaded and st.button("导入并重建索引", use_container_width=True):
        with st.spinner("正在解析文档、切分文本并生成向量……"):
            for f in uploaded:
                (config.DOCS_DIR / f.name).write_bytes(f.getvalue())
            build_index()
            load_rag.clear()
        st.success("索引重建成功！")
    if config.CHROMA_DIR.exists():
        st.caption(f"向量库位置：{config.CHROMA_DIR.relative_to(config.BASE_DIR)}")
    else:
        st.warning("尚未建立向量索引，请先导入文档。")

    st.divider()
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption("基于 RAG 的校园知识问答智能体\n\nLangChain + Chroma + DeepSeek")

# ---------------- 主界面 ----------------
st.title("🎓 校园知识问答助手")
st.caption(
    "可回答校园制度、培养方案、校园服务等问题；回答附引用来源，无法回答时会提示转人工。"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# 回放历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        for s in msg.get("sources", []):
            st.markdown(f"- 📎 {s}")

# 处理新输入
if prompt := st.chat_input("请输入你的问题，例如：国家奖学金有多少钱？"):
    # 历史必须在追加当前问题之前取出，否则第一轮也会被误判为多轮对话
    history = [
        (m["role"], m["content"])
        for m in st.session_state.messages
        if m["role"] in ("user", "assistant")
    ]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        rag = load_rag()
        with st.spinner("正在检索知识库……"):
            prepared = rag.prepare(prompt, history)

        if prepared["messages"] is None:
            answer = REFUSE_REPLY
            sources = []
            st.markdown(answer)
        else:
            # 流式输出：答案边生成边显示，不用等整段生成完
            answer = st.write_stream(rag.stream_answer(prepared["messages"]))
            if rag._is_refused(answer):
                sources = []
            else:
                sources = sorted(set(prepared["sources"]))

        for s in sources:
            st.markdown(f"- 📎 {s}")
        if prepared["retrieved"]:
            with st.expander("🔍 查看检索片段（调试用）"):
                for doc, sim in prepared["retrieved"]:
                    st.markdown(
                        f"**{doc.metadata.get('source', '未知')}**"
                        f"（相似度 {sim:.2f}）"
                    )
                    st.text(doc.page_content[:300])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )
