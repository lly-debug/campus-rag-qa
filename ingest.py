# -*- coding: utf-8 -*-
"""阶段一：离线索引流水线
加载 data/docs 下的 PDF / Word / TXT 文档 → 切分文本块 → 向量化 → 存入 Chroma
用法：uv run python ingest.py
"""
import shutil

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCS_DIR,
    EMBED_MODEL,
)

# 按扩展名选择对应的文档加载器
LOADERS = {
    ".txt": TextLoader,
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
}


def get_embeddings() -> HuggingFaceEmbeddings:
    """本地中文向量模型，CPU 即可运行；向量归一化后用余弦相似度检索"""
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_documents():
    """加载目录下所有支持格式的文档"""
    docs = []
    for path in sorted(DOCS_DIR.iterdir()):
        loader_cls = LOADERS.get(path.suffix.lower())
        if loader_cls is None:
            print(f"[跳过] 不支持的格式: {path.name}")
            continue
        loaded = loader_cls(str(path)).load()
        for d in loaded:
            d.metadata["source"] = path.name  # 记录来源文件名，用于回答引用
        docs.extend(loaded)
        print(f"[加载] {path.name}（{len(loaded)} 页/段）")
    return docs


def build_index(reset: bool = True):
    """重建向量库：清空旧库 → 切分 → 向量化 → 入库"""
    docs = load_documents()
    if not docs:
        print("data/docs 目录下没有可导入的文档，请先放入 PDF/Word/TXT 文件。")
        return

    if reset and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)  # 重建模式先清空旧向量库

    # 递归字符切分：优先按段落/句子边界切，保证块内语义完整
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"[切分] {len(docs)} 篇文档 → {len(chunks)} 个文本块")

    # 向量化并写入 Chroma（余弦距离），持久化到本地
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"[入库] 向量库已保存到 {CHROMA_DIR}，共 {vectorstore._collection.count()} 条向量")


if __name__ == "__main__":
    build_index()
