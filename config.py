# -*- coding: utf-8 -*-
"""全局配置：路径、模型、切分与检索参数"""
import os
from pathlib import Path

from dotenv import load_dotenv

# 国内网络下从镜像站下载 HuggingFace 模型（必须在导入模型库前设置）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# ---- 路径 ----
DOCS_DIR = BASE_DIR / "data" / "docs"        # 原始知识文档目录
CHROMA_DIR = BASE_DIR / "data" / "chroma"    # 向量库持久化目录

# ---- 模型 ----
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"       # 本地中文向量模型（CPU 可跑）
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
LLM_MODEL = "deepseek-chat"

# ---- 切分参数 ----
CHUNK_SIZE = 500      # 每个文本块最大字符数（容纳整组条款，避免答案脱离主题词）
CHUNK_OVERLAP = 100   # 相邻文本块重叠字符数，避免语义被切断

# ---- 检索参数 ----
TOP_K = 6             # 每次提问召回的相似片段数（稍大以覆盖同段多事实）
SIM_THRESHOLD = 0.35  # 余弦相似度阈值：最高相似度低于该值时拒答

# ---- 生成参数 ----
TEMPERATURE = 0.3     # 事实型问答用低温度，减少编造
