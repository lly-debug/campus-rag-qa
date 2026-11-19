# -*- coding: utf-8 -*-
"""阶段二：在线问答流水线
多轮问题改写 → 向量检索 Top-K → 相关性过滤（拒答）→ DeepSeek 生成带引用的回答
"""
from openai import OpenAI

from config import (
    CHROMA_DIR,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    LLM_MODEL,
    SIM_THRESHOLD,
    TEMPERATURE,
    TOP_K,
)
from ingest import get_embeddings
from langchain_chroma import Chroma

SYSTEM_PROMPT = """你是"校园知识问答助手"，只依据下面提供的【参考资料】回答用户关于校园制度、培养方案、校园服务的问题。

要求：
1. 答案必须完全来自参考资料，禁止编造。资料能回答时，给出准确、简洁的中文回答，并在句末用 [编号] 标注所引用的资料块。
2. 资料不足以回答时，必须回答："抱歉，知识库中没有找到与您问题相关的内容，暂时无法回答。建议您联系人工服务获取帮助。"不要尝试编造答案。
3. 与校园话题无关的闲聊，同样按无法回答处理。
4. 回答用简体中文，条理清晰，可以分点。"""

CONDENSE_PROMPT = """根据以下对话历史，把用户的最新问题改写成一个不依赖上下文、含义完整的独立问题。
只输出改写后的问题本身，不要回答问题。
如果最新问题本身已经完整，原样输出即可。

对话历史：
{history}

最新问题：{question}

独立问题："""

REFUSE_REPLY = (
    "抱歉，知识库中没有找到与您问题相关的内容，暂时无法回答。\n"
    "建议您联系人工服务获取帮助（如辅导员、教务处或校园服务大厅）。"
)


class CampusRAG:
    def __init__(self):
        self.client = OpenAI(
            api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL
        )
        self.vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=get_embeddings(),
            collection_metadata={"hnsw:space": "cosine"},
        )

    # ---------- 检索 ----------
    def retrieve(self, question: str):
        """返回 Top-K 相似片段 [(chunk, 相似度)]，相似度 = 1 - 余弦距离"""
        pairs = self.vectorstore.similarity_search_with_score(
            question, k=TOP_K
        )
        results = []
        for doc, distance in pairs:
            sim = 1.0 - distance
            results.append((doc, sim))
        return results

    # ---------- 多轮改写 ----------
    def condense_question(self, question: str, history: list) -> str:
        """把依赖上下文的追问（如"那第二条呢"）改写成独立完整的问题"""
        # 只有存在过助手回复时才算真正的多轮对话，否则直接用原问题，
        # 避免每次提问都多一次大模型网络往返
        if not history or not any(role == "assistant" for role, _ in history):
            return question
        lines = []
        for role, content in history[-6:]:  # 最多取最近 3 轮
            who = "用户" if role == "user" else "助手"
            lines.append(f"{who}：{content}")
        resp = self.client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": CONDENSE_PROMPT.format(
                        history="\n".join(lines), question=question
                    ),
                }
            ],
        )
        return resp.choices[0].message.content.strip()

    # ---------- 主流程 ----------
    def prepare(self, question: str, history: list | None = None) -> dict:
        """检索阶段：问题改写 → 向量检索 → 组装生成请求。
        返回 messages=None 表示触发第一层拒答（库里没有相关内容）。
        """
        history = history or []
        standalone = self.condense_question(question, history)
        retrieved = self.retrieve(standalone)

        # 第一层拒答：最好片段的相似度过低，说明库里没有相关知识
        if not retrieved or retrieved[0][1] < SIM_THRESHOLD:
            return {
                "standalone": standalone,
                "retrieved": retrieved,
                "messages": None,
                "sources": [],
            }

        # 组装参考资料：每个片段带编号和来源文件名
        blocks, sources = [], []
        for i, (doc, sim) in enumerate(retrieved, 1):
            fname = doc.metadata.get("source", "未知文档")
            sources.append(fname)
            blocks.append(f"[{i}]（来源：{fname}）\n{doc.page_content}")
        context = "\n\n".join(blocks)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        # 携带历史，保持对话语气连贯
        for role, content in history[-6:]:
            messages.append({"role": role, "content": content})
        messages.append(
            {
                "role": "user",
                "content": f"【参考资料】\n{context}\n\n【用户问题】\n{standalone}",
            }
        )
        return {
            "standalone": standalone,
            "retrieved": retrieved,
            "messages": messages,
            "sources": sources,
        }

    def stream_answer(self, messages: list):
        """流式生成回答，逐块 yield 文本增量，降低等待体感"""
        stream = self.client.chat.completions.create(
            model=LLM_MODEL,
            temperature=TEMPERATURE,
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    @staticmethod
    def _is_refused(answer: str) -> bool:
        # 第二层拒答：模型认为资料不足（按提示词约定输出固定话术）
        return "暂时无法回答" in answer and "人工" in answer

    def ask(self, question: str, history: list | None = None) -> dict:
        """返回 {answer, sources, refused, retrieved}
        sources: 实际被引用的来源文件名列表
        retrieved: 检索到的全部片段（供前端展示）
        """
        prepared = self.prepare(question, history)
        if prepared["messages"] is None:
            return {
                "answer": REFUSE_REPLY,
                "sources": [],
                "refused": True,
                "retrieved": prepared["retrieved"],
                "standalone": prepared["standalone"],
            }

        answer = "".join(self.stream_answer(prepared["messages"])).strip()
        refused = self._is_refused(answer)
        return {
            "answer": answer,
            "sources": [] if refused else sorted(set(prepared["sources"])),
            "refused": refused,
            "retrieved": prepared["retrieved"],
            "standalone": prepared["standalone"],
        }


if __name__ == "__main__":
    rag = CampusRAG()
    history = []
    print("校园知识问答助手（命令行版，输入 q 退出）")
    while True:
        q = input("\n你：").strip()
        if q.lower() in {"q", "quit", "exit"}:
            break
        result = rag.ask(q, history)
        print(f"\n助手：{result['answer']}")
        if result["sources"]:
            print(f"（引用来源：{', '.join(result['sources'])}）")
        history.append(("user", q))
        history.append(("assistant", result["answer"]))
