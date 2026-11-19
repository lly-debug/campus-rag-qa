# -*- coding: utf-8 -*-
"""评估脚本：对 50 条测试问题计算三项指标
1. 检索召回率：正确来源文档是否出现在检索结果中（仅知识内问题）
2. 回答准确率：DeepSeek 判分员对比"标准答案"与"系统回答"（仅知识内问题）
3. 拒答正确率：库外问题是否被正确拒答
结果输出到 data/eval_results.csv
运行前需已完成建库（uv run python ingest.py）
"""
import csv
import json
import time

from openai import OpenAI

from config import (
    BASE_DIR,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    LLM_MODEL,
)
from qa import CampusRAG

QUESTIONS_FILE = BASE_DIR / "data" / "test_questions.json"
RESULT_FILE = BASE_DIR / "data" / "eval_results.csv"

JUDGE_PROMPT = """你是严格的评分员。给定【标准答案】和【系统回答】，判断系统回答是否正确包含标准答案的核心信息。

判断标准：
- 核心数值、结论、对象与标准答案一致即判正确，多出的补充内容不影响判断；
- 系统回答表示"无法回答"或拒答的，判错误；
- 系统回答与标准答案核心信息矛盾的，判错误。

【标准答案】{expected}
【系统回答】{answer}

只输出 JSON，格式：{{"correct": true}} 或 {{"correct": false}}"""


def judge_answer(client: OpenAI, expected: str, answer: str) -> bool:
    """用大模型判分：系统回答是否与标准答案语义一致"""
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(expected=expected, answer=answer),
            }
        ],
    )
    text = resp.choices[0].message.content
    try:
        return "true" in text.lower()
    except Exception:
        return False


def main():
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        questions = json.load(f)

    rag = CampusRAG()
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    rows = []
    n_recall_hit = n_answer_hit = n_judged = n_refuse_hit = n_refused = 0

    for q in questions:
        result = rag.ask(q["question"], history=[])
        retrieved_files = [
            doc.metadata.get("source", "") for doc, _ in result["retrieved"]
        ]
        top1_sim = result["retrieved"][0][1] if result["retrieved"] else 0.0

        if q["type"] == "answer":
            # 检索召回：正确来源文档出现在 Top-K 中
            recall_hit = q["source"] in retrieved_files
            n_recall_hit += recall_hit
            # 回答准确：关键词粗评 + 大模型判分
            kw_ok = all(k in result["answer"] for k in q["keywords"])
            correct = judge_answer(client, q["expected"], result["answer"])
            n_judged += 1
            n_answer_hit += correct
            rows.append(
                {
                    "id": q["id"],
                    "type": "answer",
                    "question": q["question"],
                    "top1_sim": f"{top1_sim:.3f}",
                    "retrieved": "|".join(retrieved_files),
                    "recall_hit": recall_hit,
                    "keyword_ok": kw_ok,
                    "correct": correct,
                    "answer": result["answer"][:200],
                }
            )
        else:
            refused = result["refused"]
            n_refused += 1
            n_refuse_hit += refused
            rows.append(
                {
                    "id": q["id"],
                    "type": "refusal",
                    "question": q["question"],
                    "top1_sim": f"{top1_sim:.3f}",
                    "retrieved": "|".join(retrieved_files),
                    "recall_hit": "",
                    "keyword_ok": "",
                    "correct": refused,
                    "answer": result["answer"][:200],
                }
            )

        print(
            f"[{q['id']:>2}/50] {'✓' if rows[-1]['correct'] else '✗'} {q['question']}"
        )
        time.sleep(0.3)  # 轻微限速，避免触发 API 限流

    # ---- 汇总 ----
    n_answer = sum(1 for q in questions if q["type"] == "answer")
    summary = [
        "",
        "========== 评估汇总 ==========",
        f"知识内问题：{n_answer} 条",
        f"  检索召回率（正确文档进入 Top-K）：{n_recall_hit}/{n_answer} = {n_recall_hit / n_answer:.1%}",
        f"  回答准确率（大模型判分）：{n_answer_hit}/{n_judged} = {n_answer_hit / n_judged:.1%}",
        f"库外问题：{n_refused} 条",
        f"  拒答正确率：{n_refuse_hit}/{n_refused} = {n_refuse_hit / n_refused:.1%}",
    ]
    print("\n".join(summary))

    with open(RESULT_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"明细已保存到 {RESULT_FILE}")


if __name__ == "__main__":
    main()
