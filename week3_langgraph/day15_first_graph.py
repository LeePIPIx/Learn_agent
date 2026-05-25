"""
Day 15: LangGraph 入门 — State, Node, Edge
构建第一个 LangGraph 流水线：线性三节点流水线，无分支。

对比手写 agent.py: 函数调用链 vs 声明式 graph
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week1_miniagent"))

from typing import TypedDict
from langgraph.graph import StateGraph, END
from knowledge_base import kb
from memory_store import memory
from agent import LLM_model, init_kb


# ============ 1. State 定义 ============
class AgentState(TypedDict, total=False):
    user_input: str
    memory_context: str
    kb_context: str
    final_answer: str


# ============ 2. Node 函数 ============
def recall_memories(state: AgentState) -> dict:
    """Node 1: 召回长期记忆"""
    results = memory.recall(state["user_input"], k=3)
    ctx = memory.format_for_prompt(results) if results else ""
    print(f"[recall_memories] 召回 {len(results)} 条记忆")
    return {"memory_context": ctx}


def search_knowledge(state: AgentState) -> dict:
    """Node 2: 检索知识库"""
    candidates = kb.search(state["user_input"], k=10)
    ranked = kb.rerank(state["user_input"], candidates, top_k=3)
    ctx = kb.format_results(ranked) if ranked else ""
    print(f"[search_knowledge] 检索到 {len(ranked)} 个片段")
    return {"kb_context": ctx}


def generate_answer(state: AgentState) -> dict:
    """Node 3: 综合记忆和知识库，调用 LLM 生成回答"""
    client_info = LLM_model("deepseek")
    system_prompt = "你是一个有帮助的助手。请基于提供的上下文回答用户问题。"

    if state["memory_context"]:
        system_prompt += "\n\n## 用户相关记忆\n" + state["memory_context"]
    if state["kb_context"]:
        system_prompt += "\n\n## 知识库相关内容\n" + state["kb_context"]

    response = client_info["client"].chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": state["user_input"]},
        ],
        extra_body={"thinking": {"type": "disabled"}},
    )
    answer = response.choices[0].message.content
    print(f"[generate_answer] 已生成回答")
    return {"final_answer": answer}


# ============ 3. 组装 Graph ============
def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("recall_memories", recall_memories)
    graph.add_node("search_knowledge", search_knowledge)
    graph.add_node("generate_answer", generate_answer)
    graph.set_entry_point("recall_memories")
    graph.add_edge("recall_memories", "search_knowledge")
    graph.add_edge("search_knowledge", "generate_answer")
    graph.add_edge("generate_answer", END)
    return graph.compile()


# ============ 4. 交互入口 ============
if __name__ == "__main__":
    init_kb()
    app = build_graph()

    print("=" * 50)
    print("Day 15: 线性流水线 Agent")
    print("节点: recall_memories → search_knowledge → generate_answer")
    print("输入 'quit' 退出")
    print("=" * 50)

    while True:
        q = input("\nYou: ")
        if q.lower() in ("quit", "exit"):
            break
        result = app.invoke(
            {"user_input": q, "memory_context": "", "kb_context": "", "final_answer": ""}
        )
        print(f"\nAgent: {result['final_answer']}")
