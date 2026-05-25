"""
Day 16: LangGraph 条件分支 — ReAct 循环 as a Graph

将 agent.py 的 while 循环一对一映射到 LangGraph 原语：
  agent_think → (有 tool_calls?) → tool_execute → agent_think (循环)
                    ↓ (无 tool_calls)
                   END

关键概念:
  - add_conditional_edges: graph 版的 if 语句
  - Annotated[list, operator.add]: 消息列表追加模式
  - functools.partial: 把外部依赖绑定到节点函数
"""

import sys
import os
import json
from functools import partial
from typing import TypedDict, Annotated, Literal
import operator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week1_miniagent"))

from langgraph.graph import StateGraph, END
from knowledge_base import kb
from memory_store import memory
from agent import LLM_model, tools, tool_schemas, SYSTEM_PROMPT, init_kb


# ============ 1. State 定义 ============
class ReActState(TypedDict):
    messages: Annotated[list, operator.add]  # 追加模式，节点返回 {"messages": [new]} 自动拼接
    iteration_count: int
    final_answer: str


# ============ 2. Node 函数 ============
def agent_think(state: ReActState, client_info: dict) -> dict:
    """调用 LLM，返回 assistant 消息（可能带 tool_calls）"""
    response = client_info["client"].chat.completions.create(
        model="deepseek-v4-flash",
        messages=state["messages"],
        tools=tool_schemas,
        tool_choice="auto",
        extra_body={"thinking": {"type": "disabled"}},
    )
    msg = response.choices[0].message

    assistant_msg = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        assistant_msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ]
        tools_called = [tc.function.name for tc in msg.tool_calls]
        print(f"[agent_think] 调用工具: {tools_called}")
    else:
        print(f"[agent_think] 直接回答")

    return {"messages": [assistant_msg], "iteration_count": state["iteration_count"] + 1}


def tool_execute(state: ReActState) -> dict:
    """执行最后一条 assistant 消息中的所有 tool_calls"""
    last_msg = state["messages"][-1]
    if "tool_calls" not in last_msg:
        return {"messages": []}

    tool_messages = []
    for tc in last_msg["tool_calls"]:
        tool_name = tc["function"]["name"]
        tool_args = json.loads(tc["function"]["arguments"])
        print(f"[tool_execute] {tool_name}({tool_args})")

        if tool_name in tools:
            result = str(tools[tool_name]["fn"](**tool_args))
        else:
            result = f"Error: 未知工具 '{tool_name}'"
        tool_messages.append(
            {"role": "tool", "tool_call_id": tc["id"], "content": result}
        )
    return {"messages": tool_messages}


# ============ 3. 路由函数 ============
def should_continue(state: ReActState) -> Literal["tool_execute", "__end__"]:
    """条件判断：最后一条消息有 tool_calls 则继续执行工具，否则结束"""
    if state["iteration_count"] >= 10:
        return "__end__"
    last_msg = state["messages"][-1]
    if last_msg.get("role") == "assistant" and last_msg.get("tool_calls"):
        return "tool_execute"
    return "__end__"


# ============ 4. 组装 Graph ============
def build_react_graph(client_info: dict):
    graph = StateGraph(ReActState)
    graph.add_node("agent_think", partial(agent_think, client_info=client_info))
    graph.add_node("tool_execute", tool_execute)
    graph.set_entry_point("agent_think")
    graph.add_conditional_edges(
        "agent_think",
        should_continue,
        {"tool_execute": "tool_execute", "__end__": END},
    )
    graph.add_edge("tool_execute", "agent_think")
    return graph.compile()


# ============ 5. 交互入口 ============
if __name__ == "__main__":
    init_kb()
    client_info = LLM_model("deepseek")

    print("=" * 50)
    print("Day 16: ReAct 循环 Graph")
    print("agent_think → (tool_calls?) → tool_execute → agent_think")
    print("输入 'quit' 退出")
    print("=" * 50)

    while True:
        q = input("\nYou: ")
        if q.lower() in ("quit", "exit"):
            break

        relevant = memory.recall(q, k=3)
        mem_ctx = memory.format_for_prompt(relevant) if relevant else ""
        system_content = SYSTEM_PROMPT + ("\n\n" + mem_ctx if mem_ctx else "")

        graph = build_react_graph(client_info)
        result = graph.invoke(
            {
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": q},
                ],
                "iteration_count": 0,
                "final_answer": "",
            }
        )

        # 从最后一条 assistant 消息提取回答
        final = ""
        for msg in reversed(result["messages"]):
            if msg["role"] == "assistant" and msg["content"]:
                final = msg["content"]
                break
        print(f"\nAgent: {final}")
