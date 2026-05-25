"""
Day 18: 多Agent协作 — Planner Agent + Executor Agent

将 Day17 的单 LLM 三段式升级为两个独立 Agent:
  Planner Agent（规划师）→ Executor Agent（执行者）→ END

关键变化（对比 Day17）:
  - 每个 Agent 有自己固定的 system prompt，定义了职责边界
  - Planner 只规划不执行，Executor 是 LLM 驱动的（带工具调用）
  - Agent 之间通过 State 字段传递消息（plan → executor 看到 plan）
"""

import sys
import os
import json
import re
from functools import partial
from typing import TypedDict, Annotated
import operator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week1_miniagent"))

from langgraph.graph import StateGraph, END
from knowledge_base import kb
from memory_store import memory
from agent import LLM_model, tools, tool_schemas, SYSTEM_PROMPT, init_kb


# ============ 1. State：“Agent之间的消息总线” ============
class MultiAgentState(TypedDict):
    messages: Annotated[list, operator.add]
    user_input: str
    memory_context: str
    plan: list                 # Planner → Executor
    execution_results: list    # Executor 执行记录
    final_answer: str


# ============ 2. Planner Agent ============
PLANNER_SYSTEM_PROMPT = """你是一个任务规划专家（Planner Agent）。

你的唯一职责是制定执行计划，不是回答问题。

规则:
- 只输出 JSON 格式的执行计划，不要做任何其他事情
- 不要调用工具
- 每一步必须指定 tool（工具名）和 args（参数），纯推理步骤 tool 设为 ""
- 计划要完整、可执行，覆盖用户问题的所有方面

可用工具:
- get_weather(location): 查询城市天气
- get_current_time(): 获取当前时间
- calculator(expression): 数学计算
- execute_python(code): 执行Python代码
- web_search(query): 搜索互联网
- search_notes(query): 搜索个人笔记
- recall_memory(query): 搜索长期记忆
- save_memory(content, category): 保存记忆

输出格式:
```json
[
  {"step": 1, "description": "步骤描述", "tool": "工具名", "args": {"参数名": "参数值"}},
  {"step": 2, "description": "纯推理步骤", "tool": "", "args": {}}
]
```"""


def _parse_plan(text: str) -> list:
    """从 LLM 输出中提取 JSON 计划"""
    match = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text)
    if match:
        text = match.group(1)
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        try:
            plan = json.loads(match.group(0))
            if isinstance(plan, list):
                return plan
        except json.JSONDecodeError:
            pass
    return []


def planner_node(state: MultiAgentState, client_info: dict) -> dict:
    """Planner Agent: 分析问题，制定 JSON 执行计划"""
    response = client_info["client"].chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": f"请为以下问题制定执行计划:\n\n{state['user_input']}"},
        ],
        extra_body={"thinking": {"type": "disabled"}},
    )
    text = response.choices[0].message.content or ""
    plan = _parse_plan(text)

    print(f"[Planner Agent] 制定了 {len(plan)} 个步骤:")
    for s in plan:
        tool = s.get("tool", "") or "无工具"
        print(f"  Step {s.get('step','?')}: {s.get('description','')} | tool={tool}")

    return {
        "plan": plan,
        "messages": [{"role": "assistant", "content": f"[Planner] 计划:\n{text}"}],
    }


# ============ 3. Executor Agent ============
EXECUTOR_SYSTEM_PROMPT = """你是一个任务执行专家（Executor Agent）。

你的职责:
1. 严格按 Planner 给出的计划，逐步调用工具
2. 记录每个工具的执行结果
3. 全部执行完毕后，综合所有结果回答用户问题

执行规则:
- 必须为计划中 tool 非空的每一步调用对应工具（args 按计划给的值）
- 纯推理步骤（tool 为空）在最终回答中处理
- 工具调用完后，给出自然语言综合回答
- 如果工具失败，记录原因并继续

你需要先调用工具，再综合回答。不要跳过工具调用步骤。"""


def executor_node(state: MultiAgentState, client_info: dict) -> dict:
    """Executor Agent: LLM 驱动，按计划调用工具并综合回答"""
    plan = state["plan"]
    if not plan:
        return {
            "final_answer": "Planner 未能生成有效计划。",
            "messages": [{"role": "assistant", "content": "Planner 未能生成有效计划。"}],
        }

    plan_text = json.dumps(plan, ensure_ascii=False, indent=2)

    system_content = EXECUTOR_SYSTEM_PROMPT
    if state["memory_context"]:
        system_content += "\n\n## 用户记忆\n" + state["memory_context"]

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"原始问题: {state['user_input']}\n\n执行计划:\n{plan_text}\n\n请按计划逐步执行，调用完所有工具后综合回答。"},
    ]

    execution_results = []

    for iteration in range(10):
        response = client_info["client"].chat.completions.create(
            model="deepseek-v4-flash",
            messages=messages,
            tools=tool_schemas,
            tool_choice="auto",
            extra_body={"thinking": {"type": "disabled"}},
        )
        msg = response.choices[0].message

        if msg.tool_calls:
            assistant_msg = {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            }
            messages.append(assistant_msg)

            for tc in msg.tool_calls:
                tool_name = tc.function.name
                tool_args = json.loads(tc.function.arguments)
                print(f"[Executor Agent] {tool_name}({tool_args})")

                if tool_name in tools:
                    result = str(tools[tool_name]["fn"](**tool_args))
                else:
                    result = f"未知工具: {tool_name}"
                print(f"  → {result[:120]}")

                execution_results.append({
                    "tool": tool_name, "args": tool_args, "result": result,
                })
                messages.append({
                    "role": "tool", "tool_call_id": tc.id, "content": result,
                })
        else:
            final = msg.content or ""
            print(f"[Executor Agent] 综合回答完成")
            return {
                "execution_results": execution_results,
                "final_answer": final,
                "messages": [{"role": "assistant", "content": final}],
            }

    return {
        "execution_results": execution_results,
        "final_answer": "Executor 达到最大迭代次数，未能完成任务。",
        "messages": [{"role": "assistant", "content": "Executor 达到最大迭代次数。"}],
    }


# ============ 4. 组装 Graph ============
def build_multi_agent_graph(client_info: dict):
    graph = StateGraph(MultiAgentState)
    graph.add_node("planner", partial(planner_node, client_info=client_info))
    graph.add_node("executor", partial(executor_node, client_info=client_info))
    graph.set_entry_point("planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", END)
    return graph.compile()


# ============ 5. 桥接函数（签名与 agent.run_agent_loop 一致） ============
def run_agent_multi(model_name: str, user_input: str, strategy: str = "multi",
                    max_iterations: int = 10, history: list = None) -> tuple:
    init_kb()
    client_info = LLM_model(model_name)

    relevant = memory.recall(user_input, k=3)
    mem_ctx = memory.format_for_prompt(relevant) if relevant else ""

    graph = build_multi_agent_graph(client_info)
    result = graph.invoke({
        "messages": [],
        "user_input": user_input,
        "memory_context": mem_ctx,
        "plan": [],
        "execution_results": [],
        "final_answer": "",
    })

    updated_history = (history or []) + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": result["final_answer"]},
    ]
    return result["final_answer"], updated_history


# ============ 6. 交互入口 ============
if __name__ == "__main__":
    init_kb()
    client_info = LLM_model("deepseek")

    print("=" * 60)
    print("Day 18: 多Agent协作 — Planner + Executor")
    print("  Planner Agent ──→ Executor Agent ──→ END")
    print("  输入 'quit' 退出")
    print("=" * 60)

    while True:
        q = input("\nYou: ")
        if q.lower() in ("quit", "exit"):
            break

        relevant = memory.recall(q, k=3)
        mem_ctx = memory.format_for_prompt(relevant) if relevant else ""

        graph = build_multi_agent_graph(client_info)
        result = graph.invoke({
            "messages": [],
            "user_input": q,
            "memory_context": mem_ctx,
            "plan": [],
            "execution_results": [],
            "final_answer": "",
        })

        print(f"\n{'='*60}")
        print(f"最终回答:\n{result['final_answer']}")
