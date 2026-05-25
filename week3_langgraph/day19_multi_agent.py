"""
Day 19: 多Agent协作 + 审查反馈闭环

在 Day18 Planner+Executor 基础上加入 Reviewer Agent:
  Planner → Executor → Reviewer ──(通过)──→ END
                 ↑          │
                 └──(打回)──┘

新概念:
  - Reviewer Agent: 独立审查者，检查执行质量，决定通过/打回
  - 条件循环: add_conditional_edges 形成反馈闭环
  - 退出条件: max_revisions 防止无限循环
"""

import sys
import os
import json
import re
from functools import partial
from typing import TypedDict, Annotated, Literal
import operator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week1_miniagent"))

from langgraph.graph import StateGraph, END
from knowledge_base import kb
from memory_store import memory
from agent import LLM_model, tools, tool_schemas, init_kb


# ============ 1. State ============
class MultiAgentState(TypedDict):
    messages: Annotated[list, operator.add]
    user_input: str
    memory_context: str
    plan: list                    # Planner → Executor
    execution_results: list       # Executor → Reviewer（所有轮次累积）
    final_answer: str
    reviewer_feedback: str        # Reviewer → Executor（打回时的修正意见）
    decision: str                 # pass / revise
    revision_count: int           # 防止无限循环


# ============ 2. Planner Agent（同 Day18） ============
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


def _parse_json(text: str) -> dict | list | None:
    match = re.search(r"```(?:json)?\s*([\[{][\s\S]*?[\]}])\s*```", text)
    if match:
        text = match.group(1)
    match = re.search(r"[\[{][\s\S]*[\]}]", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def planner_node(state: MultiAgentState, client_info: dict) -> dict:
    response = client_info["client"].chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": f"请为以下问题制定执行计划:\n\n{state['user_input']}"},
        ],
        extra_body={"thinking": {"type": "disabled"}},
    )
    text = response.choices[0].message.content or ""
    plan = _parse_json(text)
    if not isinstance(plan, list):
        plan = []

    print(f"[Planner] 制定了 {len(plan)} 个步骤:")
    for s in plan:
        tool = s.get("tool", "") or "无工具"
        print(f"  Step {s.get('step','?')}: {s.get('description','')} | tool={tool}")

    return {
        "plan": plan,
        "messages": [{"role": "assistant", "content": f"[Planner] 计划:\n{text}"}],
    }


# ============ 3. Executor Agent（适配打回重做） ============
EXECUTOR_SYSTEM_PROMPT = """你是一个任务执行专家（Executor Agent）。

你的职责:
1. 严格按 Planner 给出的计划，逐步调用工具执行
2. 工具调用完后，综合所有结果回答用户问题

执行规则:
- 必须为计划中 tool 非空的每一步调用对应工具
- 纯推理步骤（tool 为空）在最终回答中处理
- 如果被 Reviewer 打回修改，仔细看反馈意见，补充或修正执行
- 先调用工具，再综合回答"""


def executor_node(state: MultiAgentState, client_info: dict) -> dict:
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

    # 如果是打回重做，把 Reviewer 的反馈和历史结果一起给 Executor
    user_msg = f"原始问题: {state['user_input']}\n\n执行计划:\n{plan_text}"

    if state.get("reviewer_feedback") and state.get("decision") == "revise":
        prev_results = json.dumps(state.get("execution_results", []), ensure_ascii=False, indent=2)
        user_msg += (
            f"\n\n⚠️ 上一轮被 Reviewer 打回，请根据反馈重新执行:\n"
            f"反馈意见: {state['reviewer_feedback']}\n"
            f"上一轮执行结果: {prev_results}\n"
            f"请针对反馈中提到的具体问题，补充或修正执行。"
        )
        print(f"[Executor] 收到打回意见: {state['reviewer_feedback'][:100]}...")
    else:
        user_msg += "\n\n请按计划逐步执行，调用完所有工具后综合回答。"

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_msg},
    ]

    new_results = []

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
                        "id": tc.id, "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            }
            messages.append(assistant_msg)

            for tc in msg.tool_calls:
                tool_name = tc.function.name
                tool_args = json.loads(tc.function.arguments)
                print(f"[Executor] {tool_name}({tool_args})")

                if tool_name in tools:
                    result = str(tools[tool_name]["fn"](**tool_args))
                else:
                    result = f"未知工具: {tool_name}"
                print(f"  → {result[:120]}")

                new_results.append({
                    "tool": tool_name, "args": tool_args, "result": result,
                })
                messages.append({
                    "role": "tool", "tool_call_id": tc.id, "content": result,
                })
        else:
            final = msg.content or ""
            print(f"[Executor] 执行完成")

            # 累积执行结果（保留之前轮次的，供 Reviewer 全面审查）
            accumulated = state.get("execution_results", []) + new_results
            return {
                "execution_results": accumulated,
                "final_answer": final,
                "messages": [{"role": "assistant", "content": final}],
            }

    return {
        "execution_results": state.get("execution_results", []) + new_results,
        "final_answer": "Executor 达到最大迭代次数。",
        "messages": [{"role": "assistant", "content": "Executor 达到最大迭代次数。"}],
    }


# ============ 4. Reviewer Agent（新增） ============
REVIEWER_SYSTEM_PROMPT = """你是一个质量审查专家（Reviewer Agent）。

你的职责是审查 Executor 的执行结果，判断是否完整、正确，决定通过还是打回。

审查维度:
1. 完整性: 计划中的所有步骤是否都执行了？纯推理步骤是否在最终回答中体现？
2. 正确性: 工具调用结果是否与原始问题相关？有没有明显错误？
3. 实用性: 最终回答是否清晰、有用地解答了用户的原始问题？

输出格式（必须严格 JSON）:
```json
{"decision": "pass", "reason": "所有步骤执行正确，回答完整"}
```
或
```json
{"decision": "revise", "feedback": "缺少步骤X的执行结果，请补充"}
```

decision 只能是 "pass" 或 "revise"。
如果是 revise，feedback 必须具体，指出 Executor 需要修正什么。"""


def reviewer_node(state: MultiAgentState, client_info: dict) -> dict:
    results_text = json.dumps(state["execution_results"], ensure_ascii=False, indent=2)
    plan_text = json.dumps(state["plan"], ensure_ascii=False, indent=2)

    prompt = f"""请审查以下执行结果:

## 原始问题
{state['user_input']}

## 执行计划
{plan_text}

## 执行结果
{results_text}

## 最终回答
{state['final_answer']}

请判断执行质量，输出 pass 或 revise。"""

    response = client_info["client"].chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        extra_body={"thinking": {"type": "disabled"}},
    )
    text = response.choices[0].message.content or ""
    result = _parse_json(text)

    if isinstance(result, dict) and "decision" in result:
        decision = result["decision"]
        feedback = result.get("feedback", "")
        reason = result.get("reason", "")
    else:
        decision = "pass"
        feedback = ""
        reason = "无法解析审查意见，默认通过"

    rev_count = state.get("revision_count", 0)
    if decision == "revise":
        rev_count += 1

    print(f"[Reviewer] 决定: {decision} (已修订{rev_count}次)")
    if feedback:
        print(f"  反馈: {feedback}")
    elif reason:
        print(f"  原因: {reason}")

    return {
        "decision": decision,
        "reviewer_feedback": feedback,
        "revision_count": rev_count,
    }


# ============ 5. 路由函数 ============
MAX_REVISIONS = 2

def review_decision(state: MultiAgentState) -> Literal["executor", "__end__"]:
    if state["decision"] == "pass":
        print("[路由] Reviewer 通过 → END")
        return "__end__"
    if state.get("revision_count", 0) >= MAX_REVISIONS:
        print(f"[路由] 已达最大修订次数 {MAX_REVISIONS} → 强制结束")
        return "__end__"
    print(f"[路由] Reviewer 打回 → 返回 Executor 修订 (第{state['revision_count']}次)")
    return "executor"


# ============ 6. 组装 Graph ============
def build_multi_agent_graph(client_info: dict):
    graph = StateGraph(MultiAgentState)

    graph.add_node("planner", partial(planner_node, client_info=client_info))
    graph.add_node("executor", partial(executor_node, client_info=client_info))
    graph.add_node("reviewer", partial(reviewer_node, client_info=client_info))

    graph.set_entry_point("planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "reviewer")

    graph.add_conditional_edges(
        "reviewer",
        review_decision,
        {"executor": "executor", "__end__": END},
    )

    return graph.compile()


# ============ 7. 桥接函数 ============
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
        "reviewer_feedback": "",
        "decision": "",
        "revision_count": 0,
    })

    updated_history = (history or []) + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": result["final_answer"]},
    ]
    return result["final_answer"], updated_history


# ============ 8. 交互入口 ============
if __name__ == "__main__":
    init_kb()
    client_info = LLM_model("deepseek")

    print("=" * 60)
    print("Day 19: 多Agent协作 + Reviewer 反馈闭环")
    print("  Planner → Executor → Reviewer ──(pass)──→ END")
    print("                   ↑          │")
    print("                   └──(revise)─┘")
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
            "reviewer_feedback": "",
            "decision": "",
            "revision_count": 0,
        })

        print(f"\n{'='*60}")
        print(f"最终回答:\n{result['final_answer']}")
