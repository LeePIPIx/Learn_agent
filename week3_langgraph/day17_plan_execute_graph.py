"""
Day 17: LangGraph 多阶段图 — Plan-Execute + 桥接函数

将 agent.py 的 _plan_execute_loop 映射到 graph:
  plan_phase → execute_phase → synthesize_phase → END

同时提供 run_agent_langgraph() 桥接函数，签名与 agent.run_agent_loop() 一致，
可无缝接入 Flask Web UI。
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
from agent import LLM_model, tools, tool_schemas, SYSTEM_PROMPT, init_kb


# ============ 1. State 定义 ============
class PlanExecuteState(TypedDict):
    messages: Annotated[list, operator.add]
    user_input: str
    memory_context: str
    plan: list
    execution_results: list
    final_answer: str


# ============ 2. 辅助函数 ============
def _parse_plan(text: str) -> list:
    """从 LLM 输出中提取 JSON 执行计划（从 agent.py 复用）"""
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


# ============ 3. Node 函数 ============
def plan_phase(state: PlanExecuteState, client_info: dict) -> dict:
    """Phase 1: 生成 JSON 执行计划"""
    prompt = f"""你是一个规划 agent。请为以下问题制定执行计划，只输出 JSON，不要调用工具。

可用工具: get_weather, calculator, get_current_time, execute_python, web_search, search_notes, recall_memory, save_memory

计划格式:
```json
[
  {{"step": 1, "description": "步骤描述", "tool": "工具名", "args": {{"参数名": "参数值"}}}},
  {{"step": 2, "description": "纯推理步骤", "tool": "", "args": {{}}}}
]
```

问题: {state['user_input']}"""

    response = client_info["client"].chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": "你是一个任务规划专家，只输出 JSON 格式的执行计划。"},
            {"role": "user", "content": prompt},
        ],
        extra_body={"thinking": {"type": "disabled"}},
    )
    text = response.choices[0].message.content or ""
    plan = _parse_plan(text)
    print(f"[plan_phase] 解析到 {len(plan)} 个步骤")
    return {
        "plan": plan,
        "execution_results": [],
        "messages": [{"role": "assistant", "content": f"计划:\n{text}"}],
    }


def execute_phase(state: PlanExecuteState) -> dict:
    """Phase 2: 顺序执行计划中的每个步骤"""
    results = []
    for step in state["plan"]:
        tool_name = step.get("tool", "")
        tool_args = step.get("args", {})
        desc = step.get("description", "")

        if tool_name and tool_name in tools:
            result = str(tools[tool_name]["fn"](**tool_args))
        elif tool_name:
            result = f"未知工具: {tool_name}"
        else:
            result = "跳过（纯推理步骤）"

        print(f"[execute_phase] 步骤{step.get('step','?')}: {desc} | {tool_name} → {result[:80]}")
        results.append(
            {
                "step": step.get("step"),
                "description": desc,
                "tool": tool_name,
                "args": tool_args,
                "result": result,
            }
        )
    return {"execution_results": results}


def synthesize_phase(state: PlanExecuteState, client_info: dict) -> dict:
    """Phase 3: 综合所有执行结果，生成最终回答"""
    ctx = json.dumps(state["execution_results"], ensure_ascii=False, indent=2)

    system_content = SYSTEM_PROMPT
    if state["memory_context"]:
        system_content += "\n\n## 用户记忆\n" + state["memory_context"]

    response = client_info["client"].chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": system_content},
            {
                "role": "user",
                "content": f"原始问题: {state['user_input']}\n\n所有步骤执行结果:\n{ctx}\n\n请基于以上结果给出完整的综合回答。",
            },
        ],
        extra_body={"thinking": {"type": "disabled"}},
    )
    answer = response.choices[0].message.content or ""
    print(f"[synthesize_phase] 综合完成")
    return {"final_answer": answer, "messages": [{"role": "assistant", "content": answer}]}


# ============ 4. 组装 Graph ============
def build_plan_execute_graph(client_info: dict):
    graph = StateGraph(PlanExecuteState)
    graph.add_node("plan", partial(plan_phase, client_info=client_info))
    graph.add_node("execute", execute_phase)
    graph.add_node("synthesize", partial(synthesize_phase, client_info=client_info))
    graph.set_entry_point("plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()


# ============ 5. 桥接函数：与 agent.run_agent_loop 签名一致 ============
def run_agent_langgraph(
    model_name: str,
    user_input: str,
    strategy: str = "react",
    max_iterations: int = 10,
    history: list = None,
) -> tuple:
    """使用 LangGraph 运行 agent，签名与 agent.run_agent_loop() 完全一致。

    返回: (response_text, updated_history)
    """
    from day16_react_graph import build_react_graph

    init_kb()
    client_info = LLM_model(model_name)
    # 短期会话记忆加载
    relevant = memory.recall(user_input, k=3)
    mem_ctx = memory.format_for_prompt(relevant) if relevant else ""
    system_content = SYSTEM_PROMPT + ("\n\n" + mem_ctx if mem_ctx else "")

    base_messages = [{"role": "system", "content": system_content}]
    if history:
        base_messages.extend(history)
    base_messages.append({"role": "user", "content": user_input})

    if strategy == "plan_execute":
        graph = build_plan_execute_graph(client_info)
        result = graph.invoke(
            {
                "messages": base_messages,
                "user_input": user_input,
                "memory_context": mem_ctx,
                "plan": [],
                "execution_results": [],
                "final_answer": "",
            }
        )
        final = result["final_answer"]
    else:
        graph = build_react_graph(client_info)
        result = graph.invoke(
            {
                "messages": base_messages,
                "iteration_count": 0,
                "final_answer": "",
            }
        )
        final = ""
        for msg in reversed(result["messages"]):
            if msg["role"] == "assistant" and msg["content"]:
                final = msg["content"]
                break

    updated_history = (history or []) + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": final},
    ]
    return final, updated_history


# ============ 6. 交互入口 ============
if __name__ == "__main__":
    print("=" * 50)
    print("Day 17: Plan-Execute Graph")
    print("plan_phase → execute_phase → synthesize_phase → END")
    print("输入 'quit' 退出")
    print("=" * 50)

    history = []

    while True:
        q = input("\nYou: ")
        if q.lower() in ("quit", "exit"):
            break

        strat = input("策略 (react/plan_execute, 默认 react): ").strip() or "react"
        response, history = run_agent_langgraph("deepseek", q, strategy=strat, history=history)
        history.append(history)
        
        print(f"\nAgent: {response}")
