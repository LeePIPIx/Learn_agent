"""
Day 1-2: Agent核心循环
流程: User → LLM → Thought → Action → Tool → Observation → LLM → ...

使用DeepSeek API实现
"""

import os
import json
from pyexpat import model
import re
import io
import sys
from typing import Literal
from openai import OpenAI
from datetime import datetime
import anthropic
from tavily import TavilyClient
from knowledge_base import kb
from memory_store import memory

# DeepSeek API配置
def LLM_model(model_name):
    if model_name == "deepseek":
        # 定义deepseek
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://api.deepseek.com"
        )
    # if model_name == "minimax":
    #     # 定义minimax
    #     client = OpenAI(
    #         api_key=os.getenv("MINIMAX_API_KEY"),
    #         base_url="https://api.minimax.chat/v1"
    #     )
    return {"model_name":model_name, "client":client}

# ============ 1. 定义工具 ============
def get_current_time() -> str:
    """获取当前时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def get_weather(location: str) -> str:
    """获取天气信息"""
    # 模拟天气数据
    weather_db = {
        "北京": "晴，25°C",
        "上海": "多云，28°C",
        "广州": "雨，30°C",
        "深圳": "雷阵雨，31°C",
        "杭州": "阴，26°C"
    }
    return weather_db.get(location, f"未知地区{location}的天气")

def calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        # 安全计算（仅支持基本运算）
        allowed_chars = set('0123456789+-*/.() ')
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return str(result)
        return "Error: 只支持基本数学运算"
    except:
        return "Error: 计算表达式无效"

def execute_python(code: str) -> str:
    """执行Python代码（沙箱环境）"""
    import io
    import sys
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        # 使用独立的命名空间执行代码
        namespace = {}
        exec(code, namespace)
        output = sys.stdout.getvalue()
        return output or "执行成功，无输出"
    except Exception as e:
        return f"Error: {type(e).__name__}: {str(e)}"
    finally:
        sys.stdout = old_stdout

def web_search(query: str) -> str:
    """使用Tavily搜索互联网获取信息"""
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = tavily_client.search(query)
    results = response.get("results", [])
    formatted = []
    for r in results:
        formatted.append(f"标题: {r['title']}\n内容: {r['content']}")
    return "\n\n".join(formatted)

def search_notes(query: str) -> str:
    """在个人笔记知识库中搜索相关内容（粗检索 + 重排序）"""
    candidates = kb.search(query, k=15)
    results = kb.rerank(query, candidates, top_k=5)
    return kb.format_results(results)

def rebuild_kb() -> str:
    """重建知识库索引（重新扫描 notes 目录）"""
    kb.build()
    return f"知识库已重建，共 {kb.collection.count()} 个片段"

def save_memory(content: str, category: str = "general") -> str:
    """将重要信息保存到长期记忆中"""
    return memory.remember(content, category)

def recall_memory(query: str) -> str:
    """搜索长期记忆中与查询相关的内容"""
    results = memory.recall(query, k=5)
    return memory.format_for_prompt(results) if results else "未找到相关记忆"

def forget_memory(keyword: str) -> str:
    """根据关键词删除记忆"""
    return memory.forget(keyword)

# 工具注册表
tools = {
    "get_weather": {
        "fn": get_weather,
        "description": "获取指定城市的天气信息",
        "parameters": {
            "location": {"type": "string", "description": "城市名称"}
        }
    },
    "calculator": {
        "fn": calculator,
        "description": "执行数学计算",
        "parameters": {
            "expression": {"type": "string", "description": "数学表达式，如 2+3*5"}
        }
    },
    "get_current_time": {
        "fn": get_current_time,
        "description": "获取当前时间",
        "parameters": {}
    },
    "execute_python": {
        "fn": execute_python,
        "description": "执行Python代码进行计算或数据处理",
        "parameters": {
            "code": {"type": "string", "description": "要执行的Python代码"}
        }
    },
    "web_search": {
        "fn": web_search,
        "description": "搜索互联网获取信息",
        "parameters": {
            "query": {"type": "string", "description": "搜索关键词"}
        }
    },
    "search_notes": {
        "fn": search_notes,
        "description": "在个人笔记知识库中语义搜索相关内容",
        "parameters": {
            "query": {"type": "string", "description": "搜索查询，用自然语言描述你要找的内容"}
        }
    },
    "rebuild_kb": {
        "fn": rebuild_kb,
        "description": "重新扫描 notes 目录并重建知识库索引，当用户更新了笔记文件后使用",
        "parameters": {}
    },
    "save_memory": {
        "fn": save_memory,
        "description": "将重要信息保存到长期记忆中，跨会话持久保留。用于记住用户偏好、身份、重要决定等",
        "parameters": {
            "content": {"type": "string", "description": "要记住的内容"},
            "category": {"type": "string", "description": "记忆分类，如 preference、fact、decision、personal"}
        }
    },
    "recall_memory": {
        "fn": recall_memory,
        "description": "搜索长期记忆中与查询相关的内容",
        "parameters": {
            "query": {"type": "string", "description": "搜索查询，用自然语言描述要回忆的内容"}
        }
    },
    "forget_memory": {
        "fn": forget_memory,
        "description": "根据关键词删除不再需要的记忆",
        "parameters": {
            "keyword": {"type": "string", "description": "要删除的记忆中包含的关键词"}
        }
    }
}

# 工具Schema（用于LLM识别有哪些工具可用）
tool_schemas = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息，输入城市名返回天气情况",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "城市名称，如：北京、上海"}
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前时间，返回格式化的日期时间字符串",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "执行数学计算，支持加减乘除和括号运算",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式，如 2+3*5 或 (10+5)/3"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "执行Python代码进行计算或数据处理，适用于复杂算法和数据处理任务",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "要执行的Python代码，如：print('Hello'); nums = [1,2,3]; print(sum(nums))"}
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "搜索互联网获取信息，适用于查询实时新闻、术语解释等",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词，如：DeepSeek、langchain"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": """在个人笔记知识库中进行语义搜索，找到与问题相关的笔记片段。
            适用于：
            - 里加入的身份
            - Agent 架构设计笔记
            - Python 异步编程
            - 深度学习笔记
            """,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "用自然语言描述要搜索的内容，如：深度学习优化器、Python装饰器用法"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "rebuild_kb",
            "description": "重新扫描 notes 目录并重建知识库索引。当用户添加、删除或修改了笔记文件后调用此工具更新索引",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "将重要信息保存到长期记忆中，跨会话持久保留。当用户分享个人信息、偏好、重要决定或任何需要记住的内容时，主动调用此工具",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "要记住的具体内容"},
                    "category": {"type": "string", "description": "记忆分类：preference、fact、decision、personal"}
                },
                "required": ["content", "category"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recall_memory",
            "description": "搜索长期记忆库，检索与查询相关的历史信息。在回答涉及用户个人信息或历史对话的问题前，应先使用此工具",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "用自然语言描述要回忆的内容"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "forget_memory",
            "description": "根据关键词删除长期记忆中不再需要的内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "要删除的记忆包含的关键词"}
                },
                "required": ["keyword"]
            }
        }
    }
]

# ============ 2. System Prompt ============
SYSTEM_PROMPT = """你是一个智能Agent助手。

核心循环流程：
1. Thought (思考): 分析用户问题，决定是否需要工具
2. Action (行动): 如果需要工具，选择合适的工具并执行
3. Observation (观察): 获取工具返回的结果
4. 重复直到任务完成

可用工具：
- get_weather: 查询城市天气，输入城市名返回天气情况
- get_current_time: 获取当前时间
- calculator: 计算数学表达式，支持复杂运算
- execute_python: 执行Python代码，用于复杂算法和数据处理
- web_search: 搜索互联网获取信息
- search_notes: 在个人笔记知识库中语义搜索相关内容
- rebuild_kb: 重建知识检索库
- save_memory: 将重要信息保存到长期记忆中（用户偏好、重要事实、决定等）
- recall_memory: 搜索长期记忆中的历史信息
- forget_memory: 删除不需要的记忆

当用户询问天气、计算、时间、代码执行或者搜索时，你应该使用工具。
当用户的问题可能在个人笔记中有答案时（学习笔记、技术文档等），优先使用 search_notes。
当用户说"更新笔记"、"重建索引"、"添加笔记"等涉及知识库维护的指令时，使用 rebuild_kb。

记忆管理规则（重要）：
- 每次对话开始时，自动使用 recall_memory 检索与用户问题相关的历史记忆
- 当用户分享了个人信息（如姓名、职业、偏好）或做出重要决定时，主动使用 save_memory 保存
- 如果用户纠正了你之前的错误，使用 save_memory 记住正确的信息
- 保存记忆时，content 应简洁准确，category 选择合适的分类
直接回答可以解决的问题（如常识、知识问答等）。
当需要使用工具时，必须通过 tool_calls 字段调用，不能在 content 中输出工具调用的 JSON。

特别注意：当用户询问“最近”的新闻、信息或事件时，必须先调用get_current_time获取当前时间，然后在搜索query中包含具体的年月日。搜索query格式应为“YYYY年MM月DD日+关键词”，例如“2026年4月30日 科技新闻”。

当工具执行失败时：
1. 分析错误信息，判断失败原因
2. 如果是代码错误，修复代码后重试（不要重复同样的错误）
3. 如果是工具选择错误，选择其他工具
4. 最多重试2次，如果仍然失败则返回错误原因
"""

PLAN_EXECUTE_SYSTEM_PROMPT = SYSTEM_PROMPT + """

## 计划-执行策略

在当前模式下，你将分三个阶段工作：

### 阶段1：制定计划
- 分析用户问题，输出一个JSON格式的执行计划
- 不要调用任何工具，只输出计划JSON
- 计划格式：
```json
[
  {"step": 1, "description": "步骤描述", "tool": "工具名", "args": {"参数名": "参数值"}},
  {"step": 2, "description": "步骤描述", "tool": "工具名", "args": {}}
]
```
- 如果某步骤不需要工具（纯推理），将 tool 设为空字符串 ""
- 步骤应覆盖用户问题的所有方面

### 阶段2：执行计划
- 系统会按顺序执行你的计划中的每个步骤
- 每个步骤会返回执行结果

### 阶段3：综合回答
- 你会收到所有步骤的执行结果
- 基于这些结果，给出完整、准确的最终回答
- 如果综合时需要额外信息，可以再次调用工具
"""

# ============ 3. Agent核心循环 ============
def chat_with_llm(client_info, messages: list, max_retries: int = 3, use_tools: bool = True) -> dict:
    """调用DeepSeek API"""
    client = client_info["client"]
    model_name = client_info["model_name"]
    for i in range(max_retries):
        try:
            kwargs = {
                "model": {"deepseek":"deepseek-v4-flash"}[model_name],
                "messages": messages,
                "extra_body": {"thinking":{"type":"disabled"}}
            }
            if use_tools:
                kwargs["tools"] = tool_schemas
                kwargs["tool_choice"] = "auto"
            response = client.chat.completions.create(**kwargs)
            return response
        except Exception as e:
            if i == max_retries - 1:
                raise e
            print(f"API调用失败，重试中... ({i+1}/{max_retries})")
    return None

def init_kb():
    """延迟初始化知识库"""
    if kb.collection.count() == 0:
        print("[Agent] 知识库为空，正在从 notes/ 目录构建...")
        kb.build()

def _build_assistant_msg(assistant_msg) -> dict:
    """将API返回的assistant消息转为消息历史格式"""
    return {
        "role": "assistant",
        "content": assistant_msg.content,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                }
            } for tc in assistant_msg.tool_calls
        ]
    }

def _react_loop(client_info: dict, user_input: str, memory_context: str, max_iterations: int = 10, history: list = None) -> tuple:
    """ReAct策略：边想边做，每步推理后决定是否调用工具"""
    system_content = SYSTEM_PROMPT + ("\n\n" + memory_context if memory_context else "")

    messages = [
        {"role": "system", "content": system_content},
    ]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_input})

    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        print(f"\n{'='*50}")
        print(f"[ReAct] 迭代 {iteration}")
        print(f"{'='*50}")

        response = chat_with_llm(client_info, messages)

        assistant_msg = response.choices[0].message
        print(f"[LLM输出]\n{assistant_msg.content}")
        print(f"[工具调用] {assistant_msg.tool_calls}")

        if assistant_msg.tool_calls:
            messages.append(_build_assistant_msg(assistant_msg))

            for tc in assistant_msg.tool_calls:
                tool_name = tc.function.name
                tool_args = json.loads(tc.function.arguments)

                print(f"\n[执行工具] {tool_name}")
                print(f"[工具参数] {tool_args}")

                if tool_name in tools:
                    result = tools[tool_name]["fn"](**tool_args)
                    print(f"[工具结果] {result}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": str(result)
                    })
                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"Error: 未知工具 {tool_name}"
                    })
        else:
            content = assistant_msg.content or ''
            if client_info["model_name"] == "deepseek":
                final = content
            else:
                final = re.sub(r'<think>.*?</think>', '', content, flags=re.S).strip()
            # 返回 (回答文本, messages不含system部分用于后续历史)
            return final, messages[1:]

    return "Error: 达到最大迭代次数", messages[1:]


def _parse_plan(text: str) -> list:
    """从LLM输出中提取JSON执行计划"""
    # 尝试 markdown code block
    match = re.search(r'```(?:json)?\s*(\[[\s\S]*?\])\s*```', text)
    if match:
        text = match.group(1)
    # 尝试原始JSON数组
    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        try:
            plan = json.loads(match.group(0))
            if isinstance(plan, list):
                return plan
        except json.JSONDecodeError:
            pass
    return []


def _plan_execute_loop(client_info: dict, user_input: str, memory_context: str, max_iterations: int = 10, history: list = None) -> tuple:
    """Plan-and-Execute策略：先规划再执行，最后综合回答"""
    system_content = PLAN_EXECUTE_SYSTEM_PROMPT + ("\n\n" + memory_context if memory_context else "")

    # Phase 1: 生成计划
    print(f"\n{'='*50}")
    print("[Plan&Execute] Phase 1: 生成计划")
    print(f"{'='*50}")

    plan_messages = [
        {"role": "system", "content": system_content},
    ]
    if history:
        plan_messages.extend(history)
    plan_messages.append({"role": "user", "content": f"请为以下问题制定执行计划（只输出JSON，不要调用工具）：\n\n{user_input}"})

    response = chat_with_llm(client_info, plan_messages, use_tools=False)
    plan_text = response.choices[0].message.content or ''
    print(f"[计划输出]\n{plan_text}")

    plan = _parse_plan(plan_text)
    if not plan:
        print("[Plan&Execute] 计划解析失败，回退到 ReAct")
        return _react_loop(client_info, user_input, memory_context, max_iterations, history)

    print(f"[解析到 {len(plan)} 个步骤]")

    # Phase 2: 执行计划
    print(f"\n{'='*50}")
    print("[Plan&Execute] Phase 2: 执行计划")
    print(f"{'='*50}")

    results = []
    for step in plan:
        tool_name = step.get("tool", "")
        tool_args = step.get("args", {})
        description = step.get("description", "")

        print(f"\n[步骤 {step.get('step', '?')}] {description}")
        print(f"[工具] {tool_name}, [参数] {tool_args}")

        if tool_name and tool_name in tools:
            result = tools[tool_name]["fn"](**tool_args)
        elif tool_name and tool_name not in tools:
            result = f"工具 '{tool_name}' 不存在"
        else:
            result = "跳过（无工具调用）"

        print(f"[结果] {result}")
        results.append({
            "step": step.get("step", "?"),
            "description": description,
            "tool": tool_name,
            "args": tool_args,
            "result": str(result)
        })

    # Phase 3: 综合回答
    print(f"\n{'='*50}")
    print("[Plan&Execute] Phase 3: 综合回答")
    print(f"{'='*50}")

    synthesis_messages = [
        {"role": "system", "content": system_content},
    ]
    if history:
        synthesis_messages.extend(history)
    synthesis_messages.append({"role": "user", "content": f"原始问题：{user_input}\n\n所有步骤执行结果：\n{json.dumps(results, ensure_ascii=False, indent=2)}\n\n请基于以上执行结果给出最终的综合回答。"})

    synth_iteration = 0
    while synth_iteration < 3:
        synth_iteration += 1
        response = chat_with_llm(client_info, synthesis_messages)
        assistant_msg = response.choices[0].message
        print(f"[综合LLM输出]\n{assistant_msg.content}")

        if assistant_msg.tool_calls:
            synthesis_messages.append(_build_assistant_msg(assistant_msg))
            for tc in assistant_msg.tool_calls:
                tool_name = tc.function.name
                tool_args = json.loads(tc.function.arguments)
                print(f"[综合阶段工具] {tool_name}({tool_args})")
                if tool_name in tools:
                    result = tools[tool_name]["fn"](**tool_args)
                else:
                    result = f"未知工具: {tool_name}"
                synthesis_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result)
                })
        else:
            final = assistant_msg.content or ''
            updated_history = (history or []) + [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": final}
            ]
            return final, updated_history

    final = assistant_msg.content or ''
    updated_history = (history or []) + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": final}
    ]
    return final, updated_history


def run_agent_loop(model_name: str, user_input: str, strategy: str = "react", max_iterations: int = 10, history: list = None) -> tuple:
    """Agent核心循环调度器

    strategy:
        "react" — 边想边做，适合简单任务
        "plan_execute" — 先规划再执行，适合复杂多步骤任务
    history: 之前的对话历史（不含system消息），用于多轮对话上下文
    返回: (response_text, updated_history)
    """
    init_kb()
    client_info = LLM_model(model_name)

    # 检索相关长期记忆（两种策略共享）
    relevant_memories = memory.recall(user_input, k=3)
    memory_context = memory.format_for_prompt(relevant_memories) if relevant_memories else ""
    if memory_context:
        print(f"[记忆召回]\n{memory_context}")

    if strategy == "plan_execute":
        return _plan_execute_loop(client_info, user_input, memory_context, max_iterations, history)
    else:
        return _react_loop(client_info, user_input, memory_context, max_iterations, history)

# ============ 4. 启动 ============
def main():
    """启动Flask Web服务"""
    from app import app
    app.run(debug=True, port=5000)

if __name__ == "__main__":
    main()