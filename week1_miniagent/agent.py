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

# DeepSeek API配置
def LLM_model(model_name):
    if model_name == "deepseek":
        # 定义deepseek
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            # api_key="sk-094430426f8948848c476d0d354d3566"# 设置环境变量: export DEEPSEEK_API_KEY="your-key"
            base_url="https://api.deepseek.com"
        )
    if model_name == "minimax":
        # 定义minimax
        client = OpenAI(
            api_key="sk-cp-obM5pmXT2_IEiqUIVZOzgqCGyUqMF0GEWEVq3GFHKIZ4VdRET6lfZivNPNc8DAKZSo7uKciRss1qQMkwdYDavH_L5z9XtwULQ7Zkl38X3Oxh1FpIlt6JB4g",
            base_url="https://api.minimax.chat/v1"
        )
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
    """搜索互联网获取信息（模拟）"""
    # 模拟搜索结果，实际项目中可接入百度/Google搜索API
    search_db = {
        "deepseek": "DeepSeek是国产大模型，由幻方量化研发，性能对标GPT-4",
        "langchain": "LangChain是一个用于构建LLM应用的框架，支持多种工具和组件",
        "agent": "Agent是能够自主决策和使用工具的AI系统，核心包括规划、记忆、工具调用",
        "python": "Python是一种高级编程语言，简洁易学，广泛应用于AI领域"
    }
    for key, value in search_db.items():
        if key in query.lower():
            return value
    return f"关于'{query}'的搜索结果：这是模拟数据，实际项目可接入百度/Google搜索API"

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

当用户询问天气、计算、时间、代码执行或者搜索时，你应该使用工具。
直接回答可以解决的问题（如常识、知识问答等）。
当需要使用工具时，必须通过 tool_calls 字段调用，不能在 content 中输出工具调用的 JSON。

当工具执行失败时：
1. 分析错误信息，判断失败原因
2. 如果是代码错误，修复代码后重试（不要重复同样的错误）
3. 如果是工具选择错误，选择其他工具
4. 最多重试2次，如果仍然失败则返回错误原因

输出格式：
- 使用工具时：{"thought": "你的思考过程", "action": "工具名", "action_input": {"参数": "值"}}
- 回答时：{"thought": "你的思考过程", "response": "你的回答"}
回答时请务必按照输出的格式来进行回答
"""

# ============ 3. Agent核心循环 ============
def chat_with_llm(client_info, messages: list, max_retries: int = 3) -> dict:
    """调用DeepSeek API"""
    client = client_info["client"]
    model_name = client_info["model_name"]
    for i in range(max_retries):
        try:
            response = client.chat.completions.create(
                model={"deepseek":"deepseek-v4-flash", "minimax":"MiniMax-M2.7"}[model_name],
                messages=messages,
                tools=tool_schemas,
                tool_choice="auto",
                extra_body={"thinking":{"type":"disabled"}} # 禁用LLM的内置思考过程，让它直接输出工具调用或回答
            )
            return response
        except Exception as e:
            if i == max_retries - 1:
                raise e
            print(f"API调用失败，重试中... ({i+1}/{max_retries})")
    return None

def run_agent_loop(model_name:str, user_input: str, max_iterations: int = 10) -> str:
    """运行Agent核心循环"""
    client_info = LLM_model(model_name)
    # 初始化模型的系统提示词并加入用户的问题
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]
    # 定义每个问题的对话轮次，如果对话轮次大于10轮依然没有解决问题，则报Error
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        print(f"\n{'='*50}")
        print(f"迭代 {iteration}")
        print(f"{'='*50}")

        # 1. 调用LLM
        response = chat_with_llm(client_info, messages)

        # 获取助手消息
        assistant_msg = response.choices[0].message
        print(f"[LLM输出]\n{assistant_msg.content}")
        print(f"[工具调用] {assistant_msg.tool_calls}")

        # 2. 检查是否有工具调用
        if assistant_msg.tool_calls:
            # 添加助手消息到历史
            messages.append({
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
            })

            # 3. 执行每个工具调用
            for tc in assistant_msg.tool_calls:
                tool_name = tc.function.name
                tool_args = json.loads(tc.function.arguments)

                print(f"\n[执行工具] {tool_name}")
                print(f"[工具参数] {tool_args}")

                if tool_name in tools:
                    result = tools[tool_name]["fn"](**tool_args)
                    print(f"[工具结果] {result}")

                    # 添加工具结果到消息
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
            # 没有工具调用，直接返回结果
            if client_info["model_name"] == "deepseek":
                final_response = assistant_msg.content
            else:
                final_response = assistant_msg.content.split('"response": "')[1].rstrip('"}')
            return final_response

    return "Error: 达到最大迭代次数"

# ============ 4. 演示 ============
if __name__ == "__main__":
    model_name = 'deepseek'
    print("=" * 60)
    print("Agent核心循环演示")
    print("=" * 60)

    # 测试用例
    test_queries = [
        # "北京今天天气怎么样？",
        # "请帮我计算 (123 + 456) * 789 / 2",
        # "用Python写个快速排序并执行",
        "请介绍一下DeepSeek是什么",
        # "现在几点了？",
    ]

    for query in test_queries:
        print(f"\n\n{'#'*60}")
        print(f"用户: {query}")
        print(f"{'#'*60}")
        result = run_agent_loop(model_name, query)
        print(f"\n[最终回复] {result}")