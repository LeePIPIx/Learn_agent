from openai import OpenAI

# 创建客户端
client = OpenAI(
    api_key="sk-cp-obM5pmXT2_IEiqUIVZOzgqCGyUqMF0GEWEVq3GFHKIZ4VdRET6lfZivNPNc8DAKZSo7uKciRss1qQMkwdYDavH_L5z9XtwULQ7Zkl38X3Oxh1FpIlt6JB4g",
    base_url="https://api.minimax.chat/v1"
)

# 调用 API
response = client.chat.completions.create(
    model="Minimax-M2.7",  # MiniMax 模型
    messages=[
        {"role": "user", "content": "你好"}
    ]
)

print(response.choices[0].message.content)