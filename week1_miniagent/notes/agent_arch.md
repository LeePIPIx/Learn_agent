# Agent 架构设计笔记

## 核心组件

### 1. LLM（大语言模型）
负责推理和决策，是Agent的大脑。常用模型：
- GPT-4 / GPT-4o
- Claude 3.5 Sonnet
- DeepSeek-V3
- Gemini 2.0

### 2. 工具（Tools）
Agent 可以调用的外部能力：
- 计算器、代码执行器
- 网络搜索（Tavily、SerpAPI）
- 数据库查询
- 文件操作

### 3. 记忆系统（Memory）

#### 短期记忆
- 当前会话的对话历史
- 存储在 messages 列表中
- 受上下文窗口限制

#### 长期记忆
- 向量数据库存储（Chroma、FAISS、Milvus）
- 通过 embedding 进行语义检索
- 支持记忆的增删改查

### 4. 规划（Planning）
- ReAct 模式：Reasoning + Acting
- 思维链（Chain of Thought）
- 任务分解与执行

## ReAct 循环
```
Thought → Action → Observation → Thought → Action → ... → Final Answer
```
