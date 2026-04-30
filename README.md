# Agent 学习项目

## 学习目标

学会 Agent 的相关应用，包括：
- RAG（检索增强生成）
- Tool Calling（工具调用）
- 创建工作流

## 技术栈

| 类别 | 技术 |
|------|------|
| 编程语言 | Python |
| 核心框架 | LangChain / LangGraph |
| 向量数据库 | FAISS / Chroma |
| 多Agent框架 | AutoGen |
| LLM API | OpenAI / Minimax / DeepSeek |

---

## 总目标（30天后）

1. 手写一个 Agent 系统（不用框架）
2. 用 LangGraph 做一个复杂工作流 Agent
3. 完成一个"有说服力"的项目（医疗方向）

---

## 学习计划

### 第1周：别碰框架，手写Agent

**目标**：理解"Agent到底怎么跑"

| 阶段 | 周期 | 内容 | 技术要点 |
|------|------|------|----------|
| Day 1-2 | Agent核心循环 | User → LLM → Thought → Action → Tool → Observation → LLM | API调用、System Prompt、对话程序 |
| Day 3-4 | Tool调用机制 | Agent自动选择工具（查天气、执行Python代码） | Tool Schema(JSON)、LLM判断、Tool选择策略 |
| Day 5-7 | 循环Agent | 多轮推理、Memory、错误处理 | While Loop、上下文保存、容错机制 |

**交付**："能自己查资料+总结"的Agent Demo

**技术栈**：Python + Minimax/OpenAI API

---

### 第2周：Agent能力拆解

**目标**：从"能跑"到"能设计系统"

| 阶段 | 周期 | 内容 | 技术要点 |
|------|------|------|----------|
| Day 8-9 | Memory系统 | 短期Memory（对话历史）+ 长期Memory（向量检索） | FAISS / Chroma |
| Day 10-11 | RAG | 文档切分、Embedding、检索、重排序、拼Prompt | 向量数据库、Embedding模型、重排序模型 |
| Day 12-14 | Planning | ReAct（边想边做）+ Plan-and-Execute（先规划再执行） | 推理策略对比与选择 |

**ReAct vs Plan-and-Execute 对比**：

| 策略 | 适用场景 | 优点 | 缺点 |
|------|----------|------|------|
| ReAct | 简单任务、工具少、步骤明确 | 实时调整、灵活 | 长任务效率低 |
| Plan-and-Execute | 复杂任务、多步骤、需要全局规划 | 计划清晰、可并行执行子任务 | 灵活性低、中途调整困难 |

**交付**："问论文内容"的Agent + "记住用户偏好"的Agent

**技术栈**：FAISS / Chroma + Embedding模型 + 重排序模型

---

### 第3周：上框架（工业级）

**目标**：控制复杂流程

| 阶段 | 周期 | 内容 | 技术要点 |
|------|------|------|----------|
| Day 15-17 | LangGraph | State、Node、Edge、条件分支 | LangGraph状态机 |
| Day 18-19 | 多Agent系统 | Planner Agent + Executor Agent + Reviewer Agent协作 | AutoGen / LangGraph |
| Day 20-21 | 工具系统升级 | Tool抽象（统一接口）、Tool失败重试、Tool选择优化 | 错误处理、Prompt调优 |

**交付**：多步骤任务Agent（带分支）+ 多Agent协作系统

**技术栈**：LangGraph + AutoGen

---

### 第4周：项目（决定你能不能进大厂）

**目标**：完成医疗影像Agent项目

| 阶段 | 周期 | 内容 | 技术要点 |
|------|------|------|----------|
| Day 22-25 | 系统搭建 | Agent架构：Tool(分割/报告生成/RAG)、Memory | 医疗影像处理 + Inf-Net |
| Day 26-27 | 优化 | Prompt工程、Few-shot、错误恢复、Latency优化 | 工程优化 |
| Day 28-29 | 评估与包装 | 评估方法、准备面试材料 | 系统架构图、Demo视频、README |

**评估方法**：

| 评估维度 | 方法 |
|----------|------|
| 准确性 | 工具调用准确率、回答正确率 |
| 效率 | 平均迭代次数、响应延迟 |
| 鲁棒性 | 工具失败时的错误恢复能力 |
| 用户体验 | Demo效果、交互流畅度 |

**交付**：医疗影像Agent（输入CT/DICOM/NIfTI + 医生问题 → Agent自动分割病灶+分析+生成报告+回答问题）

---

## 项目架构（最终形态）

```
Agent
├── Tool: segmentation（基于Inf-Net）
├── Tool: report generation
├── Tool: RAG（医学知识库）
└── Memory（对话历史 + 向量检索）
```

---

## 学习检查清单

### Week 1
- [ ] 调用 Minimax/OpenAI API 实现对话
- [ ] 实现 Tool Schema 并让LLM调用工具
- [ ] 完成多轮推理循环Agent

### Week 2
- [ ] 实现短期Memory（对话历史）
- [ ] 实现长期Memory（FAISS/Chroma向量检索）
- [ ] 完成RAG流程（文档→切分→Embedding→检索→重排序→生成）
- [ ] 实现ReAct和Plan-and-Execute两种策略，理解适用场景

### Week 3
- [ ] 掌握LangGraph（State/Node/Edge/条件分支）
- [ ] 用LangGraph实现多步骤任务Agent
- [ ] 实现多Agent协作系统
- [ ] 完成Tool抽象和错误处理

### Week 4
- [ ] 集成Inf-Net分割模型到Agent
- [ ] 实现医学影像RAG
- [ ] 完成报告生成模块
- [ ] 评估Agent效果（准确率、效率、鲁棒性）
- [ ] 准备面试材料（架构图、Demo）

---

## 学习资源

### 官方文档
- [LangGraph文档](https://langchain-ai.github.io/langgraph/)
- [LangChain文档](https://python.langchain.com/)
- [AutoGen文档](https://microsoft.github.io/autogen/)

### 关键论文
- ReAct: Synergizing Reasoning and Acting in Language Models
- Plan-and-Execute: 任务分解与执行分离策略

### 医疗影像（Inf-Net）
- 确认Inf-Net的接口格式（输入/输出）
- 确认是否需要DICOM/NIfTI预处理