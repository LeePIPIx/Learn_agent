"""
Flask Web 服务器 — Agent 聊天界面
使用 Flask 提供 HTTP API 和前端页面，接收用户消息并调用 agent 核心循环
"""
# 从 flask 导入所需组件：
# Flask        — Web 框架核心，创建应用对象
# render_template — 渲染 Jinja2 HTML 模板并返回给浏览器
# request      — 获取 HTTP 请求中的数据（JSON body、参数等）
# jsonify      — 将 Python 字典转换为 JSON 格式的 HTTP 响应
from flask import Flask, render_template, request, jsonify

# 从同目录的 agent 模块导入核心函数 run_agent_loop
# 该函数负责：接收用户消息 → 调用 LLM → 执行工具 → 循环直到得到最终回答
from agent import run_agent_loop

# 创建 Flask 应用实例
# __name__ 是当前模块名，Flask 用它来定位资源文件
# template_folder='templates' 指定 HTML 模板存放的目录路径
app = Flask(__name__, template_folder='templates')

# 会话存储：session_id -> 对话历史（不含system消息）
sessions = {}

# @app.route('/') 是路由装饰器，将根路径 '/'（首页）映射到下方的 index 函数
# 当用户用浏览器访问 http://localhost:5000/ 时触发
@app.route('/')
def index():
    # render_template 从 templates/ 目录找到 index.html
    # 使用 Jinja2 引擎渲染后返回 HTML 字符串，最终作为 HTTP 响应发送给浏览器
    return render_template('index.html')

# 定义 POST 类型的 API 接口 /api/chat
# method=['POST'] 限制该路由只接受 POST 请求（GET 请求会返回 405 Method Not Allowed）
# 前端通过 fetch/axios 向此地址发送 JSON 数据来与 agent 对话
@app.route('/api/chat', methods=['POST'])
def chat():
    # request.get_json() 解析请求体中的 JSON 数据，返回 Python 字典
    # 例如前端发送 {"message": "今天天气怎么样", "model": "deepseek"}
    data = request.get_json()

    # dict.get(key, default) 安全地取出 message 字段
    # 如果前端没传 message，默认为空字符串 ''，避免 KeyError 异常
    message = data.get('message', '')

    # 取出 model 字段，用于切换底层 LLM（当前只实现了 deepseek）
    # 如果前端没传，默认使用 'deepseek'
    model = data.get('model', 'deepseek')
    strategy = data.get('strategy', 'react')
    session_id = data.get('session_id', 'default')

    # 校验：如果消息为空（空字符串会被 if not 判定为 True）
    # 返回 HTTP 400 Bad Request 并附带错误信息的 JSON
    if not message:
        return jsonify({'error': 'Empty message'}), 400

    # 取出当前会话的历史消息
    history = sessions.get(session_id, [])

    # try-except 捕获 run_agent_loop 执行过程中可能抛出的任何异常
    # 例如：API 密钥未配置、网络超时、模型返回格式异常等
    try:
        response, updated_history = run_agent_loop(model, message, strategy=strategy, history=history)

        # 保存更新后的对话历史
        sessions[session_id] = updated_history

        # 将 agent 返回的文本包装为 JSON 响应 {"response": "..."}
        # 默认 HTTP 状态码 200，表示成功
        return jsonify({'response': response})

    except Exception as e:
        # 发生异常时，将异常信息转为字符串返回给前端
        # HTTP 状态码 500 表示服务器内部错误
        return jsonify({'error': str(e)}), 500

# Python 脚本的入口判断：
# 当直接运行 python app.py 时，__name__ 的值为 '__main__'，条件成立
# 当被其他模块 import 时（如 agent.py 中 from app import app），__name__ 为 'app'，条件不成立，不会启动服务器
if __name__ == '__main__':
    # 启动 Flask 内置的开发服务器
    # debug=True   — 开启调试模式：代码修改后自动重启，错误页面显示详细调用栈
    # port=5000    — 监听 5000 端口，访问地址为 http://127.0.0.1:5000
    app.run(debug=True, port=5000)
