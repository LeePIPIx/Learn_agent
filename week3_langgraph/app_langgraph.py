"""
Flask Web 服务器 — 使用 LangGraph agent

与 week1_miniagent/app.py 功能相同，但后端使用 LangGraph 编译的 graph。
运行在 5001 端口，可与原版同时运行对比。
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, render_template, request, jsonify
from day17_plan_execute_graph import run_agent_langgraph

_TEMPLATE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "week1_miniagent", "templates"
)
app = Flask(__name__, template_folder=_TEMPLATE_DIR)

sessions = {}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", "")
    model = data.get("model", "deepseek")
    strategy = data.get("strategy", "react")
    session_id = data.get("session_id", "default")

    if not message:
        return jsonify({"error": "Empty message"}), 400

    history = sessions.get(session_id, [])

    try:
        response, updated_history = run_agent_langgraph(
            model, message, strategy=strategy, history=history
        )
        sessions[session_id] = updated_history
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5001)
