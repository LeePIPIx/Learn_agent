#!/bin/bash
# 本地笔记推送脚本 — 将本地 .md 文件推送到服务器 agent 的知识库
# 用法:
#   ./push_note.sh ~/notes/my_note.md                          # 推送单个文件
#   ./push_note.sh ~/notes/my_note.md http://myserver:5000      # 指定服务器地址
#   ./push_note.sh --list http://myserver:5000                  # 列出服务器上的笔记
#   ./push_note.sh --delete agent_arch.md http://myserver:5000  # 删除服务器上的笔记

SERVER="${2:-http://localhost:5000}"
ACTION="$1"

if [ -z "$ACTION" ]; then
    echo "用法:"
    echo "  $0 <note.md> [server_url]           推送笔记"
    echo "  $0 --list [server_url]              列出服务器笔记"
    echo "  $0 --delete <filename> [server_url] 删除笔记"
    echo "  $0 --rebuild [server_url]           手动重建知识库"
    exit 1
fi

case "$ACTION" in
    --list)
        curl -s "$SERVER/api/notes" | python3 -m json.tool
        ;;
    --delete)
        if [ -z "$2" ]; then
            echo "请指定要删除的文件名"
            exit 1
        fi
        SERVER="${3:-http://localhost:5000}"
        curl -s -X DELETE "$SERVER/api/notes/$2" | python3 -m json.tool
        ;;
    --rebuild)
        curl -s -X POST "$SERVER/api/notes/rebuild" | python3 -m json.tool
        ;;
    *)
        if [ ! -f "$ACTION" ]; then
            echo "错误: 文件不存在: $ACTION"
            exit 1
        fi
        FILENAME=$(basename "$ACTION")
        echo "推送: $FILENAME → $SERVER"
        # 读取文件内容并构造 JSON，用 jq 处理转义；没有 jq 则用 python
        if command -v jq &>/dev/null; then
            CONTENT=$(< "$ACTION" jq -Rs .)
            JSON="{\"filename\":\"$FILENAME\",\"content\":$CONTENT}"
        else
            JSON=$(python3 -c "
import json, sys
with open('$ACTION', 'r') as f:
    content = f.read()
print(json.dumps({'filename': '$FILENAME', 'content': content}))
")
        fi
        curl -s -X POST "$SERVER/api/notes" \
            -H "Content-Type: application/json" \
            -d "$JSON" | python3 -m json.tool
        ;;
esac
