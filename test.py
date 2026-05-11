from pathlib import Path

root = Path("/home/ljr/workspace/DL/Project_agent/week1_miniagent/notes")

md_files = list(root.rglob("*.md"))

for f in md_files:
    print(f)