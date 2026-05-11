"""
长期记忆存储 —— 基于 Chroma 持久化，支持语义检索
从对话中自动/手动提取关键信息，跨会话持久保留
"""
import os
import chromadb
from modelscope import snapshot_download
from sentence_transformers import SentenceTransformer
from datetime import datetime

MODEL_PATH = "iic/nlp_corom_sentence-embedding_chinese-base"


class MemoryStore:
    def __init__(self, persist_dir="chroma_db"):
        local_path = snapshot_download(MODEL_PATH)
        self.embedding_model = SentenceTransformer(local_path)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="memories",
            metadata={"hnsw:space": "cosine"}
        )

    def remember(self, content: str, category: str = "general") -> str:
        """存储一条记忆"""
        timestamp = datetime.now().isoformat()
        embedding = self.embedding_model.encode([content]).tolist()
        count = self.collection.count()
        self.collection.add(
            ids=[f"mem_{count}"],
            embeddings=embedding,
            documents=[content],
            metadatas=[{"category": category, "timestamp": timestamp}]
        )
        return f"已记住 [{category}]: {content}"

    def recall(self, query: str, k: int = 5) -> list[dict]:
        """语义检索相关记忆"""
        if self.collection.count() == 0:
            return []
        query_embedding = self.embedding_model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(k, self.collection.count())
        )
        output = []
        for i in range(len(results["ids"][0])):
            output.append({
                "content": results["documents"][0][i],
                "category": results["metadatas"][0][i]["category"],
                "timestamp": results["metadatas"][0][i]["timestamp"]
            })
        return output

    def forget(self, keyword: str) -> str:
        """删除包含关键词的记忆"""
        if self.collection.count() == 0:
            return "没有可删除的记忆"
        all_data = self.collection.get()
        ids_to_delete = [
            all_data["ids"][i] for i, doc in enumerate(all_data["documents"])
            if keyword in doc
        ]
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
            return f"已删除 {len(ids_to_delete)} 条包含'{keyword}'的记忆"
        return f"未找到包含'{keyword}'的记忆"

    def format_for_prompt(self, memories: list[dict]) -> str:
        """格式化记忆为 prompt 可用的文本"""
        if not memories:
            return ""
        lines = ["[相关历史记忆]"]
        for m in memories:
            lines.append(f"- [{m['category']}] {m['content']} (记录于 {m['timestamp'][:10]})")
        return "\n".join(lines)


# 全局单例
memory = MemoryStore()
