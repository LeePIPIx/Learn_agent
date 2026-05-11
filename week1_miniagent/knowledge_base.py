"""
笔记本知识库 —— 基于 Chroma + ModelScope Embedding（国内可用）
读取 notes/ 目录下的 markdown/txt 文件，分块后向量化存储，支持语义检索。
"""
import os
import re
import chromadb
from modelscope import snapshot_download
from sentence_transformers import SentenceTransformer, CrossEncoder
from pathlib import Path

MODELSCOPE_MODEL = "iic/nlp_corom_sentence-embedding_chinese-base"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

def chunk_markdown(text: str, chunk_size: int = 500) -> list[str]:
    """将 markdown 文本按段落分块，超长的按句子再切。"""
    chunks = []
    paragraphs = re.split(r'\n{2,}', text)
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current_chunk) + len(para) <= chunk_size:
            current_chunk = (current_chunk + "\n\n" + para).strip() if current_chunk else para
        else:
            if current_chunk:
                chunks.append(current_chunk)
            if len(para) > chunk_size:
                sentences = re.split(r'(?<=[。！？.!?])', para)
                for s in sentences:
                    s = s.strip()
                    if not s:
                        continue
                    if len(current_chunk) + len(s) <= chunk_size:
                        current_chunk = (current_chunk + s).strip() if current_chunk else s
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = s
            else:
                current_chunk = para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return [c for c in chunks if len(c) >= 20]


class KnowledgeBase:
    def __init__(self, notes_dir: str = r"notes", persist_dir: str = "chroma_db"):
        self.notes_dir = Path(notes_dir)
        # 从 ModelScope 下载模型到本地，再加载
        local_path = snapshot_download(MODELSCOPE_MODEL)
        self.embedding_model = SentenceTransformer(local_path)
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.chroma_client.get_or_create_collection(
            name="notes",
            metadata={"hnsw:space": "cosine"}
        )
        self.rerank_model = None  # 延迟加载

    def _load_reranker(self):
        """延迟加载 reranker 模型（通过 ModelScope 下载）"""
        local_path = snapshot_download(RERANKER_MODEL)
        self.rerank_model = CrossEncoder(local_path)

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        """使用 cross-encoder 对候选结果重排序，返回 top_k 个最相关结果"""
        if not candidates:
            return []

        try:
            if self.rerank_model is None:
                self._load_reranker()

            pairs = [[query, c["content"]] for c in candidates]
            scores = self.rerank_model.predict(pairs, show_progress_bar=False)

            for i, score in enumerate(scores):
                candidates[i]["rerank_score"] = float(score)

            candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            return candidates[:top_k]

        except Exception as e:
            print(f"[KB] Reranker 失败: {e}，回退到原始排序")
            return candidates[:top_k]

    def build(self):
        """从 notes/ 目录读取所有文件，分块后存入 Chroma"""
        if not os.path.isdir(self.notes_dir):
            os.makedirs(self.notes_dir, exist_ok=True)
            print(f"[KB] 已创建空目录 '{self.notes_dir}'")
            return

        total_chunks = 0

        files = list(self.notes_dir.rglob("*.md"))
        if not files:
            print("[KB] notes 目录为空")
            return
        for filepath in files:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            filename = filepath.name
            chunks = chunk_markdown(content)
            if not chunks:
                continue

            ids = [f"{filename}_{i}" for i in range(len(chunks))]
            embeddings = self.embedding_model.encode(chunks).tolist()
            metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]

            existing = self.collection.get(where={"source": filename})
            if existing["ids"]:
                self.collection.delete(ids=existing["ids"])

            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas
            )
            total_chunks += len(chunks)
            print(f"[KB] 已索引: {filename} ({len(chunks)} chunks)")

        print(f"[KB] 构建完成，共 {total_chunks} 个片段")

    def search(self, query: str, k: int = 5) -> list[dict]:
        """语义检索，返回最相关的 k 个笔记片段"""
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
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "source": results["metadatas"][0][i]["source"],
                "distance": results["distances"][0][i]
            })
        return output

    def add_note(self, content: str, source: str = "manual"):
        """手动添加一条笔记"""
        chunks = chunk_markdown(content)
        if not chunks:
            return
        ids = [f"{source}_{self.collection.count() + i}" for i in range(len(chunks))]
        embeddings = self.embedding_model.encode(chunks).tolist()
        metadatas = [{"source": source, "chunk_index": i} for i in range(len(chunks))]
        self.collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
        print(f"[KB] 已添加 {len(chunks)} 条笔记")

    def format_results(self, results: list[dict]) -> str:
        """将检索结果格式化为 LLM 可读的文本"""
        if not results:
            return "知识库中没有找到相关内容。"
        lines = []
        for r in results:
            lines.append(f"[来源: {r['source']}] {r['content']}")
        return "\n\n---\n\n".join(lines)


# 全局单例 —— 路径基于本文件所在目录，不受启动位置影响
_base_dir = os.path.dirname(os.path.abspath(__file__))
kb = KnowledgeBase(
    notes_dir=os.path.join(_base_dir, "notes"),
    persist_dir=os.path.join(_base_dir, "chroma_db")
)

if __name__ == "__main__":
    kb.build()
    while True:
        q = input("\n搜索笔记 (输入 quit 退出): ")
        if q == "quit":
            break
        results = kb.search(q)
        print(kb.format_results(results))
