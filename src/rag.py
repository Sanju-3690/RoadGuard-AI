from pathlib import Path
import json
import chromadb
from sentence_transformers import SentenceTransformer
from src.config import settings

class RoadKnowledgeRAG:
    COLLECTION = "roadguard_maintenance"

    def __init__(self):
        self.vector_dir = Path(settings.vector_dir)
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.vector_dir))
        self.embedder = SentenceTransformer(settings.embedding_model)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION,
            metadata={"description": "Road maintenance knowledge for RoadGuard AI"},
        )
        if self.collection.count() == 0:
            self.build_index()

    def _load_records(self):
        path = settings.knowledge_data / "maintenance_knowledge.jsonl"
        if not path.exists():
            return []
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
        return records

    def build_index(self):
        records = self._load_records()
        if not records:
            raise FileNotFoundError("Missing data/knowledge/maintenance_knowledge.jsonl")
        vectors = self.embedder.encode([r["text"] for r in records], normalize_embeddings=True).tolist()
        self.collection.upsert(
            ids=[r["id"] for r in records],
            documents=[r["text"] for r in records],
            metadatas=[{
                "title": r["title"], "source": r["source"], "topic": r["topic"], "url": r.get("url", "")
            } for r in records],
            embeddings=vectors,
        )

    def retrieve(self, query, top_k=4):
        q = self.embedder.encode([query], normalize_embeddings=True)[0].tolist()
        result = self.collection.query(
            query_embeddings=[q], n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        output = []
        for doc, meta, distance in zip(result.get("documents", [[]])[0], result.get("metadatas", [[]])[0], result.get("distances", [[]])[0]):
            item = dict(meta or {})
            item["text"] = doc
            item["distance"] = round(float(distance), 4)
            output.append(item)
        return output
