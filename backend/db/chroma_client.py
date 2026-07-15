try:
    from chromadb import PersistentClient
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

from config import settings


class ChromaClient:
    def __init__(self):
        if CHROMA_AVAILABLE:
            self.client = PersistentClient(path=settings.CHROMA_PATH)
            self.collection = self.client.get_or_create_collection("knowledge_base")
        else:
            self.client = None
            self.collection = None

    def add_documents(self, documents: list, metadatas: list = None, ids: list = None) -> None:
        if not CHROMA_AVAILABLE:
            return
        self.collection.add(
            documents=documents,
            metadatas=metadatas or [],
            ids=ids or []
        )

    def search(self, query: str, n_results: int = 5, where: dict = None) -> list:
        if not CHROMA_AVAILABLE:
            return []
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )
        return results

    def get_by_id(self, doc_id: str) -> dict:
        if not CHROMA_AVAILABLE:
            return {}
        results = self.collection.get(ids=[doc_id])
        return results

    def delete_by_id(self, doc_id: str) -> None:
        if not CHROMA_AVAILABLE:
            return
        self.collection.delete(ids=[doc_id])

    def count(self) -> int:
        if not CHROMA_AVAILABLE:
            return 0
        return self.collection.count()