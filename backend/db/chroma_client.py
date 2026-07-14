from chromadb import PersistentClient
from config import settings

class ChromaClient:
    def __init__(self):
        self.client = PersistentClient(path=settings.CHROMA_PATH)
        self.collection = self.client.get_or_create_collection("knowledge_base")

    def add_documents(self, documents: list, metadatas: list = None, ids: list = None) -> None:
        self.collection.add(
            documents=documents,
            metadatas=metadatas or [],
            ids=ids or []
        )

    def search(self, query: str, n_results: int = 5, where: dict = None) -> list:
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )
        return results

    def get_by_id(self, doc_id: str) -> dict:
        results = self.collection.get(ids=[doc_id])
        return results

    def delete_by_id(self, doc_id: str) -> None:
        self.collection.delete(ids=[doc_id])

    def count(self) -> int:
        return self.collection.count()
