#读取data/knowledge_base/文件->切分->入 SQLite + Chroma
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from db.sqlite_client import SQLiteClient
from db.chroma_client import ChromaClient
from pathlib import Path

def init_knowledge_base():
    print("Initializing knowledge base...")
    
    sqlite_client = SQLiteClient()
    chroma_client = ChromaClient()
    
    knowledge_base_dir = Path(__file__).parent.parent / "data" / "knowledge_base"
    
    tutorials_dir = knowledge_base_dir / "tutorials"
    code_examples_dir = knowledge_base_dir / "code_examples"
    best_practices_dir = knowledge_base_dir / "best_practices"
    
    documents = []
    metadatas = []
    ids = []
    
    doc_id_counter = 1
    
    if tutorials_dir.exists():
        for file in tutorials_dir.glob("*.md"):
            with open(file, "r", encoding="utf-8") as f:
                content = f.read()
                documents.append(content)
                metadatas.append({
                    "type": "tutorial",
                    "filename": file.name,
                    "source": str(file)
                })
                ids.append(f"doc-{doc_id_counter:04d}")
                doc_id_counter += 1
    
    if code_examples_dir.exists():
        for file in code_examples_dir.glob("*.py"):
            with open(file, "r", encoding="utf-8") as f:
                content = f.read()
                documents.append(content)
                metadatas.append({
                    "type": "code",
                    "filename": file.name,
                    "source": str(file)
                })
                ids.append(f"doc-{doc_id_counter:04d}")
                doc_id_counter += 1
    
    if best_practices_dir.exists():
        for file in best_practices_dir.glob("*.md"):
            with open(file, "r", encoding="utf-8") as f:
                content = f.read()
                documents.append(content)
                metadatas.append({
                    "type": "best_practice",
                    "filename": file.name,
                    "source": str(file)
                })
                ids.append(f"doc-{doc_id_counter:04d}")
                doc_id_counter += 1
    
    if documents:
        chroma_client.add_documents(documents, metadatas, ids)
        print(f"Added {len(documents)} documents to Chroma")
    
    print("Knowledge base initialized successfully!")

if __name__ == "__main__":
    init_knowledge_base()
