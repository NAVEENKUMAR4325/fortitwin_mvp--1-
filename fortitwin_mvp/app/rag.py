import os, argparse
from typing import List
import chromadb
from chromadb.config import Settings

DB_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

def get_client():
    os.makedirs(DB_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=DB_DIR, settings=Settings(anonymized_telemetry=False))
    return client

def collection():
    client = get_client()
    return client.get_or_create_collection("fortitwin_docs")

def ingest_dir(path: str):
    col = collection()
    docs, ids = [], []
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
                    txt = fh.read()
                docs.append(txt)
                ids.append(fp)
            except Exception:
                continue
    if docs:
        col.upsert(documents=docs, ids=ids)

def retrieve(query: str, k: int = 3) -> str:
    if not query:
        return ""
    col = collection()
    res = col.query(query_texts=[query], n_results=k)
    chunks: List[str] = []
    for doc in res.get("documents", [[]])[0]:
        chunks.append(doc[:1200])
    return "\n\n---\n\n".join(chunks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest", type=str, required=True, help="Directory of text docs to ingest")
    args = parser.parse_args()
    ingest_dir(args.ingest)
    print("Ingested docs from", args.ingest)
