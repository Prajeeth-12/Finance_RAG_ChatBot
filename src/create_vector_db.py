import shutil
from pathlib import Path
import chromadb

try:
    from src.nvidia_client import get_embeddings_batch
except ImportError:
    from nvidia_client import get_embeddings_batch

DATA_DIR = Path("data")
CHROMA_DIR = Path("chroma_db")


def build_vector_database(batch_size=32):
    """Parse chunks and build ChromaDB persistent vector index using NVIDIA NIM embeddings."""
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(name="finance_documents")

    chunks_path = DATA_DIR / "chunks.txt"
    raw_blocks = [block.strip() for block in chunks_path.read_text(encoding="utf-8").split("-" * 80) if block.strip()]

    documents, ids, metadatas = [], [], []

    for idx, block in enumerate(raw_blocks):
        lines = block.splitlines()
        source, chunk_id = "unknown", str(idx)

        for line in lines:
            if line.startswith("SOURCE:"):
                source = line.replace("SOURCE:", "").strip()
            elif line.startswith("CHUNK_ID:"):
                chunk_id = line.replace("CHUNK_ID:", "").strip()

        doc_text = "\n".join(line for line in lines if not line.startswith(("SOURCE:", "CHUNK_ID:"))).strip()
        if not doc_text:
            continue

        documents.append(doc_text)
        ids.append(f"{source}_{chunk_id}")
        metadatas.append({"source": source, "chunk_id": chunk_id})

    # Batch embed via NVIDIA NIM
    embeddings = []
    print(f"Generating embeddings for {len(documents)} document chunks (Batch size: {batch_size})...")
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        batch_vectors = get_embeddings_batch(batch, input_type="passage")
        embeddings.extend(batch_vectors)
        print(f"Embedded {len(embeddings)}/{len(documents)} chunks")

    if documents:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    print(f"Successfully created vector database at {CHROMA_DIR} ({collection.count()} chunks indexed).")


if __name__ == "__main__":
    build_vector_database()