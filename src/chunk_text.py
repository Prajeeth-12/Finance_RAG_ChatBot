from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path("data")


def create_chunks(chunk_size=1000, chunk_overlap=200):
    """Split extracted text files into structured text chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    all_chunks = []
    for txt_path in sorted(DATA_DIR.glob("*.txt")):
        if txt_path.name == "chunks.txt":
            continue

        text = txt_path.read_text(encoding="utf-8")
        chunks = splitter.split_text(text)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "source": txt_path.name,
                "chunk_id": i,
                "text": chunk
            })

    # Save to consolidated chunks file
    output_path = DATA_DIR / "chunks.txt"
    with output_path.open("w", encoding="utf-8") as f:
        for item in all_chunks:
            f.write(f"SOURCE: {item['source']}\n")
            f.write(f"CHUNK_ID: {item['chunk_id']}\n")
            f.write(item["text"])
            f.write("\n" + "-" * 80 + "\n")

    print(f"Successfully processed {len(all_chunks)} chunks -> {output_path.name}")


if __name__ == "__main__":
    create_chunks()