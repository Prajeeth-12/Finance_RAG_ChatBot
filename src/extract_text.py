from pathlib import Path
from pypdf import PdfReader

DATA_DIR = Path("data")


def extract_pdf_texts():
    """Extract raw text from PDF financial reports into .txt files."""
    for pdf_path in sorted(DATA_DIR.glob("*.pdf")):
        reader = PdfReader(str(pdf_path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        
        output_path = pdf_path.with_suffix(".txt")
        output_path.write_text(text, encoding="utf-8")
        print(f"Extracted {pdf_path.name} -> {output_path.name} ({len(reader.pages)} pages, {len(text)} chars)")


if __name__ == "__main__":
    extract_pdf_texts()