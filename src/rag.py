import re
import chromadb
try:
    from src.nvidia_client import get_embedding, get_llm_completion
except ImportError:
    from nvidia_client import get_embedding, get_llm_completion


# ============================================================
# CONFIGURATION
# ============================================================

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "finance_documents"


# ============================================================
# CONNECT TO CHROMADB
# ============================================================

client = chromadb.PersistentClient(path=CHROMA_DIR)

collection = client.get_collection(
    name=COLLECTION_NAME
)


# ============================================================
# DETECT QUARTER
# ============================================================

def detect_quarter(question):
    """
    Detect Q1, Q2, Q3 or Q4 from the question.
    """

    match = re.search(
        r"\b(Q[1-4])\s*(?:FY)?26\b",
        question.upper()
    )

    if match:
        return match.group(1)

    return None


# ============================================================
# IDENTIFY QUESTION TYPE
# ============================================================

def detect_question_type(question):
    """
    Identify whether the question is about revenue,
    EBIT, or net income.
    """

    q = question.lower()

    if "revenue" in q:
        return "revenue"

    if "ebit" in q:
        return "ebit"

    if (
        "net income" in q
        or "net profit" in q
        or re.search(r"\bni\b", q)
    ):
        return "net_income"

    return "general"


# ============================================================
# SEARCH DOCUMENTS
# ============================================================

def search_documents(question, top_k=5):
    """
    Retrieve the best financial chunks.

    For a specific quarter, only that quarter's report
    is considered.

    For financial questions, explicit quarterly highlights
    are strongly preferred over annual figures.
    """

    quarter = detect_quarter(question)
    question_type = detect_question_type(question)

    # --------------------------------------------------------
    # Create question embedding
    # --------------------------------------------------------

    question_embedding = get_embedding(question, input_type="query")

    # --------------------------------------------------------
    # Search all chunks
    # --------------------------------------------------------

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=244
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    candidates = []

    # --------------------------------------------------------
    # Score every chunk
    # --------------------------------------------------------

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        source = metadata["source"].upper()
        text = document.lower()

        # ----------------------------------------------------
        # QUARTER FILTER
        # ----------------------------------------------------

        if quarter:

            expected_source = f"HCLTECH_{quarter}_FY26.TXT"

            if expected_source not in source:
                continue

        score = 0

        # ----------------------------------------------------
        # GENERAL RELEVANCE
        # ----------------------------------------------------

        if "revenue" in text:
            score += 10

        if "qoq" in text:
            score += 10

        if "yoy" in text:
            score += 10

        # ----------------------------------------------------
        # QUARTERLY FINANCIAL HIGHLIGHTS
        # ----------------------------------------------------

        # This is extremely important because the quarterly
        # highlights contain the exact quarter figures.

        if "qoq" in text and "yoy" in text:
            score += 100

        if "profitability & return metrics" in text:
            score += 100

        # ----------------------------------------------------
        # REVENUE QUESTION
        # ----------------------------------------------------

        if question_type == "revenue":

            if "inr revenue of" in text:
                score += 500

            if "revenue of" in text:
                score += 200

            # Explicit quarterly growth
            if "qoq" in text and "yoy" in text:
                score += 100

        # ----------------------------------------------------
        # EBIT QUESTION
        # ----------------------------------------------------

        elif question_type == "ebit":

            if "inr ebit at" in text:
                score += 500

            if "ebit at" in text:
                score += 200

            if "profitability & return metrics" in text:
                score += 100

        # ----------------------------------------------------
        # NET INCOME QUESTION
        # ----------------------------------------------------

        elif question_type == "net_income":

            # Exact quarterly statement:
            # NI at ₹4,488 Crores ...
            if "ni at" in text:
                score += 500

            # Net Income wording
            if "net income" in text:
                score += 200

            # Quarterly growth makes it much more likely
            # to be the requested quarter rather than FY26.
            if "qoq" in text and "yoy" in text:
                score += 300

            if "profitability & return metrics" in text:
                score += 150

            # Annual-only wording gets penalized.
            if "for the year" in text:
                score -= 300

            if "fy26 results" in text:
                score -= 300

        # ----------------------------------------------------
        # PENALIZE ANNUAL-ONLY FINANCIAL FIGURES
        # ----------------------------------------------------

        if quarter:

            if "for the year came in" in text:
                score -= 300

            if "for the year" in text:
                score -= 200

            if "fy26 results" in text:
                score -= 200

        # ----------------------------------------------------
        # COMBINE WITH SEMANTIC DISTANCE
        # ----------------------------------------------------

        final_score = score - distance

        candidates.append(
            (
                final_score,
                document,
                metadata,
                distance
            )
        )

    # --------------------------------------------------------
    # SORT BEST RESULTS
    # --------------------------------------------------------

    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    # --------------------------------------------------------
    # SELECT TOP RESULTS
    # --------------------------------------------------------

    selected = candidates[:top_k]

    return {
        "documents": [
            [item[1] for item in selected]
        ],
        "metadatas": [
            [item[2] for item in selected]
        ],
        "distances": [
            [item[3] for item in selected]
        ],
        "scores": [
            [item[0] for item in selected]
        ],
        "pipeline_info": {
            "quarter": quarter or "All Quarters (FY26)",
            "question_type": question_type.replace("_", " ").title(),
            "total_candidates": len(candidates),
            "total_scanned": len(documents)
        }
    }


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(question, results, model=None, **kwargs):
    """
    Generate an answer using ONLY retrieved context.
    """

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    if not documents:
        return (
            "I could not find this information in the "
            "provided financial reports."
        )

    # --------------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------------

    context_parts = []

    for i, document in enumerate(documents):

        source = metadatas[i]["source"]
        chunk_id = metadatas[i]["chunk_id"]

        context_parts.append(
            f"[Source: {source}, Chunk: {chunk_id}]\n"
            f"{document}"
        )

    context = "\n\n".join(context_parts)

    # --------------------------------------------------------
    # STRICT FINANCIAL PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are an expert financial research analyst for HCLTech.

Answer the user's question clearly, professionally, and accurately using ONLY the provided REPORT CONTEXT.

STRICT GROUNDING RULES:
1. Base your answer ENTIRELY on the provided REPORT CONTEXT. Do not use outside knowledge or make assumptions.
2. State exact monetary figures (e.g. ₹ / INR / Crores) and growth percentages (QoQ / YoY) as given in the report context.
3. If the user asks for a specific quarter (Q1, Q2, Q3, Q4), clearly highlight the quarterly metric for that specific quarter.
4. Do not invent, calculate, or estimate numbers.
5. Provide a clear, well-structured financial summary.

If the requested financial data is not present in the context, reply:
"I could not find this information in the provided financial reports."

USER QUESTION:
{question}

REPORT CONTEXT:
{context}

EXECUTIVE FINANCIAL SUMMARY:
"""

    if model:
        return get_llm_completion(prompt, model=model, temperature=0.0)
    return get_llm_completion(prompt, temperature=0.0)


# ============================================================
# ASK QUESTION
# ============================================================

def ask_question(question):

    results = search_documents(question)

    answer = generate_answer(
        question,
        results
    )

    print("\n" + "=" * 70)
    print("QUESTION")
    print("=" * 70)
    print(question)

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(answer)

    print("\n" + "=" * 70)
    print("SOURCES")
    print("=" * 70)

    for metadata in results["metadatas"][0]:

        print(
            f"- {metadata['source']} | "
            f"Chunk {metadata['chunk_id']}"
        )


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    question = input(
        "\nAsk a question about HCLTech's financial reports: "
    )

    ask_question(question)