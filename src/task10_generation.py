import os
import google.generativeai as genai
from src.task7_reranking import rerank


# =============================================================================
# CONFIG
# =============================================================================

TOP_K = 5
TEMPERATURE = 0.3
TOP_P = 0.9

SYSTEM_PROMPT = """
Bạn là trợ lý pháp lý AI.

Nhiệm vụ:
1. Chỉ trả lời dựa trên thông tin trong Context được cung cấp.
2. Không sử dụng kiến thức bên ngoài Context.
3. Không tự suy diễn hoặc bịa thông tin.
4. Khi đưa ra thông tin phải trích dẫn nguồn bằng định dạng:
   [Document X]
5. Nếu nhiều nguồn hỗ trợ cùng một ý:
   [Document 1, Document 2]
6. Nếu Context không đủ thông tin:
   "Tôi không tìm thấy đủ thông tin trong tài liệu được cung cấp."
7. Trả lời bằng tiếng Việt.
8. Với câu hỏi pháp luật:
   - Nêu căn cứ pháp lý nếu có.
   - Nêu điều luật, khoản, điểm nếu xuất hiện trong Context.
   - Không đưa ra lời khuyên pháp lý tuyệt đối.

Ưu tiên tính chính xác hơn độ dài câu trả lời.
"""

# =============================================================================
# REORDER
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Placeholder cho lost-in-the-middle mitigation.
    """
    return chunks


# =============================================================================
# FORMAT CONTEXT
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string.
    """

    context_parts = []

    for i, chunk in enumerate(chunks, start=1):

        metadata = chunk.get("metadata", {})

        source = metadata.get("source", f"Document-{i}")
        doc_type = metadata.get("type", "unknown")

        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk.get('content', '')}"
        )

    return "\n\n---\n\n".join(context_parts)


# =============================================================================
# GEMINI CALL
# =============================================================================

def call_gemini(prompt: str) -> str:

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY not found")

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=SYSTEM_PROMPT,
    )

    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
        },
    )

    return response.text


# =============================================================================
# RAG GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG pipeline.
    """

    # Step 1: Retrieve
    chunks = rerank(query, top_k=top_k)

    if not chunks:
        return {
            "answer": "Không tìm thấy tài liệu liên quan.",
            "sources": [],
            "retrieval_source": "none",
        }

    # Step 2: Reorder
    reordered_chunks = reorder_for_llm(chunks)

    # Step 3: Format context
    context = format_context(reordered_chunks)

    # Step 4: Build prompt
    prompt = f"""
Context:
{context}

----------------------------------------

Question:
{query}

Hãy trả lời bằng tiếng Việt và trích dẫn nguồn.
"""

    # Step 5: Gemini
    answer = call_gemini(prompt)

    # Step 6: Return
    retrieval_source = (
        chunks[0]
        .get("metadata", {})
        .get("retrieval_source", "hybrid")
    )

    return {
        "answer": answer,
        "sources": reordered_chunks,
        "retrieval_source": retrieval_source,
    }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":

    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for q in test_queries:

        print("\n" + "=" * 80)
        print("Q:", q)
        print("=" * 80)

        result = generate_with_citation(q)

        print("\nA:")
        print(result["answer"])

        print(
            f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]"
        )
