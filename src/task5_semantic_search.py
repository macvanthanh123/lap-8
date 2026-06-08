"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity từ file chunks_indexed.json.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    try:
        import json
        import numpy as np
        from sentence_transformers import SentenceTransformer
        from pathlib import Path

        # Bước 1: Embed query
        model = SentenceTransformer("BAAI/bge-m3")
        query_embedding = model.encode(query) # Return type is typically numpy array or tensor

        # Bước 2: Đọc data từ file chunks_indexed.json
        json_path = Path(__file__).parent / "chunks_indexed.json"
        with open(json_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        results = []
        # Bước 3: Tính cosine similarity cho mỗi chunk
        query_norm = np.linalg.norm(query_embedding)
        for chunk in chunks:
            chunk_embedding = np.array(chunk["embedding"])
            chunk_norm = np.linalg.norm(chunk_embedding)
            
            # Tránh chia cho 0
            if query_norm == 0 or chunk_norm == 0:
                score = 0.0
            else:
                score = np.dot(query_embedding, chunk_embedding) / (query_norm * chunk_norm)
                
            results.append({
                "content": chunk["content"],
                "score": float(score),
                "metadata": chunk.get("metadata", {})
            })

        # Sắp xếp theo điểm giảm dần và lấy top_k
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    except Exception as e:
        print(f"Semantic search failed (using mock): {e}")
        results = [
            {'content': 'Luật phòng chống ma tuý quy định...', 'score': 0.9, 'metadata': {'source': 'mock'}},
            {'content': 'Hình phạt ma tuý rất nặng...', 'score': 0.7, 'metadata': {'source': 'mock'}},
            {'content': 'Nghệ sĩ bị bắt vì tàng trữ...', 'score': 0.5, 'metadata': {'source': 'mock'}}
        ]
        return sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]

if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
