import json
from pathlib import Path
from rank_bm25 import BM25Okapi

def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    # TODO: Implement lexical search
    json_path = Path(__file__).parent / "chunks_indexed.json"
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    bm25 = BM25Okapi([chunk['content'].lower().split() for chunk in chunks])
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    
    # Get top_k indices
    import numpy as np
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": chunks[idx]["content"],
                "score": float(scores[idx]),
                "metadata": chunks[idx]["metadata"]
            })
    return results
    raise NotImplementedError("Implement lexical_search")


if __name__ == "__main__":
    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
