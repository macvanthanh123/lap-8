import os
from dotenv import load_dotenv
from pageindex import PageIndexClient
from pathlib import Path
import json

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY")
pi = PageIndexClient(api_key=PAGEINDEX_API_KEY)

def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    # TODO: Implement PageIndex query
    #
    from pageindex import PageIndex
    
    results = pi.query(query=query, top_k=top_k)
    
    return [
        {
            "content": r.text,
            "score": r.score,
            "metadata": r.metadata,
            "source": "pageindex"
        }
        for r in results
    ]
    raise NotImplementedError("Implement pageindex_search")

def upload_documents():
    """
    Upload toàn bộ chunk lên PageIndex.
    """
    LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
    legal_dir = LANDING_DIR / "legal"
    for filepath in legal_dir.iterdir():
        if filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"Converting: {filepath.name}")
    pi.submit_document(str(filepath))

    print(f"Uploaded {str(filepath)} chunks")

if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        # upload_documents()

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
