from huggingface_hub.inference._generated.types import document_question_answering
from huggingface_hub.inference._generated.types import document_question_answering
import os
from dotenv import load_dotenv
from pageindex import PageIndexClient
from pathlib import Path
import json
# Load .env
load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY")

if not PAGEINDEX_API_KEY:
    raise ValueError(
        "PAGEINDEX_API_KEY not found. Please add it to .env"
    )

# Khởi tạo 1 lần
client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
def pageindex_search(query: str, top_k: int = 5):
    docs = client.list_documents()["documents"]

    all_results = []

    for doc in docs:
        doc_id = doc["id"]

        try:
            if not client.is_retrieval_ready(doc_id):
                continue

            submit_response = client.submit_query(
                doc_id=doc_id,
                query=query
            )

            retrieval_id = submit_response["retrieval_id"]

            retrieval_result = client.get_retrieval(
                retrieval_id
            )

            print(f"\n=== {doc['name']} ===")
            print(retrieval_result)

            all_results.append(
                {
                    "doc_name": doc["name"],
                    "doc_id": doc_id,
                    "response": retrieval_result
                }
            )

        except Exception as e:
            print(f"Error querying {doc['name']}: {e}")

    return all_results

def upload_documents():
    """
    Upload toàn bộ chunk lên PageIndex.
    """
    LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
    legal_dir = LANDING_DIR / "legal"
    for filepath in legal_dir.iterdir():
        if filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"Converting: {filepath.name}")
    client.submit_document(str(filepath))

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
        print("="*20)
        print(results)
        # for r in results:
        #     print(f"[{r['score']:.3f}] {r['content'][:100]}...")
