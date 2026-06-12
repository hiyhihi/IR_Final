"""Tai san embedding model truoc khi vao mang LAN offline.

Mac dinh `submission_final/` dung BM25 nen khong bat buoc file nay.
Neu doi `RETRIEVER_BACKEND=hybrid|sbert|vector`, hay chay script nay khi con Internet.
"""
import os

from sentence_transformers import SentenceTransformer


if __name__ == "__main__":
    model_name = os.getenv("EMBEDDING_MODEL", "keepitreal/vietnamese-sbert")
    print(f"Downloading/caching: {model_name}")
    model = SentenceTransformer(model_name)
    # Encode thu de chac chan model nap duoc va warm cache.
    _ = model.encode(["xin chao", "kiem tra embedding tieng Viet"])
    print("Done. submission_final da co the nap embedding model tu local cache.")
