"""Tai san embedding model truoc khi vao mang LAN offline.

Spec BAT BUOC dung keepitreal/vietnamese-sbert. Chay khi con Internet:
    uv run python download_model.py
"""
import os

from sentence_transformers import SentenceTransformer


if __name__ == "__main__":
    model_name = os.getenv("EMBEDDING_MODEL", "keepitreal/vietnamese-sbert")
    print(f"Downloading/caching: {model_name}")
    model = SentenceTransformer(model_name)
    # Encode thu de chac chan model nap duoc va warm cache.
    _ = model.encode(["xin chao", "kiem tra embedding tieng Viet"])
    print("Done. Variant v5 da co the nap embedding model tu local cache.")
