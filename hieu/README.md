# Student RAG Server - v6 (ban thi cuoi ky)

Hop nhat diem manh cua cac variant truoc, vot cho de KHO hon (100 cau, tai lieu dai,
LLM proxy yeu ~4k context, 5 lan nop lap lai cung bo cau):

- **Retrieval hybrid** BM25 + dense vector (+ rerank tuy chon). BM25 dung
  **inverted index dung san luc upload** -> moi cau hoi khong tokenize lai corpus.
  Fusion `weighted` hoac `rrf` (`HYBRID_FUSION`). Query retrieval = **than cau hoi**
  (bo options, `RETRIEVAL_INCLUDE_OPTIONS=false`).
- **Chuan hoa van ban sau** (`text_utils.py`): NFKC, sua dinh chu-so ("15ngày"),
  loc dong metadata (`- id:`, `- chu_de:`, ...), tokenize **bo dau + bigram + trigram**
  -> BM25 van khop khi cau hoi thieu/khac dau.
- **Chunking tach theo cau**, overlap bam ranh gioi tu, manh vun < `MIN_CHUNK_LEN`
  duoc gop vao chunk ke ben (khong vut noi dung).
- **Heuristic cham phuong an** (`score_options`): khop nguyen van, khop con so,
  dieu/khoan, tu dao tieu chi khi cau hoi phu dinh (KHONG/tru/khac). Dung lam
  fallback khi LLM loi; co the bat `OVERRIDE_WITH_HEURISTIC` de de LLM khi bang
  chung cuc ro.
- **Prompt "evidence"** huong dan doi chieu chu the/tham quyen/thoi han/con so,
  xu ly cau phu dinh (`PROMPT_MODE=evidence`, doi `compact` neu muon gon).
- **Khu trung lap khi retrieve** (`DEDUP_THRESHOLD=0.92`): context da dang hon.
- **Embedding `keepitreal/vietnamese-sbert`** dung spec, load **offline-first**
  (`local_files_only` truoc, co mang moi tai) -> khong cho HF check mang khi thi.
- **Context khit cua so ~4k**: chi nhoi `CONTEXT_TOP_K` (mac dinh 4) chunk tot nhat,
  gioi han `MAX_CONTEXT_CHARS` (mac dinh 2800) -> tranh tran context lam model yeu tra bua.
- **LLM retry trong ngan sach 60s/cau**: `LLM_TIMEOUT=25` x (`LLM_RETRIES`+1) lan,
  het luot roi ve heuristic local -> `/ask` KHONG BAO GIO tra 500.
- **Answer cache xuong disk** (`cache/answer_cache.jsonl`): 5 lan nop lap lai dung lai
  dap an cu -> nhat quan + tuc thi, khong so LLM "lat" dap an.
- **Persist VectorDB xuong disk** (`vectordb/`) + auto-load khi restart (spec BAT BUOC):
  restart server roi nop lai voi `--skip-upload` van tra loi duoc, khong can upload lai.
- **Embedding/LLM chay trong threadpool** -> event loop khong nghen, health check
  `GET /` van phan hoi trong luc embed tai lieu dai.

## Cai dat bang uv

```bash
cd v5
uv sync
uv run python download_model.py   # tai keepitreal/vietnamese-sbert khi CON Internet
cp .env.example .env
```

Fallback pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python download_model.py
cp .env.example .env
```

Sua `.env`: `STUDENT_ID`, `STUDENT_SERVER_URL` (IP LAN cua may, port 5004).

## Chay

Terminal 1:

```bash
cd v5
uv run python main.py
```

Terminal 2:

```bash
cd v5
uv run python client.py register
uv run python client.py evaluate
uv run python client.py result
```

## Luong 5 lan nop

- Lan 1: `python client.py evaluate` (gui `document_received=false`) -> Teacher goi
  `/upload` roi 100 `/ask`. Server build + persist VectorDB, tra loi va luu cache.
  Neu lan dau bao loi upload (embed lau hon timeout 120s cua Teacher la BINH THUONG):
  doi server log "Persisted VectorDB" roi chay lai voi `--skip-upload`.
- Lan 2-5: `python client.py evaluate --skip-upload` (gui `document_received=true`) ->
  Teacher chi goi `/ask`, server tra lai dap an tu cache -> nhanh & on dinh.
- Muon reset diem de nop lai: `python client.py reset` (hoac `register` lai).
- Muon doi config retrieval giua cac lan nop: xoa `cache/answer_cache.jsonl` truoc,
  neu khong server se tra dap an cu tu cache.

## Test nhanh o nha

```bash
curl -s -X POST http://127.0.0.1:5004/upload \
  -H 'Content-Type: application/json' \
  -d '{"doc_id":"demo","text":"RAG gom retrieval va generation. Endpoint gui tai lieu la /upload."}'

curl -s -X POST http://127.0.0.1:5004/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"Endpoint gui tai lieu la gi?","options":{"A":"/ask","B":"/upload","C":"/result","D":"/proxy"}}'
```

## Tinh chinh khi thi

- Tai lieu rat dai, retrieve chua trung -> tang `TOP_K`/`VECTOR_CANDIDATES`, giam `CHUNK_SIZE`.
- LLM hay tra bua/tran context -> giam `MAX_CONTEXT_CHARS` (vd 2000) hoac `CONTEXT_TOP_K=3`.
- Cau hoi trich dan tu ngu nguyen van trong doc -> tang `BM25_WEIGHT` (vd 0.7-0.8).
- Cau hoi dien dat lai/y nghia -> giam `BM25_WEIGHT` (vd 0.4) de nghieng ve dense.
- Muon chinh xac hon va con thoi gian -> dien `RERANKER_MODEL=BAAI/bge-reranker-v2-m3`.
- Lam lai sach se: `rm -rf cache vectordb uploads`.
