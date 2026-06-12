# Student RAG Server

FastAPI server cho bai thi Offline RAG. Code chi phuc vu 2 endpoint Teacher Server se goi:

- `POST /upload`: nhan tai lieu, chunking, index vao VectorDB, luu VectorDB xuong disk.
- `POST /ask`: nhan cau hoi trac nghiem, retrieve context, goi LLM proxy, tra ve dap an `A/B/C/D`.

## Cai dat

```bash
cach 1:
python -m venv .venv
# source .venv/bin/activate #mac/linux
.venv\Scripts\activate # window
pip install -r requirements.txt
cp .env.example .env # chạy bằng powershell
```

```bash
cach 2:
pip install uv
uv sync
```

Sua `.env`:

```env
STUDENT_ID=B22XXXXXXX
STUDENT_SERVER_URL=http://192.168.50.xxx:5004
TEACHER_BASE_URL=http://192.168.50.218:8000/api/v1
LLM_BASE_URL=http://192.168.50.218:8000/api/v1/proxy
HOST=0.0.0.0
PORT=5004
```
python download_model.py

Neu benchmark local trong may cua ban dang dung OpenAI-compatible endpoint rieng,
hay de y `LLM_BASE_URL`. Trong log `v2` chay tot truoc do, endpoint local dung la
`http://127.0.0.1:8386/v1`, khong phai `http://127.0.0.1:8888/v1`.

## Chay server

```bash
python main.py
```

Khi dang ky voi Teacher Server, dung dia chi LAN cua may dang chay server, vi du:

```text
http://192.168.50.xxx:5004
```

## Luong xu ly

Lan dau evaluate voi `document_received=false`, Teacher Server se goi `/upload` de gui tai lieu. Server se chunk, index va luu VectorDB vao `PERSIST_DIR`.

Cac lan evaluate sau voi `document_received=true`, Teacher Server bo qua `/upload` va goi thang `/ask`. Khi server restart, VectorDB duoc load lai tu disk neu cau hinh khong doi.

## File chinh

- `main.py`: FastAPI app va logic `/upload`, `/ask`.
- `rag.py`: chunking, embedding, BM25/hybrid retrieval helpers.
- `vector_store.py`: luu chunks, search, save/load VectorDB.
- `llm_client.py`: goi LLM proxy, prompt, parse dap an `A/B/C/D`.
- `text_utils.py`: clean text va tao retrieval query.
- `config.py`: cau hinh tu `.env`.

## Flows

# 1. Chạy Student Server trước
python main.py

# 2. Đăng ký URL server của bạn
python client.py register

# 3. Bắt đầu chấm lần đầu: Teacher sẽ gọi /upload rồi hỏi /ask
python client.py evaluate

# 4. Nếu VectorDB đã có sẵn, bỏ qua upload
python client.py evaluate --doc-received

# 5. Xem điểm hiện tại
python client.py result

# 6. Reset trạng thái
python client.py reset