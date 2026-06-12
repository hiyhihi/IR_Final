# RAG IR Submission - File Structure

## 🚀 To Get Started

**1. Read this first:**
- `START_HERE.txt` - Quick overview (30 seconds)
- `SETUP.md` - Detailed setup guide (if you have questions)

**2. Just run one of these:**
- **Windows Git Bash:** `bash run.sh`
- **Windows Command Prompt:** `run.bat`
- **Windows PowerShell:** `bash run.sh`

---

## 📁 File Structure

### 🔴 CRITICAL (Don't modify)
```
run.sh              - Main bash script - RUN THIS on Linux/Mac/WSL/Git Bash
run.bat             - Main batch script - RUN THIS on Windows CMD
main.py             - FastAPI server (Student RAG endpoint)
config.py           - Load configuration from .env
rag.py              - RAG pipeline (chunking, retrieval)
vector_store.py     - Vector database (FAISS + pickle)
llm_client.py       - LLM API client wrapper
client.py           - Teacher API client (register, evaluate, result)
```

### ⚙️ Configuration
```
.env                - Configuration (IMPORTANT: Check teacher server URL!)
.env.example        - Example configuration template
pyproject.toml      - Python project metadata (uv)
requirements.txt    - Python dependencies
```

### 📚 Documentation
```
START_HERE.txt      - Quick start guide
SETUP.md            - Detailed setup & troubleshooting
README.md           - Original v5 documentation
INDEX.md            - This file
```

### 📦 Generated (after running)
```
.venv/              - Python virtual environment (auto-created)
vectordb/           - Vector database storage (auto-created)
cache/              - Answer cache (auto-created)
uploads/            - Uploaded documents (auto-created)
server.log          - Server logs (auto-created)
__pycache__/        - Python bytecode (auto-created, ignored)
```

### ⚠️ DO NOT MODIFY
```
.gitignore          - Git ignore rules
download_model.py   - Helper to download embedding model
```

---

## 🎯 What Each File Does

| File | Purpose |
|------|---------|
| `run.sh` / `run.bat` | **Everything in one script**: setup + run + submit |
| `main.py` | FastAPI server that student teacher calls via `/upload` and `/ask` |
| `config.py` | Read `.env` into Python `Config` object |
| `rag.py` | Text chunking + hybrid (BM25 + semantic) retrieval |
| `vector_store.py` | FAISS vector database with persistence |
| `llm_client.py` | Wrapper around LLM proxy (OpenAI API compatible) |
| `client.py` | CLI to call teacher: register / evaluate / result |
| `.env` | Configuration: teacher URL, student ID, LLM settings |

---

## 🔄 Execution Flow

```
run.sh / run.bat
    ↓
[1] Install uv (if needed)
    ↓
[2] Create .venv
    ↓
[3] pip install (dependencies)
    ↓
[4] python download_model.py (embedding model)
    ↓
[5] python main.py (start server in background)
    ↓
[6] python client.py register (register with teacher)
    ↓
[7] python client.py evaluate (run 100 Qs × 5 times = submit)
    ↓
[8] python client.py result (fetch score)
    ↓
[9] Kill server, done!
```

---

## ⚙️ Key Configuration (.env)

Must be set correctly:
- `TEACHER_BASE_URL` - Teacher server URL (e.g., http://192.168.50.218:8000/api/v1)
- `STUDENT_ID` - Your student ID (e.g., B22BENCHMARK)
- `STUDENT_SERVER_URL` - Your server address (default: http://127.0.0.1:5004)
- `LLM_BASE_URL` - LLM proxy URL (e.g., http://127.0.0.1:8888/v1)

Optional:
- `DEVICE` - cpu or cuda (default: cpu)
- `TOP_K` - Number of chunks to retrieve (default: 8)
- `MAX_CONTEXT_CHARS` - Max context length to LLM (default: 2800)
- `ENABLE_ANSWER_CACHE` - Cache answers (default: true)

---

## 📊 Performance Notes

- **First run:** ~5 min (model download) + 15 min (evaluation) = ~20 min total
- **Subsequent runs:** Faster due to model & answer caching
- **Evaluation:** 100 questions × 5 submissions = 500 LLM calls (~10-15 min)

---

## 🆘 Troubleshooting

See `SETUP.md` for detailed troubleshooting. Common issues:
- Port 5004 already in use → Kill process: `lsof -i :5004` (Linux) or Task Manager (Windows)
- Teacher URL wrong → Edit `.env`
- Server won't start → Check `server.log`
- Evaluation timeout → LLM is slow, increase `LLM_TIMEOUT` in `.env`

---

## 📝 Version Info

- **Base:** v5 (unified: hybrid retrieval + answer cache + vector DB persistence)
- **Built for:** Weak ~4k-context LLM + 100 questions × 5 submissions
- **Language:** Python 3.10+ (FastAPI, sentence-transformers, FAISS)

---

## ✅ Checklist Before Running

- [ ] Python 3.10+ installed
- [ ] `.env` has correct `TEACHER_BASE_URL`
- [ ] No app using port 5004 (unlikely)
- [ ] Network connection to teacher server
- [ ] Network connection to LLM proxy

Then just run: `bash run.sh` (or `run.bat`)

---

**Questions?** Check `SETUP.md` or `server.log` for details.
