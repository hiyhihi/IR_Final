# RAG IR Student Server - One-Click Setup & Submit

Đây là bản submission hoàn chỉnh từ v5 - chỉ cần **1 lệnh duy nhất** để setup, chạy, và nộp bài.

## 📋 Yêu cầu

- **Python 3.10+**
- **bash** (Git Bash trên Windows, hoặc WSL)

## 🚀 Cách chạy (Windows)

### Cách 1: Git Bash (Khuyên dùng)

1. Mở **Git Bash** (nếu chưa có, download [Git for Windows](https://git-scm.com/download/win))
2. Chuyển đến folder submission:
```bash
cd /c/Users/YourUsername/Desktop/submission
```

3. Chạy lệnh duy nhất:
```bash
bash run.sh
```

Script sẽ tự động:
- ✅ Cài uv (nếu chưa có)
- ✅ Tạo virtual environment
- ✅ Cài dependencies
- ✅ Tải model embedding
- ✅ Chạy server
- ✅ Đăng ký với teacher server
- ✅ Gửi đáp án (nộp bài)
- ✅ Lấy kết quả

### Cách 2: Windows Command Prompt

1. Mở **Command Prompt** (Win + R, gõ `cmd`, Enter)
2. Chuyển đến folder submission:
```cmd
cd C:\Users\YourUsername\Desktop\submission
```

3. Chạy:
```cmd
run.bat
```

### Cách 3: PowerShell

1. Mở **PowerShell** (Win + X, chọn PowerShell)
2. Chuyển đến folder submission:
```powershell
cd C:\Users\YourUsername\Desktop\submission
```

3. Chạy:
```powershell
bash run.sh
```

## ⚙️ Cấu hình (tùy chỉnh nếu cần)

File `.env` chứa các cấu hình quan trọng:

```env
TEACHER_BASE_URL=http://192.168.50.218:8000/api/v1  # Teacher server address
STUDENT_ID=B22BENCHMARK                              # Your student ID
STUDENT_SERVER_URL=http://127.0.0.1:5004             # Server URL
LLM_BASE_URL=http://127.0.0.1:8888/v1                # LLM proxy URL
DEVICE=cpu                                            # cpu or cuda
TOP_K=8                                               # Retrieval top-k
MAX_CONTEXT_CHARS=2800                               # Max context length
ENABLE_ANSWER_CACHE=true                             # Enable answer caching
```

## 📊 Quá trình chạy

```
1. Install uv (nếu cần)
   ↓
2. Tạo virtual environment
   ↓
3. Cài dependencies
   ↓
4. Tải model embedding
   ↓
5. Chạy server (FastAPI)
   ↓
6. Đăng ký với teacher (register)
   ↓
7. Nộp bài (evaluate) - có thể mất 10-15 phút
   ↓
8. Lấy kết quả (result)
   ↓
9. Dừng server
```

## 📝 Log files

- `server.log` - Chi tiết hoạt động server
- Để xem log khi server chạy:
```bash
tail -f server.log  # Linux/Mac/WSL
```

## 🔄 Chạy lại mà không setup

Sau lần đầu tiên, nếu chỉ muốn chạy lại:

**Git Bash / WSL:**
```bash
source .venv/bin/activate
python main.py  # Terminal 1
python client.py evaluate  # Terminal 2 khác
```

**Windows Command Prompt:**
```cmd
.venv\Scripts\activate.bat
python main.py
REM Then in another cmd window:
python client.py evaluate
```

## 🛠️ Troubleshooting

### ❌ "uv not found"
- Nếu script không tự cài được, cài bằng: `pip install uv`

### ❌ "Server failed to start"
- Kiểm tra port 5004 không bị chiếm dụng
- Xem `server.log` để chi tiết
- Cố gắng kill process cũ: `lsof -i :5004` (Linux) hoặc Task Manager (Windows)

### ❌ "Connection refused to teacher server"
- Kiểm tra `TEACHER_BASE_URL` trong `.env` có đúng không
- Kiểm tra kết nối mạng

### ❌ "Evaluation timeout"
- Có thể LLM proxy chậm, hãy chờ
- Nếu quá lâu, check `LLM_BASE_URL` trong `.env`

## 📌 Important Notes

- ⚠️ **Lần đầu**: Tải model embedding (~500MB), có thể mất vài phút
- ⚠️ **Evaluation**: Chạy 100 câu hỏi × 5 lần = 500 requests, mất 10-15 phút
- ⚠️ **LLM Timeout**: Đặt ở 45s trong `.env`, có thể tăng lên nếu LLM chậm
- ✅ **Cache**: Answers được cache, lần sau sẽ nhanh hơn

## 📧 Support

Nếu có vấn đề:
1. Xem `server.log`
2. Kiểm tra các config trong `.env`
3. Thử chạy lại script

---

**v5 unified build** - Optimized for weak ~4k-context LLM with hybrid retrieval + answer caching.
