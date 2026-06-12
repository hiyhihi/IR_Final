#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== RAG IR Student Server - One-Click Setup & Submit ===${NC}"
echo ""

# ============================================================================
# 1. Install uv if not present
# ============================================================================
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}[*] Installing uv package manager...${NC}"
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
        # Windows
        python -m pip install uv
    else
        # Linux/Mac
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
    echo -e "${GREEN}[✓] uv installed${NC}"
else
    echo -e "${GREEN}[✓] uv found${NC}"
fi

# ============================================================================
# 2. Create virtual environment with uv
# ============================================================================
echo -e "${YELLOW}[*] Creating virtual environment...${NC}"
uv venv .venv
echo -e "${GREEN}[✓] Virtual environment created${NC}"

# Activate venv
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi

# ============================================================================
# 3. Install dependencies
# ============================================================================
echo -e "${YELLOW}[*] Installing dependencies...${NC}"
uv pip install -r requirements.txt
echo -e "${GREEN}[✓] Dependencies installed${NC}"

# ============================================================================
# 4. Download embedding model if needed
# ============================================================================
echo -e "${YELLOW}[*] Downloading embedding model (keepitreal/vietnamese-sbert)...${NC}"
python download_model.py
echo -e "${GREEN}[✓] Model downloaded${NC}"

# ============================================================================
# 5. Start server in background
# ============================================================================
echo -e "${YELLOW}[*] Starting student server...${NC}"
python main.py > server.log 2>&1 &
SERVER_PID=$!
echo -e "${GREEN}[✓] Server started (PID: $SERVER_PID)${NC}"

# Wait for server to be ready
echo -e "${YELLOW}[*] Waiting for server to be ready...${NC}"
sleep 3
for i in {1..30}; do
    if python -c "import httpx; httpx.get('http://127.0.0.1:5004/docs', timeout=2)" 2>/dev/null; then
        echo -e "${GREEN}[✓] Server is ready${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}[✗] Server failed to start${NC}"
        kill $SERVER_PID 2>/dev/null || true
        cat server.log
        exit 1
    fi
    sleep 1
done

# ============================================================================
# 6. Register with teacher server
# ============================================================================
echo -e "${YELLOW}[*] Registering with teacher server...${NC}"
REGISTER_RESULT=$(python client.py register 2>&1 || echo "FAILED")
echo "$REGISTER_RESULT"
echo ""

# ============================================================================
# 7. Run evaluation (submit answers)
# ============================================================================
echo -e "${YELLOW}[*] Running evaluation (submitting answers)...${NC}"
echo "This may take 10-15 minutes. Please wait..."
echo ""
EVAL_RESULT=$(python client.py evaluate 2>&1 || echo "FAILED")
echo "$EVAL_RESULT"
echo ""

# ============================================================================
# 8. Get results
# ============================================================================
echo -e "${YELLOW}[*] Fetching results...${NC}"
sleep 2
RESULT=$(python client.py result 2>&1 || echo "FAILED")
echo -e "${GREEN}=== RESULTS ===${NC}"
echo "$RESULT"
echo ""

# ============================================================================
# 9. Cleanup
# ============================================================================
echo -e "${YELLOW}[*] Cleaning up...${NC}"
kill $SERVER_PID 2>/dev/null || true
sleep 1

echo -e "${GREEN}=== Submission Complete ===${NC}"
echo ""
echo "Server log saved in: server.log"
echo "Configuration: .env"
echo ""
echo "To run again without setup:"
echo "  source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows"
echo "  python main.py  # and in another terminal: python client.py evaluate"
