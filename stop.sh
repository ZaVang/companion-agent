#!/bin/bash
# Engram 记忆系统停止脚本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Stopping Engram Memory System...${NC}"

# 停止后端
if [ -f .backend.pid ]; then
    BACKEND_PID=$(cat .backend.pid)
    if kill -0 $BACKEND_PID 2>/dev/null; then
        kill $BACKEND_PID
        echo -e "${GREEN}Backend stopped (PID: $BACKEND_PID)${NC}"
    fi
    rm .backend.pid
fi

# 停止前端
if [ -f .frontend.pid ]; then
    FRONTEND_PID=$(cat .frontend.pid)
    if kill -0 $FRONTEND_PID 2>/dev/null; then
        kill $FRONTEND_PID
        echo -e "${GREEN}Frontend stopped (PID: $FRONTEND_PID)${NC}"
    fi
    rm .frontend.pid
fi

# 确保所有相关进程都被停止
pkill -f "run_server.py" 2>/dev/null || true
pkill -f "uvicorn service.app:app" 2>/dev/null || true
pkill -f "vite" 2>/dev/null || true

echo -e "${GREEN}All services stopped.${NC}"
