#!/bin/bash
# Engram 记忆系统一键启动脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Engram Memory System Startup Script  ${NC}"
echo -e "${BLUE}========================================${NC}"

# 检查 Python
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python not found${NC}"
    exit 1
fi

# 检查 Node.js
if ! command -v node &> /dev/null; then
    echo -e "${RED}Error: Node.js not found${NC}"
    exit 1
fi

# 创建必要的目录
mkdir -p data/agent data/memory logs

# 函数：检查端口是否被占用
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        return 0  # 端口被占用
    else
        return 1  # 端口空闲
    fi
}

# 函数：等待端口可用
wait_for_port() {
    local port=$1
    local max_wait=30
    local count=0
    while check_port $port && [ $count -lt $max_wait ]; do
        sleep 1
        count=$((count + 1))
    done
}

# 停止已有进程
echo -e "${YELLOW}Stopping existing processes...${NC}"
pkill -f "uvicorn service.app:app" 2>/dev/null || true
pkill -f "vite" 2>/dev/null || true
sleep 2

# 启动后端
echo -e "${GREEN}Starting backend server...${NC}"
if check_port 5000; then
    echo -e "${YELLOW}Port 5000 is in use, waiting...${NC}"
    wait_for_port 5000
fi

# 后台启动后端（使用简化版，不需要 torch/gradio）
nohup python run_server.py --host 0.0.0.0 --port 5000 > logs/backend.log 2>&1 &
BACKEND_PID=$!
echo -e "${GREEN}Backend started (PID: $BACKEND_PID)${NC}"
echo "Backend PID: $BACKEND_PID" > .backend.pid

# 等待后端启动
echo -e "${YELLOW}Waiting for backend to start...${NC}"
sleep 3

# 检查后端是否启动成功
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo -e "${RED}Backend failed to start. Check logs/backend.log for details.${NC}"
    exit 1
fi

# 启动前端
echo -e "${GREEN}Starting frontend server...${NC}"
cd frontend

# 检查是否需要安装依赖
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Installing frontend dependencies...${NC}"
    npm install
fi

# 后台启动前端
nohup npm run dev > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
echo -e "${GREEN}Frontend started (PID: $FRONTEND_PID)${NC}"
echo "Frontend PID: $FRONTEND_PID" > ../.frontend.pid

cd ..

# 等待前端启动
echo -e "${YELLOW}Waiting for frontend to start...${NC}"
sleep 5

# 显示状态
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Engram Memory System Started!        ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Backend:${NC}  http://localhost:5000"
echo -e "${BLUE}Frontend:${NC} http://localhost:3000"
echo -e "${BLUE}API Docs:${NC} http://localhost:5000/docs"
echo ""
echo -e "${YELLOW}Logs:${NC}"
echo "  - Backend:  logs/backend.log"
echo "  - Frontend: logs/frontend.log"
echo ""
echo -e "${YELLOW}To stop:${NC} ./stop.sh"
echo ""
