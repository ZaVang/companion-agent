"""
Engram Memory System - Simple Backend Server

只启动 Memory API，不需要完整的应用依赖（gradio, torch 等）
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import logging
from fastapi import FastAPI
import uvicorn

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI(
    title="Engram Memory System API",
    description="记忆系统可视化后端 API",
    version="1.0.0"
)

# 导入 Memory API router
try:
    from service.memory_api import router as memory_router
    app.include_router(memory_router, prefix="/api")
    logger.info("Memory API loaded successfully")
except ImportError as e:
    logger.warning(f"Memory API import failed: {e}")
    
    # 创建一个简单的健康检查端点
    @app.get("/api/health")
    async def health():
        return {"status": "ok", "message": "Engram Memory System is running"}


# 根路由
@app.get("/")
async def root():
    return {
        "name": "Engram Memory System",
        "version": "1.0.0",
        "docs": "/docs",
        "api_prefix": "/api"
    }


# 健康检查
@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Engram Memory System Backend")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    print(f"\n{'='*50}")
    print(f"  Engram Memory System Backend")
    print(f"{'='*50}")
    print(f"  Server: http://{args.host}:{args.port}")
    print(f"  API Docs: http://{args.host}:{args.port}/docs")
    print(f"{'='*50}\n")
    
    uvicorn.run(
        "run_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )
