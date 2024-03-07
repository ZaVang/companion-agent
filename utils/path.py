from pathlib import Path

# 获取当前文件的路径
CURRENT_PATH = Path(__file__).resolve()

# 获取项目的根路径
PROJECT_ROOT = CURRENT_PATH.parent.parent

# 从项目的根路径派生其他的路径
DATABASE_DIR = PROJECT_ROOT / 'db'
AGENT_DB_DIR = DATABASE_DIR / 'agent'
ENGRAM_DB_DIR = DATABASE_DIR / 'engram'
PERSONA_DB_DIR = DATABASE_DIR / 'persona'
SCHEDULE_DB_DIR = DATABASE_DIR / 'schedule'
EVENT_STREAM_DB_DIR = DATABASE_DIR / 'eventstream'
REGISTRY_DB_DIR = ENGRAM_DB_DIR / 'registry'
EMBEDDING_DB_DIR = DATABASE_DIR / 'embedding'
SHORT_TERM_MEMORY_DB_DIR = DATABASE_DIR / 'stm'
AGENT_DB_DIR = DATABASE_DIR / 'agent'

def get_database_file(filename: str) -> Path:
    """获取database目录下的文件路径"""
    return DATABASE_DIR / filename
