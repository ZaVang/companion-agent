# Sprint 13: 生产就绪优化 — Embedding/Reflection/PostgreSQL

**状态**: 活跃
**目标**: 解决生产环境关键问题：Embedding 加载超时、Reflection 属性暴露、PostgreSQL 持久化

---

## 任务清单

- [x] 1. **T1-embedding-timeout**: Embedding 加载超时优化
  - 给 `utils/model.py` 的 HTTP 请求加 timeout（5秒）
  - 加载失败时优雅降级（返回 None，不阻塞）
  - 可选：支持本地缓存路径配置

- [x] 2. **T2-reflection-attribute**: Reflection 暴露为 MemorySystem 属性
  - 添加 `ms.reflection: ReflectionTrigger` 属性
  - 暴露 `ms.trigger_reflection()` 便捷方法
  - 可选：集成到 `add_memory()` 后的自动检测

- [x] 3. **T3-postgres-storage**: PostgreSQL 持久化层
  - 创建 `memory/storage/postgres.py` 模块
  - 实现 `PostgresStorage` 类（替代/补充 JSON）
  - 支持神经元、Engram、Embedding 的 CRUD
  - 迁移脚本：JSON → PostgreSQL

- [x] 4. **T4-tests-green**: 全部测试通过
  - 新增 `tests/test_postgres_storage.py`
  - `pytest tests/ -v` 全部通过（无 regression）

---

## 验收标准

| # | 标准 | 验证方式 |
|---|------|----------|
| 1 | Embedding 超时 5s 内返回 | `timeout 5 python3 -c "from memory.system import MemorySystem; ms=MemorySystem(); ms.add_memory('test')"` 不卡住 |
| 2 | `ms.reflection` 属性存在 | `python3 -c "from memory.system import MemorySystem; ms=MemorySystem(); print('reflection:', hasattr(ms, 'reflection'))"` |
| 3 | PostgreSQL 存储能读写 | `python3 -m pytest tests/test_postgres_storage.py -v` 通过 |
| 4 | 全部测试通过 | `python3 -m pytest tests/ -v` 显示全部 passed |

---

## 验收命令

```bash
cd /tmp/companion-agent-test

# T1: Embedding 超时测试（应在 5s 内完成）
timeout 5 python3 -c "
from memory.system import MemorySystem
ms = MemorySystem()
# 模拟 embedding 加载失败场景
from memory.embedding import EmbeddingManager
original = EmbeddingManager._try_load_model
EmbeddingManager._try_load_model = lambda self: setattr(self, '_available', False) or None
n = ms.add_memory('test', event_type='chat', actor='user')
print('Embedding timeout OK, neuron created:', n.event_id)
EmbeddingManager._try_load_model = original
"

# T2: Reflection 属性检查
python3 -c "
from memory.system import MemorySystem
ms = MemorySystem()
print('has reflection:', hasattr(ms, 'reflection'))
if hasattr(ms, 'reflection'):
    print('reflection type:', type(ms.reflection).__name__)
"

# T3: PostgreSQL 存储测试
python3 -m pytest tests/test_postgres_storage.py -v --tb=short 2>&1 | tail -10

# T4: 全部测试
python3 -m pytest tests/ -v --tb=short 2>&1 | tail -5
```

---

## 依赖关系

- T1 (Embedding) 无依赖
- T2 (Reflection) 无依赖  
- T3 (PostgreSQL) 依赖 T1（存储层需要 embedding 不阻塞才能正常初始化）
- T4 (Tests) 依赖 T1、T2、T3
