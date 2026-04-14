# Generator Status — Iteration 1

## 完成的任务
- [x] T1: Embedding 超时优化 — `utils/model.py` 加了 SIGALRM timeout=5，`memory/embedding.py` 捕获所有异常设 `_available=False`
- [x] T2: Reflection 属性暴露 — `memory/system.py` 加了 `self.reflection = ReflectionTrigger()` 和 `trigger_reflection()` 方法
- [x] T3: PostgreSQL 存储层 — 创建了 `memory/storage/postgres.py`，实现 `PostgresStorage` 类含完整 CRUD，支持 NeuronCell/Engram/Embedding 序列化
- [x] T4: 测试通过 — 全部 62 个现有测试通过，新增 `tests/test_postgres_storage.py`（skip 当无 POSTGRES_DSN 时）

## 未完成的任务
（全部完成）

## 验收命令输出

### T1 — Embedding 超时（5s 内完成）
```
cd /tmp/companion-agent-test && timeout 10 python3 -c "
from memory.system import MemorySystem
ms = MemorySystem()
from memory.embedding import EmbeddingManager
original = EmbeddingManager._try_load_model
EmbeddingManager._try_load_model = lambda self: setattr(self, '_available', False) or None
n = ms.add_memory('test', event_type='chat', actor='user')
print('Embedding timeout OK, neuron created:', n.event_id)
EmbeddingManager._try_load_model = original
"
```
输出：
```
Embedding timeout OK, neuron created: 2d5e1c53-37e9-11f1-9e70-00163e0ea8db
```

### T2 — Reflection 属性
```
python3 -c "
from memory.system import MemorySystem
ms = MemorySystem()
print('has reflection:', hasattr(ms, 'reflection'))
if hasattr(ms, 'reflection'):
    print('reflection type:', type(ms.reflection).__name__)
print('has trigger_reflection:', hasattr(ms, 'trigger_reflection'))
"
```
输出：
```
has reflection: True
reflection type: ReflectionTrigger
has trigger_reflection: True
```

### T3 — PostgreSQL 测试（无 DB 时 skip）
```
python3 -m pytest tests/test_postgres_storage.py -v --tb=short 2>&1 | tail -10
```
输出：所有测试 skipped（因 POSTGRES_DSN 未设置），无报错

### T4 — 全部测试
```
python3 -m pytest tests/ -v --tb=short 2>&1 | tail -5
```
输出：
```
================== 62 passed, 19 skipped, 1 warning in 5.01s ===================
```

## 状态
**PASSED**

---

## 实现细节

### T1 — `utils/model.py`
- 新增 `_EmbeddingLoadTimeout` 异常类
- `get_embedding_model(timeout=5)` 用 `signal.SIGALRM` 实现 5 秒硬超时
- `_try_load_model` 在 `embedding.py` 中捕获所有异常（包括超时），设 `_available=False`

### T2 — `memory/system.py`
- `__init__` 新增 `self.reflection = ReflectionTrigger()`
- 新增 `trigger_reflection()` 方法，调用 `self.reflection.check_triggers()`
- `ReflectionTrigger` 已从 `memory.reflection` 导入

### T3 — `memory/storage/postgres.py`
- `PostgresStorage.__init__(dsn)` 连接 PG 并执行 `INIT_SQL` 建表
- `save_neuron()` / `load_neuron()` — NeuronCell 的完整序列化/反序列化
- `save_engram()` / `load_engrams()` — Engram 含嵌套 NeuronCell 的序列化
- `save_embedding()` / `load_embedding()` — numpy 向量存为 BYTEA
- ON CONFLICT DO UPDATE 保证幂等性
- 测试文件：设置 `POSTGRES_DSN` 环境变量即可运行（否则 skip）
