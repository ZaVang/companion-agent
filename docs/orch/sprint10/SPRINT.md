# Sprint 10: 性能优化 — 索引 + 批量操作集成

**状态**: 进行中
**目标**: 将已实现的 optimization 模块（MemoryIndex + BatchProcessor）集成到 MemorySystem 并验证性能提升

## 任务清单

- [x] 1. 在 `MemorySystem` 中添加 `MemoryIndex` 实例，记忆添加/删除时同步索引
- [x] 2. 实现 `MemorySystem.batch_retrieve()`：使用索引加速批量检索
- [x] 3. 实现 `MemorySystem.batch_decay()`：一次性对所有神经元应用衰减
- [x] 4. 验证 `BatchProcessor` 对 1000+ 神经元操作的性能（用时 < 1s）
- [x] 5. 运行 `pytest tests/ -v`，确保无 regression
- [x] 6. 编写 `tests/test_sprint10_opt_integration.py`，验证索引正确性和性能

## 验收标准

1. `MemorySystem` 实例化后，有 `index` 属性（`MemoryIndex` 实例）
2. 添加记忆后索引可搜索：`ms.index.search_by_text("关键词")` 返回相关神经元
3. `batch_decay()` 可在 1 秒内完成 1000 个神经元的衰减计算
4. `batch_retrieve()` 返回格式正确（包含 neuron_id 和 score）
5. `pytest tests/ -v` 全部通过（无 regression）
6. 新测试文件 `test_sprint10_opt_integration.py` 全部通过

## 验收命令

```bash
cd /tmp/companion-agent-test
python3 -c "from memory.system import MemorySystem; ms = MemorySystem(); assert hasattr(ms, 'index'); print('index attr OK')"
python3 -c "from memory.system import MemorySystem; from memory.optimization import MemoryIndex; idx = MemoryIndex(); result = idx.search_by_text('test'); print('search_by_text OK:', type(result))"
python3 -c "
import time, numpy as np
from memory.optimization import BatchProcessor
neurons = [{'id': str(i), 'strength': 1.0, 'decay_rate': 0.995, 'last_decay_at': None} for i in range(1000)]
start = time.time()
from memory.optimization import batch_decay
result = batch_decay(neurons)
elapsed = time.time() - start
print(f'batch_decay 1000 neurons: {elapsed:.3f}s')
assert elapsed < 1.0, f'Too slow: {elapsed}s'
print('batch_decay performance OK')
"
python3 -m pytest tests/ -v --tb=short 2>&1 | tail -5
python3 -m pytest tests/test_sprint10_opt_integration.py -v --tb=short 2>&1 | tail -10
```
