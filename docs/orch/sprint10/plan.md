# Iteration 1 Plan

## 待完成任务（按依赖顺序）
1. [T10-idx-search]: 实现 MemoryIndex.search_by_text() 方法
   - 目标：在 MemoryIndex 上添加 search_by_text 方法，支持文本关键词搜索
   - 依赖：无
   - 验收：idx.search_by_text("test") 能返回结果（list 类型），不抛 AttributeError

2. [T10-ms-batch_retrieve]: 实现 MemorySystem.batch_retrieve() 方法
   - 目标：使用 MemoryIndex 加速批量检索，返回 [{neuron_id, score}, ...]
   - 依赖：T10-idx-search（需要索引可用）
   - 验收：batch_retrieve() 返回正确格式（包含 neuron_id 和 score）

3. [T10-ms-batch_decay]: 实现 MemorySystem.batch_decay() 方法
   - 目标：一次性对所有神经元应用衰减，用时 < 1s（1000个神经元）
   - 依赖：无（独立功能）
   - 验收：batch_decay() 可在 1 秒内完成 1000 个神经元的衰减计算

4. [T10-test-integration]: 编写 tests/test_sprint10_opt_integration.py
   - 目标：验证索引正确性和 batch_retrieve/batch_decay 性能
   - 依赖：T10-ms-batch_retrieve, T10-ms-batch_decay
   - 验收：新测试文件全部通过

5. [T10-regression]: 运行 pytest tests/ 确保无 regression
   - 目标：确保所有现有 21 个测试仍然通过
   - 依赖：所有实现
   - 验收：pytest tests/ -v 全部通过

## 相关陷阱（从 pitfalls.md 筛选）
- [测试] 不要用静态数据测试动态系统 — batch_decay 测试需用真实 NeuronCell 而非字典
- [架构] 衰减参数不能一刀切 — batch_decay 应调用 decay_scheduler 的衰减逻辑

## 验收命令（从 SPRINT.md 原样复制）
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
