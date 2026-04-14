# Iteration 12 Plan

## 待完成任务（按依赖顺序）

1. **T3-Decay-Property-Fix**: 修复 `decay_scheduler` vs `decay` 属性名不一致
   - 目标：在 MemorySystem 中同时保留 `decay` 和 `decay_scheduler`（后者为别名），确保向后兼容
   - 依赖：无
   - 验收：`python3 -c "from memory.system import MemorySystem; ms = MemorySystem(); print('decay:', ms.decay); print('decay_scheduler:', ms.decay_scheduler)"` — 两行均有输出且无报错

2. **T4-Main-Py-Cleanup**: 清理过时的 `main.py`
   - 目标：在 `main.py` 中添加 DEPRECATED 标注，说明新入口是 `run_server.py`
   - 依赖：无
   - 验收：`grep -c "DEPRECATED\|废弃\|deprecated" main.py` 输出 > 0

3. **T1-Sprint1-4-Tests**: 为 Sprint 1-4 核心模块编写集成测试
   - 目标：创建 4 个集成测试文件，全部通过
     - `tests/test_sprint1_elo_integration.py` — 验证 Elo 竞争机制
     - `tests/test_sprint2_stability_integration.py` — 验证集体稳定性计算
     - `tests/test_sprint3_reflection_integration.py` — 验证 Reflection 触发与执行
     - `tests/test_sprint4_dmn_integration.py` — 验证 DMN 巩固/修剪/关联
   - 依赖：无（T3/T4 可并行，T1 本身独立）
   - 验收：4 个测试文件各自 `pytest ... -v` 全部 passed

4. **T2-E2E-Tests**: 编写端到端测试覆盖完整调用链
   - 目标：创建 `tests/test_e2e_memory_lifecycle.py`，覆盖 `add_memory() → retrieve() → decay() → dynamics` 完整链路
   - 依赖：无（T1 可并行）
   - 验收：`pytest tests/test_e2e_memory_lifecycle.py -v` 全部 passed

5. **T5-All-Tests-Green**: 运行全部测试确保无 regression
   - 目标：`pytest tests/ -v` 全部通过
   - 依赖：T1、T2、T3、T4 均完成
   - 验收：输出最后一行显示全部 passed

## 相关陷阱（从 pitfalls.md 筛选）

- **[测试]** 不要用静态数据测试动态系统 — 记忆系统本质是动态的，每次交互都在改变状态。测试应验证 Elo 上升、衰减生效等"变化"，而非固定输出
- **[测试]** 测试冲击力衰减需要长时间跨度 — 需要模拟天/周/月的长时间跨度来验证衰减曲线，而非短时间测试
- **[实现]** NeuronCell 没有 `id` 属性，只有 `event_id` — 测试中如需访问神经元 ID，应使用 `event_id` 或添加 `id` property 别名
- **[测试]** 验收命令中 positional argument 顺序必须与函数签名匹配 — 调用 `add_memory` 时应使用 keyword argument（如 `actor='test-user'`）而非依赖参数位置

## 验收命令（从 SPRINT.md 原样复制）

```bash
cd /tmp/companion-agent-test

# T1: Sprint1 Elo 集成测试
python3 -m pytest tests/test_sprint1_elo_integration.py -v --tb=short

# T2: Sprint2 Stability 集成测试
python3 -m pytest tests/test_sprint2_stability_integration.py -v --tb=short

# T3: Sprint3 Reflection 集成测试
python3 -m pytest tests/test_sprint3_reflection_integration.py -v --tb=short

# T4: Sprint4 DMN 集成测试
python3 -m pytest tests/test_sprint4_dmn_integration.py -v --tb=short

# T5: 端到端测试
python3 -m pytest tests/test_e2e_memory_lifecycle.py -v --tb=short

# T6: decay 属性一致性
python3 -c "
from memory.system import MemorySystem
ms = MemorySystem()
print('decay:', type(ms.decay).__name__)
print('decay_scheduler:', type(ms.decay_scheduler).__name__)
assert hasattr(ms, 'decay'), 'decay 属性缺失'
assert hasattr(ms, 'decay_scheduler'), 'decay_scheduler 别名缺失'
print('decay 属性一致性: OK')
"

# T7: main.py 废弃标注
grep -c "DEPRECATED\|废弃\|deprecated" main.py && echo "main.py 已标注废弃" || echo "main.py 尚未标注废弃"

# T8: 全部测试
python3 -m pytest tests/ -v --tb=short 2>&1 | tail -10
```
