# Sprint 12: 测试补全 + 系统一致性修复

**状态**: 活跃
**目标**: 为 Sprint 1-4 核心模块补充集成测试，修复已知不一致问题，建立端到端测试覆盖

---

## 任务清单

- [ ] 1. **T1-Sprint1-4-Tests**: 为 Sprint 1-4 核心模块编写集成测试
  - `tests/test_sprint1_elo_integration.py` — 验证 Elo 竞争机制（检索后竞争更新）
  - `tests/test_sprint2_stability_integration.py` — 验证集体稳定性计算
  - `tests/test_sprint3_reflection_integration.py` — 验证 Reflection 触发与执行
  - `tests/test_sprint4_dmn_integration.py` — 验证 DMN 巩固/修剪/关联

- [ ] 2. **T2-E2E-Tests**: 编写端到端测试覆盖完整调用链
  - `tests/test_e2e_memory_lifecycle.py` — `add_memory() → retrieve() → decay() → dynamics` 完整链路

- [ ] 3. **T3-Decay-Property-Fix**: 修复 `decay_scheduler` vs `decay` 属性名不一致
  - MemorySystem 中 property 名为 `decay`，但验收标准写的是 `ms.decay_scheduler`
  - 保持向后兼容：保留 `decay` 属性，新增 `decay_scheduler` 作为别名

- [ ] 4. **T4-Main-Py-Cleanup**: 清理过时的 `main.py`
  - 标注为废弃并说明新入口是 `run_server.py`

- [ ] 5. **T5-All-Tests-Green**: 运行全部测试确保无 regression
  - `pytest tests/ -v` 全部通过

---

## 验收标准

| # | 标准 | 验证方式 |
|---|------|----------|
| 1 | `test_sprint1_elo_integration.py` 存在且通过 | `pytest tests/test_sprint1_elo_integration.py -v` |
| 2 | `test_sprint2_stability_integration.py` 存在且通过 | `pytest tests/test_sprint2_stability_integration.py -v` |
| 3 | `test_sprint3_reflection_integration.py` 存在且通过 | `pytest tests/test_sprint3_reflection_integration.py -v` |
| 4 | `test_sprint4_dmn_integration.py` 存在且通过 | `pytest tests/test_sprint4_dmn_integration.py -v` |
| 5 | `test_e2e_memory_lifecycle.py` 存在且通过 | `pytest tests/test_e2e_memory_lifecycle.py -v` |
| 6 | `ms.decay_scheduler` 和 `ms.decay` 均可用 | `python3 -c "from memory.system import MemorySystem; ms = MemorySystem(); print('decay:', ms.decay); print('decay_scheduler:', ms.decay_scheduler)"` |
| 7 | `main.py` 已标注废弃 | `grep -c "DEPRECATED\|废弃\|deprecated" main.py` |
| 8 | 全部测试通过 | `pytest tests/ -v` 最后一行显示全部 passed |

---

## 验收命令

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
