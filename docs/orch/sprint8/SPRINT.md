# Sprint 8: 神经元动态增删 — 集成到 MemorySystem

**状态**: 进行中
**目标**: 将已实现的 dynamics 模块（death/birth）集成到 MemorySystem 生命周期

## 任务清单

- [ ] 1. 在 `MemorySystem` 中添加 `NeuronDynamics` 实例和定时触发
- [ ] 2. 实现 `NeuronDeathManager.evaluate()` 集成：强度/Elo 过低的神经元被淘汰
- [ ] 3. 实现 `NeuronBirthManager.evaluate()` 集成：高强度碎片生成新神经元
- [ ] 4. 运行 `pytest tests/ -v`，确保无 regression
- [ ] 5. 编写 `tests/test_sprint8_dynamics_integration.py`，验证动态增删正确性

## 验收标准

1. `MemorySystem` 实例化后，`dynamics.death_manager` 和 `dynamics.birth_manager` 存在
2. 调用 `MemorySystem.run_dynamics_cycle()` 后返回结构包含 `died_count`, `born_count`
3. `NeuronDeathManager.evaluate()` 正确识别应淘汰的神经元（strength < threshold）
4. `NeuronBirthManager.evaluate()` 正确判断是否需要生成新神经元
5. `pytest tests/ -v` 全部通过（无 regression）
6. 新测试文件 `test_sprint8_dynamics_integration.py` 全部通过

## 验收命令

```bash
cd /tmp/companion-agent-test
python3 -c "from memory.system import MemorySystem; ms = MemorySystem(); assert hasattr(ms, 'dynamics'); print('dynamics attr OK')"
python3 -c "from memory.system import MemorySystem; ms = MemorySystem(); result = ms.run_dynamics_cycle(); assert 'died_count' in result; print('run_dynamics_cycle OK:', result)"
python3 -m pytest tests/ -v --tb=short 2>&1 | tail -5
python3 -m pytest tests/test_sprint8_dynamics_integration.py -v --tb=short 2>&1 | tail -10
```
