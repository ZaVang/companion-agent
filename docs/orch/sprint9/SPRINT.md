# Sprint 9: 可视化与调试工具 — 集成到 MemorySystem

**状态**: 进行中
**目标**: 将已实现的 viz 模块（NetworkVisualizer + MemoryHistory）集成到 MemorySystem 并编写测试

## 任务清单

- [ ] 1. 在 `MemorySystem` 中添加 `NetworkVisualizer` 和 `MemoryHistory` 实例
- [ ] 2. `MemorySystem.to_graphviz()` 方法：导出当前记忆网络为 Graphviz DOT 格式
- [ ] 3. `MemorySystem.get_network_stats()` 方法：返回节点数/边数/平均度数等统计
- [ ] 4. `MemoryHistory` 与 `MemoryTracer` 集成：记录每次 add_memory / retrieve / decay 操作
- [ ] 5. 运行 `pytest tests/ -v`，确保无 regression
- [ ] 6. 编写 `tests/test_sprint9_viz_integration.py`，验证可视化输出格式正确

## 验收标准

1. `MemorySystem` 实例化后，有 `visualizer` 和 `history` 属性
2. `ms.to_graphviz()` 返回合法的 Graphviz DOT 字符串（包含 "digraph" 关键字）
3. `ms.get_network_stats()` 返回 dict 包含 `node_count`, `edge_count`, `avg_degree`
4. 添加记忆后 `MemoryHistory.get_recent()` 能返回历史记录
5. `pytest tests/ -v` 全部通过（无 regression）
6. 新测试文件 `test_sprint9_viz_integration.py` 全部通过

## 验收命令

```bash
cd /tmp/companion-agent-test
python3 -c "from memory.system import MemorySystem; ms = MemorySystem(); assert hasattr(ms, 'visualizer'); print('visualizer attr OK')"
python3 -c "from memory.system import MemorySystem; ms = MemorySystem(); dot = ms.to_graphviz(); assert 'digraph' in dot; print('to_graphviz OK, len=', len(dot))"
python3 -c "from memory.system import MemorySystem; ms = MemorySystem(); stats = ms.get_network_stats(); assert 'node_count' in stats; print('get_network_stats OK:', stats)"
python3 -m pytest tests/ -v --tb=short 2>&1 | tail -5
python3 -m pytest tests/test_sprint9_viz_integration.py -v --tb=short 2>&1 | tail -10
```
