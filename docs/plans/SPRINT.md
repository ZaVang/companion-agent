# Sprint 14: 代码质量提升 — 架构重构/文档整合/类型完善

**状态**: 活跃
**目标**: 解决代码质量问题：MemorySystem臃肿、文档碎片化、类型提示不完整

---

## 任务清单

- [ ] 1. **T1-memory-system-refactor**: MemorySystem 重构（Facade模式）
  - 分析 `memory/system.py` 424行的职责划分
  - 设计 Facade 接口，保持对外API不变
  - 拆分为：`MemorySystemFacade` + 内部模块协调器
  - 确保所有测试通过，无 breaking changes

- [ ] 2. **T2-docs-consolidation**: Sprint文档整合到Chronicle
  - 合并 `docs/orch/sprint8/`、`sprint9/`、`sprint10/` 到 `chronicle.md`
  - 统一格式，保留关键决策和pitfalls
  - 删除冗余目录，保持docs整洁

- [ ] 3. **T3-type-hints**: PostgreSQL存储层类型提示完善
  - 检查 `memory/storage/postgres.py` 所有helper函数
  - 补充参数类型、返回类型、docstring
  - 确保mypy类型检查通过（可选）

- [ ] 4. **T4-tests-green**: 全部测试通过
  - 运行 `pytest tests/ -v` 确保全部通过
  - 新增必要的测试覆盖

---

## 验收标准

| # | 标准 | 验证方式 |
|---|------|----------|
| 1 | MemorySystem行数 < 200 | `wc -l memory/system.py` |
| 2 | Facade接口稳定 | `pytest tests/` 全部通过 |
| 3 | docs/orch/下无sprint目录 | `ls docs/orch/` 只有通用文件 |
| 4 | postgres.py类型完整 | 代码review + mypy检查 |

---

## Multi-Ralph 流程

本Sprint按照三角色循环执行：

### Phase 1: Ralph (设计者)
- [ ] 分析现有代码结构
- [ ] 设计重构方案
- [ ] 更新 pitfall 知识库
- [ ] 输出：详细设计文档

### Phase 2: Ralph (实现者)
- [ ] 按设计文档实施重构
- [ ] 编写代码、更新文档
- [ ] 本地测试通过
- [ ] 输出：可运行的代码

### Phase 3: Ralph (验证者)
- [ ] 运行完整测试套件
- [ ] 检查验收标准
- [ ] 记录遗留问题
- [ ] 输出：验收报告

---

## 依赖关系

```
T1 (重构) ──┐
            ├──> T4 (测试)
T2 (文档) ──┤
            │
T3 (类型) ──┘
```

T1、T2、T3 可并行执行，最后统一验收。

---

## 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| 重构破坏现有功能 | 保持对外API不变，先写测试 |
| 文档丢失关键信息 | 合并前备份，保留原目录结构在git历史 |
| 类型提示导致运行时错误 | 使用TYPE_CHECKING，运行时不加载 |

---

## 预计工时

- T1: 2小时
- T2: 30分钟
- T3: 30分钟
- T4: 15分钟

**总计**: 约3小时
