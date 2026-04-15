# Sprint 14: MemorySystem 重构设计文档

## 设计者: Ralph (Phase 1)
**日期**: 2026-04-15
**当前状态**: 设计阶段

---

## 一、现状分析

### 1.1 MemorySystem 当前状态
- **文件**: `memory/system.py`
- **行数**: 877 行（比文档记录的 424 行还多）
- **职责**: 过于臃肿，承担了过多责任

### 1.2 职责划分分析

| 职责范围 | 行号范围 | 方法/类 | 职责描述 |
|---------|---------|---------|---------|
| **配置定义** | 58-114 | `MemorySystemConfig`, `DMNResult`, `PredictionResult` | 配置类定义 |
| **系统初始化** | 118-172 | `__init__`, `attach_episodic_memory` | 初始化所有模块 |
| **内存管理** | 174-243 | `_evict_weak_neurons`, properties | LRU驱逐、懒加载 |
| **核心操作** | 245-481 | `add_memory`, `retrieve`, `apply_decay` | 添加记忆、检索、衰减 |
| **DMN管理** | 514-553 | `run_dmn`, `predict_activation` | DMN执行、激活预测 |
| **动态管理** | 596-672 | `run_dynamics_cycle`, `trigger_reflection` | 神经元生灭、Reflection |
| **批量操作** | 675-779 | `batch_retrieve`, `batch_decay` | 批量检索、批量衰减 |
| **统计可视化** | 781-859 | `get_statistics`, `to_graphviz` | 统计信息、网络可视化 |
| **全局实例** | 862-877 | `get_memory_system`, `reset_memory_system` | 全局单例管理 |

### 1.3 问题识别

1. **职责过重**: MemorySystem 承担了存储、检索、统计、可视化等多个职责
2. **代码膨胀**: 877行代码，难以维护和测试
3. **耦合度高**: 各模块逻辑混在一起，修改一处可能影响多处
4. **测试困难**: 需要mock大量依赖才能测试单个方法

---

## 二、重构方案（Facade 模式）

### 2.1 设计原则

1. **保持API不变**: 所有公共方法签名保持一致
2. **职责分离**: 将不同职责拆分到独立的协调器
3. **最小化影响**: 重构分阶段进行，每阶段都运行测试
4. **使用TYPE_CHECKING**: 避免运行时导入循环

### 2.2 架构设计

```
┌─────────────────────────────────────────────────┐
│              MemorySystem (Facade)               │
│  - 对外API保持不变                               │
│  - 委托调用给各个协调器                          │
└──────────────────┬──────────────────────────────┘
                   │
        ┌──────────┼──────────┐
        │          │          │
        ▼          ▼          ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  StorageCoord │ │  RetrieverCo  │ │ LifecycleCo  │
│  _init: 内存  │ │  _init: 检索  │ │  _init: DMN  │
│  管理         │ │  逻辑         │ │  + 动态管理   │
└──────────────┘ └──────────────┘ └──────────────┘
        │                │                │
        └────────────────┼────────────────┘
                         │
                   ┌─────┴─────┐
                   │           │
                   ▼           ▼
            ┌──────────┐ ┌──────────┐
            │  Batch   │ │ StatsViz │
            │  Oper.   │ │  Coord.  │
            └──────────┘ └──────────┘
```

### 2.3 模块拆分方案

#### 模块 1: StorageCoordinator
**职责**: 管理内存存储、LRU驱逐、懒加载
**包含方法**:
- `_evict_weak_neurons()` → `StorageCoordinator.evict_weak_neurons()`
- 懒加载属性 → 移到 `StorageCoordinator`
**文件位置**: `memory/coordinators/storage.py`

#### 模块 2: RetrieverCoordinator
**职责**: 管理检索逻辑（包括语义检索、降级检索）
**包含方法**:
- `retrieve()` → `RetrieverCoordinator.retrieve()`
- `batch_retrieve()` → `BatchOperations.batch_retrieve()`
**文件位置**: `memory/coordinators/retriever.py`

#### 模块 3: LifecycleCoordinator
**职责**: 管理DMN执行、激活预测、动态管理
**包含方法**:
- `apply_decay()` → `LifecycleCoordinator.apply_decay()`
- `run_dmn()` → `LifecycleCoordinator.run_dmn()`
- `run_dynamics_cycle()` → `LifecycleCoordinator.run_dynamics_cycle()`
- `trigger_reflection()` → `LifecycleCoordinator.trigger_reflection()`
**文件位置**: `memory/coordinators/lifecycle.py`

#### 模块 4: BatchOperations
**职责**: 批量操作（批量检索、批量衰减）
**包含方法**:
- `batch_retrieve()` → 已归到RetrieverCoordinator
- `batch_decay()` → `BatchOperations.batch_decay()`
**文件位置**: `memory/coordinators/batch.py`

#### 模块 5: StatsVizCoordinator
**职责**: 统计信息和可视化
**包含方法**:
- `get_statistics()` → `StatsVizCoordinator.get_statistics()`
- `to_graphviz()` → `StatsVizCoordinator.to_graphviz()`
- `get_network_stats()` → `StatsVizCoordinator.get_network_stats()`
- `get_network_visualization()` → `StatsVizCoordinator.get_network_visualization()`
**文件位置**: `memory/coordinators/stats_viz.py`

---

## 三、重构实施计划

### 3.1 阶段 1: 创建协调器框架（不移动代码）
- [ ] 创建 `memory/coordinators/` 目录
- [ ] 创建 `memory/coordinators/__init__.py`
- [ ] 创建5个协调器文件（空类或占位符）
- [ ] 在 MemorySystem 中创建协调器实例
- [ ] 运行测试，确保不破坏现有功能

### 3.2 阶段 2: 移动核心操作（add_memory）
- [ ] 分析 `add_memory()` 的依赖
- [ ] 将 `add_memory()` 逻辑拆分到协调器
- [ ] 更新 MemorySystem 委托调用
- [ ] 运行测试，确保功能不变

### 3.3 阶段 3: 移动检索逻辑
- [ ] 将 `retrieve()` 移到 `RetrieverCoordinator`
- [ ] 更新 MemorySystem 委托调用
- [ ] 运行测试

### 3.4 阶段 4: 移动生命周期管理
- [ ] 将 `apply_decay()`, `run_dmn()`, `run_dynamics_cycle()`, `trigger_reflection()` 移到 `LifecycleCoordinator`
- [ ] 更新 MemorySystem 委托调用
- [ ] 运行测试

### 3.5 阶段 5: 移动批量操作和统计可视化
- [ ] 将 `batch_decay()` 移到 `BatchOperations`
- [ ] 将 `get_statistics()`, `to_graphviz()` 等移到 `StatsVizCoordinator`
- [ ] 更新 MemorySystem 委托调用
- [ ] 运行测试

### 3.6 阶段 6: 清理和验证
- [ ] 删除 MemorySystem 中已移动的方法
- [ ] 验证行数 < 200
- [ ] 运行完整测试套件
- [ ] 更新文档

---

## 四、风险与缓解措施

| 风险 | 概率 | 影响 | 缓解措施 |
|-----|------|------|---------|
| 破坏现有API | 低 | 高 | 保持所有公共方法签名不变，只改变内部实现 |
| 测试失败 | 中 | 中 | 每个阶段都运行测试，及时回滚 |
| 循环导入 | 低 | 中 | 使用 TYPE_CHECKING 延迟导入 |
| 性能下降 | 低 | 低 | 协调器只是委托调用，无额外开销 |

---

## 五、验收标准

| # | 标准 | 当前 | 目标 |
|---|------|------|------|
| 1 | MemorySystem行数 | 877 | < 200 |
| 2 | Facade接口稳定 | N/A | 所有测试通过 |
| 3 | 协调器数量 | 0 | 5 |
| 4 | 代码可读性 | N/A | 每个协调器 < 150 行 |

---

## 六、T2: Sprint 文档整合方案

### 6.1 需要合并的目录
- `docs/orch/sprint8/` - 包含 pitfalls.md, SPRINT.md
- `docs/orch/sprint9/` - 包含 pitfalls.md, SPRINT.md
- `docs/orch/sprint10/` - 包含 eval.md, gen_status.md, pitfalls.md, plan.md, SPRINT.md

### 6.2 合并策略
1. 读取各Sprint的文档内容
2. 按时间顺序整合到 `docs/chronicle.md`
3. 保留关键决策和pitfalls
4. 删除冗余目录 `sprint8/`, `sprint9/`, `sprint10/`

### 6.3 输出格式
```markdown
## Sprint 8: 神经元动态 (2026-04-XX)
### 关键决策
- ...
### Pitfalls
- ...
### 成果
- ...
```

---

## 七、T3: PostgreSQL 类型提示完善方案

### 7.1 需要完善的函数
根据 `memory/storage/postgres.py` 分析：

| 函数名 | 当前状态 | 需要补充 |
|-------|---------|---------|
| `_serialize_connection()` | 有参数类型 | 补充返回类型 |
| `_deserialize_connection()` | 有参数类型 | 补充返回类型 |
| `_neuron_to_row()` | 有参数类型 | 补充返回类型 |
| `_row_to_neuron()` | 有参数类型 | 补充返回类型 |
| `_engram_to_row()` | 有参数类型 | 补充返回类型 |
| `_row_to_engram()` | 有参数类型 | 补充返回类型 |
| `save_neuron()` | 有参数类型 | 补充返回类型、docstring |
| `load_neuron()` | 有参数类型 | 补充返回类型、docstring |
| `delete_neuron()` | 有参数类型 | 补充返回类型、docstring |
| `list_neurons()` | 有参数类型 | 补充返回类型、docstring |
| `save_engram()` | 有参数类型 | 补充返回类型、docstring |
| `load_engram()` | 有参数类型 | 补充返回类型、docstring |
| `load_engrams()` | 有参数类型 | 补充返回类型、docstring |
| `delete_engram()` | 有参数类型 | 补充返回类型、docstring |
| `save_embedding()` | 有参数类型 | 补充返回类型、docstring |
| `load_embedding()` | 有参数类型 | 补充返回类型、docstring |

### 7.2 实施步骤
1. 为所有helper函数补充返回类型
2. 为所有PostgresStorage方法补充完整的docstring
3. 使用TYPE_CHECKING避免循环导入
4. 运行mypy检查（可选）

---

## 八、总结

### 关键成果
1. **设计文档**: 完整的重构方案，包括架构图、实施计划
2. **风险缓解**: 识别风险并制定应对措施
3. **验收标准**: 明确的量化指标
4. **并行任务**: T2、T3的实施方案

### 下一步（Phase 2）
开始实施重构，按照阶段逐步执行，每阶段都运行测试验证。
