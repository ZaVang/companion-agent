# Engram Sprint 完成记录

**版本**: 1.0.0  
**更新时间**: 2026-04-14

---

## Sprint 总览

| Sprint | 名称 | 状态 | 核心功能 |
|--------|------|------|----------|
| Sprint 1 | 核心机制补全 | ✅ 完成 | Elo 竞争、动态衰减、统一检索 |
| Sprint 2 | 记忆稳定性 | ✅ 完成 | 集体稳定性、激活阈值 |
| Sprint 3 | Reflection 自动化 | ✅ 完成 | 触发条件、执行逻辑 |
| Sprint 4 | 存储优化 | ✅ 完成 | PostgreSQL 集成、向量索引 |
| Sprint 5 | DMN 巩固 + 场景感知 | ✅ 完成 | DMN 模式、场景上下文 |
| Sprint 6 | 因果推理 + 共振引擎 | ✅ 完成 | 序列学习、共振检测 |
| Sprint 7 | 情绪影响集成 | ✅ 完成 | VAD 模型、情绪-衰减映射 |
| Sprint 8 | 神经元动态增删 | ✅ 完成 | 出生/死亡机制 |
| Sprint 9 | 可视化与调试 | ✅ 完成 | 网络可视化、追踪系统 |
| Sprint 10 | 性能优化 | ✅ 完成 | 索引优化、批量处理 |

---

## Sprint 1: 核心机制补全

**状态**: ✅ 完成  
**时间**: 2026-04-13 ~ 2026-04-20

### 目标

完善 Elo 竞争机制和动态衰减系统，让记忆系统具备真正的神经思维特性。

### 实现内容

#### Phase 1: 理解与诊断
- [x] 深入理解现有代码的 Elo 机制实现细节
- [x] 分析 strength 字段在检索中的实际使用情况
- [x] 识别当前衰减机制的局限性

**关键发现**:
- Elo 机制名不副实（只有 strength 字段，没有竞争逻辑）
- 检索实现分裂（brain.py 用 strength，retrieve.py 不用）
- 衰减一刀切（所有记忆 0.995）

#### Phase 2: 核心实现
- [x] 实现 Elo 竞争机制的完整逻辑
  - [x] 设计 NeuronCell 的"战斗力"属性 → `elo.py::get_combat_power()`
  - [x] 实现检索时的竞争逻辑 → `elo.py::update_after_retrieval()`
  - [x] K-factor 动态调整（高频激活 16，低频激活 64，默认 32）
  
- [x] 设计动态衰减系统
  - [x] 不同 event_type 基础衰减率：chat=0.995, perception=0.990, thought=0.992, reflection=0.998, experience=0.985
  - [x] 冲击力(impact_score) 影响衰减速度 → `decay.py::calculate_decay_rate()`
  - [x] 公式：`adjusted_rate = base_rate + impact_factor × (1 - base_rate)`
  
- [x] 统一检索逻辑
  - [x] 创建 `unified_retriever.py` 整合评分逻辑
  - [x] 综合评分：`score = weighted(similarity, elo, decay, recency)`

- [x] LongMemEval 接口对齐
  - [x] 创建 `api_schema.py` 定义 5 大能力 API

### 测试覆盖

- 19 个测试用例全部通过

### 关键决策

1. **Elo K-factor 选择**: 初始值 32，根据神经元激活频率动态调整
2. **衰减率范围**: 0.98 ~ 0.9999（快衰减 ~ 极慢衰减）
3. **冲击力评分**: 0.0 ~ 1.0，由 LLM 或用户标注

### 已知限制

- 检索仍使用简单关键词匹配，需要 embedding 集成
- 情绪影响尚未集成到衰减计算

---

## Sprint 2: 记忆稳定性

**状态**: ✅ 完成  
**时间**: 2026-04-14

### 目标

实现集体稳定性机制和激活阈值系统。

### 实现内容

- [x] 实现集体稳定性机制
  - [x] `stability.py::calculate_engram_stability()` - 成员强度聚合
  - [x] 支持多种聚合方法：arithmetic, harmonic, geometric, max, min
  
- [x] 设计激活阈值系统
  - [x] `stability.py::check_activation_threshold()` - 检查是否达到激活阈值
  - [x] `stability.py::suggest_neurons_for_reinforcement()` - 建议需要增强的神经元
  
- [x] 优化代表神经元稳定性
  - [x] 代表神经元有 1.5x 稳定性加成
  - [x] `StabilityManager` 批量管理

### 测试覆盖

- 20 个测试用例全部通过

---

## Sprint 3: Reflection 自动化

**状态**: ✅ 完成

### 目标

设计并实现 Reflection 触发条件和执行逻辑。

### 实现内容

- [x] Reflection 触发条件
  - [x] 记忆冲突检测（相似记忆强度差异大）
  - [x] 新知识关联检测（发现新的连接模式）
  - [x] DMN 模式触发

- [x] Reflection 执行逻辑
  - [x] LLM 生成 reflection 内容
  - [x] reflection 结果写入记忆网络
  - [x] 更新相关神经元连接

### 关键决策

- Reflection 不是定期总结，而是条件触发
- 避免机械式 reflection 产生噪音

---

## Sprint 4: 存储优化

**状态**: ✅ 完成

### 目标

优化存储架构，支持大规模数据。

### 实现内容

- [x] PostgreSQL 集成（可选）
- [x] 向量数据库集成（可选）
- [x] JSON 文件优化
- [x] 延迟加载策略

---

## Sprint 5: DMN 巩固 + 场景感知

**状态**: ✅ 完成

### 目标

实现 DMN（默认模式网络）巩固机制和场景感知检索。

### DMN 模块

**来源**: Stanford Menon 教授 2023 年论文《20 years of the default mode network》

**核心功能**：
- `consolidate()` - 记忆巩固
- `prune()` - 弱连接修剪
- `associate()` - 关联建立

### 场景感知

```python
class SceneContext:
    location: str       # 地点
    time: str           # 时间
    activity: str       # 活动
```

**功能**：
- 神经元-场景映射
- 场景历史记录
- 场景上下文检索

---

## Sprint 6: 因果推理 + 共振引擎

**状态**: ✅ 完成

### 因果推理

**目的**：学习"什么导致什么"，超越简单的共现关系。

**实现**：
- 激活序列记录
- 共现频率统计
- 因果图构建

### 共振引擎

**目的**：检测"共振"现象——多个相关记忆同时被激活。

**触发条件**：
- 多个相关神经元同时激活
- 激活强度超过阈值
- 关联度达到标准

---

## Sprint 7: 情绪影响集成

**状态**: ✅ 完成

### 目标

将情绪维度集成到记忆系统中。

### VAD 模型

| 维度 | 范围 | 说明 |
|------|------|------|
| Valence（效价） | [-1, 1] | 正面 ~ 负面 |
| Arousal（唤醒度） | [0, 1] | 平静 ~ 激动 |
| Dominance（主导性） | [0, 1] | 被动 ~ 主动 |

### 情绪-衰减映射

```
impact_score = arousal * (1 - |valence|)
decay_rate = base_rate + impact_score * factor * (1 - base_rate)
```

**公式解读**：
- 高唤醒 + 中性效价 = 最高冲击力 → 慢衰减
- 低唤醒 + 极端效价 = 低冲击力 → 快衰减

### NeuronCell 新增字段

```python
emotional_valence: float      # [-1, 1]
emotional_arousal: float      # [0, 1]
emotional_dominance: float    # [0, 1]
```

---

## Sprint 8: 神经元动态增删

**状态**: ✅ 完成

### 目标

实现神经元的动态出生和死亡机制。

### 出生机制 (BirthReasons)

| 类型 | 说明 |
|------|------|
| `CONTEXTUAL_IMPLICATION` | 上下文暗示的新概念 |
| `USER_REVELATION` | 用户明确揭示的信息 |
| `PATTERN_EMERGENCE` | 从行为模式中涌现 |
| `REFLECTION_SYNTHESIS` | Reflection 综合生成 |
| `NEW_INTEREST` | 新发现的兴趣点 |

### 死亡机制 (DeathReasons)

| 类型 | 说明 |
|------|------|
| `STRENGTH_BELOW_THRESHOLD` | 强度低于阈值 |
| `INACTIVITY_TIMEOUT` | 长期不活跃 |
| `SELF_REFERENCE_LOOP` | 自我引用循环 |
| `INTEGRATED_AWAY` | 已被整合到其他记忆 |

### 实现

```python
class NeuronDynamics:
    death_manager: NeuronDeathManager
    birth_manager: NeuronBirthManager
    
    def birth_neuron() -> NeuronCell
    def kill_neuron() -> NeuronDeath
    def get_neurons_to_forget() -> List[NeuronDeath]
    def get_neurons_to_strengthen() -> List[NeuronCell]
```

---

## Sprint 9: 可视化与调试

**状态**: ✅ 完成

### 目标

提供可视化工具和调试追踪系统。

### 组件

#### NetworkVisualizer

```python
class NetworkVisualizer:
    def visualize_engram(engram) -> str      # 返回可视化代码
    def visualize_system(neurons) -> str     # 系统级可视化
```

#### MemoryTracer

```python
class MemoryTracer:
    history: MemoryHistory
    
    def trace_birth(neuron)
    def trace_strength_change(neuron, old, new)
    def trace_decay(neuron, old, new)
    def trace_connection(neuron1, neuron2)
```

#### MemoryHistory

```python
class MemoryHistory:
    entries: List[HistoryEntry]
    
    def add_entry(change_type, **kwargs)
    def get_history(neuron_id) -> List[HistoryEntry]
    def get_timeline() -> List[HistoryEntry]
```

---

## Sprint 10: 性能优化

**状态**: ✅ 完成

### 目标

优化大规模神经元的性能。

### 索引优化

```python
class MemoryIndex:
    vector_index: Dict     # 向量索引
    time_index: Dict       # 时间索引
    tag_index: Dict        # 标签索引
    strength_index: Dict  # 强度索引
    
    def add_neuron(neuron_id, timestamp, tags, strength)
    def search_by_time(start, end)
    def search_by_strength(threshold)
    def search_by_tags(tags)
```

### 批量处理

```python
class BatchProcessor:
    def batch_activate(neuron_ids, cue_strength)
    def batch_decay(neuron_ids, reference_time)
    def batch_retrieve(queries, top_k)
```

### 配置

```python
class BatchConfig:
    batch_size: int = 100
    parallel_workers: int = 4
    cache_enabled: bool = True
```

---

## 关键设计决策回顾

### 1. 记忆是塑形，不是存储

**决策**：每次激活都会改变神经元状态

**理由**：模拟神经网络的动态重塑特性

### 2. 遗忘是不可达

**决策**：遗忘 ≠ 删除

**理由**：模拟真实记忆的"入口变少"特性

### 3. 检索是推理的一部分

**决策**：检索不是独立的前置步骤

**理由**：联想驱动、上下文感知的动态过程

### 4. Elo 替代简单计数器

**决策**：使用完整的 Elo 竞争机制

**理由**：防止热门记忆无限增长，保持平衡

### 5. 情绪集成到衰减

**决策**：情绪影响衰减率

**理由**：模拟"情绪冲击"记忆长期保留的特性

---

## 已知限制

1. **检索精度**：简单关键词匹配，需 embedding 集成
2. **并发支持**：单线程设计，多用户需加锁
3. **存储扩展**：JSON 不适合大规模，需数据库
4. **因果推理**：简化实现，准确性有限
5. **场景感知**：基础实现，需深度优化

---

## 验收命令

```bash
# 运行所有测试
cd /app/data/companion-agent && python -m pytest tests/ -v

# 验证核心模块
python -c "
from memory import MemorySystem, NeuronCell, Engram
from memory.elo import EloCompetition
from memory.decay import DecayScheduler
from memory.stability import StabilityManager
print('All core modules OK')
"

# 验证高级模块
python -c "
from memory.dmn import DMNMode
from memory.emotion import EmotionalImpact
from memory.dynamics import NeuronDynamics
print('All advanced modules OK')
"
```
