# Generator Status — Iteration 1

## 完成的任务
- [x] 深入理解现有代码的 Elo 机制实现细节 — 发现：代码中有 strength 字段但没有真正的 Elo 竞争机制
- [x] 分析 strength 字段在检索中的实际使用情况 — 发现：检索函数使用了 combined_scores = similarity * strength，但 retrieve.py 纯基于 embedding
- [x] 识别当前衰减机制的局限性 — 发现：衰减率硬编码为 0.995/0.999，一刀切

## 关键发现

### 1. Elo 机制现状：概念存在、实现残缺

**代码位置：**
- `memory/neuron.py`: `NeuronCell.strength` (默认 1.0) + `calculate_connection_strength()`
- `memory/engram.py`: `Engram.strength` (默认 1.0)
- `agent/brain.py`: 检索时使用 `combined_scores = similarity * strength`

**实际实现：**
- `calculate_connection_strength()` 使用 harmonic mean 聚合两个神经元的 strength
- 检索后通过 `np.log(np.exp(strength) + increase_factor)` 调整值

**差距分析（与 pitfalls.md 对比）：**
- ❌ **没有 Elo 竞争力机制**：只有简单的 strength 乘数，没有"战斗"概念
- ❌ **没有排名更新**：检索后只是 log 增长，没有与他者竞争后的相对排名变化
- ❌ **检索结果不受记忆间竞争影响**：热门记忆不会因为被频繁检索而压制冷门记忆

### 2. strength 使用情况：部分使用、不完整

**使用 strength 的检索函数 (brain.py)：**
- `retrieve_from_stm()`: ✅ 使用 combined_scores
- `retrieve_from_ltm()`: ✅ 使用 combined_scores  
- `retrieve_from_engrams()`: ✅ 使用 combined_scores

**不使用 strength 的检索函数 (retrieve.py)：**
- `retrieve_top_k_embeddings()`: ❌ 纯 embedding 相似度

**核心问题：**
- `retrieve.py` 是独立模块，未集成 strength 机制
- 检索后更新逻辑是单向增长 (`log`)，不是竞争式调整

### 3. 衰减机制问题：一刀切、缺动态性

**硬编码位置：**
- `neuron.py` line 79: `decay_rate: float = 0.995`
- `brain.py` line 182: `decrease_factor: float = 0.999`

**问题分析：**
- 所有记忆类型（chat、thought、reflection、experience、perception）使用相同衰减率
- 没有根据情绪类型/冲击力调整衰减
- "差点被撞" 与 "每天遛狗" 应该有不同的衰减曲线，但当前实现没有区分

**pitfalls.md 要求的正确做法：**
> 根据冲击力/情绪类型动态调整衰减率。"差点被撞"应该衰减极慢，"每天遛狗"刺激停止后快速衰减

## 未完成的任务
- Phase 1 全部完成

## 下一步建议

### Phase 2 优先实现：
1. **Elo 竞争机制核心**
   - 设计 NeuronCell/Egram 的 "战斗力" 属性（可基于 strength + 连接数 + 激活频率计算）
   - 实现检索时的竞争逻辑：当前检索结果 vs 候选结果的战斗力比较
   - 参考 Elo 算法：K-factor、胜率期望值 E = 1 / (1 + 10^(R_opponent - R_self) / 400)

2. **动态衰减系统**
   - 为不同 event_type 设计不同的基础衰减率
   - 添加"情绪/冲击力"字段影响衰减速度
   - 或引入时间窗口衰减：短期记忆快衰减，长期记忆慢衰减

3. **统一 retrieve.py 和 brain.py 的检索逻辑**
   - 让 retrieve.py 的函数也能使用 strength × similarity × decay 评分

### 验收命令测试预期：
当前运行验收命令预期能通过基本导入，但 Phase 2 实现后需要新的测试来验证 Elo 竞争逻辑。

## 状态

**PASSED** — Phase 1 诊断任务全部完成，发现了关键实现差距，为 Phase 2 提供了明确的重构方向。
