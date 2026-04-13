# Engram 记忆系统重构 Sprint

**Goal**: 重构 companion-agent 的 engram 记忆系统，实现真正的神经思维记忆架构

**Architecture**: 基于神经元网络 + Elo 竞争 + 动态衰减 + 因果推断

**Tech Stack**: Python 3.13, PostgreSQL, NetworkX, NumPy

---

## 任务清单

### Phase 1: 理解与诊断

- [x] 深入理解现有代码的 Elo 机制实现细节
- [x] 分析 strength 字段在检索中的实际使用情况
- [x] 识别当前衰减机制的局限性

### Phase 2: 核心机制补全

- [ ] 实现 Elo 竞争机制的完整逻辑（信号竞争 → 战斗力计算 → Elo 更新）
- [ ] 设计动态衰减系统（根据冲击力/情绪类型调整衰减率）
- [ ] 实现 reflection 自动化触发机制

### Phase 3: 高级特性

- [ ] 设计因果推断层（记忆 → 因果图）
- [ ] 实现场景敏感激活（不只是 audience 维度）
- [ ] 实现"共振"机制（碎片记忆的自动激活）

### Phase 4: 验证与集成

- [ ] 编写测试用例验证记忆竞争逻辑
- [ ] 模拟"差点被撞"场景测试冲击力衰减
- [ ] 集成到 Ari 的运行环境

---

## 验收命令

```bash
# 1. 运行单元测试
cd /app/data/companion-agent && python -m pytest tests/ -v

# 2. 验证 Elo 竞争机制
python -c "from memory.engram import Engram; print('Elo mechanism OK')"

# 3. 验证衰减系统
python -c "from memory.neuron import calculate_connection_strength; print('Decay OK')"
```

---

## 关键问题

1. **Elo 如何真正影响行为？**
   - 当前 strength 字段存在但未在检索中使用
   - 需要将 Elo 融入检索排序逻辑

2. **衰减参数从哪来？**
   - 当前硬编码 decay_rate=0.995
   - 需要根据记忆类型动态调整

3. **Reflection 如何自动触发？**
   - 有 ReflectionEvent 类型但无触发逻辑
   - 需要设计触发条件和频率

4. **因果推断如何实现？**
   - 完全缺失
   - 需要设计从记忆到因果图的映射

---

## 时间线

- Week 1: Phase 1（理解与诊断）
- Week 2: Phase 2（核心机制补全）
- Week 3: Phase 3（高级特性）
- Week 4: Phase 4（验证与集成）

---

