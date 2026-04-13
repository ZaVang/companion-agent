# Sprint 1: 核心机制补全

**状态**: 进行中
**开始时间**: 2026-04-13
**预计完成**: 2026-04-20

---

## 目标

完善 Elo 竞争机制和动态衰减系统，让记忆系统具备真正的神经思维特性。

---

## 任务清单

### Phase 1: 理解与诊断 ✅

- [x] 深入理解现有代码的 Elo 机制实现细节
- [x] 分析 strength 字段在检索中的实际使用情况
- [x] 识别当前衰减机制的局限性

**关键发现**:
- Elo 机制名不副实（只有 strength 字段，没有竞争逻辑）
- 检索实现分裂（brain.py 用 strength，retrieve.py 不用）
- 衰减一刀切（所有记忆 0.995）

### Phase 2: 核心实现

- [ ] 实现 Elo 竞争机制的完整逻辑
  - [ ] 设计 NeuronCell 的"战斗力"属性
  - [ ] 实现检索时的竞争逻辑（信号竞争 → 战斗力计算 → Elo 更新）
  - [ ] 参考 Elo 算法：K-factor + 胜率期望值
  
- [ ] 设计动态衰减系统
  - [ ] 为不同 event_type 设计不同的基础衰减率
  - [ ] 添加"冲击力"字段影响衰减速度
  - [ ] 实现 `calculate_decay_rate(neuron, impact_score)` 函数
  
- [ ] 统一检索逻辑
  - [ ] 让 retrieve.py 使用 strength × similarity × decay 评分
  - [ ] 合并 brain.py 和 retrieve.py 的检索函数

### Phase 3: Reflection 自动化

- [ ] 设计 reflection 触发条件
  - [ ] 记忆冲突（相似记忆强度差异大）
  - [ ] 新知识关联（发现新的连接模式）
  - [ ] 定期触发（DMN 模式的一部分）
  
- [ ] 实现 reflection 执行逻辑
  - [ ] LLM 生成 reflection 内容
  - [ ] reflection 结果写入记忆网络
  - [ ] 更新相关神经元连接

### Phase 4: 测试与验证

- [ ] 编写 Elo 竞争机制测试用例
- [ ] 编写动态衰减测试用例
- [ ] 模拟"差点被撞"场景（高冲击力，慢衰减）
- [ ] 模拟"每天遛狗"场景（低冲击力，快衰减）

---

## 验收命令

```bash
# 1. 运行单元测试
cd /app/data/companion-agent && python -m pytest tests/ -v

# 2. 验证 Elo 竞争机制
python -c "
from memory.elo import EloCompetitor
elo = EloCompetitor()
print('Elo mechanism OK')
"

# 3. 验证动态衰减
python -c "
from memory.decay import calculate_decay_rate
rate = calculate_decay_rate(event_type='chat', impact_score=0.9)
assert rate > 0.99, 'High impact should decay slowly'
print('Dynamic decay OK')
"
```

---

## 关键决策

1. **Elo K-factor 选择**: 初始值 32，根据神经元激活频率动态调整
2. **衰减率范围**: 0.98 ~ 0.9999（快衰减 ~ 极慢衰减）
3. **冲击力评分**: 0.0 ~ 1.0，由 LLM 或用户标注

---

## 相关文档

- [architecture.md](../architecture.md) - 系统架构设计
- [pitfalls.md](./pitfalls.md) - 陷阱知识库
- [FUTURE.md](../FUTURE.md) - 后续 Sprint 规划

---

