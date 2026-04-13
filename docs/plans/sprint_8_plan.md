# Sprint 8: 神经元动态增删

**状态**: Phase 1 规划中
**开始时间**: 2026-04-15
**预计完成**: 2026-04-22

## 目标

支持神经元数量的动态变化。

## 核心功能

### 1. 神经元"死亡"机制

```python
class NeuronDeathCondition(BaseModel):
    """神经元死亡条件"""
    strength_threshold: float = 0.05
    elo_threshold: float = 50.0
    no_activation_days: int = 30
```

### 2. 神经元"新生"机制

```python
class NeuronBirthPolicy(BaseModel):
    """神经元新生策略"""
    max_neurons: int = 10000
    birth_rate: float = 0.1  # 每天最多新增比例
```

## 验收标准

- [ ] 死亡条件正确判断
- [ ] 新生策略正确执行
- [ ] 连接正确继承/迁移
