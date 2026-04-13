# Sprint 7: 情绪与冲击力

**状态**: Phase 1 规划中
**开始时间**: 2026-04-15
**预计完成**: 2026-04-22

## 目标

实现情绪/冲击力对记忆的影响机制。

## 核心功能

### 1. 情绪评分机制

```python
class EmotionalImpact(BaseModel):
    """情绪影响"""
    valence: float = 0.0    # 效价 [-1, 1]
    arousal: float = 0.0     # 唤醒度 [0, 1]
    dominance: float = 0.0   # 主导性 [0, 1]
    
    @property
    def emotional_intensity(self) -> float:
        """情绪强度"""
```

### 2. 冲击力 → 衰减率映射

```python
class ImpactMapper:
    """冲击力映射器"""
    
    def map_impact_to_decay(
        self,
        impact_score: float,
        base_decay: float
    ) -> float:
        """将冲击力映射到衰减率"""
```

## 验收标准

- [ ] 情绪评分计算正确
- [ ] 冲击力正确影响衰减率
- [ ] 高情绪记忆衰减更慢
