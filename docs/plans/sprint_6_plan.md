# Sprint 6: "共振"机制

**状态**: Phase 1 规划中
**开始时间**: 2026-04-15
**预计完成**: 2026-04-22

## 目标

实现碎片记忆的自动激活（神经共振/同步发放）。

## 核心功能

### 1. 能量最小化模型

```python
class ResonanceEngine:
    """共振引擎"""
    
    def calculate_activation_energy(self, neuron: NeuronCell) -> float:
        """计算激活能量"""
        
    def spread_activation(self, activated_neurons: List[NeuronCell]):
        """扩散激活"""
```

### 2. 扩散激活算法

```python
class ActivationDiffuser:
    """激活扩散器"""
    
    def diffuse(self, seed_neurons: List[NeuronCell], depth: int = 3):
        """从种子神经元扩散激活"""
```

## 验收标准

- [ ] 能量计算正确
- [ ] 激活正确扩散
- [ ] 共振阈值可配置
