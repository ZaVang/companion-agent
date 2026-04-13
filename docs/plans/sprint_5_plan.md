# Sprint 5: 场景敏感激活

**状态**: Phase 1 规划中
**开始时间**: 2026-04-15
**预计完成**: 2026-04-22

## 目标

实现多维度场景感知，让记忆激活更加智能和精准。

## 核心功能

### 1. 场景维度扩展

```python
class SceneContext(BaseModel):
    """场景上下文"""
    location: str = ""           # 地点
    time_of_day: str = ""        # 时段
    day_of_week: str = ""        # 星期
    activity: str = ""           # 活动
    mood: str = ""               # 情绪
    social_context: str = ""     # 社交场景
```

### 2. 场景 → 神经元映射

```python
class SceneSensitiveMapper:
    """场景敏感映射器"""
    
    def map_neuron_to_scene(self, neuron_id, scene: SceneContext):
        """将神经元映射到场景"""
        
    def get_neurons_for_scene(self, scene: SceneContext) -> List[NeuronCell]:
        """获取适合当前场景的神经元"""
```

### 3. 场景敏感检索权重

```python
class SceneAwareRetrieval:
    """场景感知检索"""
    
    def calculate_scene_weight(
        self,
        neuron: NeuronCell,
        current_scene: SceneContext
    ) -> float:
        """计算场景权重"""
```

## 验收标准

- [ ] 场景维度可配置
- [ ] 神经元与场景正确关联
- [ ] 场景敏感检索权重计算正确
