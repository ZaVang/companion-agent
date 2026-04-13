# Sprint 4: 因果推断层

**状态**: Phase 3 完成 ✅
**开始时间**: 2026-04-15
**完成时间**: 2026-04-15

---

## 1. 目标

实现因果推断层，从记忆网络中学习因果关系。因果推断允许系统：
- 理解事件之间的因果链条
- 预测行动的结果
- 推断隐藏的原因

## 2. 核心功能设计

### 2.1 激活序列记录

```python
class ActivationSequence:
    """激活序列记录"""
    sequence_id: UUID1
    timestamps: List[datetime]
    neuron_ids: List[UUID1]
    event_types: List[str]
    contexts: List[str]  # 场景上下文
    
    def add_activation(self, neuron_id: UUID1, timestamp: datetime, context: str = ""):
        """添加激活记录"""
        
    def get_temporal_order(self) -> List[UUID1]:
        """获取时间顺序的激活序列"""
```

### 2.2 共现频率统计

```python
class CoOccurrenceMatrix:
    """共现矩阵"""
    matrix: Dict[str, Dict[str, int]]  # {neuron1: {neuron2: count}}
    total_activations: int
    
    def record_cooccurrence(self, neuron1: UUID1, neuron2: UUID1):
        """记录共现"""
        
    def get_frequency(self, neuron1: UUID1, neuron2: UUID1) -> float:
        """获取共现频率"""
        
    def get_strongest_correlations(self, neuron_id: UUID1, top_k: int = 5) -> List[Tuple[UUID1, float]]:
        """获取最强关联"""
```

### 2.3 因果图数据结构

```python
class CausalGraph:
    """因果图"""
    nodes: Set[UUID1]                    # 神经元节点
    edges: Dict[Tuple[UUID1, UUID1], CausalEdge]  # (cause, effect) -> edge
    
    # 统计信息
    activation_counts: Dict[UUID1, int]
    transition_counts: Dict[Tuple[UUID1, UUID1], int]
    
    def add_causal_link(self, cause: UUID1, effect: UUID1, strength: float = 1.0):
        """添加因果链接"""
        
    def get_causal_parents(self, node: UUID1) -> List[UUID1]:
        """获取节点的因"""
        
    def get_causal_children(self, node: UUID1) -> List[UUID1]:
        """获取节点的果"""
        
    def infer_probable_effects(self, cause: UUID1) -> List[Tuple[UUID1, float]]:
        """推断可能的果"""


class CausalEdge:
    """因果边"""
    cause_id: UUID1
    effect_id: UUID1
    strength: float               # 因果强度
    confidence: float             # 置信度
    evidence_count: int           # 证据数量
    temporal_delay_ms: int        # 时间延迟（毫秒）
```

### 2.4 因果推断算法

```python
class CausalInference:
    """因果推断引擎"""
    
    def __init__(self, graph: CausalGraph, config: CausalConfig):
        self.graph = graph
        self.config = config
    
    def detect_causal_direction(
        self, 
        neuron_a: UUID1, 
        neuron_b: UUID1,
        activation_sequences: List[ActivationSequence]
    ) -> Tuple[UUID1, UUID1, float]:
        """
        检测因果方向
        
        使用时间顺序和统计方法判断 A→B 还是 B→A。
        返回: (cause, effect, confidence)
        """
        
    def compute_causal_strength(
        self,
        cause: UUID1,
        effect: UUID1,
        sequences: List[ActivationSequence]
    ) -> float:
        """
        计算因果强度
        
        使用条件概率: P(effect | cause) / P(effect | ~cause)
        """
        
    def find_causal_chains(
        self,
        start: UUID1,
        max_length: int = 5
    ) -> List[List[UUID1]]:
        """
        查找因果链
        
        找出从起点出发的所有可能因果链。
        """
        
    def predict_effects(
        self,
        cause: UUID1,
        current_time: datetime,
        max_predictions: int = 5
    ) -> List[CausalPrediction]:
        """
        预测效果
        
        给定原因，预测可能的未来效果。
        """
```

### 2.5 因果网络可视化

```python
class CausalVisualizer:
    """因果网络可视化"""
    
    def generate_causal_graphviz(
        self,
        graph: CausalGraph,
        highlight_nodes: Set[UUID1] = None
    ) -> str:
        """生成 Graphviz DOT 格式"""
        
    def generate_causal_d3_json(
        self,
        graph: CausalGraph
    ) -> dict:
        """生成 D3.js 可视化 JSON"""
```

## 3. 实现方案

### 3.1 目录结构

```
memory/causal/
├── __init__.py
├── core.py          # CausalGraph 主类
├── sequence.py      # ActivationSequence
├── statistics.py    # CoOccurrenceMatrix
├── inference.py     # CausalInference
├── config.py        # 配置
└── visualize.py     # 可视化
```

## 4. 验收标准

- [ ] 激活序列正确记录和检索
- [ ] 共现矩阵正确统计
- [ ] 因果图正确构建
- [ ] 因果方向检测准确
- [ ] 因果强度计算正确
- [ ] 单元测试覆盖率 > 60%
- [ ] 集成测试 53+ 通过

## 5. 测试场景

1. **序列记录**: 记录多个激活事件，验证时间顺序
2. **共现统计**: 验证共现频率计算
3. **因果检测**: 给定激活序列，正确判断因果方向
4. **因果链**: 从起点找到完整的因果链
5. **预测**: 给定原因，预测可能的效果

## 6. 关键决策

1. **因果判定**: 使用 Granger 因果检验思想 - 如果 A 预测 B 比 B 预测 A 更准，则 A→B
2. **时间窗口**: 激活在 5 秒内视为相关
3. **最小证据**: 需要至少 3 次共现才能建立因果关系
4. **置信度**: 基于证据数量动态计算
