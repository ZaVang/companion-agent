# Sprint 3: DMN 模式（空闲整合）

**状态**: Phase 3 完成 ✅
**开始时间**: 2026-04-15
**完成时间**: 2026-04-15

---

## 1. 目标

实现 Default Mode Network (DMN) 模拟，即空闲/睡眠期间的自动记忆整合机制。DMN 在人类大脑中负责：
- 记忆巩固 (memory consolidation)
- 场景想象 (scene construction)
- 社会认知 (social cognition)

## 2. 核心功能设计

### 2.1 DMN 触发条件

```python
class DMNTriggerCondition(BaseModel):
    """DMN 触发条件"""
    idle_duration_min: int = 30          # 空闲持续时间（分钟）
    sleep_mode: bool = False             # 睡眠模式标志
    low_activity_threshold: float = 0.1 # 低活跃度阈值
    scheduled_times: List[str] = []       # 定时触发时间 ["02:00", "06:00"]
```

**触发条件**:
1. **空闲触发**: 系统空闲超过 `idle_duration_min` 分钟
2. **睡眠触发**: `sleep_mode=True` 且系统活跃度低于阈值
3. **定时触发**: 每天指定时间（如凌晨 2:00、6:00）

### 2.2 自动 Reflection 机制

```python
class DMNMode:
    """DMN 模式管理器"""
    
    def should_trigger(self, activity_level: float, idle_time: int) -> bool:
        """判断是否应触发 DMN"""
        
    def consolidate_stm_to_ltm(self, stm: ShortTermMemory, ltm: EpisodicMemory):
        """STM → LTM 固化流程"""
        
    def run_automatic_reflection(self, ltm: EpisodicMemory) -> ReflectionResult:
        """自动运行 reflection"""
```

**Reflection 流程**:
1. 分析最近活跃的神经元
2. 检测记忆冲突或新关联
3. 生成 reflection 内容
4. 将 reflection 写入记忆网络

### 2.3 记忆关联发现

```python
class AssociationFinder:
    """关联发现器"""
    
    def find_hidden_connections(self, neurons: List[NeuronCell]) -> List[Tuple[NeuronCell, NeuronCell]]:
        """发现隐藏的关联"""
        
    def build_semantic_bridges(self, engram1: Engram, engram2: Engram) -> List[NeuronCell]:
        """构建语义桥梁"""
```

### 2.4 弱神经元清理

```python
class NeuronPruner:
    """神经元修剪器"""
    
    WEAK_STRENGTH_THRESHOLD: float = 0.1
    LOW_ELO_THRESHOLD: float = 100.0
    
    def identify_prunable_neurons(self, neurons: List[NeuronCell]) -> List[NeuronCell]:
        """识别可清理的神经元"""
        
    def prune_weak_neurons(self, engram: Engram) -> int:
        """执行清理"""
```

## 3. 实现方案

### 3.1 目录结构

```
memory/dmn/
├── __init__.py
├── core.py          # DMNMode 主类
├── trigger.py       # 触发条件判断
├── consolidator.py  # STM→LTM 固化器
├── associator.py    # 关联发现器
└── pruner.py        # 弱神经元清理
```

### 3.2 核心接口

```python
# memory/dmn/core.py
class DMNMode:
    def __init__(self, config: Optional[DMNConfig] = None):
        self.config = config or DMNConfig()
        self.last_dmn_run: Optional[datetime] = None
        self.idle_start_time: Optional[datetime] = None
        
    def check_and_run(self, 
                     memory_manager: 'MemoryManager',
                     current_time: Optional[datetime] = None) -> Optional[DMNResult]:
        """检查条件并运行 DMN"""
        
    def run_dmn_cycle(self, 
                     stm: ShortTermMemory,
                     ltm: EpisodicMemory,
                     current_time: datetime) -> DMNResult:
        """执行一个完整的 DMN 周期"""
```

## 4. 验收标准

- [ ] DMN 触发条件可配置（空闲时间、睡眠模式、定时）
- [ ] 自动 reflection 能检测冲突和关联
- [ ] STM → LTM 固化流程正确执行
- [ ] 关联发现器能找到隐藏连接
- [ ] 弱神经元清理机制工作正常
- [ ] 单元测试覆盖率 > 60%
- [ ] 集成测试 34+ 通过

## 5. 测试场景

1. **空闲触发**: 模拟 30 分钟空闲后 DMN 自动运行
2. **睡眠模式**: 设置睡眠模式，活跃度低于阈值时触发
3. **定时触发**: 凌晨 2:00 自动触发
4. **固化验证**: STM 中的神经元正确迁移到 LTM
5. **关联发现**: 两个相似 engram 发现共享神经元
6. **弱清理**: 清理 strength < 0.1 的神经元

## 6. 关键决策

1. **DMN 触发频率**: 最小间隔 6 小时，避免过度整合
2. **固化阈值**: strength > 0.5 的神经元才固化到 LTM
3. **清理安全**: 只清理完全没有连接的孤立神经元
4. **Reflection 优先级**: 冲突 > 关联 > 定期整合
