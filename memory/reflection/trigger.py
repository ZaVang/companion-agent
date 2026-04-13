"""
Reflection 触发条件检测

实现 ReflectionTrigger 类，检测各种触发条件。
"""

from typing import Dict, List, Optional, Set, Tuple, Callable
from datetime import datetime
from pydantic import UUID1

from memory.engram import Engram
from memory.neuron import NeuronCell
from memory.schemas import (
    ReflectionConfig, TriggerCondition,
    CONFLICT_STRENGTH_DIFF, CONFLICT_SIMILARITY_MIN,
    NEW_CONNECTION_THRESHOLD, REFLECTION_INTERVAL_HOURS
)


# ============== 触发条件检测函数 ==============

def detect_conflict(
    engrams: List[Engram],
    similarity_fn: Callable[[Engram, Engram], float],
    config: Optional[ReflectionConfig] = None
) -> List[Tuple[Engram, Engram, float]]:
    """
    检测记忆冲突
    
    查找相似但强度差异大的记忆对。
    """
    if config is None:
        config = ReflectionConfig()
    
    conflicts = []
    
    for i, engram1 in enumerate(engrams):
        for engram2 in engrams[i + 1:]:
            similarity = similarity_fn(engram1, engram2)
            
            if similarity < config.conflict_similarity_min:
                continue
            
            strength_diff = abs(engram1.strength - engram2.strength)
            
            if strength_diff >= config.conflict_strength_diff:
                conflicts.append((engram1, engram2, strength_diff))
    
    conflicts.sort(key=lambda x: x[2], reverse=True)
    return conflicts


def detect_new_associations(
    recent_neurons: List[NeuronCell],
    baseline_connections: Dict[str, Set[str]],
    config: Optional[ReflectionConfig] = None
) -> List[Tuple[NeuronCell, List[str]]]:
    """
    检测新关联
    
    查找新增连接的神经元。
    """
    if config is None:
        config = ReflectionConfig()
    
    new_associations = []
    
    for neuron in recent_neurons:
        current_connections = {str(conn.target_id) for conn in neuron.outgoing_connections}
        baseline = baseline_connections.get(str(neuron.event_id), set())
        new_connections = current_connections - baseline
        
        if len(new_connections) >= config.new_connection_threshold:
            new_associations.append((neuron, list(new_connections)))
    
    return new_associations


def should_trigger_scheduled(
    last_reflection_time: Optional[datetime],
    config: Optional[ReflectionConfig] = None
) -> bool:
    """判断是否应该触发定期 reflection"""
    if config is None:
        config = ReflectionConfig()
    
    if last_reflection_time is None:
        return True
    
    now = datetime.now()
    hours_since = (now - last_reflection_time).total_seconds() / 3600
    
    return hours_since >= config.reflection_interval_hours


# ============== Reflection 触发器 ==============

class ReflectionTrigger:
    """
    Reflection 触发器
    
    负责检测各种触发条件并决定是否触发 reflection。
    """
    
    def __init__(self, config: Optional[ReflectionConfig] = None):
        self.config = config or ReflectionConfig()
        self._last_reflection_time: Optional[datetime] = None
        self._baseline_connections: Dict[str, Set[str]] = {}
        self._trigger_history: List[TriggerCondition] = []
    
    def check_triggers(
        self,
        engrams: List[Engram],
        recent_neurons: List[NeuronCell],
        similarity_fn: Optional[Callable] = None
    ) -> List[TriggerCondition]:
        """
        检查所有触发条件
        
        Returns:
            触发的条件列表（按优先级排序）
        """
        triggers = []
        
        # 1. 检查记忆冲突
        if similarity_fn:
            conflicts = detect_conflict(engrams, similarity_fn, self.config)
            if conflicts:
                triggers.append(TriggerCondition(
                    type='conflict',
                    description=f"检测到 {len(conflicts)} 个记忆冲突",
                    priority=3,
                    metadata={'conflict_count': len(conflicts), 'conflicts': conflicts}
                ))
        
        # 2. 检查新关联
        new_assocs = detect_new_associations(
            recent_neurons, 
            self._baseline_connections, 
            self.config
        )
        if new_assocs:
            triggers.append(TriggerCondition(
                type='association',
                description=f"发现 {len(new_assocs)} 个新关联模式",
                priority=2,
                metadata={'association_count': len(new_assocs), 'associations': new_assocs}
            ))
        
        # 3. 检查定期触发
        if should_trigger_scheduled(self._last_reflection_time, self.config):
            triggers.append(TriggerCondition(
                type='scheduled',
                description="定期 reflection 触发",
                priority=1
            ))
        
        self._trigger_history.extend(triggers)
        triggers.sort(key=lambda x: x.priority, reverse=True)
        return triggers
    
    def update_baseline(self, neurons: List[NeuronCell]):
        """更新基线连接状态"""
        for neuron in neurons:
            self._baseline_connections[str(neuron.event_id)] = {
                str(conn.target_id) for conn in neuron.outgoing_connections
            }
    
    @property
    def last_reflection_time(self) -> Optional[datetime]:
        return self._last_reflection_time
    
    def mark_reflection_completed(self):
        """标记 reflection 已完成"""
        self._last_reflection_time = datetime.now()
    
    @property
    def trigger_history(self) -> List[TriggerCondition]:
        return self._trigger_history.copy()
