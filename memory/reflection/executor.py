"""
Reflection 执行器

实现 ReflectionExecutor 类，执行 reflection 逻辑并生成结果。
"""

import uuid
from typing import Dict, List, Optional, TYPE_CHECKING
from datetime import datetime

from memory.schemas import (
    ReflectionConfig, ReflectionResult, TriggerCondition,
    MIN_CONFLICT_PAIRS, MAX_REFLECTION_LENGTH
)

if TYPE_CHECKING:
    from memory.engram import Engram


class ReflectionExecutor:
    """
    Reflection 执行器
    
    负责执行 reflection 逻辑并生成结果。
    """
    
    def __init__(self, config: Optional[ReflectionConfig] = None):
        self.config = config or ReflectionConfig()
    
    def execute(
        self,
        triggers: List[TriggerCondition],
        engrams: List['Engram'],
        memory_manager: Optional[object] = None
    ) -> ReflectionResult:
        """
        执行 reflection
        
        Args:
            triggers: 触发的条件
            engrams: 相关的记忆列表
            memory_manager: 记忆管理器（用于更新记忆）
        
        Returns:
            ReflectionResult
        """
        if not triggers:
            return ReflectionResult(
                reflection_type='consolidation',
                content='No triggers activated',
                triggered_conditions=[],
                involved_engrams=[],
                involved_neurons=[]
            )
        
        primary_trigger = max(triggers, key=lambda x: x.priority)
        
        if primary_trigger.type == 'conflict':
            return self._resolve_conflicts(triggers, engrams, memory_manager)
        elif primary_trigger.type == 'association':
            return self._discover_associations(triggers, engrams, memory_manager)
        else:
            return self._consolidate_memory(triggers, engrams, memory_manager)
    
    def _resolve_conflicts(
        self,
        triggers: List[TriggerCondition],
        engrams: List['Engram'],
        memory_manager: Optional[object]
    ) -> ReflectionResult:
        """解决记忆冲突"""
        conflict_trigger = next(
            (t for t in triggers if t.type == 'conflict'), 
            None
        )
        
        if not conflict_trigger:
            return self._consolidate_memory(triggers, engrams, memory_manager)
        
        conflicts = conflict_trigger.metadata.get('conflicts', [])
        involved_engrams = []
        involved_neurons = []
        resolution_content = []
        
        for engram1, engram2, strength_diff in conflicts[:self.config.min_conflict_pairs]:
            involved_engrams.extend([str(engram1.uuid), str(engram2.uuid)])
            
            neurons1 = list(engram1.get_all_neurons())
            neurons2 = list(engram2.get_all_neurons())
            involved_neurons.extend([n.event_id for n in neurons1])
            involved_neurons.extend([n.event_id for n in neurons2])
            
            stronger = engram1 if engram1.strength > engram2.strength else engram2
            weaker = engram2 if engram1.strength > engram2.strength else engram1
            
            resolution = (
                f"冲突检测：记忆 '{engram1.summary[:30]}' 和 '{engram2.summary[:30]}' "
                f"存在强度差异 {strength_diff:.2f}。"
                f"建议：增强较弱记忆的连接，或重新评估较强记忆的权重。"
            )
            resolution_content.append(resolution)
            
            if memory_manager:
                for neuron in list(weak.get_all_neurons()):
                    neuron.strength = min(1.0, neuron.strength * 1.1)
        
        content = "\n".join(resolution_content)
        if len(content) > self.config.max_reflection_length:
            content = content[:self.config.max_reflection_length] + "..."
        
        return ReflectionResult(
            reflection_type='conflict_resolution',
            content=content,
            triggered_conditions=triggers,
            involved_engrams=[uuid.UUID(e) for e in set(involved_engrams)],
            involved_neurons=list(set(involved_neurons)),
            conflict_resolved=True,
            impact_score=0.8
        )
    
    def _discover_associations(
        self,
        triggers: List[TriggerCondition],
        engrams: List['Engram'],
        memory_manager: Optional[object]
    ) -> ReflectionResult:
        """发现新关联"""
        assoc_trigger = next(
            (t for t in triggers if t.type == 'association'),
            None
        )
        
        involved_neurons = []
        new_connections = 0
        strengthened = []
        association_content = []
        
        if assoc_trigger and memory_manager:
            associations = assoc_trigger.metadata.get('associations', [])
            for neuron_id, new_ids in associations:
                involved_neurons.append(neuron_id)
                
                neuron = memory_manager.get_neuron(neuron_id)
                if neuron:
                    existing = {conn.target_id for conn in neuron.outgoing_connections}
                    truly_new = [nid for nid in new_ids if nid not in existing]
                    new_connections += len(truly_new)
                    
                    for target_id in truly_new:
                        target = memory_manager.get_neuron(target_id)
                        if target:
                            neuron.strength = min(1.0, neuron.strength * 1.05)
                            target.strength = min(1.0, target.strength * 1.05)
                            strengthened.append(target_id)
        
        for neuron_id in involved_neurons:
            if memory_manager:
                engram = memory_manager.get_engram_for_neuron(neuron_id)
                if engram:
                    strengthened.append(str(engram.uuid))
        
        association_content.append(
            f"新关联发现：检测到 {new_connections} 个新的神经连接。"
            f"这些连接涉及 {len(set(involved_neurons))} 个神经元。"
            f"建议：强化这些新形成的连接路径。"
        )
        
        return ReflectionResult(
            reflection_type='association_discovery',
            content="\n".join(association_content),
            triggered_conditions=triggers,
            involved_engrams=[uuid.UUID(e) for e in set(strengthened)],
            involved_neurons=list(set(involved_neurons)),
            new_connections_created=new_connections,
            strengthened_engrams=[uuid.UUID(e) for e in set(strengthened)]
        )
    
    def _consolidate_memory(
        self,
        triggers: List[TriggerCondition],
        engrams: List['Engram'],
        memory_manager: Optional[object]
    ) -> ReflectionResult:
        """记忆整合"""
        involved_engrams = [e.uuid for e in engrams]
        involved_neurons = []
        strengthened = []
        
        for engram in engrams:
            neurons = list(engram.get_all_neurons())
            involved_neurons.extend([n.event_id for n in neurons])
            
            if memory_manager:
                for neuron in neurons:
                    neuron.strength = min(1.0, neuron.strength * 1.02)
                strengthened.append(engram.uuid)
        
        content = (
            f"定期整合：检查了 {len(engrams)} 个记忆，"
            f"涉及 {len(involved_neurons)} 个神经元。"
            f"记忆网络整体稳定。"
        )
        
        return ReflectionResult(
            reflection_type='consolidation',
            content=content,
            triggered_conditions=triggers,
            involved_engrams=involved_engrams,
            involved_neurons=list(set(involved_neurons)),
            strengthened_engrams=strengthened
        )
