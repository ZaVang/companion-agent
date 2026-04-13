"""
STM → LTM 固化器

负责将短时记忆中的重要神经元迁移到长时记忆。
"""

from typing import Optional, List, Dict, TYPE_CHECKING
from datetime import datetime
import uuid
from pydantic import UUID1

if TYPE_CHECKING:
    from memory.memory import ShortTermMemory, EpisodicMemory
    from memory.neuron import NeuronCell
    from memory.engram import Engram

from memory.dmn.models import ConsolidationRecord
from memory.memory import RegistryMetadata
from memory.engram import Engram


class STMConsolidator:
    """
    STM → LTM 固化器
    
    负责将短时记忆中的神经元迁移到长时记忆。
    """
    
    def __init__(
        self,
        strength_threshold: float = 0.5,
        max_per_cycle: int = 50
    ):
        """
        Args:
            strength_threshold: 固化阈值，只有 strength >= threshold 的神经元才会固化
            max_per_cycle: 每周期最大固化数量
        """
        self.strength_threshold = strength_threshold
        self.max_per_cycle = max_per_cycle
    
    def consolidate(
        self,
        stm: 'ShortTermMemory',
        ltm: 'EpisodicMemory',
        audience: Optional[str] = None
    ) -> List['ConsolidationRecord']:
        """
        执行 STM → LTM 固化
        
        Args:
            stm: 短时记忆
            ltm: 长时记忆
            audience: 特定用户的记忆，如果不指定则处理所有
        
        Returns:
            固化记录列表
        """
        records = []
        audiences_to_process = [audience] if audience else list(stm.sequences.keys())
        
        for aud in audiences_to_process:
            if aud not in stm.sequences:
                continue
            
            engram = stm.sequences[aud]
            consolidation_candidates = self._get_candidates(engram)
            
            # 按 strength 排序，优先固化强的
            consolidation_candidates.sort(key=lambda x: x.strength, reverse=True)
            consolidation_candidates = consolidation_candidates[:self.max_per_cycle]
            
            for neuron in consolidation_candidates:
                record = self._consolidate_neuron(
                    neuron=neuron,
                    source_engram_uuid=str(engram.uuid),
                    ltm=ltm,
                    audience=aud
                )
                if record:
                    records.append(record)
        
        return records
    
    def _get_candidates(self, engram: 'Engram') -> List['NeuronCell']:
        """
        获取可以固化的神经元候选
        
        条件：
        1. strength >= threshold
        2. 尚未固化到 LTM
        """
        candidates = []
        for neuron in engram.get_all_neurons():
            if neuron.strength >= self.strength_threshold:
                # 检查是否有足够的连接（确保是有意义的记忆）
                if self._is_meaningful(neuron):
                    candidates.append(neuron)
        return candidates
    
    def _is_meaningful(self, neuron: 'NeuronCell') -> bool:
        """
        判断神经元是否是有意义的记忆
        
        有意义的记忆应该有：
        1. 至少一个出向或入向连接
        2. 或者 strength 非常高（>= 0.8）
        """
        if neuron.strength >= 0.8:
            return True
        return len(neuron.outgoing_connections) > 0 or len(neuron.incoming_connections) > 0
    
    def _consolidate_neuron(
        self,
        neuron: 'NeuronCell',
        source_engram_uuid: str,
        ltm: 'EpisodicMemory',
        audience: str
    ) -> Optional['ConsolidationRecord']:
        """
        固化单个神经元到 LTM
        
        如果目标 audience 的 engram 不存在则创建。
        """
        # 检查是否已经在 LTM 中
        existing = self._find_in_ltm(ltm, audience, neuron.event_id)
        if existing:
            # 更新 existing neuron
            strength_before = existing.strength
            existing.strength = max(existing.strength, neuron.strength)
            existing.is_consolidated = True
            return ConsolidationRecord(
                neuron_id=str(neuron.event_id),
                source_engram=source_engram_uuid,
                target_engram=str(neuron.event_id),
                strength_before=strength_before,
                strength_after=existing.strength,
                timestamp=datetime.now()
            )
        
        # 复制到 LTM - 创建新 Engram
        neuron.is_consolidated = True
        
        if audience not in ltm.engram_managers:
            from memory.engram import EngramManager
            ltm.engram_managers[audience] = EngramManager()
        
        # 创建一个包含单个神经元的新 Engram
        new_engram = Engram(
            uuid=uuid.uuid1(),
            time=datetime.now(),
            actor=[neuron.actor]
        )
        new_engram.add_neurons(neuron)
        ltm.engram_managers[audience].add_engram(new_engram)
        
        # 注册
        if audience not in ltm.registry:
            ltm.registry[audience] = {}
        ltm.registry[audience][new_engram.uuid] = RegistryMetadata(
            time=new_engram.time,
            summary=f"Consolidated neuron {neuron.event_id}",
            strength=neuron.strength,
            actor=[neuron.actor]
        )
        
        return ConsolidationRecord(
            neuron_id=str(neuron.event_id),
            source_engram=source_engram_uuid,
            target_engram=str(new_engram.uuid),
            strength_before=neuron.strength,
            strength_after=neuron.strength,
            timestamp=datetime.now()
        )
    
    def _find_in_ltm(
        self,
        ltm: 'EpisodicMemory',
        audience: str,
        neuron_id: UUID1
    ) -> Optional['NeuronCell']:
        """在 LTM 中查找神经元"""
        if audience not in ltm.engram_managers:
            return None
        return ltm.engram_managers[audience].engram_dict.get(str(neuron_id))
    
    def estimate_consolidation_needed(self, stm: 'ShortTermMemory') -> int:
        """
        估算需要固化的神经元数量
        """
        count = 0
        for engram in stm.sequences.values():
            for neuron in engram.get_all_neurons():
                if neuron.strength >= self.strength_threshold and self._is_meaningful(neuron):
                    count += 1
        return count
