"""
弱神经元清理器

清理弱的、孤立的或低价值的神经元。
"""

from typing import Optional, List, Tuple, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from memory.neuron import NeuronCell
    from memory.engram import Engram

from memory.dmn.models import PruneRecord


class NeuronPruner:
    """
    弱神经元清理器
    
    清理满足以下条件的神经元：
    1. strength < threshold（太弱）
    2. elo < threshold（太低价值）
    3. 完全孤立（没有任何连接）
    """
    
    def __init__(
        self,
        strength_threshold: float = 0.1,
        elo_threshold: float = 100.0,
        max_prune_per_cycle: int = 20,
        preserve_consolidated: bool = True
    ):
        """
        Args:
            strength_threshold: 强度阈值
            elo_threshold: Elo 阈值
            max_prune_per_cycle: 每周期最大清理数量
            preserve_consolidated: 是否保留已固化的神经元
        """
        self.strength_threshold = strength_threshold
        self.elo_threshold = elo_threshold
        self.max_prune_per_cycle = max_prune_per_cycle
        self.preserve_consolidated = preserve_consolidated
    
    def identify_prunable(
        self,
        neurons: List['NeuronCell']
    ) -> List[Tuple['NeuronCell', str]]:
        """
        识别可以清理的神经元
        
        Returns:
            (neuron, reason) 列表
        """
        prunable = []
        
        for neuron in neurons:
            # 跳过已固化的神经元
            if self.preserve_consolidated and neuron.is_consolidated:
                continue
            
            reason = self._get_prune_reason(neuron)
            if reason:
                prunable.append((neuron, reason))
        
        return prunable
    
    def _get_prune_reason(self, neuron: 'NeuronCell') -> Optional[str]:
        """判断神经元应该被清理的原因"""
        # 检查是否孤立
        if self._is_isolated(neuron):
            return "isolated"
        
        # 检查强度
        if neuron.strength < self.strength_threshold:
            return "weak_strength"
        
        return None
    
    def _is_isolated(self, neuron: 'NeuronCell') -> bool:
        """判断神经元是否是孤立的（没有任何连接）"""
        return (
            len(neuron.outgoing_connections) == 0 and 
            len(neuron.incoming_connections) == 0
        )
    
    def prune_from_engram(
        self,
        engram: 'Engram'
    ) -> List['PruneRecord']:
        """
        从 engram 中清理弱神经元
        
        Returns:
            清理记录列表
        """
        neurons = list(engram.get_all_neurons())
        prunable = self.identify_prunable(neurons)
        
        # 按优先级排序：孤立 > 弱强度
        prunable.sort(key=lambda x: (
            0 if x[1] == "isolated" else 1,
            x[0].strength
        ))
        
        # 截取最大数量
        prunable = prunable[:self.max_prune_per_cycle]
        
        records = []
        for neuron, reason in prunable:
            record = PruneRecord(
                neuron_id=str(neuron.event_id),
                reason=reason,
                strength=neuron.strength,
                elo=neuron.strength * 1000,  # 估算
                connections_count=len(neuron.outgoing_connections) + len(neuron.incoming_connections),
                timestamp=datetime.now()
            )
            records.append(record)
        
        # 从 engram 中移除 - 直接操作底层数据结构避免类型问题
        neuron_ids_to_remove = {n.event_id for n, _ in prunable}
        for event_type, neuron_list in engram.engram.items():
            engram.engram[event_type] = [n for n in neuron_list if n.event_id not in neuron_ids_to_remove]
        
        return records
    
    def estimate_pruning_needed(
        self,
        engrams: List['Engram']
    ) -> int:
        """
        估算需要清理的神经元数量
        """
        count = 0
        for engram in engrams:
            neurons = list(engram.get_all_neurons())
            prunable = self.identify_prunable(neurons)
            count += len(prunable)
        return count
    
    def get_statistics(
        self,
        neurons: List['NeuronCell']
    ) -> dict:
        """
        获取神经元统计信息
        
        Returns:
            统计信息字典
        """
        total = len(neurons)
        if total == 0:
            return {
                "total": 0,
                "isolated": 0,
                "weak_strength": 0,
                "consolidated": 0,
                "prunable": 0
            }
        
        isolated = 0
        weak_strength = 0
        consolidated = 0
        
        for neuron in neurons:
            if self._is_isolated(neuron):
                isolated += 1
            if neuron.strength < self.strength_threshold:
                weak_strength += 1
            if neuron.is_consolidated:
                consolidated += 1
        
        prunable = len(self.identify_prunable(neurons))
        
        return {
            "total": total,
            "isolated": isolated,
            "weak_strength": weak_strength,
            "consolidated": consolidated,
            "prunable": prunable,
            "isolation_rate": isolated / total,
            "weak_rate": weak_strength / total
        }
