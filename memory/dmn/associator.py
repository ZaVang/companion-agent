"""
关联发现器

发现记忆之间的隐藏关联。
"""

from typing import Optional, List, Tuple, Dict, Set, TYPE_CHECKING
from datetime import datetime
import uuid

if TYPE_CHECKING:
    from memory.neuron import NeuronCell
    from memory.engram import Engram

from memory.dmn.models import AssociationRecord


class AssociationFinder:
    """
    关联发现器
    
    发现记忆之间的隐藏关联：
    1. 语义关联：通过内容相似度
    2. 时间关联：通过激活时间接近
    3. 上下文关联：通过共享连接
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.5,
        temporal_window_minutes: int = 30,
        max_associations: int = 10
    ):
        """
        Args:
            similarity_threshold: 最小相似度阈值
            temporal_window_minutes: 时间窗口（分钟）
            max_associations: 每周期最大关联数
        """
        self.similarity_threshold = similarity_threshold
        self.temporal_window_minutes = temporal_window_minutes
        self.max_associations = max_associations
    
    def find_associations(
        self,
        neurons: List['NeuronCell'],
        engrams: Optional[List['Engram']] = None
    ) -> List['AssociationRecord']:
        """
        发现神经元之间的关联
        
        Args:
            neurons: 神经元列表
            engrams: 可选的 engram 列表（用于更深入的关联分析）
        
        Returns:
            关联记录列表
        """
        records = []
        seen_pairs: Set[Tuple[str, str]] = set()
        
        # 1. 通过共享连接发现关联
        for neuron in neurons:
            for other in neurons:
                if neuron.event_id == other.event_id:
                    continue
                
                pair = tuple(sorted([str(neuron.event_id), str(other.event_id)]))
                if pair in seen_pairs:
                    continue
                
                # 检查是否共享连接
                shared = self._count_shared_connections(neuron, other)
                if shared > 0:
                    similarity = self._calculate_similarity(neuron, other)
                    if similarity >= self.similarity_threshold:
                        records.append(AssociationRecord(
                            neuron1_id=str(neuron.event_id),
                            neuron2_id=str(other.event_id),
                            similarity=similarity,
                            connection_type="contextual",
                            timestamp=datetime.now()
                        ))
                        seen_pairs.add(pair)
        
        # 2. 通过时间接近发现关联
        neurons_by_type: Dict[str, List['NeuronCell']] = {}
        for neuron in neurons:
            if neuron.event_type not in neurons_by_type:
                neurons_by_type[neuron.event_type] = []
            neurons_by_type[neuron.event_type].append(neuron)
        
        for event_type, typed_neurons in neurons_by_type.items():
            sorted_neurons = sorted(typed_neurons, key=lambda n: n.create_time)
            
            for i, neuron in enumerate(sorted_neurons):
                for other in sorted_neurons[i+1:i+5]:  # 最多检查后面4个
                    pair = tuple(sorted([str(neuron.event_id), str(other.event_id)]))
                    if pair in seen_pairs:
                        continue
                    
                    time_diff = abs((other.create_time - neuron.create_time).total_seconds() / 60)
                    if time_diff <= self.temporal_window_minutes:
                        similarity = self._calculate_temporal_similarity(neuron, other)
                        if similarity >= self.similarity_threshold:
                            records.append(AssociationRecord(
                                neuron1_id=str(neuron.event_id),
                                neuron2_id=str(other.event_id),
                                similarity=similarity,
                                connection_type="temporal",
                                timestamp=datetime.now()
                            ))
                            seen_pairs.add(pair)
        
        # 按相似度排序，截取最大数量
        records.sort(key=lambda x: x.similarity, reverse=True)
        return records[:self.max_associations]
    
    def find_semantic_bridges(
        self,
        engram1: 'Engram',
        engram2: 'Engram'
    ) -> List[Tuple['NeuronCell', 'NeuronCell']]:
        """
        在两个 engram 之间构建语义桥梁
        
        找出连接两个 engram 的神经元对。
        """
        bridges = []
        
        # 获取两个 engram 的所有神经元
        neurons1 = list(engram1.get_all_neurons())
        neurons2 = list(engram2.get_all_neurons())
        
        # 构建连接索引
        connections1: Dict[str, Set[str]] = {}
        for neuron in neurons1:
            connections1[str(neuron.event_id)] = {
                str(c.target_id) for c in neuron.outgoing_connections
            }
        
        connections2: Dict[str, Set[str]] = {}
        for neuron in neurons2:
            connections2[str(neuron.event_id)] = {
                str(c.target_id) for c in neuron.outgoing_connections
            }
        
        # 找出跨 engram 的连接
        for nid1, targets1 in connections1.items():
            for nid2, targets2 in connections2.items():
                # 检查是否有共同的目标
                shared = targets1 & targets2
                if shared:
                    n1 = engram1.get_neuron_by_id(uuid.UUID(nid1))
                    n2 = engram2.get_neuron_by_id(uuid.UUID(nid2))
                    if n1 and n2:
                        bridges.append((n1, n2))
        
        return bridges
    
    def _count_shared_connections(
        self,
        neuron1: 'NeuronCell',
        neuron2: 'NeuronCell'
    ) -> int:
        """计算两个神经元共享的连接数"""
        targets1 = {c.target_id for c in neuron1.outgoing_connections}
        targets2 = {c.target_id for c in neuron2.outgoing_connections}
        
        incoming1 = {c.target_id for c in neuron1.incoming_connections}
        incoming2 = {c.target_id for c in neuron2.incoming_connections}
        
        return len(targets1 & targets2) + len(incoming1 & incoming2)
    
    def _calculate_similarity(
        self,
        neuron1: 'NeuronCell',
        neuron2: 'NeuronCell'
    ) -> float:
        """
        计算两个神经元之间的相似度
        
        基于：
        1. event_type 是否相同
        2. 共享连接数
        3. 强度差异
        """
        # 类型相同加分
        type_score = 0.3 if neuron1.event_type == neuron2.event_type else 0.0
        
        # 连接相似度
        shared = self._count_shared_connections(neuron1, neuron2)
        max_conn = max(
            len(neuron1.outgoing_connections) + len(neuron1.incoming_connections),
            len(neuron2.outgoing_connections) + len(neuron2.incoming_connections),
            1
        )
        conn_score = 0.4 * (shared / max_conn)
        
        # 强度相似度
        strength_diff = abs(neuron1.strength - neuron2.strength)
        strength_score = 0.3 * (1 - strength_diff)
        
        return min(1.0, type_score + conn_score + strength_score)
    
    def _calculate_temporal_similarity(
        self,
        neuron1: 'NeuronCell',
        neuron2: 'NeuronCell'
    ) -> float:
        """计算时间相似度"""
        time_diff_min = abs((neuron2.create_time - neuron1.create_time).total_seconds() / 60)
        
        # 时间越近相似度越高
        if time_diff_min <= 5:
            return 0.9
        elif time_diff_min <= 15:
            return 0.7
        elif time_diff_min <= 30:
            return 0.5
        else:
            return 0.3
