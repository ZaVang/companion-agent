"""
共现频率统计

统计神经元之间的共现频率。
"""

from typing import Dict, List, Tuple, Set, Optional
from pydantic import BaseModel, Field, UUID1
from collections import defaultdict
import uuid


class CoOccurrenceMatrix(BaseModel):
    """
    共现矩阵
    
    统计神经元之间的共现频率。
    """
    # {neuron1_id: {neuron2_id: count}}
    matrix: Dict[str, Dict[str, int]] = Field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    activation_counts: Dict[str, int] = Field(default_factory=lambda: defaultdict(int))
    total_activations: int = 0
    
    def record_activation(self, neuron_id: str) -> None:
        """记录单个神经元激活"""
        self.activation_counts[neuron_id] += 1
        self.total_activations += 1
    
    def record_cooccurrence(self, neuron1_id: str, neuron2_id: str) -> None:
        """记录两个神经元共现"""
        if neuron1_id == neuron2_id:
            return
        
        # 双向记录（无向图）
        self.matrix[neuron1_id][neuron2_id] += 1
        self.matrix[neuron2_id][neuron1_id] += 1
        
        # 更新激活计数
        self.activation_counts[neuron1_id] += 1
        self.activation_counts[neuron2_id] += 1
        self.total_activations += 2
    
    def get_frequency(self, neuron1_id: str, neuron2_id: str) -> float:
        """
        获取两个神经元的共现频率
        
        频率 = 共现次数 / min(各自激活次数)
        """
        cooccurrence = self.matrix.get(neuron1_id, {}).get(neuron2_id, 0)
        count1 = self.activation_counts.get(neuron1_id, 0)
        count2 = self.activation_counts.get(neuron2_id, 0)
        
        if count1 == 0 or count2 == 0:
            return 0.0
        
        min_count = min(count1, count2)
        return cooccurrence / min_count if min_count > 0 else 0.0
    
    def get_joint_probability(self, neuron1_id: str, neuron2_id: str) -> float:
        """
        获取联合概率 P(A ∩ B)
        """
        cooccurrence = self.matrix.get(neuron1_id, {}).get(neuron2_id, 0)
        return cooccurrence / self.total_activations if self.total_activations > 0 else 0.0
    
    def get_conditional_probability(self, neuron1_id: str, neuron2_id: str) -> float:
        """
        获取条件概率 P(A|B) = P(A ∩ B) / P(B)
        """
        cooccurrence = self.matrix.get(neuron1_id, {}).get(neuron2_id, 0)
        count2 = self.activation_counts.get(neuron2_id, 0)
        
        return cooccurrence / count2 if count2 > 0 else 0.0
    
    def get_strongest_correlations(
        self,
        neuron_id: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        获取与指定神经元最强关联的神经元列表
        
        返回: [(neuron_id, frequency), ...]
        """
        if neuron_id not in self.matrix:
            return []
        
        correlations = [
            (neighbor, self.get_frequency(neuron_id, neighbor))
            for neighbor in self.matrix[neuron_id].keys()
            if neighbor != neuron_id
        ]
        
        correlations.sort(key=lambda x: x[1], reverse=True)
        return correlations[:top_k]
    
    def get_all_nodes(self) -> Set[str]:
        """获取所有出现过的神经元"""
        nodes = set(self.activation_counts.keys())
        for neighbors in self.matrix.values():
            nodes.update(neighbors.keys())
        return nodes
    
    def get_degree(self, neuron_id: str) -> int:
        """获取节点的度（连接数）"""
        if neuron_id not in self.matrix:
            return 0
        return len(self.matrix[neuron_id])
    
    def get_normalized_matrix(self) -> Dict[str, Dict[str, float]]:
        """
        获取归一化后的矩阵
        
        每个值除以该行的最大值。
        """
        normalized = defaultdict(lambda: defaultdict(float))
        
        for neuron1 in self.matrix:
            max_val = max(self.matrix[neuron1].values()) if self.matrix[neuron1] else 1
            if max_val > 0:
                for neuron2, count in self.matrix[neuron1].items():
                    normalized[neuron1][neuron2] = count / max_val
        
        return normalized
    
    class Config:
        arbitrary_types_allowed = True


def build_cooccurrence_from_sequence(
    sequence,
    window_size: int = 10
) -> CoOccurrenceMatrix:
    """
    从激活序列构建共现矩阵
    
    Args:
        sequence: ActivationSequence 对象
        window_size: 窗口大小（考虑窗口内的所有共现）
    """
    matrix = CoOccurrenceMatrix()
    
    temporal_order = sequence.get_temporal_order()
    timestamps = [e.timestamp for e in sequence.events]
    
    for i, neuron1 in enumerate(temporal_order):
        matrix.record_activation(str(neuron1))
        
        # 窗口内的所有神经元
        for j in range(max(0, i - window_size + 1), min(len(temporal_order), i + window_size)):
            if i != j:
                neuron2 = temporal_order[j]
                matrix.record_cooccurrence(str(neuron1), str(neuron2))
    
    return matrix
