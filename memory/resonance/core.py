"""
共振机制核心模块
"""

from typing import List, Dict, Set, Optional
from pydantic import BaseModel


class ResonanceConfig(BaseModel):
    """共振配置"""
    resonance_threshold: float = 0.3      # 共振阈值
    decay_rate: float = 0.8             # 激活衰减率
    max_depth: int = 5                  # 最大扩散深度
    energy_base: float = 1.0             # 基础能量
    connection_strength_boost: float = 0.2  # 连接强度增强


class ResonanceEngine:
    """共振引擎"""
    
    def __init__(self, config: Optional[ResonanceConfig] = None):
        self.config = config or ResonanceConfig()
    
    def calculate_activation_energy(
        self,
        strength: float,
        elo: float,
        connections_count: int
    ) -> float:
        """
        计算激活能量
        
        能量 = 基础能量 * (强度因子 + Elo因子 + 连接因子)
        """
        strength_factor = strength * 0.4
        elo_factor = min(elo / 2000.0, 1.0) * 0.3
        connection_factor = min(connections_count / 10.0, 1.0) * 0.3
        
        return self.config.energy_base * (strength_factor + elo_factor + connection_factor)
    
    def check_resonance(
        self,
        energy1: float,
        energy2: float
    ) -> bool:
        """检查是否产生共振"""
        avg_energy = (energy1 + energy2) / 2
        return avg_energy >= self.config.resonance_threshold
    
    def calculate_resonance_strength(
        self,
        energy1: float,
        energy2: float
    ) -> float:
        """计算共振强度"""
        if not self.check_resonance(energy1, energy2):
            return 0.0
        
        return (energy1 + energy2) / 2


class ActivationDiffuser:
    """激活扩散器"""
    
    def __init__(
        self,
        resonance_config: Optional[ResonanceConfig] = None,
        neuron_connections: Optional[Dict[str, Set[str]]] = None
    ):
        self.resonance = ResonanceEngine(resonance_config)
        # neuron_id -> Set[connected_neuron_ids]
        self._connections = neuron_connections or {}
    
    def add_connection(self, from_id: str, to_id: str):
        """添加连接"""
        if from_id not in self._connections:
            self._connections[from_id] = set()
        self._connections[from_id].add(to_id)
    
    def diffuse(
        self,
        seed_neurons: List[str],
        neuron_states: Optional[Dict[str, dict]] = None,
        depth: Optional[int] = None
    ) -> Set[str]:
        """
        从种子神经元扩散激活
        
        Args:
            seed_neurons: 种子神经元 ID 列表
            neuron_states: 神经元状态 {id: {strength, elo, connections_count}}
            depth: 最大扩散深度
        
        Returns:
            被激活的神经元集合
        """
        max_depth = depth or self.resonance.config.max_depth
        activated = set(seed_neurons)
        
        current_level = set(seed_neurons)
        
        for d in range(max_depth):
            next_level = set()
            
            for neuron_id in current_level:
                if neuron_id not in self._connections:
                    continue
                
                # 获取当前神经元能量
                current_energy = self._get_neuron_energy(
                    neuron_id, neuron_states
                )
                
                for connected_id in self._connections[neuron_id]:
                    if connected_id in activated:
                        continue
                    
                    connected_energy = self._get_neuron_energy(
                        connected_id, neuron_states
                    )
                    
                    # 检查共振
                    resonance_strength = self.resonance.calculate_resonance_strength(
                        current_energy, connected_energy
                    )
                    
                    if resonance_strength >= self.resonance.config.resonance_threshold:
                        next_level.add(connected_id)
            
            activated.update(next_level)
            current_level = next_level
            
            if not current_level:
                break
        
        return activated
    
    def _get_neuron_energy(
        self,
        neuron_id: str,
        neuron_states: Optional[Dict[str, dict]]
    ) -> float:
        """获取神经元能量"""
        if neuron_states and neuron_id in neuron_states:
            state = neuron_states[neuron_id]
            return self.resonance.calculate_activation_energy(
                strength=state.get('strength', 0.5),
                elo=state.get('elo', 1000.0),
                connections_count=len(self._connections.get(neuron_id, set()))
            )
        
        # 默认能量
        return 0.5 * self.resonance.config.energy_base
    
    def find_resonance_pairs(
        self,
        neuron_ids: List[str],
        neuron_states: Optional[Dict[str, dict]] = None
    ) -> List[tuple]:
        """找出所有共振对"""
        pairs = []
        
        for i, id1 in enumerate(neuron_ids):
            for id2 in neuron_ids[i+1:]:
                e1 = self._get_neuron_energy(id1, neuron_states)
                e2 = self._get_neuron_energy(id2, neuron_states)
                
                if self.resonance.check_resonance(e1, e2):
                    pairs.append((
                        id1, id2,
                        self.resonance.calculate_resonance_strength(e1, e2)
                    ))
        
        return pairs
