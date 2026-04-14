"""
Sprint 8: 神经元动态增删模块

神经元动态管理：
- death: 神经元死亡机制
- birth: 神经元新生机制
"""

from typing import Dict, List, Optional, Set, Tuple, Any

from memory.dynamics.death import (
    NeuronDeath,
    DeathCriteria,
    NeuronDeathManager,
    DeathRecord,
    DeathReason,
    should_neuron_die,
)
from memory.dynamics.birth import (
    NeuronBirth,
    BirthCriteria,
    NeuronBirthManager,
    BirthRecord,
    BirthReason,
    should_create_new_neuron,
    split_high_intensity_memory,
    abstract_concept,
)


class NeuronDynamics:
    """
    神经元动态管理器
    
    整合死亡和新生机制。
    """
    
    def __init__(
        self,
        death_criteria: DeathCriteria = None,
        birth_criteria: BirthCriteria = None
    ):
        self.death_manager = NeuronDeathManager(death_criteria)
        self.birth_manager = NeuronBirthManager(birth_criteria)
    
    def evaluate_dynamics(self, neuron_data: Dict) -> Dict:
        """
        评估神经元动态
        
        Args:
            neuron_data: {
                'id': str,
                'elo': float,
                'strength': float,
                'last_activation': datetime,
                'connections_count': int
            }
        
        Returns:
            {
                'should_die': bool,
                'death_reason': DeathReason,
                'should_birth': bool,
                'birth_reason': BirthReason
            }
        """
        # 检查死亡
        should_die, death_reason, _ = self.death_manager.evaluate_neuron(
            neuron_id=neuron_data['id'],
            elo=neuron_data.get('elo'),
            strength=neuron_data.get('strength'),
            last_activation=neuron_data.get('last_activation'),
            connections_count=neuron_data.get('connections_count', 0)
        )
        
        # 检查新生
        should_birth, birth_record = self.birth_manager.evaluate_birth(
            neuron_id=neuron_data['id'],
            strength=neuron_data.get('strength', 0),
            related_neurons=neuron_data.get('related_neurons', []),
            content=neuron_data.get('content', '')
        )
        
        return {
            'should_die': should_die,
            'death_reason': death_reason,
            'should_birth': should_birth,
            'birth_reason': birth_record.reason if birth_record else None
        }
    
    def get_statistics(self) -> Dict:
        """获取动态统计"""
        return {
            'death': self.death_manager.get_statistics(),
            'birth': self.birth_manager.get_statistics()
        }


__all__ = [
    # Death
    'NeuronDeath',
    'DeathCriteria',
    'NeuronDeathManager',
    'DeathRecord',
    'DeathReason',
    'should_neuron_die',
    # Birth
    'NeuronBirth',
    'BirthCriteria',
    'NeuronBirthManager',
    'BirthRecord',
    'BirthReason',
    'should_create_new_neuron',
    'split_high_intensity_memory',
    'abstract_concept',
    # Combined
    'NeuronDynamics',
]
