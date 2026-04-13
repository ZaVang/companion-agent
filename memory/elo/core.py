"""
Elo 核心算法

实现神经元的 Elo 评分系统和竞争逻辑。
"""

import math
from typing import Dict, List, Optional, Tuple, Tuple
import uuid

from memory.schemas import (
    EloConfig, NeuronEloState,
    INITIAL_ELO, MIN_ELO, MAX_ELO
)


# ============== 核心算法 ==============

def expected_win_probability(player_elo: float, opponent_elo: float) -> float:
    """
    计算期望胜率（Elo 公式）
    
    E = 1 / (1 + 10^((R_opponent - R_player) / 400))
    """
    return 1.0 / (1.0 + 10 ** ((opponent_elo - player_elo) / 400))


def calculate_combat_score(
    neuron_elo: float,
    activation_count: int
) -> float:
    """
    计算战斗评分
    
    score = sqrt(elo) * log(activation_count + 1)
    """
    if activation_count == 0:
        return math.sqrt(neuron_elo)
    return math.sqrt(neuron_elo) * math.log(activation_count + 1)


def adjust_elo_ratings(
    winner: NeuronEloState,
    loser: NeuronEloState,
    winner_wins: Optional[bool] = None,
    config: EloConfig = None
) -> Tuple[NeuronEloState, NeuronEloState]:
    """
    调整两个神经元的 Elo 评分
    
    Args:
        winner: 胜者状态
        loser: 败者状态
        winner_wins: None 表示平局
        config: 配置
    
    Returns:
        (new_winner, new_loser)
    """
    if config is None:
        config = EloConfig()
    
    # 深拷贝避免修改原对象
    winner = NeuronEloState(**winner.model_dump())
    loser = NeuronEloState(**loser.model_dump())
    
    if winner_wins is None:
        # 平局：两人向中间靠拢
        diff = winner.elo - loser.elo
        adjustment = abs(diff) * 0.1
        
        if diff > 0:
            winner.elo -= adjustment
            loser.elo += adjustment
        else:
            winner.elo += adjustment
            loser.elo -= adjustment
    else:
        # 计算期望胜率
        expected_winner = expected_win_probability(winner.elo, loser.elo)
        
        # 使用胜者 K-factor
        k_factor = winner.get_k_factor(config)
        
        # 计算调整量
        actual = 1.0 if winner_wins else 0.0
        adjustment = k_factor * (actual - expected_winner)
        
        # 更新 Elo
        winner.elo = min(config.max_elo, winner.elo + adjustment)
        loser.elo = max(config.min_elo, loser.elo - adjustment)
    
    # 更新统计
    winner.activation_count += 1
    loser.activation_count += 1
    
    if winner_wins is not None:
        winner.win_count += 1
    
    return winner, loser


# ============== Elo 竞争管理器 ==============

class EloCompetition:
    """
    Elo 竞争管理器
    
    负责管理所有神经元的 Elo 状态和竞争逻辑。
    """
    
    def __init__(self, config: EloConfig = None):
        self.config = config or EloConfig()
        self._states: Dict[str, NeuronEloState] = {}
    
    def register_neuron(self, neuron_id: str) -> NeuronEloState:
        """注册神经元"""
        if neuron_id not in self._states:
            self._states[neuron_id] = NeuronEloState(
                event_id=uuid.uuid4(),
                elo=self.config.initial_elo
            )
        return self._states[neuron_id]
    
    def get_state(self, neuron_id: str) -> Optional[NeuronEloState]:
        """获取神经元状态"""
        return self._states.get(neuron_id)
    
    def battle(
        self, 
        neuron1_id: str, 
        neuron2_id: str,
        winner_id: Optional[str] = None
    ) -> str:
        """
        执行一次竞争
        
        Args:
            neuron1_id: 神经元1 ID
            neuron2_id: 神经元2 ID
            winner_id: 预设胜者（None 表示根据 Elo 概率决定）
        
        Returns:
            胜者 ID
        """
        # 注册神经元
        self.register_neuron(neuron1_id)
        self.register_neuron(neuron2_id)
        
        state1 = self._states[neuron1_id]
        state2 = self._states[neuron2_id]
        
        # 决定胜者
        if winner_id is None:
            prob = expected_win_probability(state1.elo, state2.elo)
            winner_id = neuron1_id if prob > 0.5 else neuron2_id
        
        # 更新状态
        winner_wins = (winner_id == neuron1_id)
        new1, new2 = adjust_elo_ratings(
            state1, state2, winner_wins=winner_wins, config=self.config
        )
        
        self._states[neuron1_id] = new1
        self._states[neuron2_id] = new2
        
        return winner_id
    
    def get_top_neurons(self, n: int = 10) -> List[Tuple[str, NeuronEloState]]:
        """获取排名前 n 的神经元"""
        sorted_states = sorted(
            self._states.items(),
            key=lambda x: x[1].elo,
            reverse=True
        )
        return sorted_states[:n]
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        if not self._states:
            return {'total': 0}
        
        elos = [s.elo for s in self._states.values()]
        return {
            'total': len(self._states),
            'avg_elo': sum(elos) / len(elos),
            'max_elo': max(elos),
            'min_elo': min(elos),
        }
