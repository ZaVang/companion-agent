"""
Elo 竞争机制模块

实现神经元的 Elo 评分系统和竞争逻辑，模拟神经信号的竞争和适应。

核心概念:
- 神经元通过检索过程中的竞争来调整强度
- 胜者增强，败者减弱（但不完全相同于传统 Elo）
- K-factor 控制调整速度，根据激活频率动态调整
"""

import math
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
import uuid


# ============== 配置常量 ==============

# Elo 基础参数
INITIAL_ELO: float = 1000.0  # 初始 Elo 评分
MIN_ELO: float = 100.0       # 最低 Elo 评分
MAX_ELO: float = 2000.0     # 最高 Elo 评分

# K-factor 配置
DEFAULT_K_FACTOR: float = 32.0    # 默认 K-factor（新手调整幅度）
HIGH_ACTIVITY_K: float = 16.0     # 高频激活神经元（调整幅度减小）
LOW_ACTIVITY_K: float = 64.0     # 低频激活神经元（调整幅度增大）

# K-factor 阈值
HIGH_ACTIVITY_THRESHOLD: int = 50   # 激活次数超过此值视为高频
LOW_ACTIVITY_THRESHOLD: int = 5     # 激活次数低于此值视为低频


# ============== 数据模型 ==============

class EloConfig(BaseModel):
    """Elo 系统配置"""
    initial_elo: float = INITIAL_ELO
    min_elo: float = MIN_ELO
    max_elo: float = MAX_ELO
    default_k_factor: float = DEFAULT_K_FACTOR
    high_activity_k: float = HIGH_ACTIVITY_K
    low_activity_k: float = LOW_ACTIVITY_K
    high_activity_threshold: int = HIGH_ACTIVITY_THRESHOLD
    low_activity_threshold: int = LOW_ACTIVITY_THRESHOLD


class NeuronEloState(BaseModel):
    """神经元的 Elo 状态"""
    event_id: uuid.UUID
    elo: float = INITIAL_ELO
    activation_count: int = 0  # 激活次数
    win_count: int = 0         # 获胜次数
    last_activation: Optional[float] = None  # 上次激活时间戳
    
    def get_k_factor(self, config: EloConfig = EloConfig()) -> float:
        """根据激活频率动态计算 K-factor"""
        if self.activation_count >= config.high_activity_threshold:
            return config.high_activity_k
        elif self.activation_count <= config.low_activity_threshold:
            return config.low_activity_k
        else:
            return config.default_k_factor
    
    def get_combat_power(self) -> float:
        """
        计算神经元的"战斗力"
        
        战斗力 = Elo ^ (1/2) * log(activation_count + 1)
        
        设计原理:
        - Elo 提供基础评分
        - 激活次数提供经验加成（对数增长防止无限膨胀）
        """
        if self.activation_count == 0:
            return math.sqrt(self.elo)
        return math.sqrt(self.elo) * math.log(self.activation_count + 1)


# ============== 核心算法 ==============

def expected_win_probability(player_elo: float, opponent_elo: float) -> float:
    """
    计算期望胜率（Elo 公式）
    
    E = 1 / (1 + 10^((R_opponent - R_player) / 400))
    
    Args:
        player_elo: 玩家 Elo
        opponent_elo: 对手 Elo
    
    Returns:
        期望胜率 [0, 1]
    """
    return 1.0 / (1.0 + 10 ** ((opponent_elo - player_elo) / 400))


def calculate_combat_score(
    neuron_elo: float,
    activation_count: int,
    base_k: float = DEFAULT_K_FACTOR
) -> float:
    """
    计算神经元的综合竞争分数
    
    公式: score = Elo^(0.5) * (1 + 0.1 * min(activation_count, 100))
    
    Args:
        neuron_elo: 神经元 Elo 评分
        activation_count: 激活次数
        base_k: 基础调整系数
    
    Returns:
        综合竞争分数
    """
    # 基础战斗力
    base_power = math.sqrt(neuron_elo)
    
    # 经验加成（上限 10 倍）
    experience_factor = 1 + 0.1 * min(activation_count / 10, 10)
    
    return base_power * experience_factor


class EloCompetitor:
    """
    Elo 竞争系统
    
    负责:
    1. 管理所有神经元的 Elo 状态
    2. 处理竞争更新
    3. 计算战斗力排名
    """
    
    def __init__(self, config: Optional[EloConfig] = None):
        self.config = config or EloConfig()
        self._elo_states: Dict[uuid.UUID, NeuronEloState] = {}
    
    def register_neuron(self, event_id: uuid.UUID) -> NeuronEloState:
        """
        注册新神经元，分配初始 Elo
        
        Args:
            event_id: 神经元事件 ID
        
        Returns:
            新创建的 Elo 状态
        """
        if event_id not in self._elo_states:
            self._elo_states[event_id] = NeuronEloState(
                event_id=event_id,
                elo=self.config.initial_elo
            )
        return self._elo_states[event_id]
    
    def get_elo_state(self, event_id: uuid.UUID) -> Optional[NeuronEloState]:
        """获取神经元的 Elo 状态"""
        return self._elo_states.get(event_id)
    
    def get_combat_power(self, event_id: uuid.UUID) -> float:
        """获取神经元的战斗力"""
        state = self._elo_states.get(event_id)
        if state is None:
            return math.sqrt(self.config.initial_elo)
        return state.get_combat_power()
    
    def update_after_competition(
        self,
        winner_id: uuid.UUID,
        loser_id: uuid.UUID,
        tie: bool = False
    ) -> Tuple[float, float]:
        """
        竞争后更新 Elo
        
        Args:
            winner_id: 胜者 ID
            loser_id: 败者 ID
            tie: 是否平局
        
        Returns:
            (winner_elo_change, loser_elo_change)
        """
        # 确保神经元已注册
        winner_state = self.register_neuron(winner_id)
        loser_state = self.register_neuron(loser_id)
        
        # 获取 K-factor
        winner_k = winner_state.get_k_factor(self.config)
        loser_k = loser_state.get_k_factor(self.config)
        
        # 计算期望胜率
        expected_win = expected_win_probability(winner_state.elo, loser_state.elo)
        
        # 计算 Elo 变化
        if tie:
            winner_change = winner_k * (0.5 - expected_win)
            loser_change = loser_k * (0.5 - (1 - expected_win))
        else:
            winner_change = winner_k * (1 - expected_win)
            loser_change = loser_k * (0 - (1 - expected_win))
        
        # 更新 Elo（带边界限制）
        winner_state.elo = max(
            self.config.min_elo,
            min(self.config.max_elo, winner_state.elo + winner_change)
        )
        loser_state.elo = max(
            self.config.min_elo,
            min(self.config.max_elo, loser_state.elo + loser_change)
        )
        
        # 更新统计
        winner_state.activation_count += 1
        winner_state.win_count += 1
        loser_state.activation_count += 1
        
        return winner_change, loser_change
    
    def update_after_retrieval(
        self,
        retrieved_ids: List[uuid.UUID],
        all_candidate_ids: List[uuid.UUID],
        retrieval_score: float = 1.0
    ) -> Dict[uuid.UUID, float]:
        """
        检索后更新所有候选神经元的 Elo
        
        核心逻辑:
        - 被检索到的神经元之间进行竞争
        - 被检索的神经元 vs 未被检索的神经元
        - 检索得分越高，竞争优势越大
        
        Args:
            retrieved_ids: 被检索到的神经元 IDs
            all_candidate_ids: 所有候选神经元 IDs
            retrieval_score: 检索得分 [0, 1]，影响竞争激烈程度
        
        Returns:
            每个神经元的 Elo 变化
        """
        retrieved_set = set(retrieved_ids)
        not_retrieved_ids = [uid for uid in all_candidate_ids if uid not in retrieved_set]
        
        elo_changes: Dict[uuid.UUID, float] = {}
        
        # 被检索到的神经元之间进行竞争（内部排名）
        if len(retrieved_ids) > 1:
            for i in range(len(retrieved_ids)):
                for j in range(i + 1, len(retrieved_ids)):
                    _, loser_change = self.update_after_competition(
                        retrieved_ids[i], retrieved_ids[j], tie=True
                    )
                    elo_changes[retrieved_ids[j]] = elo_changes.get(retrieved_ids[j], 0) + loser_change
        
        # 被检索的 vs 未被检索的
        if retrieved_ids and not_retrieved_ids:
            # 计算平均竞争优势
            avg_retrieved_power = sum(self.get_combat_power(uid) for uid in retrieved_ids) / len(retrieved_ids)
            avg_not_retrieved_power = sum(self.get_combat_power(uid) for uid in not_retrieved_ids) / len(not_retrieved_ids)
            
            # 竞争强度系数
            intensity = retrieval_score * 0.5  # 最高 0.5
            
            for ret_id in retrieved_ids:
                for non_ret_id in not_retrieved_ids:
                    winner_change, loser_change = self.update_after_competition(
                        ret_id, non_ret_id
                    )
                    # 根据竞争强度调整
                    winner_change *= intensity
                    loser_change *= intensity
                    
                    elo_changes[ret_id] = elo_changes.get(ret_id, 0) + winner_change
                    elo_changes[non_ret_id] = elo_changes.get(non_ret_id, 0) + loser_change
        
        return elo_changes
    
    def get_ranking(self, top_k: Optional[int] = None) -> List[Tuple[uuid.UUID, float]]:
        """
        获取神经元战斗力排名
        
        Args:
            top_k: 返回前 k 名，None 则返回全部
        
        Returns:
            [(event_id, combat_power), ...] 按战斗力降序
        """
        rankings = [
            (event_id, state.get_combat_power())
            for event_id, state in self._elo_states.items()
        ]
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        if top_k is not None:
            return rankings[:top_k]
        return rankings
    
    def merge_from(self, other: 'EloCompetitor') -> None:
        """
        从另一个 EloCompetitor 合并状态
        
        用于分布式训练或状态恢复
        """
        for event_id, state in other._elo_states.items():
            if event_id in self._elo_states:
                # 取较高的 Elo
                self._elo_states[event_id].elo = max(
                    self._elo_states[event_id].elo,
                    state.elo
                )
                self._elo_states[event_id].activation_count = max(
                    self._elo_states[event_id].activation_count,
                    state.activation_count
                )
            else:
                self._elo_states[event_id] = state.model_copy(deep=True)
    
    def reset(self) -> None:
        """重置所有 Elo 状态"""
        self._elo_states.clear()
    
    def get_statistics(self) -> Dict:
        """获取 Elo 系统统计信息"""
        if not self._elo_states:
            return {
                'total_neurons': 0,
                'avg_elo': 0,
                'avg_activation_count': 0,
                'top_elo': None,
            }
        
        elos = [s.elo for s in self._elo_states.values()]
        activations = [s.activation_count for s in self._elo_states.values()]
        
        return {
            'total_neurons': len(self._elo_states),
            'avg_elo': sum(elos) / len(elos),
            'min_elo': min(elos),
            'max_elo': max(elos),
            'avg_activation_count': sum(activations) / len(activations),
            'total_activations': sum(activations),
            'top_elo': max(elos),
        }


# ============== 全局实例 ==============

# 默认配置的全局 Elo 竞争器
_default_competitor: Optional[EloCompetitor] = None


def get_global_competitor() -> EloCompetitor:
    """获取全局 Elo 竞争器实例"""
    global _default_competitor
    if _default_competitor is None:
        _default_competitor = EloCompetitor()
    return _default_competitor


def reset_global_competitor() -> None:
    """重置全局 Elo 竞争器"""
    global _default_competitor
    if _default_competitor:
        _default_competitor.reset()
