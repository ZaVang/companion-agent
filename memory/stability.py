"""
记忆稳定性模块

实现集体稳定性机制和激活阈值系统。

核心概念:
- 记忆可想起程度 = 成员神经元强度聚合值
- 代表神经元比成员神经元更稳定
- 激活阈值：超过阈值才能被想起
"""

import math
from typing import Dict, List, Literal, Optional, Tuple
from datetime import datetime
from pydantic import BaseModel, Field, UUID1
import numpy as np

from memory.neuron import NeuronCell
from memory.engram import Engram


# ============== 配置常量 ==============

# 聚合方法
AGGREGATION_METHODS = ['arithmetic', 'harmonic', 'geometric', 'max', 'min']

# 代表神经元稳定性加成
REPRESENTATIVE_STABILITY_BOOST = 1.5  # 代表神经元稳定性乘数

# 激活阈值配置
DEFAULT_ACTIVATION_THRESHOLD = 0.3  # 默认激活阈值
MIN_ACTIVATION_THRESHOLD = 0.1
MAX_ACTIVATION_THRESHOLD = 0.9

# 稳定性计算参数
STABILITY_EXPONENT = 0.5  # 稳定性指数


# ============== 数据模型 ==============

class StabilityConfig(BaseModel):
    """稳定性配置"""
    default_threshold: float = DEFAULT_ACTIVATION_THRESHOLD
    representative_boost: float = REPRESENTATIVE_STABILITY_BOOST
    stability_exponent: float = STABILITY_EXPONENT
    aggregation_method: str = 'arithmetic'


class ActivationResult(BaseModel):
    """激活结果"""
    engram_id: UUID1
    is_activated: bool
    stability_score: float
    member_contributions: Dict[str, float]  # {neuron_id: contribution}
    threshold: float
    below_threshold_neurons: List[str]  # 未激活的神经元 IDs


# ============== 核心算法 ==============

def aggregate_strengths(
    strengths: List[float],
    method: str = 'arithmetic'
) -> float:
    """
    聚合多个神经元强度
    
    Args:
        strengths: 强度列表
        method: 聚合方法
    
    Returns:
        聚合后的强度值
    """
    if not strengths:
        return 0.0
    
    if len(strengths) == 1:
        return strengths[0]
    
    if method == 'arithmetic':
        return sum(strengths) / len(strengths)
    
    elif method == 'harmonic':
        # 调和平均：强调较小值
        return len(strengths) / sum(1 / s for s in strengths if s > 0)
    
    elif method == 'geometric':
        # 几何平均：折中方案
        return math.prod(max(s, 0.01) for s in strengths) ** (1 / len(strengths))
    
    elif method == 'max':
        return max(strengths)
    
    elif method == 'min':
        return min(strengths)
    
    else:
        return sum(strengths) / len(strengths)


def calculate_neuron_stability(
    neuron: NeuronCell,
    config: Optional[StabilityConfig] = None
) -> float:
    """
    计算单个神经元的稳定性
    
    公式: stability = strength^exponent * decay_rate^days * boost
    
    Args:
        neuron: 神经元
        config: 稳定性配置
    
    Returns:
        稳定性评分 [0, 1]
    """
    if config is None:
        config = StabilityConfig()
    
    # 基础稳定性来自强度
    base_stability = neuron.strength ** config.stability_exponent
    
    # 衰减因子
    if neuron.last_decay_at:
        if isinstance(neuron.last_decay_at, str):
            last_time = datetime.strptime(neuron.last_decay_at, '%Y-%m-%d %H:%M:%S')
        else:
            last_time = neuron.last_decay_at
        days = (datetime.now() - last_time).total_seconds() / (24 * 3600)
        decay_factor = neuron.decay_rate ** max(0, days)
    else:
        decay_factor = 1.0
    
    return base_stability * decay_factor


def calculate_engram_stability(
    engram: Engram,
    member_weights: Optional[Dict[str, float]] = None,
    config: Optional[StabilityConfig] = None
) -> Tuple[float, Dict[str, float]]:
    """
    计算记忆（Engram）的稳定性
    
    公式: stability = aggregate(contributions)
    其中 contribution = stability_i * weight_i
    
    Args:
        engram: 记忆
        member_weights: 成员权重（可选）
        config: 稳定性配置
    
    Returns:
        (stability_score, {neuron_id: contribution})
    """
    if config is None:
        config = StabilityConfig()
    
    # 获取所有神经元
    neurons = list(engram.get_all_neurons())
    
    if not neurons:
        return 0.0, {}
    
    # 计算每个神经元的贡献
    contributions = {}
    for neuron in neurons:
        stability = calculate_neuron_stability(neuron, config)
        
        # 应用权重
        weight = member_weights.get(str(neuron.event_id), 1.0) if member_weights else 1.0
        
        # 检查是否是代表神经元
        if neuron.event_id == engram.represent:
            weight *= config.representative_boost
        
        contributions[str(neuron.event_id)] = stability * weight
    
    # 聚合所有贡献
    stability_score = aggregate_strengths(
        list(contributions.values()),
        method=config.aggregation_method
    )
    
    return stability_score, contributions


def check_activation_threshold(
    engram: Engram,
    threshold: float = DEFAULT_ACTIVATION_THRESHOLD,
    config: Optional[StabilityConfig] = None
) -> ActivationResult:
    """
    检查记忆是否达到激活阈值
    
    Args:
        engram: 记忆
        threshold: 激活阈值
        config: 稳定性配置
    
    Returns:
        ActivationResult
    """
    if config is None:
        config = StabilityConfig()
    
    stability_score, contributions = calculate_engram_stability(engram, config=config)
    
    # 判断是否激活
    is_activated = stability_score >= threshold
    
    # 找出未达到阈值的神经元
    if contributions:
        threshold_per_neuron = threshold / len(contributions)
        below_threshold = [
            neuron_id for neuron_id, contrib in contributions.items()
            if contrib < threshold_per_neuron
        ]
    else:
        below_threshold = []
    
    return ActivationResult(
        engram_id=engram.uuid,
        is_activated=is_activated,
        stability_score=stability_score,
        member_contributions=contributions,
        threshold=threshold,
        below_threshold_neurons=below_threshold
    )


def suggest_neurons_for_reinforcement(
    engram: Engram,
    target_stability: float = 0.5,
    config: Optional[StabilityConfig] = None
) -> List[Tuple[str, float]]:
    """
    建议需要增强的神经元
    
    找出稳定性最低的神经元，建议增强它们。
    
    Args:
        engram: 记忆
        target_stability: 目标稳定性
        config: 稳定性配置
    
    Returns:
        [(neuron_id, recommended_boost), ...] 按增强优先级排序
    """
    if config is None:
        config = StabilityConfig()
    
    _, contributions = calculate_engram_stability(engram, config=config)
    
    # 计算平均贡献
    if not contributions:
        return []
    
    avg_contribution = sum(contributions.values()) / len(contributions)
    
    # 找出低于平均的神经元
    below_avg = [
        (neuron_id, contrib)
        for neuron_id, contrib in contributions.items()
        if contrib < avg_contribution
    ]
    
    # 按贡献升序排列（最需要增强的在前）
    below_avg.sort(key=lambda x: x[1])
    
    # 计算建议的增强量
    suggestions = []
    for neuron_id, contrib in below_avg:
        # 增强量 = 目标 - 当前
        boost = max(0.1, (target_stability - contrib) / contrib)
        suggestions.append((neuron_id, min(boost, 2.0)))  # 限制最大增强量
    
    return suggestions


# ============== 稳定性管理器 ==============

class StabilityManager:
    """
    稳定性管理器
    
    负责:
    1. 管理所有记忆的稳定性
    2. 批量检查激活状态
    3. 触发神经元增强
    """
    
    def __init__(self, config: Optional[StabilityConfig] = None):
        self.config = config or StabilityConfig()
        self._engram_stabilities: Dict[str, float] = {}  # {engram_id: stability}
    
    def register_engram(self, engram: Engram) -> float:
        """注册记忆并计算初始稳定性"""
        stability, _ = calculate_engram_stability(engram, config=self.config)
        self._engram_stabilities[str(engram.uuid)] = stability
        return stability
    
    def get_stability(self, engram_id: str) -> float:
        """获取记忆的稳定性"""
        return self._engram_stabilities.get(engram_id, 0.0)
    
    def update_stability(self, engram: Engram) -> float:
        """更新记忆的稳定性"""
        stability, _ = calculate_engram_stability(engram, config=self.config)
        self._engram_stabilities[str(engram.uuid)] = stability
        return stability
    
    def batch_check_activation(
        self,
        engrams: List[Engram],
        threshold: Optional[float] = None
    ) -> List[ActivationResult]:
        """
        批量检查激活状态
        
        Args:
            engrams: 记忆列表
            threshold: 激活阈值（None 则使用默认）
        
        Returns:
            所有记忆的激活结果
        """
        if threshold is None:
            threshold = self.config.default_threshold
        
        results = []
        for engram in engrams:
            result = check_activation_threshold(
                engram, threshold, self.config
            )
            results.append(result)
        
        # 按稳定性降序排列
        results.sort(key=lambda x: x.stability_score, reverse=True)
        
        return results
    
    def get_activatable_engrams(
        self,
        engrams: List[Engram],
        threshold: Optional[float] = None,
        top_k: Optional[int] = None
    ) -> List[Tuple[Engram, float]]:
        """
        获取可激活的记忆
        
        Args:
            engrams: 记忆列表
            threshold: 激活阈值
            top_k: 返回前 k 个
        
        Returns:
            [(engram, stability_score), ...] 按稳定性降序
        """
        results = self.batch_check_activation(engrams, threshold)
        
        activatable = [
            (engram, result.stability_score)
            for engram, result in zip(engrams, results)
            if result.is_activated
        ]
        
        if top_k is not None:
            return activatable[:top_k]
        
        return activatable
    
    def get_statistics(self) -> Dict:
        """获取稳定性统计"""
        if not self._engram_stabilities:
            return {
                'total_engrams': 0,
                'avg_stability': 0,
            }
        
        stabilities = list(self._engram_stabilities.values())
        return {
            'total_engrams': len(stabilities),
            'avg_stability': sum(stabilities) / len(stabilities),
            'min_stability': min(stabilities),
            'max_stability': max(stabilities),
            'above_threshold': sum(1 for s in stabilities if s >= self.config.default_threshold),
        }


# ============== 全局实例 ==============

_default_manager: Optional[StabilityManager] = None


def get_global_stability_manager() -> StabilityManager:
    """获取全局稳定性管理器"""
    global _default_manager
    if _default_manager is None:
        _default_manager = StabilityManager()
    return _default_manager


def reset_global_stability_manager() -> None:
    """重置全局稳定性管理器"""
    global _default_manager
    if _default_manager:
        StabilityManager.__init__(_default_manager)
