"""
Decay 核心算法

实现记忆的动态衰减机制。
"""

import math
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np

from memory.schemas import (
    DecayConfig, NeuronDecayState,
    DECAY_RATE_RANGE, BASE_DECAY_RATES, IMPACT_DECAY_FACTOR
)
from memory.utils import now as utc_now, from_naive


# ============== 核心算法 ==============

def calculate_decay_rate(
    event_type: str,
    impact_score: float,
    config: Optional[DecayConfig] = None
) -> float:
    """
    计算单个神经元的衰减率
    
    公式: adjusted_rate = base_rate + impact_score * factor * (1 - base_rate)
    """
    if config is None:
        config = DecayConfig()
    
    # 获取基础衰减率
    base_rate = config.base_decay_rates.get(event_type, 0.995)
    
    # 计算调整
    impact_factor = impact_score * config.impact_decay_factor
    adjusted = base_rate + impact_factor * (1 - base_rate)
    
    # 限制范围
    min_rate, max_rate = config.decay_rate_range
    return max(min_rate, min(max_rate, adjusted))


def apply_decay(
    neuron: NeuronDecayState,
    reference_time: Optional[datetime] = None
) -> float:
    """
    应用衰减，更新 strength
    
    公式: strength = strength * decay_rate^days
    """
    # 统一使用 UTC 时区
    if reference_time is None:
        reference_time = utc_now()
    reference_time = from_naive(reference_time)
    
    # 计算时间差（天）
    if neuron.last_decay_at:
        if isinstance(neuron.last_decay_at, str):
            last_time = datetime.strptime(neuron.last_decay_at, '%Y-%m-%d %H:%M:%S')
        else:
            last_time = neuron.last_decay_at
        # 确保 last_time 是 aware datetime
        last_time = from_naive(last_time)
        time_days = (reference_time - last_time).total_seconds() / (24 * 3600)
    else:
        time_days = 0.0
    
    # 应用衰减
    neuron.strength *= (neuron.decay_rate ** time_days)
    neuron.last_decay_at = reference_time
    
    return neuron.strength


def calculate_multi_event_decay(
    neurons: List[NeuronDecayState],
    reference_time: Optional[datetime] = None
) -> List[float]:
    """
    批量计算多个神经元的衰减
    """
    return [apply_decay(n, reference_time) for n in neurons]


def estimate_decay_curve(
    days: int,
    decay_rate: float,
    initial_strength: float = 1.0
) -> float:
    """
    估计衰减曲线上的某点值
    
    Args:
        days: 天数
        decay_rate: 衰减率
        initial_strength: 初始强度
    
    Returns:
        估计的强度值
    """
    return initial_strength * (decay_rate ** days)


def get_decay_half_life(decay_rate: float) -> float:
    """
    计算衰减半衰期（强度减半所需天数）
    
    公式: 0.5 = decay_rate^days
          days = log(0.5) / log(decay_rate)
    """
    if decay_rate >= 1.0:
        return float('inf')
    return math.log(0.5) / math.log(decay_rate)


# ============== 衰减调度器 ==============

class DecayScheduler:
    """
    衰减调度器
    
    负责批量调度神经元的衰减计算。
    """
    
    def __init__(self, config: Optional[DecayConfig] = None):
        self.config = config or DecayConfig()
        self._last_batch_time: Optional[datetime] = None
    
    def schedule_batch_decay(
        self,
        neurons: List[NeuronDecayState],
        reference_time: Optional[datetime] = None
    ) -> List[float]:
        """
        调度批量衰减
        """
        if reference_time is None:
            reference_time = utc_now()
        
        self._last_batch_time = reference_time
        return calculate_multi_event_decay(neurons, reference_time)
    
    def get_decay_statistics(
        self,
        neurons: List[NeuronDecayState]
    ) -> Dict:
        """获取衰减统计"""
        if not neurons:
            return {'count': 0}
        
        strengths = [n.strength for n in neurons]
        return {
            'count': len(neurons),
            'avg_strength': np.mean(strengths),
            'min_strength': np.min(strengths),
            'max_strength': np.max(strengths),
            'median_strength': np.median(strengths),
        }
    
    @property
    def last_batch_time(self) -> Optional[datetime]:
        return self._last_batch_time
    
    def apply_decay_to_neuron(
        self,
        neuron_id: str,
        current_strength: float,
        reference_time: Optional[datetime] = None
    ) -> float:
        """
        对单个神经元应用衰减
        
        Args:
            neuron_id: 神经元 ID
            current_strength: 当前强度
            reference_time: 参考时间
        
        Returns:
            衰减后的新强度
        """
        if reference_time is None:
            reference_time = utc_now()
        
        # 计算时间因子（使用配置的默认衰减率）
        # 如果需要更精确的计算，可以使用存储的 decay_rate
        base_decay_rate = self.config.base_decay_rates.get('default', 0.995)
        
        # 简化计算：使用默认衰减率
        # 在实际应用中，应该根据神经元类型获取对应的衰减率
        decay_factor = base_decay_rate ** 1.0  # 默认 1 天
        
        return current_strength * decay_factor


# ============== 全局调度器 ==============

_global_scheduler: Optional[DecayScheduler] = None


def get_global_scheduler() -> DecayScheduler:
    """获取全局衰减调度器"""
    global _global_scheduler
    if _global_scheduler is None:
        _global_scheduler = DecayScheduler()
    return _global_scheduler


def reset_global_scheduler():
    """重置全局调度器"""
    global _global_scheduler
    _global_scheduler = None
