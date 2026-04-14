"""
动态衰减系统模块

实现记忆的动态衰减机制，模拟神经元的自然遗忘过程。

核心概念:
- 不同 event_type 有不同的基础衰减率
- 冲击力(impact_score) 越高，衰减越慢
- 衰减是可叠加的，时间越长影响越大
"""

from memory.decay.core import (
    calculate_decay_rate,
    apply_decay,
    calculate_multi_event_decay,
    estimate_decay_curve,
    get_decay_half_life,
    DecayScheduler,
    get_global_scheduler,
    reset_global_scheduler,
    # Sprint 7 集成
    calculate_emotion_aware_decay,
    apply_decay_with_emotion,
)

# 为了向后兼容
from memory.schemas import (
    DecayConfig, NeuronDecayState,
    DECAY_RATE_RANGE, BASE_DECAY_RATES, IMPACT_DECAY_FACTOR,
)

__all__ = [
    # 核心算法
    'calculate_decay_rate',
    'apply_decay',
    'calculate_multi_event_decay',
    'estimate_decay_curve',
    'get_decay_half_life',
    'DecayScheduler',
    'get_global_scheduler',
    'reset_global_scheduler',
    
    # Sprint 7 集成
    'calculate_emotion_aware_decay',
    'apply_decay_with_emotion',
    
    # 配置和模型
    'DecayConfig',
    'NeuronDecayState',
    
    # 配置常量
    'DECAY_RATE_RANGE', 'BASE_DECAY_RATES', 'IMPACT_DECAY_FACTOR',
]
