"""
记忆稳定性模块

实现集体稳定性机制和激活阈值系统。

核心概念:
- 记忆可想起程度 = 成员神经元强度聚合值
- 代表神经元比成员神经元更稳定
- 激活阈值：超过阈值才能被想起
"""

from memory.stability.core import (
    aggregate_strengths,
    calculate_neuron_stability,
    calculate_engram_stability,
    check_activation_threshold,
    suggest_neurons_for_reinforcement,
)

# 为了向后兼容，导出 schema 中的内容
from memory.schemas import (
    StabilityConfig,
    ActivationResult,
    AGGREGATION_METHODS,
    REPRESENTATIVE_STABILITY_BOOST,
    DEFAULT_ACTIVATION_THRESHOLD,
    MIN_ACTIVATION_THRESHOLD, MAX_ACTIVATION_THRESHOLD,
    STABILITY_EXPONENT,
)

__all__ = [
    # 核心算法
    'aggregate_strengths',
    'calculate_neuron_stability',
    'calculate_engram_stability',
    'check_activation_threshold',
    'suggest_neurons_for_reinforcement',
    
    # 配置和模型
    'StabilityConfig',
    'ActivationResult',
    
    # 配置常量
    'AGGREGATION_METHODS',
    'REPRESENTATIVE_STABILITY_BOOST',
    'DEFAULT_ACTIVATION_THRESHOLD',
    'MIN_ACTIVATION_THRESHOLD', 'MAX_ACTIVATION_THRESHOLD',
    'STABILITY_EXPONENT',
]
