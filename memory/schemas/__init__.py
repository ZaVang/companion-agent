"""
记忆系统 Schema 模块

导出所有配置类和数据模型。
"""

from memory.schemas.config import (
    # Elo 配置
    EloConfig,
    INITIAL_ELO, MIN_ELO, MAX_ELO,
    DEFAULT_K_FACTOR, HIGH_ACTIVITY_K, LOW_ACTIVITY_K,
    HIGH_ACTIVITY_THRESHOLD, LOW_ACTIVITY_THRESHOLD,
    
    # Decay 配置
    DecayConfig,
    DECAY_RATE_RANGE, BASE_DECAY_RATES, IMPACT_DECAY_FACTOR,
    
    # Reflection 配置
    ReflectionConfig,
    CONFLICT_STRENGTH_DIFF, CONFLICT_SIMILARITY_MIN,
    NEW_CONNECTION_THRESHOLD, ASSOCIATION_COUNT_MIN,
    REFLECTION_INTERVAL_HOURS, MIN_REFLECTION_INTERVAL,
    MAX_REFLECTION_LENGTH, MIN_CONFLICT_PAIRS,
    
    # Stability 配置
    StabilityConfig,
    AGGREGATION_METHODS,
    REPRESENTATIVE_STABILITY_BOOST,
    DEFAULT_ACTIVATION_THRESHOLD,
    MIN_ACTIVATION_THRESHOLD, MAX_ACTIVATION_THRESHOLD,
    STABILITY_EXPONENT,
)

from memory.schemas.models import (
    # Elo 模型
    NeuronEloState,
    
    # Decay 模型
    NeuronDecayState,
    
    # Reflection 模型
    TriggerCondition, ReflectionResult,
    
    # Stability 模型
    ActivationResult,
    
    # Engram 模型
    EngramMember, EngramSummary,
)

__all__ = [
    # 配置类
    'EloConfig', 'DecayConfig', 'ReflectionConfig', 'StabilityConfig',
    
    # 配置常量
    'INITIAL_ELO', 'MIN_ELO', 'MAX_ELO',
    'DEFAULT_K_FACTOR', 'HIGH_ACTIVITY_K', 'LOW_ACTIVITY_K',
    'HIGH_ACTIVITY_THRESHOLD', 'LOW_ACTIVITY_THRESHOLD',
    'DECAY_RATE_RANGE', 'BASE_DECAY_RATES', 'IMPACT_DECAY_FACTOR',
    'CONFLICT_STRENGTH_DIFF', 'CONFLICT_SIMILARITY_MIN',
    'NEW_CONNECTION_THRESHOLD', 'ASSOCIATION_COUNT_MIN',
    'REFLECTION_INTERVAL_HOURS', 'MIN_REFLECTION_INTERVAL',
    'MAX_REFLECTION_LENGTH', 'MIN_CONFLICT_PAIRS',
    'AGGREGATION_METHODS', 'REPRESENTATIVE_STABILITY_BOOST',
    'DEFAULT_ACTIVATION_THRESHOLD', 'MIN_ACTIVATION_THRESHOLD',
    'MAX_ACTIVATION_THRESHOLD', 'STABILITY_EXPONENT',
    
    # 数据模型
    'NeuronEloState', 'NeuronDecayState',
    'TriggerCondition', 'ReflectionResult',
    'ActivationResult', 'EngramMember', 'EngramSummary',
]
