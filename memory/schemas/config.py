"""
记忆系统配置模块

集中管理所有配置类。
"""

from typing import Dict, Literal, Optional, Tuple
from pydantic import BaseModel, Field


# ============== Elo 配置 ==============

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


# ============== Decay 配置 ==============

# 衰减率配置
DECAY_RATE_RANGE: Tuple[float, float] = (0.98, 0.9999)  # 快衰减 ~ 极慢衰减

# event_type 基础衰减率（每天）
BASE_DECAY_RATES: Dict[str, float] = {
    'chat': 0.995,           # 对话衰减较快（日常闲聊）
    'perception': 0.990,     # 感知记忆衰减中等
    'thought': 0.992,        # 思考记忆衰减较慢
    'reflection': 0.998,    # 反思记忆衰减很慢
    'experience': 0.985,    # 体验记忆衰减较快
}

# 冲击力对衰减率的影响系数
IMPACT_DECAY_FACTOR: float = 0.5  # 调整：impact_score * factor * decay_rate


class DecayConfig(BaseModel):
    """衰减系统配置"""
    decay_rate_range: Tuple[float, float] = DECAY_RATE_RANGE
    base_decay_rates: Dict[str, float] = Field(
        default_factory=lambda: BASE_DECAY_RATES.copy()
    )
    impact_decay_factor: float = Field(default=0.5)  # 提高冲击力影响


# ============== Reflection 配置 ==============

# 触发条件阈值
CONFLICT_STRENGTH_DIFF: float = 0.3  # 强度差异超过此值触发冲突检测
CONFLICT_SIMILARITY_MIN: float = 0.6  # 最小相似度才认为是"相似"记忆

# 关联发现阈值
NEW_CONNECTION_THRESHOLD: int = 3  # 新增连接数超过此值触发
ASSOCIATION_COUNT_MIN: int = 2     # 最小关联数

# 定期触发配置
REFLECTION_INTERVAL_HOURS: int = 24  # 默认 24 小时触发一次
MIN_REFLECTION_INTERVAL: int = 6     # 最小间隔（小时）

# Reflection 内容配置
MAX_REFLECTION_LENGTH: int = 500     # 最大反思内容长度
MIN_CONFLICT_PAIRS: int = 2          # 最少冲突对数触发深度反思


class ReflectionConfig(BaseModel):
    """Reflection 系统配置"""
    conflict_strength_diff: float = CONFLICT_STRENGTH_DIFF
    conflict_similarity_min: float = CONFLICT_SIMILARITY_MIN
    new_connection_threshold: int = NEW_CONNECTION_THRESHOLD
    association_count_min: int = ASSOCIATION_COUNT_MIN
    reflection_interval_hours: int = REFLECTION_INTERVAL_HOURS
    min_reflection_interval: int = MIN_REFLECTION_INTERVAL
    max_reflection_length: int = MAX_REFLECTION_LENGTH
    min_conflict_pairs: int = MIN_CONFLICT_PAIRS


# ============== Stability 配置 ==============

# 聚合方法
AGGREGATION_METHODS: list = ['arithmetic', 'harmonic', 'geometric', 'max', 'min']

# 代表神经元稳定性加成
REPRESENTATIVE_STABILITY_BOOST: float = 1.5  # 代表神经元稳定性乘数

# 激活阈值配置
DEFAULT_ACTIVATION_THRESHOLD: float = 0.3  # 默认激活阈值
MIN_ACTIVATION_THRESHOLD: float = 0.1
MAX_ACTIVATION_THRESHOLD: float = 0.9

# 稳定性计算参数
STABILITY_EXPONENT: float = 0.5  # 稳定性指数


class StabilityConfig(BaseModel):
    """稳定性配置"""
    default_threshold: float = DEFAULT_ACTIVATION_THRESHOLD
    representative_boost: float = REPRESENTATIVE_STABILITY_BOOST
    stability_exponent: float = STABILITY_EXPONENT
    aggregation_method: str = 'arithmetic'
