"""
Elo 竞争机制模块

实现神经元的 Elo 评分系统和竞争逻辑，模拟神经信号的竞争和适应。
"""

from memory.elo.core import (
    expected_win_probability,
    calculate_combat_score,
    adjust_elo_ratings,
    EloCompetition,
)

# 为了向后兼容，导出 schema 中的内容
from memory.schemas import (
    EloConfig,
    NeuronEloState,
    INITIAL_ELO, MIN_ELO, MAX_ELO,
    DEFAULT_K_FACTOR, HIGH_ACTIVITY_K, LOW_ACTIVITY_K,
    HIGH_ACTIVITY_THRESHOLD, LOW_ACTIVITY_THRESHOLD,
)

__all__ = [
    # 核心算法
    'expected_win_probability',
    'calculate_combat_score',
    'adjust_elo_ratings',
    'EloCompetition',
    
    # 配置和模型
    'EloConfig',
    'NeuronEloState',
    
    # 配置常量
    'INITIAL_ELO', 'MIN_ELO', 'MAX_ELO',
    'DEFAULT_K_FACTOR', 'HIGH_ACTIVITY_K', 'LOW_ACTIVITY_K',
    'HIGH_ACTIVITY_THRESHOLD', 'LOW_ACTIVITY_THRESHOLD',
]
