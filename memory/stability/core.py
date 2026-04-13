"""
Stability 核心算法

实现集体稳定性机制和激活阈值系统。
"""

import math
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from datetime import datetime
from pydantic import UUID1

from memory.schemas import (
    StabilityConfig, ActivationResult,
    DEFAULT_ACTIVATION_THRESHOLD, AGGREGATION_METHODS,
    REPRESENTATIVE_STABILITY_BOOST, STABILITY_EXPONENT
)

if TYPE_CHECKING:
    from memory.neuron import NeuronCell
    from memory.engram import Engram


# ============== 核心算法 ==============

def aggregate_strengths(
    strengths: List[float],
    method: str = 'arithmetic'
) -> float:
    """
    聚合多个神经元强度
    """
    if not strengths:
        return 0.0
    
    if len(strengths) == 1:
        return strengths[0]
    
    if method == 'arithmetic':
        return sum(strengths) / len(strengths)
    
    elif method == 'harmonic':
        return len(strengths) / sum(1 / s for s in strengths if s > 0)
    
    elif method == 'geometric':
        return math.prod(max(s, 0.01) for s in strengths) ** (1 / len(strengths))
    
    elif method == 'max':
        return max(strengths)
    
    elif method == 'min':
        return min(strengths)
    
    else:
        return sum(strengths) / len(strengths)


def calculate_neuron_stability(
    neuron: 'NeuronCell',
    config: Optional[StabilityConfig] = None
) -> float:
    """
    计算单个神经元的稳定性
    
    公式: stability = strength^exponent * decay_rate^days
    """
    if config is None:
        config = StabilityConfig()
    
    base_stability = neuron.strength ** config.stability_exponent
    
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
    engram: 'Engram',
    member_weights: Optional[Dict[str, float]] = None,
    config: Optional[StabilityConfig] = None
) -> Tuple[float, Dict[str, float]]:
    """
    计算记忆（Engram）的稳定性
    
    公式: stability = aggregate(contributions)
    其中 contribution = stability_i * weight_i
    """
    if config is None:
        config = StabilityConfig()
    
    neurons = list(engram.get_all_neurons())
    
    if not neurons:
        return 0.0, {}
    
    contributions = {}
    for neuron in neurons:
        stability = calculate_neuron_stability(neuron, config)
        
        weight = member_weights.get(str(neuron.event_id), 1.0) if member_weights else 1.0
        
        if neuron.event_id == engram.represent:
            weight *= config.representative_boost
        
        contributions[str(neuron.event_id)] = stability * weight
    
    stability_score = aggregate_strengths(
        list(contributions.values()),
        method=config.aggregation_method
    )
    
    return stability_score, contributions


def check_activation_threshold(
    engram: 'Engram',
    threshold: float = DEFAULT_ACTIVATION_THRESHOLD,
    config: Optional[StabilityConfig] = None
) -> ActivationResult:
    """
    检查记忆是否达到激活阈值
    """
    if config is None:
        config = StabilityConfig()
    
    stability_score, contributions = calculate_engram_stability(engram, config=config)
    
    is_activated = stability_score >= threshold
    
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
    engram: 'Engram',
    target_stability: float = 0.5,
    config: Optional[StabilityConfig] = None
) -> List[Tuple[str, float]]:
    """
    建议需要增强的神经元
    
    找出稳定性最低的神经元，建议增强它们。
    """
    if config is None:
        config = StabilityConfig()
    
    _, contributions = calculate_engram_stability(engram, config=config)
    
    if not contributions:
        return []
    
    avg_contribution = sum(contributions.values()) / len(contributions)
    
    below_avg = [
        (neuron_id, contrib)
        for neuron_id, contrib in contributions.items()
        if contrib < avg_contribution
    ]
    
    below_avg.sort(key=lambda x: x[1])
    
    suggestions = []
    for neuron_id, contrib in below_avg:
        boost = max(0.1, (target_stability - contrib) / contrib)
        suggestions.append((neuron_id, min(boost, 2.0)))
    
    return suggestions
