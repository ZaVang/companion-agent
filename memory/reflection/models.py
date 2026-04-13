"""
Reflection 辅助函数

提供 Reflection 相关的工具函数和工厂方法。
"""

import uuid
from datetime import datetime
from typing import Optional, TYPE_CHECKING

from memory.neuron import NeuronCell
from memory.schemas import ReflectionResult
from memory.decay import calculate_decay_rate

if TYPE_CHECKING:
    pass


def create_reflection_neuron(
    result: ReflectionResult,
    actor: str = 'system'
) -> NeuronCell:
    """
    从 ReflectionResult 创建神经元
    
    Args:
        result: Reflection 执行结果
        actor: 执行 reflection 的角色
    
    Returns:
        NeuronCell (event_type='reflection')
    """
    from utils.common import DEFAULT_AREA
    from utils.schema import DateTime
    
    decay_rate = calculate_decay_rate('reflection', result.impact_score)
    
    return NeuronCell(
        event_id=result.event_id,
        event_type='reflection',
        create_time=result.created_at,
        strength=result.impact_score,
        decay_rate=decay_rate,
        impact_score=result.impact_score,
        actor=actor
    )


def run_reflection_cycle(
    engrams: list,
    recent_neurons: list,
    memory_manager: Optional[object] = None,
    similarity_fn: Optional[callable] = None,
    config: Optional = None
):
    """
    运行一个完整的 reflection 周期
    
    Returns:
        (triggers, result)
    """
    from memory.reflection.trigger import ReflectionTrigger
    from memory.reflection.executor import ReflectionExecutor
    from memory.schemas import ReflectionConfig
    
    if config is None:
        config = ReflectionConfig()
    
    trigger = ReflectionTrigger(config)
    executor = ReflectionExecutor(config)
    
    triggers = trigger.check_triggers(engrams, recent_neurons, similarity_fn)
    result = executor.execute(triggers, engrams, memory_manager)
    
    trigger.mark_reflection_completed()
    
    return triggers, result
