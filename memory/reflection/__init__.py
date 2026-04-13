"""
Reflection 自动化模块

实现 Reflection 触发条件和执行逻辑，模拟大脑的默认模式网络（DMN）活动。

触发条件:
1. 记忆冲突检测（相似记忆强度差异大）
2. 新知识关联发现（发现新的连接模式）
3. 定期触发（DMN 模式的一部分）
"""

from memory.reflection.trigger import (
    ReflectionTrigger,
    detect_conflict,
    detect_new_associations,
    should_trigger_scheduled,
)

from memory.reflection.executor import ReflectionExecutor

from memory.reflection.models import (
    create_reflection_neuron,
    run_reflection_cycle,
)

# 为了向后兼容，导出 schema 中的内容
from memory.schemas import (
    ReflectionConfig,
    TriggerCondition,
    ReflectionResult,
    CONFLICT_STRENGTH_DIFF, CONFLICT_SIMILARITY_MIN,
    NEW_CONNECTION_THRESHOLD, ASSOCIATION_COUNT_MIN,
    REFLECTION_INTERVAL_HOURS, MIN_REFLECTION_INTERVAL,
    MAX_REFLECTION_LENGTH, MIN_CONFLICT_PAIRS,
)

__all__ = [
    # 触发器
    'ReflectionTrigger',
    'detect_conflict',
    'detect_new_associations',
    'should_trigger_scheduled',
    
    # 执行器
    'ReflectionExecutor',
    
    # 辅助函数
    'create_reflection_neuron',
    'run_reflection_cycle',
    
    # 配置和模型
    'ReflectionConfig',
    'TriggerCondition',
    'ReflectionResult',
    
    # 配置常量
    'CONFLICT_STRENGTH_DIFF', 'CONFLICT_SIMILARITY_MIN',
    'NEW_CONNECTION_THRESHOLD', 'ASSOCIATION_COUNT_MIN',
    'REFLECTION_INTERVAL_HOURS', 'MIN_REFLECTION_INTERVAL',
    'MAX_REFLECTION_LENGTH', 'MIN_CONFLICT_PAIRS',
]
