"""
记忆系统数据模型

集中管理所有 Pydantic 数据模型。
"""

import uuid
import math
from typing import Dict, List, Literal, Optional, Set, Tuple
from datetime import datetime
from pydantic import BaseModel, Field, UUID1

from memory.schemas.config import (
    EloConfig, DecayConfig, ReflectionConfig, StabilityConfig,
    INITIAL_ELO, DECAY_RATE_RANGE
)


# ============== Elo 模型 ==============

class NeuronEloState(BaseModel):
    """神经元的 Elo 状态"""
    event_id: uuid.UUID
    elo: float = INITIAL_ELO
    activation_count: int = 0  # 激活次数
    win_count: int = 0         # 获胜次数
    last_activation: Optional[float] = None  # 上次激活时间戳
    
    def get_k_factor(self, config: EloConfig = None) -> float:
        """根据激活频率动态计算 K-factor"""
        if config is None:
            config = EloConfig()
        
        if self.activation_count >= config.high_activity_threshold:
            return config.high_activity_k
        elif self.activation_count <= config.low_activity_threshold:
            return config.low_activity_k
        else:
            return config.default_k_factor
    
    def get_combat_power(self) -> float:
        """
        计算神经元的"战斗力"
        战斗力 = Elo ^ (1/2) * log(activation_count + 1)
        """
        if self.activation_count == 0:
            return math.sqrt(self.elo)
        return math.sqrt(self.elo) * math.log(self.activation_count + 1)


# ============== Decay 模型 ==============

class NeuronDecayState(BaseModel):
    """神经元的衰减状态"""
    event_id: uuid.UUID
    event_type: str
    strength: float = 1.0  # 神经元强度
    base_decay_rate: float = 0.995  # 基础衰减率
    impact_score: float = 0.5        # 冲击力评分 [0, 1]
    decay_rate: float = 0.995        # 当前实际衰减率
    created_at: datetime = Field(default_factory=datetime.now)
    last_decay_at: datetime = Field(default_factory=datetime.now)
    
    def calculate_adjusted_decay_rate(self, config: DecayConfig = None) -> float:
        """
        计算调整后的衰减率
        公式: adjusted_rate = base_rate + impact_score * factor * (1 - base_rate)
        """
        if config is None:
            config = DecayConfig()
        
        base = self.base_decay_rate
        impact_factor = self.impact_score * config.impact_decay_factor
        adjusted = base + impact_factor * (1 - base)
        
        min_rate, max_rate = DECAY_RATE_RANGE
        return max(min_rate, min(max_rate, adjusted))
    
    def time_elapsed_days(self, reference_time: datetime = None) -> float:
        """计算自上次衰减以来的天数"""
        if reference_time is None:
            reference_time = datetime.now()
        delta = reference_time - self.last_decay_at
        return delta.total_seconds() / (24 * 3600)


# ============== Reflection 模型 ==============

class TriggerCondition(BaseModel):
    """触发条件类型"""
    type: Literal['conflict', 'association', 'scheduled', 'manual']
    description: str
    priority: int = 0  # 优先级，数字越大优先级越高
    metadata: Dict = Field(default_factory=dict)


class ReflectionResult(BaseModel):
    """Reflection 执行结果"""
    event_id: UUID1 = Field(default_factory=uuid.uuid1)
    reflection_type: str  # 'conflict_resolution', 'association_discovery', 'consolidation'
    content: str
    triggered_conditions: List[TriggerCondition] = Field(default_factory=list)
    involved_engrams: List[UUID1] = Field(default_factory=list)
    involved_neurons: List[UUID1] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    impact_score: float = 0.6  # 反思的冲击力
    
    # 执行结果
    conflict_resolved: bool = False
    new_connections_created: int = 0
    strengthened_engrams: List[UUID1] = Field(default_factory=list)


# ============== Stability 模型 ==============

class ActivationResult(BaseModel):
    """激活结果"""
    engram_id: UUID1
    is_activated: bool
    stability_score: float
    member_contributions: Dict[str, float]  # {neuron_id: contribution}
    threshold: float
    below_threshold_neurons: List[str]  # 未激活的神经元 IDs


# ============== Engram 模型 ==============

class EngramMember(BaseModel):
    """Engram 成员信息"""
    neuron_id: UUID1
    event_type: str
    strength: float
    is_representative: bool = False


class EngramSummary(BaseModel):
    """Engram 摘要"""
    uuid: UUID1
    summary: str
    time: datetime
    strength: float
    member_count: int
    is_retrievable: bool
