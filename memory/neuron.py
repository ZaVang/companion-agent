import uuid
import numpy as np
from typing import Literal, Optional, Set, List
from pydantic import BaseModel, Field, UUID1, validator
from datetime import datetime
from zoneinfo import ZoneInfo

from utils.common import DEFAULT_AREA
from utils.schema import DateTime
from memory.event import EventStream


class Connection(BaseModel):
    target_id: UUID1
    create_time: DateTime
    
    @validator('create_time', pre=True, always=True)
    def parse_create_time(cls, v):
        if isinstance(v, str):
            naive_datetime = datetime.strptime(v, '%Y-%m-%d %H:%M:%S')
            return naive_datetime.replace(tzinfo=ZoneInfo(DEFAULT_AREA))
        return v
    
    def __hash__(self):
        return hash((self.target_id, self.create_time))

    def __eq__(self, other):
        if not isinstance(other, Connection):
            return NotImplemented
        return (self.target_id, self.create_time) == (other.target_id, other.create_time)

class NeuronCell(BaseModel):
    event_id: UUID1 = Field(default_factory=uuid.uuid1)
    event_type: Literal['chat', 'perception', 'thought', 'reflection', 'experience']
    create_time: DateTime
    strength: float = 1.0
    
    # === Sprint 1 Phase 2 新增字段 ===
    # 衰减率（由 decay.py 动态计算）
    decay_rate: float = 0.995
    # 冲击力评分 [0, 1]，影响衰减速度
    impact_score: float = 0.5
    # 上次衰减时间，用于批量计算衰减
    last_decay_at: Optional[DateTime] = None
    
    # === Sprint 2 新增字段 ===
    # 激活阈值 [0, 1]，只有超过此阈值才能被想起
    activation_threshold: float = 0.3
    # 是否为稳固记忆（代表神经元）
    is_consolidated: bool = False
    
    # === 原有字段 ===
    actor: str
    audience: Optional[List[str]] = None
    outgoing_connections: Set[Connection] = Field(default_factory=set)
    incoming_connections: Set[Connection] = Field(default_factory=set)
    
    @validator('create_time', pre=True, always=True)
    def parse_create_time(cls, v):
        if isinstance(v, str):
            naive_datetime = datetime.strptime(v, '%Y-%m-%d %H:%M:%S')
            return naive_datetime.replace(tzinfo=ZoneInfo(DEFAULT_AREA))
        return v
    
    def __hash__(self):
        return hash(self.event_id)

    def __eq__(self, other):
        return isinstance(other, NeuronCell) and self.event_id == other.event_id

    def connect_to(self, other: 'NeuronCell'):
        """Create a directed connection from this neuron to another neuron."""
        connection = Connection(target_id=other.event_id,)
        self.outgoing_connections.add(connection)
        other_connection = Connection(target_id=self.event_id)
        other.incoming_connections.add(other_connection)

    def disconnect_from(self, other: 'NeuronCell'):
        """Remove the directed connection from this neuron to another neuron, if connected."""
        self.outgoing_connections = {conn for conn in self.outgoing_connections if conn.target_id != other.event_id}
        other.incoming_connections = {conn for conn in other.incoming_connections if conn.target_id != self.event_id}

    def disconnect_incoming(self, other: 'NeuronCell'):
        """Remove the directed connection from another neuron to this neuron, if connected."""
        self.incoming_connections = {conn for conn in self.incoming_connections if conn.target_id != other.event_id}
        other.outgoing_connections = {conn for conn in other.outgoing_connections if conn.target_id != self.event_id}
    
    # === Sprint 1 Phase 2: 衰减与强度更新方法 ===
    
    def apply_decay(self, reference_time: Optional[datetime] = None) -> float:
        """
        应用衰减，更新 strength
        
        Args:
            reference_time: 参考时间，默认为当前时间
        
        Returns:
            衰减后的新 strength
        """
        if reference_time is None:
            reference_time = datetime.now()
        
        # 计算时间差（天）
        if self.last_decay_at:
            if isinstance(self.last_decay_at, str):
                last_time = datetime.strptime(self.last_decay_at, '%Y-%m-%d %H:%M:%S')
            else:
                last_time = self.last_decay_at
            time_days = (reference_time - last_time).total_seconds() / (24 * 3600)
        else:
            time_days = 0.0
        
        # 应用衰减公式: strength = strength * decay_rate^time_days
        self.strength *= (self.decay_rate ** time_days)
        
        # 更新上次衰减时间
        self.last_decay_at = reference_time
        
        return self.strength
    
    def boost_strength(self, factor: float = 1.1) -> float:
        """
        增强神经元强度（用于检索命中时）
        
        Args:
            factor: 增强因子
        
        Returns:
            增强后的 strength
        """
        # 使用对数增长防止无限膨胀
        self.strength = np.log(np.exp(self.strength) + factor - 1)
        return self.strength
    
    def update_decay_rate(self, decay_rate: float) -> None:
        """更新衰减率"""
        self.decay_rate = max(0.98, min(0.9999, decay_rate))
    
    def update_impact_score(self, impact_score: float) -> None:
        """更新冲击力评分"""
        self.impact_score = max(0.0, min(1.0, impact_score))
    
    def is_retrievable(self, cue_strength: float = 1.0) -> bool:
        """
        判断神经元是否可以被想起（可检索）
        
        可检索条件: effective_strength >= activation_threshold
        
        effective_strength = strength * cue_strength * consolidation_bonus
        
        Args:
            cue_strength: 线索强度 [0, 1]，外部提供的激活线索强度
        
        Returns:
            是否可检索
        """
        # 稳固记忆有加成
        consolidation_bonus = 1.5 if self.is_consolidated else 1.0
        
        # 有效强度 = 强度 * 线索强度 * 稳固加成
        effective_strength = self.strength * cue_strength * consolidation_bonus
        
        return effective_strength >= self.activation_threshold
    
    def get_retrievability_score(self, cue_strength: float = 1.0) -> float:
        """
        获取可检索性评分
        
        Args:
            cue_strength: 线索强度
        
        Returns:
            retrievability = min(1.0, effective_strength / threshold)
        """
        consolidation_bonus = 1.5 if self.is_consolidated else 1.0
        effective_strength = self.strength * cue_strength * consolidation_bonus
        
        if self.activation_threshold <= 0:
            return 1.0
        
        return min(1.0, effective_strength / self.activation_threshold)
    
    def mark_consolidated(self):
        """标记为稳固记忆"""
        self.is_consolidated = True
        # 稳固记忆阈值可以适当降低
        self.activation_threshold = max(0.1, self.activation_threshold * 0.8)


def calculate_connection_strength(neuron1: NeuronCell, 
                                  neuron2: NeuronCell, 
                                  eventstream: EventStream,
                                  similarity: float, 
                                  average_method: str = 'harmonic',
                                  apply_time_decay: bool = False,
                                  decay_rate: float = 0.995) -> float:
    """
    Calculate the connection strength between two neurons.

    Args:
        neuron1, neuron2: NeuronCell objects to calculate the connection strength between.
        eventstream: EventStream object to retrieve event details.
        average_method: Method for averaging strengths ('harmonic', 'arithmetic', 'geometric').
        apply_time_decay: Whether to apply time decay based on event creation time difference.
        decay_rate: Decay rate per day.

    Returns:
        Connection strength as a float.

    Note:
        - Harmonic mean emphasizes smaller values.
        - Arithmetic mean is the standard average.
        - Geometric mean is less sensitive to extremely high values.
    """
    # Retrieve events
    event1 = eventstream.get_event(neuron1.event_id)
    event2 = eventstream.get_event(neuron2.event_id)

    # Calculate average strength
    if average_method == 'harmonic':
        average_strength = 2 / (1/neuron1.strength + 1/neuron2.strength)
    elif average_method == 'arithmetic':
        average_strength = (neuron1.strength + neuron2.strength) / 2
    elif average_method == 'geometric':
        average_strength = np.sqrt(neuron1.strength * neuron2.strength)
    else:
        raise ValueError("Invalid average method. Choose 'harmonic', 'arithmetic', or 'geometric'.")

    # Calculate connection strength
    connection_strength = similarity * average_strength

    # Apply time decay if enabled
    if apply_time_decay and event1.create_time and event2.create_time:
        time_diff = abs(datetime.fromisoformat(event1.create_time) - datetime.fromisoformat(event2.create_time)).days
        decay_factor = decay_rate ** time_diff
        connection_strength *= decay_factor

    return connection_strength