"""
激活序列记录

记录神经元的激活顺序和时间。
"""

import uuid
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, UUID1, ConfigDict
from datetime import datetime


class ActivationEvent(BaseModel):
    """单个激活事件"""
    neuron_id: UUID1
    timestamp: datetime
    context: str = ""
    event_type: str = ""
    activation_strength: float = 1.0


class ActivationSequence(BaseModel):
    """
    激活序列记录
    
    记录一段时间内的神经元激活顺序。
    """
    sequence_id: UUID1 = Field(default_factory=uuid.uuid1)
    events: List[ActivationEvent] = Field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def add_activation(
        self,
        neuron_id: UUID1,
        timestamp: datetime,
        context: str = "",
        event_type: str = "",
        activation_strength: float = 1.0
    ) -> None:
        """添加激活记录"""
        event = ActivationEvent(
            neuron_id=neuron_id,
            timestamp=timestamp,
            context=context,
            event_type=event_type,
            activation_strength=activation_strength
        )
        self.events.append(event)
        
        if self.start_time is None or timestamp < self.start_time:
            self.start_time = timestamp
        if self.end_time is None or timestamp > self.end_time:
            self.end_time = timestamp
    
    def get_temporal_order(self) -> List[UUID1]:
        """获取按时间排序的神经元 ID 列表"""
        sorted_events = sorted(self.events, key=lambda e: e.timestamp)
        return [e.neuron_id for e in sorted_events]
    
    def get_neuron_timestamps(self, neuron_id: UUID1) -> List[datetime]:
        """获取特定神经元的所有激活时间"""
        return [e.timestamp for e in self.events if e.neuron_id == neuron_id]
    
    def get_duration_ms(self) -> int:
        """获取序列持续时间（毫秒）"""
        if self.start_time is None or self.end_time is None:
            return 0
        return int((self.end_time - self.start_time).total_seconds() * 1000)
    
    def filter_by_time_range(
        self,
        start: datetime,
        end: datetime
    ) -> 'ActivationSequence':
        """按时间范围过滤"""
        filtered_events = [
            e for e in self.events
            if start <= e.timestamp <= end
        ]
        new_seq = ActivationSequence(
            sequence_id=uuid.uuid1(),
            events=filtered_events,
            metadata=self.metadata.copy()
        )
        if filtered_events:
            new_seq.start_time = min(e.timestamp for e in filtered_events)
            new_seq.end_time = max(e.timestamp for e in filtered_events)
        return new_seq
    
    def get_contexts(self) -> List[str]:
        """获取所有上下文"""
        return [e.context for e in self.events if e.context]
    
    def __len__(self) -> int:
        return len(self.events)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class SequenceDatabase(BaseModel):
    """
    序列数据库
    
    管理多个激活序列。
    """
    sequences: Dict[UUID1, ActivationSequence] = Field(default_factory=dict)
    
    def add_sequence(self, sequence: ActivationSequence) -> UUID1:
        """添加序列"""
        self.sequences[sequence.sequence_id] = sequence
        return sequence.sequence_id
    
    def get_sequence(self, sequence_id: UUID1) -> Optional[ActivationSequence]:
        """获取序列"""
        return self.sequences.get(sequence_id)
    
    def get_all_neurons(self) -> set:
        """获取所有出现过的神经元"""
        neurons = set()
        for seq in self.sequences.values():
            neurons.update(seq.get_temporal_order())
        return neurons
    
    def get_sequences_for_neuron(self, neuron_id: UUID1) -> List[ActivationSequence]:
        """获取包含特定神经元的所有序列"""
        return [
            seq for seq in self.sequences.values()
            if neuron_id in seq.get_temporal_order()
        ]
