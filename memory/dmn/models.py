"""
DMN 配置和数据模型
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class DMNState(str, Enum):
    """DMN 运行状态"""
    IDLE = "idle"              # 空闲状态
    MONITORING = "monitoring"  # 监控中
    RUNNING = "running"        # 运行中
    SUSPENDED = "suspended"    # 暂停


class DMNConfig(BaseModel):
    """DMN 系统配置"""
    # 触发条件
    idle_duration_min: int = 30           # 空闲持续时间（分钟）
    sleep_mode: bool = False              # 睡眠模式标志
    low_activity_threshold: float = 0.1    # 低活跃度阈值
    
    # 定时触发
    scheduled_times: List[str] = []       # 定时触发时间 ["02:00", "06:00"]
    enable_scheduled: bool = True          # 启用定时触发
    
    # 固化配置
    consolidation_strength_threshold: float = 0.5  # 固化阈值
    max_consolidation_per_cycle: int = 50         # 每周期最大固化数量
    
    # 清理配置
    prune_strength_threshold: float = 0.1  # 清理强度阈值
    prune_elo_threshold: float = 100.0     # 清理 Elo 阈值
    max_prune_per_cycle: int = 20          # 每周期最大清理数量
    
    # 关联发现配置
    association_similarity_min: float = 0.5  # 最小相似度
    max_associations_per_cycle: int = 10     # 每周期最大关联数
    
    # 运行限制
    min_interval_hours: int = 6             # 最小运行间隔（小时）
    max_runtime_seconds: int = 300          # 最大运行时间（秒）


class DMNTriggerCondition(BaseModel):
    """DMN 触发条件"""
    trigger_type: str                       # "idle" | "sleep" | "scheduled" | "manual"
    idle_duration_min: int = 0              # 空闲持续时间
    activity_level: float = 1.0            # 当前活跃度
    current_time: datetime                 # 触发时间
    
    @property
    def priority(self) -> int:
        """触发优先级"""
        priorities = {"manual": 3, "idle": 2, "scheduled": 2, "sleep": 1}
        return priorities.get(self.trigger_type, 0)


class ConsolidationRecord(BaseModel):
    """固化记录"""
    neuron_id: str
    source_engram: str
    target_engram: Optional[str] = None
    strength_before: float
    strength_after: float
    timestamp: datetime


class PruneRecord(BaseModel):
    """清理记录"""
    neuron_id: str
    reason: str                             # "weak_strength" | "low_elo" | "isolated"
    strength: float
    elo: float
    connections_count: int
    timestamp: datetime


class AssociationRecord(BaseModel):
    """关联记录"""
    neuron1_id: str
    neuron2_id: str
    similarity: float
    connection_type: str                   # "semantic" | "temporal" | "contextual"
    timestamp: datetime


class DMNResult(BaseModel):
    """DMN 运行结果"""
    triggered_conditions: List[DMNTriggerCondition] = Field(default_factory=list)
    state: DMNState = DMNState.IDLE
    
    # 执行统计
    neurons_consolidated: int = 0
    neurons_pruned: int = 0
    associations_found: int = 0
    reflections_generated: int = 0
    
    # 详细记录
    consolidation_records: List[ConsolidationRecord] = Field(default_factory=list)
    prune_records: List[PruneRecord] = Field(default_factory=list)
    association_records: List[AssociationRecord] = Field(default_factory=list)
    
    # 执行信息
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    
    # 错误信息
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        return self.error is None and self.state == DMNState.RUNNING
    
    @property
    def summary(self) -> str:
        return (
            f"DMN Cycle: consolidated {self.neurons_consolidated} neurons, "
            f"pruned {self.neurons_pruned}, found {self.associations_found} associations, "
            f"generated {self.reflections_generated} reflections"
        )
