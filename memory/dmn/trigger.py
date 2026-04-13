"""
DMN 触发器

负责判断是否应触发 DMN 模式。
"""

from typing import Optional, List, Tuple
from datetime import datetime, timedelta
import time

from memory.dmn.models import DMNConfig, DMNTriggerCondition, DMNState


class DMNTrigger:
    """
    DMN 触发器
    
    判断是否应触发 DMN 模式，包括：
    - 空闲触发
    - 睡眠触发
    - 定时触发
    """
    
    def __init__(self, config: Optional[DMNConfig] = None):
        self.config = config or DMNConfig()
        self._idle_start: Optional[datetime] = None
        self._last_run: Optional[datetime] = None
        self._current_state = DMNState.IDLE
        self._activity_history: List[Tuple[datetime, float]] = []  # (timestamp, activity)
    
    def mark_active(self, current_time: Optional[datetime] = None) -> None:
        """
        标记系统为活跃状态
        
        重置空闲计时器。
        """
        now = current_time or datetime.now()
        self._idle_start = None
        self._activity_history.append((now, 1.0))
        self._trim_history(now)
    
    def mark_idle(self, current_time: Optional[datetime] = None) -> None:
        """
        标记系统为空闲状态
        
        开始空闲计时。
        """
        now = current_time or datetime.now()
        if self._idle_start is None:
            self._idle_start = now
        self._activity_history.append((now, 0.0))
        self._trim_history(now)
    
    def set_activity_level(self, level: float, current_time: Optional[datetime] = None) -> None:
        """
        设置当前活跃度等级
        
        Args:
            level: 活跃度 [0.0, 1.0]
        """
        now = current_time or datetime.now()
        self._activity_history.append((now, max(0.0, min(1.0, level))))
        self._trim_history(now)
    
    def _trim_history(self, current_time: datetime, max_age_hours: int = 24) -> None:
        """清理过期的历史记录"""
        cutoff = current_time - timedelta(hours=max_age_hours)
        self._activity_history = [
            (ts, lvl) for ts, lvl in self._activity_history
            if ts > cutoff
        ]
    
    def get_current_activity_level(self) -> float:
        """
        获取当前活跃度等级
        
        基于最近的活动历史计算。
        """
        if not self._activity_history:
            return 1.0
        
        # 最近 5 分钟的活动
        recent = [
            lvl for ts, lvl in self._activity_history[-10:]
        ]
        return sum(recent) / len(recent) if recent else 0.0
    
    def get_idle_duration_minutes(self, current_time: Optional[datetime] = None) -> int:
        """
        获取当前空闲持续时间（分钟）
        """
        if self._idle_start is None:
            return 0
        now = current_time or datetime.now()
        delta = now - self._idle_start
        return int(delta.total_seconds() / 60)
    
    def should_trigger(
        self,
        current_time: Optional[datetime] = None,
        force: bool = False
    ) -> Tuple[bool, Optional[DMNTriggerCondition]]:
        """
        判断是否应触发 DMN
        
        Args:
            current_time: 当前时间
            force: 强制触发
        
        Returns:
            (should_trigger, trigger_condition)
        """
        now = current_time or datetime.now()
        
        # 强制触发
        if force:
            return True, DMNTriggerCondition(
                trigger_type="manual",
                current_time=now,
                activity_level=self.get_current_activity_level()
            )
        
        # 检查最小间隔
        if self._last_run is not None:
            elapsed = (now - self._last_run).total_seconds() / 3600
            if elapsed < self.config.min_interval_hours:
                return False, None
        
        # 检查空闲触发
        idle_trigger = self._check_idle_trigger(now)
        if idle_trigger:
            return True, idle_trigger
        
        # 检查睡眠触发
        sleep_trigger = self._check_sleep_trigger(now)
        if sleep_trigger:
            return True, sleep_trigger
        
        # 检查定时触发
        if self.config.enable_scheduled:
            scheduled_trigger = self._check_scheduled_trigger(now)
            if scheduled_trigger:
                return True, scheduled_trigger
        
        return False, None
    
    def _check_idle_trigger(self, now: datetime) -> Optional[DMNTriggerCondition]:
        """检查空闲触发条件"""
        if self._idle_start is None:
            return None
        
        idle_min = self.get_idle_duration_minutes(now)
        if idle_min >= self.config.idle_duration_min:
            return DMNTriggerCondition(
                trigger_type="idle",
                idle_duration_min=idle_min,
                activity_level=self.get_current_activity_level(),
                current_time=now
            )
        return None
    
    def _check_sleep_trigger(self, now: datetime) -> Optional[DMNTriggerCondition]:
        """检查睡眠触发条件"""
        if not self.config.sleep_mode:
            return None
        
        activity = self.get_current_activity_level()
        if activity <= self.config.low_activity_threshold:
            return DMNTriggerCondition(
                trigger_type="sleep",
                activity_level=activity,
                current_time=now
            )
        return None
    
    def _check_scheduled_trigger(self, now: datetime) -> Optional[DMNTriggerCondition]:
        """检查定时触发条件"""
        if not self.config.scheduled_times:
            return None
        
        current_time_str = now.strftime("%H:%M")
        if current_time_str in self.config.scheduled_times:
            return DMNTriggerCondition(
                trigger_type="scheduled",
                activity_level=self.get_current_activity_level(),
                current_time=now
            )
        return None
    
    def mark_dmn_started(self, current_time: Optional[datetime] = None) -> None:
        """标记 DMN 开始运行"""
        self._last_run = current_time or datetime.now()
        self._current_state = DMNState.RUNNING
    
    def mark_dmn_finished(self) -> None:
        """标记 DMN 运行完成"""
        self._current_state = DMNState.IDLE
    
    @property
    def last_run_time(self) -> Optional[datetime]:
        return self._last_run
    
    @property
    def state(self) -> DMNState:
        return self._current_state
