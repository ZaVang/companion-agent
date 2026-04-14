"""
Default Mode Network (DMN) 核心模块

整合触发器、固化器、关联发现器和清理器，实现完整的 DMN 循环。
"""

from typing import Optional, TYPE_CHECKING, List
from datetime import datetime
import time

from memory.dmn.models import (
    DMNConfig, DMNResult, DMNTriggerCondition, DMNState,
    ConsolidationRecord, PruneRecord, AssociationRecord
)
from memory.dmn.trigger import DMNTrigger
from memory.dmn.consolidator import STMConsolidator
from memory.dmn.associator import AssociationFinder
from memory.dmn.pruner import NeuronPruner
from memory.reflection.executor import ReflectionExecutor
from memory.reflection.trigger import ReflectionTrigger
from memory.schemas import ReflectionConfig

if TYPE_CHECKING:
    from memory.memory import ShortTermMemory, EpisodicMemory
    from memory.engram import Engram
    from memory.neuron import NeuronCell


class DMNMode:
    """
    Default Mode Network (DMN) 模式管理器
    
    整合所有 DMN 组件，执行空闲/睡眠期间的自动记忆整合。
    
    典型用法:
        dmn = DMNMode()
        
        # 标记活动
        dmn.mark_active()
        
        # 检查并运行
        result = dmn.check_and_run(stm, ltm)
    """
    
    def __init__(self, config: Optional[DMNConfig] = None):
        """
        Args:
            config: DMN 配置
        """
        self.config = config or DMNConfig()
        
        # 初始化子组件
        self.trigger = DMNTrigger(self.config)
        self.consolidator = STMConsolidator(
            strength_threshold=self.config.consolidation_strength_threshold,
            max_per_cycle=self.config.max_consolidation_per_cycle
        )
        self.associator = AssociationFinder(
            similarity_threshold=self.config.association_similarity_min,
            max_associations=self.config.max_associations_per_cycle
        )
        self.pruner = NeuronPruner(
            strength_threshold=self.config.prune_strength_threshold,
            elo_threshold=self.config.prune_elo_threshold,
            max_prune_per_cycle=self.config.max_prune_per_cycle
        )
        self.reflection_executor = ReflectionExecutor(ReflectionConfig())
        self.reflection_trigger = ReflectionTrigger(ReflectionConfig())
        
        # 状态
        self._current_result: Optional[DMNResult] = None
    
    def mark_active(self, current_time: Optional[datetime] = None) -> None:
        """标记系统为活跃状态"""
        self.trigger.mark_active(current_time)
    
    def mark_idle(self, current_time: Optional[datetime] = None) -> None:
        """标记系统为空闲状态"""
        self.trigger.mark_idle(current_time)
    
    def set_sleep_mode(self, enabled: bool) -> None:
        """设置睡眠模式"""
        self.config.sleep_mode = enabled
        if enabled:
            self.trigger.set_activity_level(0.0)
    
    def check_and_run(
        self,
        stm: Optional['ShortTermMemory'],
        ltm: Optional['EpisodicMemory'],
        current_time: Optional[datetime] = None,
        force: bool = False
    ) -> Optional[DMNResult]:
        """
        检查条件并运行 DMN
        
        Args:
            stm: 短时记忆
            ltm: 长时记忆
            current_time: 当前时间
            force: 强制触发
        
        Returns:
            DMNResult 或 None（如果不需要运行）
        """
        should_run, trigger_condition = self.trigger.should_trigger(
            current_time=current_time,
            force=force
        )
        
        if not should_run:
            return None
        
        return self.run_dmn_cycle(
            stm=stm,
            ltm=ltm,
            trigger_condition=trigger_condition,
            current_time=current_time
        )
    
    def run_dmn_cycle(
        self,
        stm: Optional['ShortTermMemory'],
        ltm: Optional['EpisodicMemory'],
        trigger_condition: DMNTriggerCondition,
        current_time: Optional[datetime] = None
    ) -> DMNResult:
        """
        执行一个完整的 DMN 周期
        
        步骤:
        1. 固化 STM → LTM
        2. 发现关联
        3. 清理弱神经元
        4. 生成 reflection
        
        Args:
            stm: 短时记忆
            ltm: 长时记忆
            trigger_condition: 触发条件
            current_time: 当前时间
        
        Returns:
            DMNResult
        """
        start_time = current_time or datetime.now()
        self.trigger.mark_dmn_started(start_time)
        
        result = DMNResult(
            triggered_conditions=[trigger_condition],
            state=DMNState.RUNNING,
            start_time=start_time
        )
        
        try:
            # 1. 收集所有神经元
            neurons = self._collect_all_neurons(stm, ltm)
            engrams = self._collect_all_engrams(stm, ltm)
            
            # 2. 执行固化 (STM → LTM)
            if stm and ltm:
                consolidation_records = self._run_consolidation(stm, ltm)
                result.consolidation_records = consolidation_records
                result.neurons_consolidated = len(consolidation_records)
            
            # 3. 发现关联
            association_records = self.associator.find_associations(neurons, engrams)
            result.association_records = association_records
            result.associations_found = len(association_records)
            
            # 4. 清理弱神经元
            if engrams:
                prune_records = self._run_pruning(engrams)
                result.prune_records = prune_records
                result.neurons_pruned = len(prune_records)
            
            # 5. 生成 reflection
            if ltm and not ltm.is_empty():
                reflection_result = self._run_reflection(ltm)
                if reflection_result:
                    result.reflections_generated = 1
            
        except Exception as e:
            result.error = str(e)
            result.state = DMNState.IDLE
        
        end_time = datetime.now()
        result.end_time = end_time
        result.duration_seconds = (end_time - start_time).total_seconds()
        result.state = DMNState.IDLE
        
        self.trigger.mark_dmn_finished()
        self._current_result = result
        
        return result
    
    def _collect_all_neurons(
        self,
        stm: Optional['ShortTermMemory'],
        ltm: Optional['EpisodicMemory']
    ) -> List['NeuronCell']:
        """收集所有神经元"""
        neurons = []
        
        if stm:
            for engram in stm.sequences.values():
                neurons.extend(list(engram.get_all_neurons()))
        
        if ltm:
            for manager in ltm.engram_managers.values():
                for engram in manager.engram_dict.values():
                    neurons.extend(list(engram.get_all_neurons()))
        
        return neurons
    
    def _collect_all_engrams(
        self,
        stm: Optional['ShortTermMemory'],
        ltm: Optional['EpisodicMemory']
    ) -> List['Engram']:
        """收集所有 engram"""
        engrams = []
        
        if stm:
            engrams.extend(list(stm.sequences.values()))
        
        if ltm:
            for manager in ltm.engram_managers.values():
                engrams.extend(list(manager.engram_dict.values()))
        
        return engrams
    
    def _run_consolidation(
        self,
        stm: 'ShortTermMemory',
        ltm: 'EpisodicMemory'
    ) -> List[ConsolidationRecord]:
        """运行固化"""
        return self.consolidator.consolidate(stm, ltm)
    
    def _run_pruning(
        self,
        engrams: List['Engram']
    ) -> List[PruneRecord]:
        """运行清理"""
        all_records = []
        for engram in engrams:
            records = self.pruner.prune_from_engram(engram)
            all_records.extend(records)
        return all_records
    
    def _run_reflection(
        self,
        ltm: 'EpisodicMemory'
    ):
        """运行 reflection。

        调用链：
          1. _collect_all_engrams()  → 收集所有 engram
          2. reflection_trigger.check_triggers()  → 评估触发条件
          3. reflection_executor.execute()        → 执行 reflection

        如果没有触发条件则静默返回（不算错误）。
        """
        # 收集 engrams 和神经元
        engrams = []
        recent_neurons = []
        for manager in ltm.engram_managers.values():
            engrams.extend(list(manager.engram_dict.values()))
            for engram in manager.engram_dict.values():
                recent_neurons.extend(list(engram.get_all_neurons()))

        if not engrams:
            return None

        # 限制神经元数量，避免过大计算
        recent_neurons = recent_neurons[:100]

        # 使用默认相似度函数（基于 engram strength）
        def default_similarity_fn(e1: 'Engram', e2: 'Engram') -> float:
            # 简单相似度：两个 engram 的强度越接近，相似度越高
            # 范围 [0, 1]
            diff = abs(e1.strength - e2.strength)
            return max(0.0, 1.0 - diff)

        # Step 1: 评估触发条件
        triggers = self.reflection_trigger.check_triggers(
            engrams=engrams,
            recent_neurons=recent_neurons,
            similarity_fn=default_similarity_fn,
        )

        if not triggers:
            # 没有触发条件，不执行 reflection
            return None

        # Step 2: 执行 reflection（将 ltm 作为 memory_manager 传入）
        return self.reflection_executor.execute(
            triggers=triggers,
            engrams=engrams,
            memory_manager=ltm,
        )
    
    @property
    def last_result(self) -> Optional[DMNResult]:
        """获取最近一次 DMN 运行结果"""
        return self._current_result
    
    @property
    def state(self) -> DMNState:
        """获取当前状态"""
        return self.trigger.state
    
    @property
    def idle_duration_minutes(self) -> int:
        """获取当前空闲持续时间"""
        return self.trigger.get_idle_duration_minutes()
    
    def get_diagnostics(self) -> dict:
        """获取诊断信息"""
        return {
            "state": self.state.value,
            "idle_duration_minutes": self.idle_duration_minutes,
            "current_activity_level": self.trigger.get_current_activity_level(),
            "last_run_time": self.trigger.last_run_time,
            "sleep_mode": self.config.sleep_mode,
            "config": self.config.model_dump()
        }
