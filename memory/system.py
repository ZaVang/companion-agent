"""
MemorySystem 统一入口 (Facade 模式)

整合所有 Sprint 功能的统一记忆系统。
重构后行数精简，使用协调器分离职责。
"""

from typing import Dict, List, Optional, Any, TYPE_CHECKING
from datetime import datetime
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from memory.neuron import NeuronCell
    from memory.emotion import EmotionalImpact

from memory.elo import EloCompetition, EloConfig
from memory.decay import DecayScheduler, DecayConfig
from memory.stability import StabilityManager, StabilityConfig, calculate_neuron_stability
from memory.dmn import DMNMode, DMNConfig
from memory.causal import CausalInference, CausalConfig
from memory.scene import SceneAwareRetrieval
from memory.resonance import ResonanceEngine, ResonanceConfig
from memory.emotion import EmotionalImpact
from memory.dynamics import NeuronDynamics, DeathCriteria, BirthCriteria
from memory.viz import MemoryHistory, MemoryTracer, NetworkVisualizer
from memory.optimization import MemoryIndex, BatchProcessor, BatchConfig
from memory.reflection import ReflectionTrigger
from memory.coordinators.lifecycle import DMNResult, PredictionResult


# ============== 配置类 ==============

class MemorySystemConfig(BaseModel):
    """MemorySystem 配置"""
    elo_config: EloConfig = Field(default_factory=EloConfig)
    decay_config: DecayConfig = Field(default_factory=DecayConfig)
    stability_config: StabilityConfig = Field(default_factory=StabilityConfig)
    dmn_config: DMNConfig = Field(default_factory=DMNConfig)
    causal_config: CausalConfig = Field(default_factory=CausalConfig)
    enable_scene: bool = True
    resonance_config: ResonanceConfig = Field(default_factory=ResonanceConfig)
    death_criteria: DeathCriteria = Field(default_factory=DeathCriteria)
    birth_criteria: BirthCriteria = Field(default_factory=BirthCriteria)
    batch_config: BatchConfig = Field(default_factory=BatchConfig)
    enable_index: bool = True
    auto_decay: bool = True
    auto_consolidation: bool = True
    auto_dynamics: bool = True
    max_neurons: Optional[int] = None


# ============== 主类 (Facade) ==============

class MemorySystem:
    """
    统一记忆系统入口 (Facade 模式)
    
    整合所有 Sprint 功能，提供统一接口。
    内部委托给各个协调器处理具体逻辑。
    """
    
    def __init__(
        self,
        config: Optional[MemorySystemConfig] = None,
        episodic_memory: Optional[Any] = None,
        _use_parallel: bool = True,
    ):
        self.config = config or MemorySystemConfig()
        self._use_parallel = _use_parallel
        
        # 初始化核心模块
        self.elo = EloCompetition(self.config.elo_config)
        self.decay = DecayScheduler(self.config.decay_config)
        self.stability = StabilityManager(self.config.stability_config)
        self.dmn = DMNMode(self.config.dmn_config)
        self.causal = CausalInference(self.config.causal_config)
        self.scene = SceneAwareRetrieval() if self.config.enable_scene else None
        self.resonance = ResonanceEngine(self.config.resonance_config)
        
        # Sprint 8: 动态管理
        self.dynamics = NeuronDynamics(
            death_criteria=self.config.death_criteria,
            birth_criteria=self.config.birth_criteria
        )
        
        # Sprint 9: 可视化与追踪
        self.history = MemoryHistory()
        self.tracer = MemoryTracer(self.history)
        self.visualizer = NetworkVisualizer()
        
        # Sprint 10: 优化
        self.index = MemoryIndex() if self.config.enable_index else None
        self.batch = BatchProcessor(self.config.batch_config)
        
        # Sprint 11: Reflection 触发器
        self.reflection = ReflectionTrigger()
        
        # LTM 注入
        self._episodic_memory = episodic_memory
        
        # 初始化协调器
        self._init_coordinators()
        
        # 内部状态
        self._initialized = True
    
    def _init_coordinators(self) -> None:
        """初始化所有协调器"""
        from memory.coordinators import (
            StorageCoordinator,
            RetrieverCoordinator,
            LifecycleCoordinator,
            BatchOperations,
            StatsVizCoordinator,
        )
        
        # StorageCoordinator
        self._storage = StorageCoordinator(
            config=self.config,
            index=self.index,
        )
        self._storage.set_elo_reference(self.elo)
        self._storage.set_decay_reference(self.decay)
        self._storage.set_scene_reference(self.scene)
        
        # RetrieverCoordinator
        self._retriever = RetrieverCoordinator(self._storage)
        
        # LifecycleCoordinator
        self._lifecycle = LifecycleCoordinator(
            storage=self._storage,
            decay=self.decay,
            dmn=self.dmn,
            elo=self.elo,
            dynamics=self.dynamics,
            scene=self.scene,
            reflection=self.reflection,
            stability_fn=calculate_neuron_stability,
        )
        
        # BatchOperations
        self._batch = BatchOperations(self._storage)
        
        # StatsVizCoordinator
        self._stats_viz = StatsVizCoordinator(
            storage=self._storage,
            elo=self.elo,
            dynamics=self.dynamics,
            history=self.history,
            visualizer=self.visualizer,
            index=self.index,
        )
    
    def attach_episodic_memory(self, episodic_memory: Any) -> None:
        """注入 EpisodicMemory 实例，使 retrieve() 能从 LTM 召回"""
        self._episodic_memory = episodic_memory
    
    @property
    def _neurons(self) -> Dict:
        """兼容层：支持旧的 _neurons 访问方式"""
        return self._storage.get_all_neurons()
    
    @property
    def decay_scheduler(self) -> DecayScheduler:
        """decay 的别名 property，确保向后兼容"""
        return self.decay
    
    def add_memory(
        self,
        content: str,
        event_type: str = 'chat',
        emotion: Optional[EmotionalImpact] = None,
        scene: Optional[Any] = None,
        actor: str = 'system',
        audience: List[str] = None,
        metadata: Dict = None
    ) -> 'NeuronCell':
        """添加记忆（委托给 StorageCoordinator）"""
        if audience is None:
            audience = []
        if metadata is None:
            metadata = {}
        
        # 追踪历史
        self.tracer.history.add_entry(
            neuron_id='',  # 先占位
            change_type='created',
            event_type=event_type,
            emotion={
                'valence': emotion.valence if emotion else 0,
                'arousal': emotion.arousal if emotion else 0.5,
                'dominance': emotion.dominance if emotion else 0.5
            },
            **metadata
        )
        
        neuron = self._storage.add_memory(
            content=content,
            event_type=event_type,
            actor=actor,
            audience=audience,
            emotion=emotion,
            scene=scene,
            metadata=metadata,
        )
        
        # 更新历史中的neuron_id
        self.tracer.history.add_entry(
            neuron_id=str(neuron.event_id),
            change_type='created',
            event_type=event_type,
            emotion={
                'valence': emotion.valence if emotion else 0,
                'arousal': emotion.arousal if emotion else 0.5,
                'dominance': emotion.dominance if emotion else 0.5
            },
            **metadata
        )
        
        return neuron
    
    def retrieve(
        self,
        query: str,
        scene: Optional[Any] = None,
        top_k: int = 10,
        event_types: List[str] = None
    ) -> List['NeuronCell']:
        """检索记忆"""
        ltm_recaller = None
        if self._episodic_memory:
            def ltm_recaller(query_embedding, embedding_manager, event_stream, top_k, event_types):
                return self._episodic_memory.retrieve_from_engrams(
                    query_embedding=query_embedding,
                    embedding_manager=embedding_manager,
                    event_stream=event_stream,
                    top_k=top_k,
                    event_types=event_types,
                )
        
        return self._retriever.retrieve(
            query=query,
            scene=scene,
            top_k=top_k,
            event_types=event_types,
            ltm_recaller=ltm_recaller,
        )
    
    def apply_decay(self, reference_time: datetime = None) -> int:
        """应用衰减"""
        return self._lifecycle.apply_decay(reference_time)
    
    def run_dmn(self) -> DMNResult:
        """运行 DMN"""
        return self._lifecycle.run_dmn(
            auto_consolidation=self.config.auto_consolidation,
            auto_dynamics=self.config.auto_dynamics,
        )
    
    def predict_activation(self, cue: str) -> List[PredictionResult]:
        """预测激活"""
        return self._lifecycle.predict_activation(cue)
    
    def run_dynamics_cycle(self) -> Dict[str, int]:
        """运行动态管理周期"""
        return self._lifecycle.run_dynamics_cycle()
    
    def trigger_reflection(self, similarity_fn=None) -> List:
        """触发 Reflection 检测"""
        engrams = list(getattr(self, '_engrams', {}).values()) if hasattr(self, '_engrams') else []
        return self._lifecycle.trigger_reflection(engrams=engrams, similarity_fn=similarity_fn)
    
    def batch_retrieve(self, query: str, top_k: int = 10, event_types: List[str] = None) -> List[Dict]:
        """批量检索"""
        return self._batch.batch_retrieve(query=query, top_k=top_k, event_types=event_types)
    
    def batch_decay(self, reference_time: datetime = None, neurons: List['NeuronCell'] = None) -> Dict:
        """批量衰减"""
        return self._batch.batch_decay(reference_time=reference_time, neurons=neurons)
    
    def get_statistics(self) -> Dict:
        """获取系统统计"""
        return self._stats_viz.get_statistics()
    
    def to_graphviz(self) -> str:
        """导出为 Graphviz DOT"""
        return self._stats_viz.to_graphviz()
    
    def get_network_stats(self) -> Dict:
        """获取网络统计"""
        return self._stats_viz.get_network_stats()
    
    def get_network_visualization(self) -> str:
        """获取网络可视化"""
        return self._stats_viz.get_network_visualization()


# 全局实例
_global_memory_system: Optional[MemorySystem] = None


def get_memory_system() -> MemorySystem:
    """获取全局 MemorySystem"""
    global _global_memory_system
    if _global_memory_system is None:
        _global_memory_system = MemorySystem()
    return _global_memory_system


def reset_memory_system() -> None:
    """重置全局 MemorySystem"""
    global _global_memory_system
    _global_memory_system = None
