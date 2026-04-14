"""
Engram 记忆系统

核心模块:
- neuron: 神经元单元
- engram: 记忆痕迹
- event: 事件流
- memory: 短/长期记忆
- elo: Elo 竞争机制 (Sprint 1 Phase 2)
- decay: 动态衰减系统 (Sprint 1 Phase 2)
- unified_retriever: 统一检索 (Sprint 1 Phase 2)
- reflection: Reflection 自动化 (Sprint 1 Phase 3)
- stability: 记忆稳定性 (Sprint 2)
- schemas: 配置和数据模型
- api_schema: LongMemEval 兼容接口
- dmn: 记忆巩固 (Sprint 5)
- causal: 因果推理 (Sprint 6)
- scene: 场景感知 (Sprint 5)
- resonance: 共振引擎 (Sprint 6)
- emotion: 情绪处理 (Sprint 7)
- dynamics: 神经元动态增删 (Sprint 8)
- viz: 可视化与调试 (Sprint 9)
- optimization: 性能优化 (Sprint 10)
- system: 统一入口 (Sprint 8-10)
"""

# Schema 模块（配置和数据模型）
from memory.schemas import (
    # 配置类
    EloConfig, DecayConfig, ReflectionConfig, StabilityConfig,
    # 数据模型
    NeuronEloState, NeuronDecayState,
    TriggerCondition, ReflectionResult,
    ActivationResult, EngramMember, EngramSummary,
)

# 核心模块
from memory.neuron import NeuronCell, Connection, calculate_connection_strength
from memory.engram import Engram, EngramManager, RegistryMetadata
from memory.event import (
    EventStream, Events, BaseEvent,
    ChatEvent, PerceptionEvent, ThoughtEvent,
    ReflectionEvent, ExperienceEvent,
)
from memory.memory import ShortTermMemory, EpisodicMemory

# Elo 竞争机制
from memory.elo import (
    EloCompetition,
    expected_win_probability,
    calculate_combat_score,
    adjust_elo_ratings,
)

# 动态衰减系统
from memory.decay import (
    DecayScheduler,
    calculate_decay_rate,
    apply_decay,
    calculate_multi_event_decay,
    estimate_decay_curve,
    get_decay_half_life,
    get_global_scheduler,
    reset_global_scheduler,
)

# 统一检索系统
from memory.unified_retriever import (
    UnifiedRetriever,
    RetrievalConfig,
    RetrievalResult,
    get_global_retriever,
    reset_global_retriever,
)

# Reflection 自动化
from memory.reflection import (
    ReflectionTrigger,
    ReflectionExecutor,
    create_reflection_neuron,
    run_reflection_cycle,
    detect_conflict,
    detect_new_associations,
)

# 记忆稳定性
from memory.stability import (
    aggregate_strengths,
    calculate_neuron_stability,
    calculate_engram_stability,
    check_activation_threshold,
    suggest_neurons_for_reinforcement,
)

# Sprint 5: DMN 和 Scene
from memory.dmn import DMNMode, DMNConfig
from memory.scene import SceneAwareRetrieval, SceneContext

# Sprint 6: Causal 和 Resonance
from memory.causal import CausalInference, CausalConfig
from memory.resonance import ResonanceEngine, ResonanceConfig

# Sprint 7: Emotion
from memory.emotion import EmotionalImpact, ImpactMapper

# Sprint 8: Dynamics
from memory.dynamics import (
    NeuronDynamics,
    NeuronDeath,
    NeuronBirth,
    DeathCriteria,
    BirthCriteria,
    NeuronDeathManager,
    NeuronBirthManager,
    DeathReason,
    BirthReason,
    DeathRecord,
    BirthRecord,
)

# Sprint 9: Visualization
from memory.viz import (
    NetworkVisualizer,
    NetworkStats,
    visualize_engram_network,
    get_network_statistics,
    MemoryHistory,
    MemoryTracer,
    HistoryEntry,
    ChangeType,
    track_memory_change,
)

# Sprint 10: Optimization
from memory.optimization import (
    MemoryIndex,
    VectorIndex,
    TimeIndex,
    TagIndex,
    IndexConfig,
    get_memory_index,
    BatchProcessor,
    BatchConfig,
    BatchResult,
    batch_activate,
    batch_decay,
    batch_retrieve,
)

# System (Sprint 8-10)
from memory.system import (
    MemorySystem,
    MemorySystemConfig,
    DMNResult,
    PredictionResult,
    get_memory_system,
    reset_memory_system,
)


__all__ = [
    # Schema
    'EloConfig', 'DecayConfig', 'ReflectionConfig', 'StabilityConfig',
    'NeuronEloState', 'NeuronDecayState',
    'TriggerCondition', 'ReflectionResult',
    'ActivationResult', 'EngramMember', 'EngramSummary',
    
    # 核心
    'NeuronCell', 'Connection', 'calculate_connection_strength',
    'Engram', 'EngramManager', 'RegistryMetadata',
    'EventStream', 'Events',
    'ShortTermMemory', 'EpisodicMemory',
    
    # Elo
    'EloCompetition', 'expected_win_probability',
    'calculate_combat_score', 'adjust_elo_ratings',
    
    # Decay
    'DecayScheduler', 'calculate_decay_rate', 'apply_decay',
    'calculate_multi_event_decay', 'estimate_decay_curve',
    'get_decay_half_life', 'get_global_scheduler', 'reset_global_scheduler',
    
    # Retriever
    'UnifiedRetriever', 'RetrievalConfig', 'RetrievalResult',
    'get_global_retriever', 'reset_global_retriever',
    
    # Reflection
    'ReflectionTrigger', 'ReflectionExecutor',
    'create_reflection_neuron', 'run_reflection_cycle',
    'detect_conflict', 'detect_new_associations',
    
    # Stability
    'aggregate_strengths', 'calculate_neuron_stability',
    'calculate_engram_stability', 'check_activation_threshold',
    'suggest_neurons_for_reinforcement',
    
    # DMN & Scene (Sprint 5)
    'DMNMode', 'DMNConfig',
    'SceneAwareRetrieval', 'SceneContext',
    
    # Causal & Resonance (Sprint 6)
    'CausalInference', 'CausalConfig',
    'ResonanceEngine', 'ResonanceConfig',
    
    # Emotion (Sprint 7)
    'EmotionalImpact', 'ImpactMapper',
    
    # Dynamics (Sprint 8)
    'NeuronDynamics', 'NeuronDeath', 'NeuronBirth',
    'DeathCriteria', 'BirthCriteria',
    'NeuronDeathManager', 'NeuronBirthManager',
    'DeathReason', 'BirthReason', 'DeathRecord', 'BirthRecord',
    
    # Visualization (Sprint 9)
    'NetworkVisualizer', 'NetworkStats',
    'visualize_engram_network', 'get_network_statistics',
    'MemoryHistory', 'MemoryTracer', 'HistoryEntry',
    'ChangeType', 'track_memory_change',
    
    # Optimization (Sprint 10)
    'MemoryIndex', 'VectorIndex', 'TimeIndex', 'TagIndex',
    'IndexConfig', 'get_memory_index',
    'BatchProcessor', 'BatchConfig', 'BatchResult',
    'batch_activate', 'batch_decay', 'batch_retrieve',
    
    # System
    'MemorySystem', 'MemorySystemConfig',
    'DMNResult', 'PredictionResult',
    'get_memory_system', 'reset_memory_system',
]
