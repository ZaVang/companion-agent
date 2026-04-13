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
- stability: 记忆稳定性 (Sprint 2)
- api_schema: LongMemEval 兼容接口
"""

# 核心模块
from memory.neuron import NeuronCell, Connection, calculate_connection_strength
from memory.engram import Engram, EngramManager, RegistryMetadata
from memory.event import (
    EventStream,
    Events,
    BaseEvent,
    ChatEvent,
    PerceptionEvent,
    ThoughtEvent,
    ReflectionEvent,
    ExperienceEvent,
)
from memory.memory import ShortTermMemory, EpisodicMemory

# Sprint 1 Phase 2 新增
from memory.elo import (
    EloCompetitor,
    EloConfig,
    NeuronEloState,
    calculate_combat_score,
    expected_win_probability,
    get_global_competitor,
    reset_global_competitor,
)
from memory.decay import (
    DecayScheduler,
    DecayConfig,
    NeuronDecayState,
    calculate_decay_rate,
    apply_decay,
    estimate_decay_curve,
    get_global_scheduler,
    reset_global_scheduler,
)
from memory.unified_retriever import (
    UnifiedRetriever,
    RetrievalConfig,
    RetrievalResult,
    get_global_retriever,
    reset_global_retriever,
)

# Sprint 2 新增
from memory.stability import (
    StabilityManager,
    StabilityConfig,
    ActivationResult,
    aggregate_strengths,
    calculate_neuron_stability,
    calculate_engram_stability,
    check_activation_threshold,
    suggest_neurons_for_reinforcement,
    get_global_stability_manager,
    reset_global_stability_manager,
)

# LongMemEval 兼容接口
from memory.api_schema import (
    MemoryCapability,
    EventType,
    MemoryScope,
    AddMemoryRequest,
    RetrieveMemoryRequest,
    UpdateMemoryRequest,
    RetrieveMemoryResponse,
    AddMemoryResponse,
    SystemStatus,
)

__all__ = [
    # 核心
    'NeuronCell',
    'Connection',
    'Engram',
    'EngramManager',
    'RegistryMetadata',
    'EventStream',
    'Events',
    'ShortTermMemory',
    'EpisodicMemory',
    # Sprint 1 Phase 2
    'EloCompetitor',
    'EloConfig',
    'NeuronEloState',
    'DecayScheduler',
    'DecayConfig',
    'NeuronDecayState',
    'calculate_decay_rate',
    'apply_decay',
    'UnifiedRetriever',
    'RetrievalConfig',
    'RetrievalResult',
    # Sprint 2
    'StabilityManager',
    'StabilityConfig',
    'ActivationResult',
    'aggregate_strengths',
    'calculate_neuron_stability',
    'calculate_engram_stability',
    'check_activation_threshold',
    'suggest_neurons_for_reinforcement',
    # LongMemEval 接口
    'MemoryCapability',
    'EventType',
    'MemoryScope',
    'AddMemoryRequest',
    'RetrieveMemoryRequest',
    'UpdateMemoryRequest',
    'RetrieveMemoryResponse',
    'AddMemoryResponse',
    'SystemStatus',
]
