"""
MemorySystem 统一入口

整合所有 Sprint 功能的统一记忆系统。
"""

from typing import Dict, List, Optional, Set, Tuple, Any, Literal
from pydantic import BaseModel, Field, UUID1
from datetime import datetime
from dataclasses import dataclass, field
import uuid

from memory.utils import now as utc_now

# 导入现有模块
from memory.elo import EloCompetition, EloConfig
from memory.decay import DecayScheduler, DecayConfig, BASE_DECAY_RATES, calculate_emotion_aware_decay
from memory.stability import (
    StabilityManager,
    StabilityConfig,
    calculate_neuron_stability,
    calculate_engram_stability,
)
from memory.dmn import DMNMode, DMNConfig
from memory.causal import CausalInference, CausalConfig
from memory.scene import SceneAwareRetrieval, SceneContext
from memory.resonance import ResonanceEngine, ResonanceConfig
from memory.emotion import EmotionalImpact

# Sprint 8: 神经元动态
from memory.dynamics import (
    NeuronDynamics,
    NeuronDeathManager,
    NeuronBirthManager,
    DeathCriteria,
    BirthCriteria,
    DeathReason,
    BirthReason,
)

# Sprint 9: 可视化
from memory.viz import MemoryHistory, MemoryTracer, NetworkVisualizer

# Sprint 10: 优化
from memory.optimization import MemoryIndex, BatchProcessor, BatchConfig

# 核心类
from memory.neuron import NeuronCell
from memory.engram import Engram
from memory.unified_retriever import UnifiedRetriever, RetrievalConfig, RetrievalResult
from memory.embedding import EmbeddingManager
from memory.event import EventStream


# ============== 配置类 ==============

class MemorySystemConfig(BaseModel):
    """MemorySystem 配置"""
    # Elo 配置
    elo_config: EloConfig = Field(default_factory=EloConfig)
    
    # Decay 配置
    decay_config: DecayConfig = Field(default_factory=DecayConfig)
    
    # Stability 配置
    stability_config: StabilityConfig = Field(default_factory=StabilityConfig)
    
    # DMN 配置
    dmn_config: DMNConfig = Field(default_factory=DMNConfig)
    
    # Causal 配置
    causal_config: CausalConfig = Field(default_factory=CausalConfig)
    
    # Scene 配置
    enable_scene: bool = True
    
    # Resonance 配置
    resonance_config: ResonanceConfig = Field(default_factory=ResonanceConfig)
    
    # Dynamics 配置
    death_criteria: DeathCriteria = Field(default_factory=DeathCriteria)
    birth_criteria: BirthCriteria = Field(default_factory=BirthCriteria)
    
    # Optimization 配置
    batch_config: BatchConfig = Field(default_factory=BatchConfig)
    enable_index: bool = True
    
    # 行为配置
    auto_decay: bool = True          # 自动衰减
    auto_consolidation: bool = True  # 自动固化
    auto_dynamics: bool = True       # 自动动态管理
    max_neurons: Optional[int] = None  # 神经元容量上限，None=无限制


class DMNResult(BaseModel):
    """DMN 运行结果"""
    success: bool
    consolidations: int = 0
    prunings: int = 0
    new_associations: int = 0
    messages: List[str] = Field(default_factory=list)


@dataclass
class PredictionResult:
    """预测激活结果"""
    neuron_id: str
    predicted_strength: float
    confidence: float
    related_neurons: List[str] = field(default_factory=list)


# ============== 主类 ==============

class MemorySystem:
    """
    统一记忆系统入口
    
    整合所有 Sprint 功能，提供统一接口。
    """
    
    def __init__(
        self,
        config: Optional[MemorySystemConfig] = None,
        episodic_memory: Optional['EpisodicMemory'] = None,
        _use_parallel: bool = True,
    ):
        self.config = config or MemorySystemConfig()
        self._use_parallel = _use_parallel

        # 核心模块（Sprint 1-7）
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

        # 核心依赖：Embedding + EventStream + 统一检索器
        self._embedding_manager: Optional[EmbeddingManager] = None
        self._event_stream: Optional[EventStream] = None
        self._unified_retriever: Optional[UnifiedRetriever] = None

        # LTM 注入：支持直接传入（优雅）或后期 attach（向后兼容）
        self._episodic_memory: Optional['EpisodicMemory'] = episodic_memory

        # 内部状态
        self._neurons: Dict[str, NeuronCell] = {}
        self._engrams: Dict[str, Engram] = {}
        self._initialized = True
    
    def attach_episodic_memory(self, episodic_memory: 'EpisodicMemory') -> None:
        """注入 EpisodicMemory 实例，使 retrieve() 能从 LTM 召回"""
        self._episodic_memory = episodic_memory

    def _evict_weak_neurons(self) -> int:
        """
        LRU-style 容量上限 eviction。

        当 self._neurons 超出 max_neurons 时，
        按 strength 从低到高排序，删除最弱的神经元直到回到容量以内。

        Returns:
            被驱逐的神经元数量。
        """
        max_cap = self.config.max_neurons
        if max_cap is None:
            return 0

        current_count = len(self._neurons)
        if current_count <= max_cap:
            return 0

        excess = current_count - max_cap
        # 按 strength 升序排列，最弱的在前面
        sorted_neurons = sorted(
            self._neurons.items(),
            key=lambda item: item[1].strength
        )

        evicted = 0
        for neuron_id, neuron in sorted_neurons:
            if evicted >= excess:
                break
            self._neurons.pop(neuron_id, None)
            if self.index:
                self.index.remove_neuron(neuron_id)
            evicted += 1

        return evicted
    
    @property
    def embedding_manager(self) -> EmbeddingManager:
        """延迟初始化 EmbeddingManager"""
        if self._embedding_manager is None:
            self._embedding_manager = EmbeddingManager()
        return self._embedding_manager
    
    @property
    def event_stream(self) -> EventStream:
        """延迟初始化 EventStream"""
        if self._event_stream is None:
            self._event_stream = EventStream()
        return self._event_stream
    
    @property
    def unified_retriever(self) -> UnifiedRetriever:
        """延迟初始化统一检索器"""
        if self._unified_retriever is None:
            self._unified_retriever = UnifiedRetriever(
                retrieval_config=RetrievalConfig(),
                elo_competitor=self.elo,
                decay_scheduler=self.decay,
                scene_retrieval=self.scene
            )
        return self._unified_retriever
    
    # ============== 核心操作 ==============
    
    def add_memory(
        self,
        content: str,
        event_type: str = 'chat',
        emotion: Optional[EmotionalImpact] = None,
        scene: Optional[SceneContext] = None,
        actor: str = 'system',
        audience: List[str] = None,
        metadata: Dict = None
    ) -> NeuronCell:
        """
        添加记忆
        
        Args:
            content: 记忆内容
            event_type: 事件类型
            emotion: 情绪影响
            scene: 场景上下文
            actor: 行动者
            audience: 受众
            metadata: 额外元数据
        
        Returns:
            创建的 NeuronCell（内部包含 scene/emotion/resonance 元数据）
        """
        if audience is None:
            audience = []
        if metadata is None:
            metadata = {}
        
        # 创建神经元
        neuron = NeuronCell(
            event_type=event_type,
            create_time=utc_now(),
            actor=actor,
            audience=audience
        )
        
        # 设置情绪
        if emotion:
            neuron.emotional_valence = emotion.valence
            neuron.emotional_arousal = emotion.arousal
            neuron.emotional_dominance = emotion.dominance
            
            # 计算冲击力
            impact_score = emotion.to_impact_score()
            neuron.impact_score = impact_score
        
        # 计算衰减率
        if emotion:
            neuron.decay_rate = calculate_emotion_aware_decay(
                event_type=event_type,
                emotional_valence=emotion.valence,
                emotional_arousal=emotion.arousal,
                emotional_dominance=emotion.dominance
            )
        else:
            neuron.decay_rate = BASE_DECAY_RATES.get(event_type, 0.995)
        
        # 注册到 Elo 系统
        self.elo.register_neuron(neuron.event_id)
        
        # 更新索引
        if self.index:
            self.index.add_neuron(
                neuron_id=str(neuron.event_id),
                timestamp=neuron.create_time,
                tags={event_type},
                strength=neuron.strength,
                event_type=event_type
            )
        
        # Sprint 5: 场景映射
        if self.scene and scene:
            self.scene.map_neuron_to_scene(str(neuron.event_id), scene)
        
        # Sprint 6: 共振分析 - 计算新神经元的激活能量
        resonance_data: Optional[Dict] = None
        try:
            elo_state = self.elo.get_neuron_state(str(neuron.event_id))
            elo_value = elo_state.elo if elo_state else 1000.0
        except Exception:
            elo_value = 1000.0
        try:
            activation_energy = self.resonance.calculate_activation_energy(
                strength=neuron.strength,
                elo=elo_value,
                connections_count=len(neuron.outgoing_connections)
            )
            resonance_data = {
                'activation_energy': activation_energy,
                'resonance_threshold': self.resonance.config.resonance_threshold,
            }
        except Exception:
            resonance_data = None
        
        # Sprint 5/6/7 元数据：存储在神经元实例上（供检索层读取）
        # 注意：不通过 Pydantic 字段存储，避免验证错误
        neuron._scene = scene.model_dump() if scene else None
        neuron._emotion = emotion.model_dump() if emotion else None
        neuron._resonance = resonance_data
        
        # 追踪历史
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
        
        # 存储神经元
        self._neurons[str(neuron.event_id)] = neuron

        # LRU-style 容量上限 eviction：超出时删除最弱的神经元
        if self.config.max_neurons is not None:
            self._evict_weak_neurons()
        
        # 生成并存储语义 embedding（关键修复：语义检索的前提）
        #健壮性：embedding 失败不影响记忆存储
        try:
            embedding = self.embedding_manager.embed(content)
            if embedding is not None:
                self.embedding_manager.add_embeddings({neuron.event_id: embedding})
        except Exception:
            pass  # embedding 生成失败不影响记忆存储
        
        # 将事件写入 EventStream（供 UnifiedRetriever 获取内容）
        self.event_stream.add_event(neuron)
        
        return neuron
    
    def retrieve(
        self,
        query: str,
        scene: Optional[SceneContext] = None,
        top_k: int = 10,
        event_types: List[str] = None
    ) -> List[NeuronCell]:
        """
        检索记忆
        
        Args:
            query: 查询文本
            scene: 场景上下文（可选）
            top_k: 返回数量
            event_types: 事件类型过滤
        
        Returns:
            匹配的 NeuronCell 列表
        """
        if not self._neurons:
            return []
        
        candidates = list(self._neurons.values())
        
        # 按事件类型过滤
        if event_types:
            candidates = [n for n in candidates if n.event_type in event_types]
        
        # 场景过滤（提前裁剪候选集，减少后续计算量）
        if self.scene and scene:
            scene_neuron_ids = self.scene.get_neurons_for_scene(scene)
            neuron_ids_set = {str(n.event_id) for n in candidates}
            common = neuron_ids_set & scene_neuron_ids
            candidates = [n for n in candidates if str(n.event_id) in common]
        
        if not candidates:
            return []
        
        # 关键修复：使用 UnifiedRetriever 进行语义检索
        try:
            query_embedding = self.embedding_manager.embed(query)
            
            results = self.unified_retriever.retrieve(
                neurons=candidates,
                query_embedding=query_embedding,
                embedding_manager=self.embedding_manager,
                event_stream=self.event_stream,
                current_scene=scene
            )
            
            # 通过 event_id 找到 NeuronCell 对象
            retrieved_ids = [r.event_id for r in results]
            neuron_map = {str(n.event_id): n for n in candidates}
            retrieved = [neuron_map[str(rid)] for rid in retrieved_ids if str(rid) in neuron_map]
            
            if retrieved:
                return retrieved[:top_k]
        except Exception:
            pass  # 降级到词重叠检索
        
        # 降级：词重叠检索（无 embedding 或检索失败时）
        query_words = set(query.lower().split())
        scores = []
        for neuron in candidates:
            # 从 event_stream 拿真实内容做匹配
            try:
                event = self.event_stream.get_event(neuron.event_id)
                content_words = set(event.content.lower().split())
            except Exception:
                content_words = set(neuron.event_type.lower().split())
            
            overlap = len(query_words & content_words)
            score = overlap / max(len(query_words), 1) if query_words else 0
            score *= neuron.strength
            scores.append((neuron, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        stm_results = [n for n, _ in scores[:top_k]]
        
        # 🔑 关键修复：STM 结果不足时，自动从 LTM 召回
        if len(stm_results) < top_k:
            try:
                query_emb = self.embedding_manager.embed(query)
                ltm_results = self._episodic_memory.retrieve_from_engrams(
                    query_embedding=query_emb,
                    embedding_manager=self.embedding_manager,
                    event_stream=self.event_stream,
                    top_k=top_k - len(stm_results),
                    event_types=event_types,
                )
                # 过滤掉已经在 STM 结果中的
                stm_ids = {str(n.event_id) for n in stm_results}
                for neuron in ltm_results:
                    if str(neuron.event_id) not in stm_ids:
                        stm_results.append(neuron)
            except Exception:
                pass
        
        return stm_results[:top_k]
    
    def apply_decay(self, reference_time: datetime = None) -> int:
        """
        应用衰减
        
        Args:
            reference_time: 参考时间
        
        Returns:
            衰减的神经元数量
        """
        if reference_time is None:
            reference_time = utc_now()
        
        decayed_count = 0
        for neuron_id, neuron in list(self._neurons.items()):
            old_strength = neuron.strength
            neuron.apply_decay(reference_time)
            
            if neuron.strength < old_strength:
                decayed_count += 1
                
                # 追踪
                self.tracer.trace_decay(
                    neuron_id=neuron_id,
                    old_strength=old_strength,
                    new_strength=neuron.strength,
                    decay_rate=neuron.decay_rate
                )
        
        return decayed_count
    
    def run_dmn(self) -> DMNResult:
        """
        运行 DMN
        
        Returns:
            DMNResult
        """
        # 简化实现
        result = DMNResult(success=True)
        
        # 评估需要固化的记忆
        if self.config.auto_consolidation:
            consolidation_threshold = 0.7  # 默认阈值
            for neuron_id, neuron in self._neurons.items():
                stability = calculate_neuron_stability(neuron)
                if stability > consolidation_threshold:
                    result.consolidations += 1
        
        # 评估需要清理的记忆
        if self.config.auto_dynamics:
            neurons_to_check = [
                {
                    'id': neuron_id,
                    'elo': self.elo.get_neuron_state(neuron_id).elo if hasattr(self.elo, 'get_neuron_state') else 1000,
                    'strength': neuron.strength,
                    'last_activation': None,
                    'connections_count': len(neuron.outgoing_connections)
                }
                for neuron_id, neuron in self._neurons.items()
            ]
            
            death_results = self.dynamics.death_manager.batch_evaluate(neurons_to_check)
            for neuron_id, should_die, reason, record in death_results:
                if should_die:
                    result.prunings += 1
                    self._neurons.pop(neuron_id, None)
                    if self.index:
                        self.index.remove_neuron(neuron_id)
        
        return result
    
    def predict_activation(self, cue: str) -> List[PredictionResult]:
        """
        预测激活
        
        使用共振机制预测给定线索会激活哪些记忆。
        
        Args:
            cue: 激活线索
        
        Returns:
            预测结果列表
        """
        # 检索相关记忆
        candidates = self.retrieve(cue, top_k=20)
        
        results = []
        for neuron in candidates:
            # 计算激活能量
            energy = self.resonance.calculate_activation_energy(
                strength=neuron.strength,
                elo=1000.0,  # 默认值
                connections_count=len(neuron.outgoing_connections)
            )
            
            # 使用能量作为共振分数
            resonance_score = min(1.0, energy)
            
            # 预测激活后的强度
            predicted_strength = neuron.strength * (1 + resonance_score)
            
            results.append(PredictionResult(
                neuron_id=str(neuron.event_id),
                predicted_strength=predicted_strength,
                confidence=resonance_score,
                related_neurons=[str(c.target_id) for c in neuron.outgoing_connections]
            ))
        
        # 按置信度排序
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results
    
    # ============== 动态管理 ==============
    
    def run_dynamics_cycle(self) -> Dict:
        """
        运行动态管理周期
        
        Returns:
            统计信息
        """
        stats = {
            'births': 0,
            'deaths': 0,
            'splits': 0,
            'abstractions': 0
        }
        
        # 评估死亡
        neurons_data = [
            {
                'id': neuron_id,
                'elo': self.elo.get_neuron_state(neuron_id).elo if hasattr(self.elo, 'get_neuron_state') else 1000,
                'strength': neuron.strength,
                'last_activation': None,
                'connections_count': len(neuron.outgoing_connections)
            }
            for neuron_id, neuron in self._neurons.items()
        ]
        
        death_results = self.dynamics.death_manager.batch_evaluate(neurons_data)
        for neuron_id, should_die, reason, record in death_results:
            if should_die:
                stats['deaths'] += 1
                self._neurons.pop(neuron_id, None)
                if self.index:
                    self.index.remove_neuron(neuron_id)
        
        # 评估新生
        for neuron_id, neuron in list(self._neurons.items()):
            birth_record = self.dynamics.birth_manager.execute_birth(
                neuron_id=neuron_id,
                strength=neuron.strength,
                content=""
            )
            if birth_record:
                stats['births'] += 1
        
        return stats
    
    def batch_retrieve(
        self,
        query: str,
        top_k: int = 10,
        event_types: List[str] = None
    ) -> List[Dict]:
        """
        使用索引加速的批量检索。

        优先使用 search_by_text（无需 embedding，可靠快速）。

        Args:
            query: 查询文本
            top_k: 返回数量
            event_types: 事件类型过滤

        Returns:
            List[{'neuron_id': str, 'score': float}]
        """
        if not self._neurons:
            return []

        try:
            candidates = list(self._neurons.values())

            if event_types:
                candidates = [n for n in candidates if n.event_type in event_types]

            neuron_map = {str(n.event_id): n for n in candidates}

            # 路径1: search_by_text（最快，无需 embedding）
            if self.index:
                text_results = self.index.search_by_text(query, top_k=top_k)
                if text_results:
                    retrieved = []
                    for neuron_id, score in text_results:
                        if neuron_id in neuron_map:
                            retrieved.append({
                                'neuron_id': neuron_id,
                                'score': score,
                                'neuron': neuron_map[neuron_id]
                            })
                    return retrieved

            # 路径2: 回退到简单内存扫描（不依赖 embedding）
            # 基于 event_type 关键词匹配
            query_lower = query.lower()
            results = []
            for neuron_id, neuron in neuron_map.items():
                event_type_lower = neuron.event_type.lower()
                if query_lower in event_type_lower:
                    results.append({
                        'neuron_id': neuron_id,
                        'score': 0.5,
                        'neuron': neuron
                    })
            return results[:top_k]

        except Exception:
            return []

    def batch_decay(
        self,
        reference_time: datetime = None,
        neurons: List[NeuronCell] = None
    ) -> Dict:
        """
        批量对神经元应用衰减（使用 BatchProcessor）。

        Args:
            reference_time: 参考时间
            neurons: 要衰减的神经元列表，None=所有

        Returns:
            {'applied': int, 'skipped': int, 'total_time_ms': float}
        """
        import time
        start = time.time()

        if neurons is None:
            neurons = list(self._neurons.values())

        if reference_time is None:
            reference_time = utc_now()

        applied = 0
        skipped = 0

        for neuron in neurons:
            try:
                old = neuron.strength
                neuron.apply_decay(reference_time)
                if neuron.strength < old:
                    applied += 1
                else:
                    skipped += 1
            except Exception:
                skipped += 1

        elapsed_ms = (time.time() - start) * 1000
        return {
            'applied': applied,
            'skipped': skipped,
            'total_time_ms': elapsed_ms
        }

    # ============== 统计和调试 ==============
    
    def get_statistics(self) -> Dict:
        """获取系统统计"""
        total_neurons = len(self._neurons)
        
        neurons_by_type: Dict[str, int] = {}
        total_strength = 0.0
        for neuron in self._neurons.values():
            neurons_by_type[neuron.event_type] = neurons_by_type.get(neuron.event_type, 0) + 1
            total_strength += neuron.strength
        
        return {
            'total_neurons': total_neurons,
            'neurons_by_type': neurons_by_type,
            'avg_strength': total_strength / total_neurons if total_neurons > 0 else 0,
            'elo_statistics': self.elo.get_statistics() if hasattr(self.elo, 'get_statistics') else {},
            'dynamics': {
                'total_deaths': self.dynamics.death_manager.get_statistics()['total_deaths'],
                'total_births': self.dynamics.birth_manager.get_statistics()['total_births']
            },
            'history': self.history.get_statistics(),
            'index': self.index.get_statistics() if self.index else None
        }
    
    def to_graphviz(self) -> str:
        """
        导出记忆网络为 Graphviz DOT 格式字符串。

        Returns:
            DOT 格式字符串（包含 "digraph" 关键字）
        """
        neurons = list(self._neurons.values())
        connections = [
            (str(n.event_id), str(c.target_id))
            for n in neurons
            for c in n.outgoing_connections
        ]
        self.visualizer.build_from_neurons(neurons, connections)
        return self.visualizer.render_graphviz()

    def get_network_stats(self) -> Dict:
        """
        获取记忆网络的统计信息。

        Returns:
            dict 包含 node_count, edge_count, avg_degree 等
        """
        neurons = list(self._neurons.values())
        connections = [
            (str(n.event_id), str(c.target_id))
            for n in neurons
            for c in n.outgoing_connections
        ]
        self.visualizer.build_from_neurons(neurons, connections)
        stats = self.visualizer.get_statistics()
        return {
            'node_count': stats.total_neurons,
            'edge_count': stats.total_connections,
            'avg_degree': stats.avg_connections_per_neuron,
            'density': stats.density,
            'isolated_nodes': stats.isolated_neurons,
            'neurons_by_type': stats.neurons_by_type,
            'avg_strength': stats.avg_strength,
        }

    def get_network_visualization(self) -> str:
        """获取网络可视化"""
        viz = NetworkVisualizer()
        
        neurons = list(self._neurons.values())
        connections = []
        
        for neuron in neurons:
            for conn in neuron.outgoing_connections:
                connections.append((str(neuron.event_id), str(conn.target_id)))
        
        viz.build_from_neurons(neurons, connections)
        return viz.render_text()


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
