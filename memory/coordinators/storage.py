"""
存储协调器 - 管理内存存储、LRU驱逐、懒加载

负责：
- NeuronCell 的内存存储
- LRU 容量上限驱逐
- EmbeddingManager/EventStream/UnifiedRetriever 的懒加载
- add_memory 完整流程
"""

from typing import TYPE_CHECKING, Dict, Optional, List, Any

if TYPE_CHECKING:
    from memory.embedding import EmbeddingManager
    from memory.event import EventStream
    from memory.unified_retriever import UnifiedRetriever
    from memory.neuron import NeuronCell
    from memory.system import MemorySystemConfig
    from memory.indexing import MemoryIndex


class StorageCoordinator:
    """
    管理 MemorySystem 的内存存储和懒加载组件。
    
    职责：
    - 管理 _neurons 字典
    - LRU 容量上限驱逐
    - EmbeddingManager/EventStream/UnifiedRetriever 懒加载
    - add_memory 完整流程
    """
    
    def __init__(
        self,
        config: "MemorySystemConfig",
        index: Optional["MemoryIndex"] = None,
    ) -> None:
        self.config = config
        self.index = index
        
        # 内存存储
        self._neurons: Dict[str, "NeuronCell"] = {}
        
        # 懒加载组件
        self._embedding_manager: Optional["EmbeddingManager"] = None
        self._event_stream: Optional["EventStream"] = None
        self._unified_retriever: Optional["UnifiedRetriever"] = None
        
        # 引用（由 MemorySystem 设置）
        self._elo_reference: Any = None
        self._decay_reference: Any = None
        self._scene_reference: Any = None
    
    def add_neuron(self, neuron: "NeuronCell") -> None:
        """存储神经元到内存"""
        self._neurons[str(neuron.event_id)] = neuron
    
    def get_neuron(self, event_id: str) -> Optional["NeuronCell"]:
        """获取神经元"""
        return self._neurons.get(str(event_id))
    
    def remove_neuron(self, event_id: str) -> bool:
        """移除神经元，返回是否成功"""
        neuron_id = str(event_id)
        if neuron_id in self._neurons:
            self._neurons.pop(neuron_id)
            if self.index:
                self.index.remove_neuron(neuron_id)
            return True
        return False
    
    def get_all_neurons(self) -> Dict[str, "NeuronCell"]:
        """获取所有神经元"""
        return self._neurons
    
    def evict_weak_neurons(self) -> int:
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
    def embedding_manager(self) -> "EmbeddingManager":
        """延迟初始化 EmbeddingManager"""
        if self._embedding_manager is None:
            from memory.embedding import EmbeddingManager
            self._embedding_manager = EmbeddingManager()
        return self._embedding_manager
    
    @property
    def event_stream(self) -> "EventStream":
        """延迟初始化 EventStream"""
        if self._event_stream is None:
            from memory.event import EventStream
            self._event_stream = EventStream()
        return self._event_stream
    
    @property
    def unified_retriever(self) -> "UnifiedRetriever":
        """延迟初始化统一检索器"""
        if self._unified_retriever is None:
            from memory.unified_retriever import UnifiedRetriever, RetrievalConfig
            from memory.elo import EloCompetition
            from memory.decay import DecayScheduler
            from memory.scene import SceneAwareRetrieval
            
            self._unified_retriever = UnifiedRetriever(
                retrieval_config=RetrievalConfig(),
                elo_competitor=self._elo_reference,
                decay_scheduler=self._decay_reference,
                scene_retrieval=self._scene_reference,
            )
        return self._unified_retriever
    
    def set_elo_reference(self, elo: "EloCompetition") -> None:
        """设置 Elo 引用（用于 unified_retriever）"""
        self._elo_reference = elo
    
    def set_decay_reference(self, decay: "DecayScheduler") -> None:
        """设置 Decay 引用"""
        self._decay_reference = decay
    
    def set_scene_reference(self, scene: "SceneAwareRetrieval") -> None:
        """设置 Scene 引用"""
        self._scene_reference = scene
    
    def get_statistics(self) -> Dict:
        """获取存储统计信息"""
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
        }
    
    def add_memory(
        self,
        content: str,
        event_type: str,
        actor: str,
        audience: List[str],
        emotion: Any = None,
        scene: Any = None,
        metadata: Dict = None,
    ) -> 'NeuronCell':
        """
        添加记忆的完整流程。
        
        包含：创建神经元、设置元数据、注册ELO、更新索引、场景映射、共振分析、追踪、存储。
        """
        from memory.neuron import NeuronCell
        from memory.utils import now as utc_now
        from memory.decay import BASE_DECAY_RATES, calculate_emotion_aware_decay
        
        if metadata is None:
            metadata = {}
        
        neuron = NeuronCell(
            event_type=event_type,
            create_time=utc_now(),
            actor=actor,
            audience=audience or []
        )
        
        if emotion:
            neuron.emotional_valence = emotion.valence
            neuron.emotional_arousal = emotion.arousal
            neuron.emotional_dominance = emotion.dominance
            neuron.impact_score = emotion.to_impact_score()
            neuron.decay_rate = calculate_emotion_aware_decay(
                event_type=event_type,
                emotional_valence=emotion.valence,
                emotional_arousal=emotion.arousal,
                emotional_dominance=emotion.dominance
            )
        else:
            neuron.decay_rate = BASE_DECAY_RATES.get(event_type, 0.995)
        
        # 注册到ELO
        if hasattr(self._elo_reference, 'register_neuron'):
            self._elo_reference.register_neuron(neuron.event_id)
        
        # 更新索引
        if self.index:
            self.index.add_neuron(
                neuron_id=str(neuron.event_id),
                timestamp=neuron.create_time,
                tags={event_type},
                strength=neuron.strength,
                event_type=event_type
            )
        
        # 场景映射
        if self._scene_reference and scene:
            self._scene_reference.map_neuron_to_scene(str(neuron.event_id), scene)
        
        # 共振分析
        resonance_data: Optional[Dict] = None
        try:
            if hasattr(self._elo_reference, 'get_state'):
                elo_state = self._elo_reference.get_state(str(neuron.event_id))
                elo_value = elo_state.elo if elo_state else 1000.0
            else:
                elo_value = 1000.0
            
            from memory.resonance import ResonanceEngine, ResonanceConfig
            resonance = ResonanceEngine(ResonanceConfig())
            activation_energy = resonance.calculate_activation_energy(
                strength=neuron.strength,
                elo=elo_value,
                connections_count=len(neuron.outgoing_connections)
            )
            resonance_data = {
                'activation_energy': activation_energy,
                'resonance_threshold': resonance.config.resonance_threshold,
            }
        except Exception:
            pass
        
        # 存储元数据
        neuron._scene = scene.model_dump() if scene else None
        neuron._emotion = emotion.model_dump() if emotion else None
        neuron._resonance = resonance_data
        
        # 存储神经元
        self.add_neuron(neuron)
        
        # LRU驱逐
        self.evict_weak_neurons()
        
        # 生成embedding
        try:
            embedding = self.embedding_manager.embed(content)
            if embedding is not None:
                self.embedding_manager.add_embeddings({neuron.event_id: embedding})
        except Exception:
            pass
        
        # 写入事件流
        self.event_stream.add_event(neuron)
        
        return neuron
