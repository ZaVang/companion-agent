"""
生命周期协调器 - 管理DMN、衰减、动态管理

负责：
- 衰减调度
- DMN 执行
- 神经元生灭
- Reflection 触发
"""

from typing import TYPE_CHECKING, Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from memory.neuron import NeuronCell
    from memory.decay import DecayScheduler, DecayConfig
    from memory.dmn import DMNMode, DMNConfig
    from memory.elo import EloCompetition
    from memory.dynamics import NeuronDynamics, NeuronDeathManager, NeuronBirthManager
    from memory.scene import SceneAwareRetrieval
    from memory.reflection import ReflectionTrigger
    from memory.stability import calculate_neuron_stability
    from memory.coordinators.storage import StorageCoordinator


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


class LifecycleCoordinator:
    """
    管理 MemorySystem 的生命周期操作。
    
    职责：
    - 衰减调度
    - DMN 执行
    - 神经元生灭动态
    - Reflection 触发
    """
    
    def __init__(
        self,
        storage: "StorageCoordinator",
        decay: "DecayScheduler",
        dmn: "DMNMode",
        elo: "EloCompetition",
        dynamics: "NeuronDynamics",
        scene: Optional["SceneAwareRetrieval"] = None,
        reflection: Optional["ReflectionTrigger"] = None,
        stability_fn: Optional[callable] = None,
    ) -> None:
        self.storage = storage
        self.decay = decay
        self.dmn = dmn
        self.elo = elo
        self.dynamics = dynamics
        self.scene = scene
        self.reflection = reflection
        self.stability_fn = stability_fn or (lambda n: getattr(n, 'strength', 1.0))
        
        # Resonance engine for prediction
        from memory.resonance import ResonanceEngine, ResonanceConfig
        self.resonance = ResonanceEngine(ResonanceConfig())
    
    def apply_decay(self, reference_time: Optional[datetime] = None) -> int:
        """
        应用衰减
        
        Args:
            reference_time: 参考时间
        
        Returns:
            衰减的神经元数量
        """
        from memory.utils import now as utc_now
        
        if reference_time is None:
            reference_time = utc_now()
        
        decayed_count = 0
        for neuron in list(self.storage.get_all_neurons().values()):
            old_strength = neuron.strength
            neuron.apply_decay(reference_time)
            
            if neuron.strength < old_strength:
                decayed_count += 1
        
        return decayed_count
    
    def run_dmn(self, auto_consolidation: bool = True, auto_dynamics: bool = True) -> DMNResult:
        """
        运行 DMN
        
        Args:
            auto_consolidation: 是否自动固化
            auto_dynamics: 是否自动动态管理
        
        Returns:
            DMNResult
        """
        result = DMNResult(success=True)
        
        # 评估需要固化的记忆
        if auto_consolidation:
            consolidation_threshold = 0.7
            for neuron in self.storage.get_all_neurons().values():
                stability = self.stability_fn(neuron)
                if stability > consolidation_threshold:
                    result.consolidations += 1
        
        # 评估需要清理的记忆
        if auto_dynamics:
            neurons_to_check = []
            for neuron_id, neuron in self.storage.get_all_neurons().items():
                elo_state = self.elo.get_state(neuron_id) if hasattr(self.elo, 'get_state') else None
                neurons_to_check.append({
                    'id': neuron_id,
                    'elo': elo_state.elo if elo_state else 1000,
                    'strength': neuron.strength,
                    'last_activation': None,
                    'connections_count': len(neuron.outgoing_connections)
                })
            
            death_results = self.dynamics.death_manager.batch_evaluate(neurons_to_check)
            for neuron_id, should_die, reason, record in death_results:
                if should_die:
                    result.prunings += 1
                    self.storage.remove_neuron(neuron_id)
        
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
        from memory.coordinators.retriever import RetrieverCoordinator
        
        retriever = RetrieverCoordinator(self.storage)
        candidates = retriever.retrieve(cue, top_k=20)
        
        results = []
        for neuron in candidates:
            energy = self.resonance.calculate_activation_energy(
                strength=neuron.strength,
                elo=1000.0,
                connections_count=len(neuron.outgoing_connections)
            )
            
            resonance_score = min(1.0, energy)
            predicted_strength = neuron.strength * (1 + resonance_score)
            
            results.append(PredictionResult(
                neuron_id=str(neuron.event_id),
                predicted_strength=predicted_strength,
                confidence=resonance_score,
                related_neurons=[str(c.target_id) for c in neuron.outgoing_connections]
            ))
        
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results
    
    def run_dynamics_cycle(self) -> Dict[str, int]:
        """
        运行动态管理周期
        
        Returns:
            统计信息 {'births': int, 'deaths': int, 'splits': int, 'abstractions': int}
        """
        stats = {
            'births': 0,
            'deaths': 0,
            'splits': 0,
            'abstractions': 0
        }
        
        # 评估死亡
        neurons_data = []
        for neuron_id, neuron in self.storage.get_all_neurons().items():
            elo_state = self.elo.get_state(neuron_id) if hasattr(self.elo, 'get_state') else None
            neurons_data.append({
                'id': neuron_id,
                'elo': elo_state.elo if elo_state else 1000,
                'strength': neuron.strength,
                'last_activation': None,
                'connections_count': len(neuron.outgoing_connections)
            })
        
        death_results = self.dynamics.death_manager.batch_evaluate(neurons_data)
        for neuron_id, should_die, reason, record in death_results:
            if should_die:
                stats['deaths'] += 1
                self.storage.remove_neuron(neuron_id)
        
        # 评估新生
        for neuron in list(self.storage.get_all_neurons().values()):
            birth_record = self.dynamics.birth_manager.execute_birth(
                neuron_id=str(neuron.event_id),
                strength=neuron.strength,
                content=""
            )
            if birth_record:
                stats['births'] += 1
        
        return stats
    
    def trigger_reflection(self, engrams: List[Any] = None, similarity_fn=None) -> List:
        """
        触发 Reflection 检测流程。
        
        Args:
            engrams: Engram 列表
            similarity_fn: 可选，相似度计算函数
        
        Returns:
            触发的条件列表
        """
        if self.reflection is None:
            return []
        
        recent_neurons = list(self.storage.get_all_neurons().values())
        
        self.reflection.update_baseline(recent_neurons)
        
        triggers = self.reflection.check_triggers(
            engrams=engrams or [],
            recent_neurons=recent_neurons,
            similarity_fn=similarity_fn
        )
        
        return triggers
