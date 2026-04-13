"""
因果推断引擎

实现因果推断算法。
"""

from typing import List, Tuple, Optional, Dict
from datetime import datetime
from pydantic import BaseModel

from memory.causal.core import CausalGraph, CausalEdge
from memory.causal.sequence import ActivationSequence
from memory.causal.statistics import CoOccurrenceMatrix
from memory.causal.config import CausalConfig


class CausalPrediction(BaseModel):
    """因果预测"""
    cause_id: str
    predicted_effects: List[Tuple[str, float, float]]  # [(effect_id, probability, confidence), ...]
    timestamp: datetime
    based_on_evidence: int  # 基于的证据数量


class CausalInference:
    """
    因果推断引擎
    
    从激活序列中学习因果关系并推断。
    """
    
    def __init__(
        self,
        graph: Optional[CausalGraph] = None,
        config: Optional[CausalConfig] = None
    ):
        self.graph = graph or CausalGraph()
        self.config = config or CausalConfig()
    
    def learn_from_sequence(
        self,
        sequence: ActivationSequence,
        temporal_window_ms: Optional[int] = None
    ) -> int:
        """
        从激活序列学习因果关系
        
        Returns:
            新增的因果边数量
        """
        window = temporal_window_ms or self.config.temporal_window_ms
        events = sorted(sequence.events, key=lambda e: e.timestamp)
        
        new_edges = 0
        
        for i, event1 in enumerate(events):
            for j, event2 in enumerate(events):
                if i >= j:
                    continue
                
                time_diff_ms = (event2.timestamp - event1.timestamp).total_seconds() * 1000
                
                # 检查是否在时间窗口内
                if 0 < time_diff_ms <= window:
                    cause = str(event1.neuron_id)
                    effect = str(event2.neuron_id)
                    
                    # 检查是否已经有边
                    if not self.graph.has_edge(cause, effect):
                        strength = self._compute_causal_strength(
                            cause, effect, events, time_diff_ms
                        )
                        
                        if strength >= self.config.min_causal_strength:
                            self.graph.add_causal_link(
                                cause, effect,
                                strength=strength,
                                temporal_delay_ms=int(time_diff_ms)
                            )
                            new_edges += 1
                    else:
                        # 更新现有边
                        strength = self._compute_causal_strength(
                            cause, effect, events, time_diff_ms
                        )
                        edge = self.graph.get_edge(cause, effect)
                        if edge:
                            edge.update(strength)
        
        return new_edges
    
    def _compute_causal_strength(
        self,
        cause: str,
        effect: str,
        events: List,
        time_diff_ms: float
    ) -> float:
        """
        计算因果强度
        
        使用多个因素：
        1. 时间接近度
        2. 激活频率
        3. 事件类型匹配
        """
        # 时间接近度（越近越强）
        temporal_score = 1.0 - (time_diff_ms / self.config.temporal_window_ms)
        temporal_score = max(0.1, temporal_score)
        
        # 事件类型匹配
        cause_events = [e for e in events if str(e.neuron_id) == cause]
        effect_events = [e for e in events if str(e.neuron_id) == effect]
        
        # 计算条件概率 P(effect | cause)
        co_occurrences = 0
        for ce in cause_events:
            for ee in effect_events:
                diff = abs((ee.timestamp - ce.timestamp).total_seconds() * 1000)
                if 0 < diff <= self.config.temporal_window_ms:
                    co_occurrences += 1
        
        p_effect_given_cause = co_occurrences / len(cause_events) if cause_events else 0
        p_effect = len(effect_events) / len(events) if events else 0
        
        # 提升度 (lift)
        if p_effect > 0:
            lift = p_effect_given_cause / p_effect
        else:
            lift = 1.0
        
        # 综合强度
        strength = (temporal_score * 0.3 + min(lift, 2.0) * 0.5 + p_effect_given_cause * 0.2)
        return min(1.0, max(0.0, strength))
    
    def detect_causal_direction(
        self,
        neuron_a: str,
        neuron_b: str,
        sequences: List[ActivationSequence]
    ) -> Tuple[str, str, float]:
        """
        检测因果方向
        
        判断 A→B 还是 B→A。
        返回: (cause, effect, confidence)
        """
        ab_strength = 0.0
        ba_strength = 0.0
        
        for sequence in sequences:
            ab_strength += self._compute_sequence_direction_strength(
                sequence, neuron_a, neuron_b
            )
            ba_strength += self._compute_sequence_direction_strength(
                sequence, neuron_b, neuron_a
            )
        
        if ab_strength > ba_strength:
            return (neuron_a, neuron_b, ab_strength / len(sequences) if sequences else 0)
        else:
            return (neuron_b, neuron_a, ba_strength / len(sequences) if sequences else 0)
    
    def _compute_sequence_direction_strength(
        self,
        sequence: ActivationSequence,
        cause: str,
        effect: str
    ) -> float:
        """计算序列中的方向强度"""
        cause_timestamps = []
        effect_timestamps = []
        
        for event in sequence.events:
            if str(event.neuron_id) == cause:
                cause_timestamps.append(event.timestamp)
            elif str(event.neuron_id) == effect:
                effect_timestamps.append(event.timestamp)
        
        # 统计 A 总是先于 B 出现的次数
        correct_order = 0
        total_pairs = 0
        
        for ct in cause_timestamps:
            for et in effect_timestamps:
                time_diff = (et - ct).total_seconds()
                if 0 < time_diff <= self.config.temporal_window_ms / 1000:
                    correct_order += 1
                total_pairs += 1
        
        return correct_order / total_pairs if total_pairs > 0 else 0.5
    
    def predict_effects(
        self,
        cause: str,
        top_k: Optional[int] = None
    ) -> CausalPrediction:
        """
        预测效果
        
        给定原因，预测可能的未来效果。
        """
        max_preds = top_k or self.config.max_predictions
        predictions = self.graph.infer_probable_effects(cause, max_preds)
        
        return CausalPrediction(
            cause_id=cause,
            predicted_effects=predictions,
            timestamp=datetime.now(),
            based_on_evidence=sum(
                e.evidence_count for e in self.graph.edges.values()
                if e.cause_id == cause
            )
        )
    
    def predict_causes(
        self,
        effect: str,
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float, float]]:
        """
        预测原因
        
        给定效果，反推可能的过去原因。
        """
        max_preds = top_k or self.config.max_predictions
        return self.graph.infer_probable_causes(effect, max_preds)
    
    def explain_connection(
        self,
        neuron_a: str,
        neuron_b: str
    ) -> Dict[str, any]:
        """
        解释两个神经元之间的连接
        
        返回连接的解释信息。
        """
        if self.graph.has_edge(neuron_a, neuron_b):
            cause, effect = neuron_a, neuron_b
        elif self.graph.has_edge(neuron_b, neuron_a):
            cause, effect = neuron_b, neuron_a
        else:
            return {"connected": False}
        
        edge = self.graph.get_edge(cause, effect)
        
        return {
            "connected": True,
            "direction": f"{cause} -> {effect}",
            "cause": cause,
            "effect": effect,
            "strength": edge.strength if edge else 0,
            "confidence": edge.confidence if edge else 0,
            "evidence_count": edge.evidence_count if edge else 0,
            "avg_temporal_delay_ms": edge.temporal_delay_ms if edge else 0
        }
    
    def get_causal_explanations(
        self,
        neuron_id: str,
        depth: int = 2
    ) -> Dict[str, List]:
        """
        获取关于神经元的因果解释
        
        包括：直接原因、直接效果、深层原因、深层效果。
        """
        explanations = {
            "direct_causes": self.graph.get_causal_parents(neuron_id)[:5],
            "direct_effects": self.graph.get_causal_children(neuron_id)[:5],
            "root_causes": [],
            "final_effects": []
        }
        
        # 追溯深层原因
        visited = set()
        def find_roots(current: str, path: List[str]):
            if len(path) >= depth:
                return
            parents = self.graph.get_causal_parents(current)
            if not parents:
                if path:
                    explanations["root_causes"].append(path[0])
                return
            for parent, _ in parents:
                if parent not in visited:
                    visited.add(parent)
                    find_roots(parent, path + [parent])
        
        visited.add(neuron_id)
        find_roots(neuron_id, [])
        
        # 追溯深层效果
        visited = {neuron_id}
        def find_effects(current: str, path: List[str]):
            if len(path) >= depth:
                return
            children = self.graph.get_causal_children(current)
            if not children:
                if path:
                    explanations["final_effects"].append(path[-1])
                return
            for child, _ in children:
                if child not in visited:
                    visited.add(child)
                    find_effects(child, path + [child])
        
        find_effects(neuron_id, [])
        
        return explanations
