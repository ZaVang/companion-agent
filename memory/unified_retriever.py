"""
统一检索系统

整合检索逻辑，支持：
- Elo 竞争机制
- 动态衰减
- 综合评分排序
- 场景敏感检索（Sprint 5）

检索评分公式:
    score = similarity × Elo_strength × decay_factor × recency_factor × scene_weight
"""

import math
from typing import Dict, List, Literal, Optional, Tuple, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, UUID1
import numpy as np

from memory.neuron import NeuronCell
from memory.engram import Engram
from memory.elo import EloCompetition, calculate_combat_score
from memory.decay import calculate_decay_rate, DecayScheduler, apply_decay
from memory.decay import calculate_emotion_aware_decay  # Sprint 7 集成
from memory.utils import now as utc_now, from_naive, ensure_aware

# Sprint 5 集成: 导入场景模块
from memory.scene import SceneContext, SceneSensitiveMapper, SceneAwareRetrieval


# ============== 配置常量 ==============

# 检索参数
DEFAULT_TOP_K = 5
DEFAULT_THRESHOLD = 0.3

# 评分权重
SIMILARITY_WEIGHT = 1.0
ELO_WEIGHT = 0.5
DECAY_WEIGHT = 0.3
RECENCY_WEIGHT = 0.2

# Recency 配置（近期偏好）
RECENCY_HALF_LIFE_DAYS = 7  # 7 天后 recency 减半


# ============== 数据模型 ==============

class RetrievalResult(BaseModel):
    """检索结果"""
    event_id: UUID1
    content: str
    event_type: str
    score: float
    similarity: float
    elo_strength: float
    decay_factor: float
    recency_factor: float
    timestamp: datetime


class RetrievalConfig(BaseModel):
    """检索配置"""
    top_k: int = DEFAULT_TOP_K
    threshold: float = DEFAULT_THRESHOLD
    similarity_weight: float = SIMILARITY_WEIGHT
    elo_weight: float = ELO_WEIGHT
    decay_weight: float = DECAY_WEIGHT
    recency_weight: float = RECENCY_WEIGHT
    recency_half_life_days: float = RECENCY_HALF_LIFE_DAYS
    enable_elo_competition: bool = True
    enable_decay: bool = True
    # Sprint 5 集成: 场景敏感检索
    enable_scene_sensitivity: bool = True
    scene_weight: float = 0.2  # 场景权重


# ============== 核心算法 ==============

def calculate_recency_factor(
    event_timestamp: datetime,
    reference_time: Optional[datetime] = None,
    half_life_days: float = RECENCY_HALF_LIFE_DAYS
) -> float:
    """
    计算时间衰减因子
    
    公式: factor = 0.5^(days / half_life)
    
    Args:
        event_timestamp: 事件时间戳
        reference_time: 参考时间（默认当前时间）
        half_life_days: 半衰期（天）
    
    Returns:
        时间因子 [0, 1]
    """
    if reference_time is None:
        reference_time = utc_now()
    
    days_diff = (reference_time - event_timestamp).total_seconds() / (24 * 3600)
    
    if days_diff < 0:
        days_diff = 0  # 未来事件视为现在
    
    return 0.5 ** (days_diff / half_life_days)


def calculate_combined_score(
    similarity: float,
    elo_strength: float,
    decay_rate: float,
    time_days: float,
    config: Optional[RetrievalConfig] = None
) -> float:
    """
    计算综合检索评分
    
    公式:
        decay_factor = decay_rate^time_days
        score = (w1 * similarity) * (w2 * elo_strength) * (w3 * decay_factor) * (w4 * recency)
    
    Args:
        similarity: 语义相似度 [0, 1]
        elo_strength: Elo 战斗力（归一化后）
        decay_rate: 衰减率
        time_days: 距离创建的天数
        config: 检索配置
    
    Returns:
        综合评分 [0, 1]
    """
    if config is None:
        config = RetrievalConfig()
    
    # 计算衰减因子
    decay_factor = decay_rate ** time_days
    
    # 归一化 Elo 战斗力到 [0, 1]
    # 假设 Elo 范围 100-2000，映射到 0.5-1.5
    normalized_elo = 0.5 + 0.5 * (elo_strength - 100) / 1900
    normalized_elo = max(0.1, min(2.0, normalized_elo))
    
    # 综合评分
    score = (
        config.similarity_weight * similarity +
        config.elo_weight * normalized_elo +
        config.decay_weight * decay_factor
    ) / (config.similarity_weight + config.elo_weight + config.decay_weight)
    
    return score


class UnifiedRetriever:
    """
    统一检索器
    
    整合 Elo 竞争和动态衰减的检索逻辑。
    """
    
    def __init__(
        self,
        retrieval_config: Optional[RetrievalConfig] = None,
        elo_competitor: Optional[EloCompetition] = None,
        decay_scheduler: Optional[DecayScheduler] = None,
        # Sprint 5 集成: 场景敏感检索
        scene_retrieval: Optional[SceneAwareRetrieval] = None
    ):
        self.config = retrieval_config or RetrievalConfig()
        self.elo = elo_competitor or EloCompetition()
        self.decay = decay_scheduler or DecayScheduler()
        # Sprint 5 集成
        self.scene_retrieval = scene_retrieval or SceneAwareRetrieval()
    
    def register_neuron(
        self,
        neuron: NeuronCell,
        impact_score: float = 0.5,
        # Sprint 5 集成: 场景注册
        scene: Optional[SceneContext] = None
    ) -> None:
        """
        注册神经元到检索系统
        
        Args:
            neuron: 神经元
            impact_score: 冲击力评分
            scene: 场景上下文（可选）
        """
        # 注册到 Elo 系统
        self.elo.register_neuron(neuron.event_id)
        
        # 更新神经元的衰减率（Sprint 7: 支持情绪感知）
        # 如果神经元有情绪信息，使用情绪感知衰减
        if hasattr(neuron, 'emotional_arousal') and neuron.emotional_arousal != 0.5:
            decay_rate = calculate_emotion_aware_decay(
                event_type=neuron.event_type,
                emotional_valence=getattr(neuron, 'emotional_valence', 0.0),
                emotional_arousal=neuron.emotional_arousal,
                base_decay=0.995
            )
        else:
            decay_rate = calculate_decay_rate(neuron.event_type, impact_score)
        
        neuron.decay_rate = decay_rate
        neuron.impact_score = impact_score
        
        # Sprint 5 集成: 注册到场景系统
        if scene and self.config.enable_scene_sensitivity:
            self.scene_retrieval.mapper.map_neuron_to_scene(
                str(neuron.event_id), scene
            )
    
    def calculate_retrieval_score(
        self,
        neuron: NeuronCell,
        similarity: float,
        reference_time: Optional[datetime] = None,
        # Sprint 5 集成: 场景感知
        current_scene: Optional[SceneContext] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        计算单个神经元的检索评分
        
        Args:
            neuron: 神经元
            similarity: 语义相似度
            reference_time: 参考时间
            current_scene: 当前场景（可选）
        
        Returns:
            (score, breakdown_dict)
        """
        if reference_time is None:
            reference_time = utc_now()
        
        # 获取 Elo 战斗力
        elo_strength = self.elo.get_combat_power(neuron.event_id)
        
        # 计算时间差
        if isinstance(neuron.create_time, str):
            create_time = datetime.strptime(neuron.create_time, '%Y-%m-%d %H:%M:%S')
        else:
            create_time = neuron.create_time
        time_days = (reference_time - create_time).total_seconds() / (24 * 3600)
        
        # 计算各因子
        decay_factor = neuron.decay_rate ** max(0, time_days)
        recency_factor = calculate_recency_factor(
            create_time, reference_time, self.config.recency_half_life_days
        )
        
        # 归一化 Elo
        normalized_elo = 0.5 + 0.5 * (elo_strength - 100) / 1900
        normalized_elo = max(0.1, min(2.0, normalized_elo))
        
        # Sprint 5 集成: 计算场景权重
        scene_weight = 1.0
        if current_scene and self.config.enable_scene_sensitivity:
            scene_weight = self.scene_retrieval.calculate_scene_weight(
                str(neuron.event_id), current_scene
            )
        
        # 加权综合评分
        total_weight = (
            self.config.similarity_weight +
            self.config.elo_weight +
            self.config.decay_weight +
            self.config.recency_weight +
            (self.config.scene_weight if self.config.enable_scene_sensitivity else 0)
        )
        
        score = (
            self.config.similarity_weight * similarity +
            self.config.elo_weight * normalized_elo +
            self.config.decay_weight * decay_factor +
            self.config.recency_weight * recency_factor +
            (self.config.scene_weight * scene_weight if self.config.enable_scene_sensitivity else 0)
        ) / total_weight
        
        breakdown = {
            'similarity': similarity,
            'elo_strength': normalized_elo,
            'decay_factor': decay_factor,
            'recency_factor': recency_factor,
            'scene_weight': scene_weight,
        }
        
        return score, breakdown
    
    def retrieve(
        self,
        neurons: List[NeuronCell],
        query_embedding: np.ndarray,
        embedding_manager: Any,
        event_stream: Any,
        reference_time: Optional[datetime] = None,
        # Sprint 5 集成: 场景感知检索
        current_scene: Optional[SceneContext] = None
    ) -> List[RetrievalResult]:
        """
        执行统一检索
        
        Args:
            neurons: 候选神经元列表
            query_embedding: 查询向量
            embedding_manager: embedding 管理器（有 calculate_similarities 方法）
            event_stream: 事件流（有 get_event 方法）
            reference_time: 参考时间
            current_scene: 当前场景（可选）
        
        Returns:
            按评分排序的检索结果
        """
        if reference_time is None:
            reference_time = utc_now()
        
        if not neurons:
            return []
        
        # 1. 获取所有神经元的 embedding IDs
        neuron_ids = [n.event_id for n in neurons]
        
        # 2. 批量计算相似度
        similarities = embedding_manager.calculate_similarities(query_embedding, neuron_ids)
        
        # 3. 计算每个神经元的检索评分
        scored_neurons = []
        for neuron, similarity in zip(neurons, similarities):
            if similarity < self.config.threshold:
                continue
            
            score, breakdown = self.calculate_retrieval_score(
                neuron, similarity, reference_time, current_scene
            )
            
            # 获取事件内容
            try:
                event = event_stream.get_event(neuron.event_id)
                content = event.content
            except Exception:
                content = ""
            
            scored_neurons.append({
                'neuron': neuron,
                'score': score,
                'similarity': breakdown['similarity'],
                'elo_strength': breakdown['elo_strength'],
                'decay_factor': breakdown['decay_factor'],
                'recency_factor': breakdown['recency_factor'],
                'scene_weight': breakdown.get('scene_weight', 1.0),
                'content': content,
            })
        
        # 4. 排序
        scored_neurons.sort(key=lambda x: x['score'], reverse=True)
        
        # 5. 取 top-k
        top_neurons = scored_neurons[:self.config.top_k]
        
        # 6. 更新 Elo（竞争机制）
        if self.config.enable_elo_competition:
            all_ids = [n.event_id for n in neurons]
            retrieved_ids = [n['neuron'].event_id for n in top_neurons]
            self.elo.update_after_retrieval(
                retrieved_ids=retrieved_ids,
                all_candidate_ids=all_ids,
                retrieval_score=top_neurons[0]['score'] if top_neurons else 0.5
            )
        
        # 7. 构建结果
        results = []
        for item in top_neurons:
            neuron = item['neuron']
            results.append(RetrievalResult(
                event_id=neuron.event_id,
                content=item['content'],
                event_type=neuron.event_type,
                score=item['score'],
                similarity=item['similarity'],
                elo_strength=item['elo_strength'],
                decay_factor=item['decay_factor'],
                recency_factor=item['recency_factor'],
                timestamp=neuron.create_time if isinstance(neuron.create_time, datetime) else datetime.strptime(neuron.create_time, '%Y-%m-%d %H:%M:%S')
            ))
        
        return results
    
    def apply_decay_to_all(
        self,
        neurons: List[NeuronCell],
        reference_time: Optional[datetime] = None
    ) -> Dict[UUID1, float]:
        """
        对所有神经元应用衰减
        
        Args:
            neurons: 神经元列表
            reference_time: 参考时间
        
        Returns:
            {neuron_id: new_strength}
        """
        if reference_time is None:
            reference_time = utc_now()
        
        updates = {}
        for neuron in neurons:
            old_strength = neuron.strength
            new_strength = self.decay.apply_decay_to_neuron(
                neuron.event_id,
                neuron.strength,
                reference_time
            )
            updates[neuron.event_id] = new_strength
        
        return updates


# ============== 全局实例 ==============

_default_retriever: Optional[UnifiedRetriever] = None


def get_global_retriever() -> UnifiedRetriever:
    """获取全局统一检索器"""
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = UnifiedRetriever()
    return _default_retriever


def reset_global_retriever() -> None:
    """重置全局统一检索器"""
    global _default_retriever
    if _default_retriever:
        _default_retriever = UnifiedRetriever()
