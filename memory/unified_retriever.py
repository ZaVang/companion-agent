"""
统一检索系统

整合检索逻辑，支持：
- Elo 竞争机制
- 动态衰减
- 综合评分排序

检索评分公式:
    score = similarity × Elo_strength × decay_factor × recency_factor
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
        reference_time = datetime.now()
    
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
        decay_scheduler: Optional[DecayScheduler] = None
    ):
        self.config = retrieval_config or RetrievalConfig()
        self.elo = elo_competitor or EloCompetition()
        self.decay = decay_scheduler or DecayScheduler()
    
    def register_neuron(
        self,
        neuron: NeuronCell,
        impact_score: float = 0.5
    ) -> None:
        """
        注册神经元到检索系统
        
        Args:
            neuron: 神经元
            impact_score: 冲击力评分
        """
        # 注册到 Elo 系统
        self.elo.register_neuron(neuron.event_id)
        
        # 注册到衰减系统
        self.decay.register_neuron(
            event_id=neuron.event_id,
            event_type=neuron.event_type,
            impact_score=impact_score,
            created_at=neuron.create_time
        )
        
        # 更新神经元的衰减率
        decay_rate = calculate_decay_rate(neuron.event_type, impact_score)
        neuron.decay_rate = decay_rate
        neuron.impact_score = impact_score
    
    def calculate_retrieval_score(
        self,
        neuron: NeuronCell,
        similarity: float,
        reference_time: Optional[datetime] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        计算单个神经元的检索评分
        
        Args:
            neuron: 神经元
            similarity: 语义相似度
            reference_time: 参考时间
        
        Returns:
            (score, breakdown_dict)
        """
        if reference_time is None:
            reference_time = datetime.now()
        
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
        
        # 加权综合评分
        score = (
            self.config.similarity_weight * similarity +
            self.config.elo_weight * normalized_elo +
            self.config.decay_weight * decay_factor +
            self.config.recency_weight * recency_factor
        ) / (
            self.config.similarity_weight +
            self.config.elo_weight +
            self.config.decay_weight +
            self.config.recency_weight
        )
        
        breakdown = {
            'similarity': similarity,
            'elo_strength': normalized_elo,
            'decay_factor': decay_factor,
            'recency_factor': recency_factor,
        }
        
        return score, breakdown
    
    def retrieve(
        self,
        neurons: List[NeuronCell],
        query_embedding: np.ndarray,
        embedding_manager: Any,
        event_stream: Any,
        reference_time: Optional[datetime] = None
    ) -> List[RetrievalResult]:
        """
        执行统一检索
        
        Args:
            neurons: 候选神经元列表
            query_embedding: 查询向量
            embedding_manager: embedding 管理器（有 calculate_similarities 方法）
            event_stream: 事件流（有 get_event 方法）
            reference_time: 参考时间
        
        Returns:
            按评分排序的检索结果
        """
        if reference_time is None:
            reference_time = datetime.now()
        
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
                neuron, similarity, reference_time
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
            reference_time = datetime.now()
        
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
