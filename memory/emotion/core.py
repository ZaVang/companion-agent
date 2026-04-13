"""
情绪与冲击力核心模块
"""

from typing import Optional, Tuple
from pydantic import BaseModel


class EmotionalImpact(BaseModel):
    """情绪影响"""
    valence: float = 0.0      # 效价 [-1, 1], -1=负面, 0=中性, 1=正面
    arousal: float = 0.5     # 唤醒度 [0, 1], 0=平静, 1=激动
    dominance: float = 0.5    # 主导性 [0, 1], 0=被动, 1=主动
    
    @property
    def emotional_intensity(self) -> float:
        """情绪强度 = 唤醒度 * (1 - |效价|)"""
        return self.arousal * (1.0 - abs(self.valence))
    
    @property
    def is_positive(self) -> bool:
        """是否正面情绪"""
        return self.valence > 0.3
    
    @property
    def is_negative(self) -> bool:
        """是否负面情绪"""
        return self.valence < -0.3
    
    def to_impact_score(self) -> float:
        """
        转换为冲击力评分
        
        冲击力与情绪强度和唤醒度正相关
        """
        base_intensity = self.emotional_intensity
        valence_boost = abs(self.valence) * 0.2
        
        return min(1.0, base_intensity + valence_boost)


class ImpactMapper:
    """冲击力映射器"""
    
    def __init__(
        self,
        high_impact_floor: float = 0.7,
        low_impact_ceiling: float = 0.3
    ):
        """
        Args:
            high_impact_floor: 高冲击力下限（高于此值衰减减慢）
            low_impact_ceiling: 低冲击力上限（低于此值衰减加快）
        """
        self.high_impact_floor = high_impact_floor
        self.low_impact_ceiling = low_impact_ceiling
    
    def map_impact_to_decay(
        self,
        impact_score: float,
        base_decay: float
    ) -> float:
        """
        将冲击力映射到衰减率
        
        高冲击力 -> 慢衰减
        低冲击力 -> 快衰减
        
        Args:
            impact_score: 冲击力评分 [0, 1]
            base_decay: 基础衰减率
        
        Returns:
            调整后的衰减率
        """
        if impact_score >= self.high_impact_floor:
            # 高冲击力：衰减减慢 10-20%
            adjustment = 1.0 + (impact_score - self.high_impact_floor) * 0.2
            return min(0.9999, base_decay * adjustment)
        
        elif impact_score <= self.low_impact_ceiling:
            # 低冲击力：衰减加快 10-30%
            adjustment = 1.0 - (self.low_impact_ceiling - impact_score) * 0.3
            return max(0.98, base_decay * adjustment)
        
        else:
            # 中等冲击力：基础衰减
            return base_decay
    
    def map_emotion_to_decay(
        self,
        emotion: EmotionalImpact,
        base_decay: float
    ) -> float:
        """从情绪对象映射到衰减率"""
        return self.map_impact_to_decay(emotion.to_impact_score(), base_decay)
    
    def calculate_retention_bonus(
        self,
        impact_score: float
    ) -> float:
        """
        计算保留加成
        
        高冲击力记忆获得额外的 Elo 加成
        """
        if impact_score >= self.high_impact_floor:
            return (impact_score - self.high_impact_floor) * 0.5
        return 0.0
    
    def get_decay_multiplier(
        self,
        impact_score: float
    ) -> Tuple[float, str]:
        """
        获取衰减乘数及其描述
        
        Returns:
            (multiplier, description)
        """
        if impact_score >= self.high_impact_floor:
            return 1.2, "慢衰减"
        elif impact_score <= self.low_impact_ceiling:
            return 0.7, "快衰减"
        else:
            return 1.0, "正常衰减"
