"""
Sprint 7: 情绪与冲击力测试
"""

import pytest
from memory.emotion.core import EmotionalImpact, ImpactMapper


class TestEmotionalImpact:
    """情绪影响测试"""
    
    def test_emotional_intensity(self):
        """测试情绪强度计算"""
        # 高唤醒 + 中性效价 = 高强度
        emotion = EmotionalImpact(valence=0.0, arousal=0.9)
        assert emotion.emotional_intensity == pytest.approx(0.9, rel=0.1)
        
        # 低唤醒 = 低强度
        emotion = EmotionalImpact(valence=0.0, arousal=0.2)
        assert emotion.emotional_intensity < 0.3
    
    def test_is_positive(self):
        """测试正面情绪判断"""
        emotion = EmotionalImpact(valence=0.5)
        assert emotion.is_positive is True
        
        emotion = EmotionalImpact(valence=0.0)
        assert emotion.is_positive is False
    
    def test_is_negative(self):
        """测试负面情绪判断"""
        emotion = EmotionalImpact(valence=-0.5)
        assert emotion.is_negative is True
        
        emotion = EmotionalImpact(valence=0.0)
        assert emotion.is_negative is False
    
    def test_to_impact_score(self):
        """测试冲击力评分转换"""
        # 高唤醒 + 中性效价 = 高冲击力（因为 emotional_intensity = arousal * (1 - |valence|)）
        emotion = EmotionalImpact(valence=0.0, arousal=0.9)
        score = emotion.to_impact_score()
        
        assert score > 0.5
        
        # 高唤醒 + 高效价绝对值 = 中等冲击力
        emotion2 = EmotionalImpact(valence=-0.8, arousal=0.9)
        score2 = emotion2.to_impact_score()
        
        assert score2 > 0.1  # 至少有一定冲击力


class TestImpactMapper:
    """冲击力映射器测试"""
    
    def test_map_high_impact_to_decay(self):
        """测试高冲击力衰减映射"""
        mapper = ImpactMapper()
        
        # 高冲击力 = 慢衰减
        decay = mapper.map_impact_to_decay(
            impact_score=0.9,
            base_decay=0.995
        )
        
        # 应该比基础衰减更慢
        assert decay >= 0.995
    
    def test_map_low_impact_to_decay(self):
        """测试低冲击力衰减映射"""
        mapper = ImpactMapper()
        
        # 低冲击力 = 快衰减
        decay = mapper.map_impact_to_decay(
            impact_score=0.1,
            base_decay=0.995
        )
        
        # 应该比基础衰减更快
        assert decay <= 0.995
    
    def test_map_emotion_to_decay(self):
        """测试从情绪对象映射衰减"""
        mapper = ImpactMapper()
        
        # 高冲击情绪
        emotion = EmotionalImpact(valence=-0.8, arousal=0.9)
        decay = mapper.map_emotion_to_decay(emotion, base_decay=0.995)
        
        assert decay >= 0.995
    
    def test_calculate_retention_bonus(self):
        """测试保留加成计算"""
        mapper = ImpactMapper()
        
        # 高冲击力 = 高加成
        bonus = mapper.calculate_retention_bonus(0.9)
        assert bonus > 0
        
        # 低冲击力 = 无加成
        bonus = mapper.calculate_retention_bonus(0.1)
        assert bonus == 0.0
    
    def test_get_decay_multiplier(self):
        """测试衰减乘数"""
        mapper = ImpactMapper()
        
        # 高冲击力
        mult, desc = mapper.get_decay_multiplier(0.9)
        assert mult > 1.0
        assert "慢" in desc
        
        # 低冲击力
        mult, desc = mapper.get_decay_multiplier(0.1)
        assert mult < 1.0
        assert "快" in desc
        
        # 中等冲击力
        mult, desc = mapper.get_decay_multiplier(0.5)
        assert mult == 1.0
        assert "正常" in desc
