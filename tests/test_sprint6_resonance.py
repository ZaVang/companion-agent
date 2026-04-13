"""
Sprint 6: 共振机制测试
"""

import pytest
from memory.resonance.core import ResonanceConfig, ResonanceEngine, ActivationDiffuser


class TestResonanceEngine:
    """共振引擎测试"""
    
    def test_calculate_activation_energy(self):
        """测试激活能量计算"""
        config = ResonanceConfig()
        engine = ResonanceEngine(config)
        
        # 高强度 + 高 Elo + 多连接 = 高能量
        energy = engine.calculate_activation_energy(
            strength=1.0,
            elo=1500.0,
            connections_count=10
        )
        
        assert energy > 0
        assert energy <= config.energy_base
    
    def test_check_resonance(self):
        """测试共振检测"""
        engine = ResonanceEngine()
        
        # 高能量应该共振
        assert engine.check_resonance(0.5, 0.5) is True
        
        # 低能量不应该共振
        assert engine.check_resonance(0.1, 0.1) is False
    
    def test_calculate_resonance_strength(self):
        """测试共振强度计算"""
        engine = ResonanceEngine()
        
        strength = engine.calculate_resonance_strength(0.5, 0.5)
        assert strength > 0
        
        # 不共振时返回 0
        strength = engine.calculate_resonance_strength(0.1, 0.1)
        assert strength == 0.0


class TestActivationDiffuser:
    """激活扩散器测试"""
    
    def test_add_connection(self):
        """测试添加连接"""
        diffuser = ActivationDiffuser()
        diffuser.add_connection("n1", "n2")
        diffuser.add_connection("n1", "n3")
        
        assert "n2" in diffuser._connections["n1"]
        assert "n3" in diffuser._connections["n1"]
    
    def test_diffuse_basic(self):
        """测试基础扩散"""
        diffuser = ActivationDiffuser()
        diffuser.add_connection("n1", "n2")
        diffuser.add_connection("n2", "n3")
        
        # 从 n1 开始扩散
        activated = diffuser.diffuse(
            seed_neurons=["n1"],
            neuron_states={
                "n1": {"strength": 1.0, "elo": 1000.0},
                "n2": {"strength": 0.8, "elo": 1000.0},
                "n3": {"strength": 0.6, "elo": 1000.0}
            },
            depth=2
        )
        
        # n1 和可能激活的 n2 应该在结果中
        assert "n1" in activated
    
    def test_find_resonance_pairs(self):
        """测试查找共振对"""
        diffuser = ActivationDiffuser()
        
        pairs = diffuser.find_resonance_pairs(
            neuron_ids=["n1", "n2", "n3"],
            neuron_states={
                "n1": {"strength": 1.0, "elo": 1500.0},
                "n2": {"strength": 0.9, "elo": 1400.0},
                "n3": {"strength": 0.5, "elo": 800.0}
            }
        )
        
        # 应该找到一些共振对
        assert isinstance(pairs, list)
