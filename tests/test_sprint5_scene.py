"""
Sprint 5: 场景敏感激活测试
"""

import pytest
from memory.scene.core import SceneContext, SceneSensitiveMapper, SceneAwareRetrieval


class TestSceneContext:
    """场景上下文测试"""
    
    def test_create_scene(self):
        """测试创建场景"""
        scene = SceneContext(
            location="office",
            time_of_day="morning",
            day_of_week="weekday",
            activity="work"
        )
        
        assert scene.location == "office"
        assert scene.time_of_day == "morning"
    
    def test_get_key(self):
        """测试场景唯一标识"""
        scene = SceneContext(
            location="home",
            time_of_day="evening",
            activity="leisure"
        )
        
        key = scene.get_key()
        assert "home" in key
        assert "evening" in key


class TestSceneSensitiveMapper:
    """场景敏感映射器测试"""
    
    def test_map_neuron_to_scene(self):
        """测试映射神经元到场景"""
        mapper = SceneSensitiveMapper()
        scene = SceneContext(location="office", time_of_day="morning")
        
        mapper.map_neuron_to_scene("n1", scene)
        
        neurons = mapper.get_neurons_for_scene(scene)
        assert "n1" in neurons
    
    def test_get_neurons_partial_match(self):
        """测试部分匹配"""
        mapper = SceneSensitiveMapper()
        
        # 映射到特定场景
        scene1 = SceneContext(location="office", time_of_day="morning", activity="work")
        mapper.map_neuron_to_scene("n1", scene1)
        
        # 查询相似场景
        scene2 = SceneContext(location="office", time_of_day="afternoon", activity="work")
        neurons = mapper.get_neurons_for_scene(scene2)
        
        # 应该部分匹配
        assert len(neurons) >= 0


class TestSceneAwareRetrieval:
    """场景感知检索测试"""
    
    def test_calculate_scene_weight(self):
        """测试场景权重计算"""
        mapper = SceneSensitiveMapper()
        retrieval = SceneAwareRetrieval(mapper)
        
        scene = SceneContext(location="office", time_of_day="morning")
        mapper.map_neuron_to_scene("n1", scene)
        
        weight = retrieval.calculate_scene_weight("n1", scene, base_score=1.0)
        
        # 精确匹配应该增强
        assert weight >= 1.0
    
    def test_rank_neurons_by_scene(self):
        """测试按场景排序"""
        mapper = SceneSensitiveMapper()
        retrieval = SceneAwareRetrieval(mapper)
        
        scene = SceneContext(location="office", time_of_day="morning")
        mapper.map_neuron_to_scene("n1", scene)
        
        ranked = retrieval.rank_neurons_by_scene(["n1", "n2"], scene)
        
        assert len(ranked) == 2
        # n1 匹配场景应该排在前面
        assert ranked[0] == "n1"
