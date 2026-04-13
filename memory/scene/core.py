"""
场景敏感激活核心模块
"""

from typing import List, Dict, Optional, Set
from pydantic import BaseModel, Field
from datetime import datetime


class SceneContext(BaseModel):
    """场景上下文"""
    location: str = ""
    time_of_day: str = ""  # morning, afternoon, evening, night
    day_of_week: str = ""  # weekday, weekend
    activity: str = ""     # work, leisure, travel
    mood: str = ""         # happy, sad, neutral
    social_context: str = ""  # alone, with_friends, with_family, meeting
    
    def get_key(self) -> str:
        """获取场景唯一标识"""
        parts = [
            self.location,
            self.time_of_day,
            self.day_of_week,
            self.activity,
            self.mood,
            self.social_context
        ]
        return "|".join(p for p in parts if p)


class SceneSensitiveMapper:
    """场景敏感映射器"""
    
    def __init__(self):
        # neuron_id -> Set[scene_keys]
        self._neuron_scene_map: Dict[str, Set[str]] = {}
        # scene_key -> Set[neuron_ids]
        self._scene_neuron_map: Dict[str, Set[str]] = {}
    
    def map_neuron_to_scene(self, neuron_id: str, scene: SceneContext) -> None:
        """将神经元映射到场景"""
        scene_key = scene.get_key()
        
        if neuron_id not in self._neuron_scene_map:
            self._neuron_scene_map[neuron_id] = set()
        self._neuron_scene_map[neuron_id].add(scene_key)
        
        if scene_key not in self._scene_neuron_map:
            self._scene_neuron_map[scene_key] = set()
        self._scene_neuron_map[scene_key].add(neuron_id)
    
    def get_neurons_for_scene(self, scene: SceneContext) -> Set[str]:
        """获取适合当前场景的神经元"""
        scene_key = scene.get_key()
        
        # 精确匹配
        if scene_key in self._scene_neuron_map:
            return self._scene_neuron_map[scene_key]
        
        # 部分匹配
        matched = set()
        for stored_key, neurons in self._scene_neuron_map.items():
            if self._is_similar(scene_key, stored_key):
                matched.update(neurons)
        
        return matched
    
    def _is_similar(self, key1: str, key2: str) -> bool:
        """检查两个场景是否相似"""
        parts1 = set(key1.split("|"))
        parts2 = set(key2.split("|"))
        
        # 至少有一个非空部分相同
        common = parts1 & parts2
        non_empty = {p for p in common if p}
        return len(non_empty) >= 2


class SceneAwareRetrieval:
    """场景感知检索"""
    
    def __init__(self, mapper: Optional[SceneSensitiveMapper] = None):
        self.mapper = mapper or SceneSensitiveMapper()
    
    def calculate_scene_weight(
        self,
        neuron_id: str,
        current_scene: SceneContext,
        base_score: float = 1.0
    ) -> float:
        """计算场景权重"""
        scene_key = current_scene.get_key()
        
        if neuron_id not in self.mapper._neuron_scene_map:
            return base_score
        
        neuron_scenes = self.mapper._neuron_scene_map[neuron_id]
        
        if scene_key in neuron_scenes:
            return base_score * 1.5  # 精确匹配增强
        
        # 计算相似场景的匹配度
        similarity = 0.0
        for stored_key in neuron_scenes:
            sim = self._calculate_similarity(scene_key, stored_key)
            similarity = max(similarity, sim)
        
        return base_score * (1.0 + similarity * 0.5)
    
    def _calculate_similarity(self, key1: str, key2: str) -> float:
        """计算场景相似度"""
        parts1 = set(key1.split("|"))
        parts2 = set(key2.split("|"))
        
        if not parts1 or not parts2:
            return 0.0
        
        common = parts1 & parts2
        return len(common) / max(len(parts1), len(parts2))
    
    def rank_neurons_by_scene(
        self,
        neuron_ids: List[str],
        current_scene: SceneContext,
        base_scores: Optional[Dict[str, float]] = None
    ) -> List[str]:
        """根据场景对神经元排序"""
        scored = []
        for nid in neuron_ids:
            base = base_scores.get(nid, 1.0) if base_scores else 1.0
            scene_weight = self.calculate_scene_weight(nid, current_scene, base)
            scored.append((nid, scene_weight))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [nid for nid, _ in scored]
