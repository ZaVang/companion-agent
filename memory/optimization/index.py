"""
索引系统

提供多种索引类型：
1. VectorIndex: 向量索引（基于 embedding 相似度）
2. TimeIndex: 时间索引（基于时间范围）
3. TagIndex: 标签索引（基于标签/分类）
"""

from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import bisect
import numpy as np


class IndexConfig(BaseModel):
    """索引配置"""
    vector_dim: int = 384              # 向量维度
    vector_threshold: float = 0.7    # 向量相似度阈值
    time_buckets: int = 24            # 时间分桶数（每小时一个桶）
    max_results: int = 100            # 最大返回结果数
    
    # 缓存配置
    enable_cache: bool = True
    cache_size: int = 1000
    
    # 更新配置
    rebuild_interval_hours: int = 6   # 重建间隔


@dataclass
class IndexedNeuron:
    """索引神经元"""
    neuron_id: str
    embedding: Optional[np.ndarray] = None
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Set[str] = field(default_factory=set)
    strength: float = 1.0
    elo: float = 1000.0
    event_type: str = "unknown"
    metadata: Dict = field(default_factory=dict)


class VectorIndex:
    """
    向量索引
    
    使用简单的余弦相似度实现。
    实际生产环境可替换为 FAISS、Annoy 等专业向量索引库。
    """
    
    def __init__(self, dim: int = 384, threshold: float = 0.7):
        self.dim = dim
        self.threshold = threshold
        
        self._vectors: Dict[str, np.ndarray] = {}  # neuron_id -> embedding
        self._inverse_idx: Dict[str, List[str]] = {}  # 用于快速查找
    
    def add(self, neuron_id: str, embedding: np.ndarray) -> None:
        """添加向量"""
        if embedding.shape[0] != self.dim:
            # 调整维度
            if embedding.shape[0] > self.dim:
                embedding = embedding[:self.dim]
            else:
                embedding = np.pad(embedding, (0, self.dim - embedding.shape[0]))
        
        self._vectors[neuron_id] = embedding
    
    def remove(self, neuron_id: str) -> None:
        """移除向量"""
        self._vectors.pop(neuron_id, None)
    
    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
        threshold: float = None
    ) -> List[Tuple[str, float]]:
        """
        搜索最相似的向量
        
        Returns:
            List[(neuron_id, similarity)]
        """
        if threshold is None:
            threshold = self.threshold
        
        if not self._vectors:
            return []
        
        # 调整查询向量维度
        if len(query.shape) == 1:
            query = query.reshape(1, -1)
        
        if query.shape[1] != self.dim:
            # 调整维度
            if query.shape[1] > self.dim:
                query = query[:, :self.dim]
            else:
                query = np.pad(query, ((0, 0), (0, self.dim - query.shape[1])))
        
        results = []
        for neuron_id, vec in self._vectors.items():
            similarity = self._cosine_similarity(query[0], vec)
            if similarity >= threshold:
                results.append((neuron_id, float(similarity)))
        
        # 排序并返回 top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def rebuild(self, neurons: List[Tuple[str, np.ndarray]]) -> None:
        """重建索引"""
        self._vectors.clear()
        for neuron_id, embedding in neurons:
            self.add(neuron_id, embedding)


class TimeIndex:
    """
    时间索引
    
    使用分桶策略实现快速时间范围查询。
    """
    
    def __init__(self, num_buckets: int = 24):
        self.num_buckets = num_buckets
        self._buckets: Dict[int, List[str]] = {i: [] for i in range(num_buckets)}
        self._neuron_times: Dict[str, datetime] = {}
        self._min_time: Optional[datetime] = None
        self._max_time: Optional[datetime] = None
    
    def add(self, neuron_id: str, timestamp: datetime) -> None:
        """添加时间记录"""
        self._neuron_times[neuron_id] = timestamp
        
        # 更新全局时间范围
        if self._min_time is None or timestamp < self._min_time:
            self._min_time = timestamp
        if self._max_time is None or timestamp > self._max_time:
            self._max_time = timestamp
        
        # 计算桶索引
        bucket_idx = self._get_bucket_index(timestamp)
        if bucket_idx not in self._buckets:
            self._buckets[bucket_idx] = []
        self._buckets[bucket_idx].append(neuron_id)
    
    def remove(self, neuron_id: str) -> None:
        """移除时间记录"""
        timestamp = self._neuron_times.pop(neuron_id, None)
        if timestamp:
            bucket_idx = self._get_bucket_index(timestamp)
            if bucket_idx in self._buckets:
                self._buckets[bucket_idx] = [
                    n for n in self._buckets[bucket_idx] if n != neuron_id
                ]
    
    def _get_bucket_index(self, timestamp: datetime) -> int:
        """计算桶索引"""
        if self._min_time is None or self._max_time is None:
            return 0
        
        total_seconds = (self._max_time - self._min_time).total_seconds()
        if total_seconds == 0:
            return 0
        
        elapsed = (timestamp - self._min_time).total_seconds()
        bucket = int((elapsed / total_seconds) * self.num_buckets)
        return min(bucket, self.num_buckets - 1)
    
    def search_range(
        self,
        start_time: datetime,
        end_time: datetime,
        inclusive: str = "both"
    ) -> List[str]:
        """
        搜索时间范围内的神经元
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            inclusive: "both", "left", "right", "neither"
        
        Returns:
            匹配的 neuron_id 列表
        """
        results = []
        
        for neuron_id, timestamp in self._neuron_times.items():
            if self._in_range(timestamp, start_time, end_time, inclusive):
                results.append(neuron_id)
        
        return results
    
    def _in_range(
        self,
        t: datetime,
        start: datetime,
        end: datetime,
        inclusive: str
    ) -> bool:
        """检查时间是否在范围内"""
        if inclusive == "both":
            return start <= t <= end
        elif inclusive == "left":
            return start <= t < end
        elif inclusive == "right":
            return start < t <= end
        else:
            return start < t < end
    
    def get_recent(self, hours: int = 24) -> List[str]:
        """获取最近 N 小时的神经元"""
        if self._max_time is None:
            return []
        
        cutoff = self._max_time - timedelta(hours=hours)
        return self.search_range(cutoff, self._max_time, inclusive="right")


class TagIndex:
    """
    标签索引
    
    支持多标签索引和前缀搜索。
    """
    
    def __init__(self):
        self._tag_to_neurons: Dict[str, Set[str]] = {}  # tag -> neuron_ids
        self._neuron_to_tags: Dict[str, Set[str]] = {}   # neuron_id -> tags
    
    def add(self, neuron_id: str, tags: Set[str]) -> None:
        """添加标签"""
        if neuron_id not in self._neuron_to_tags:
            self._neuron_to_tags[neuron_id] = set()
        
        for tag in tags:
            if tag not in self._tag_to_neurons:
                self._tag_to_neurons[tag] = set()
            self._tag_to_neurons[tag].add(neuron_id)
            self._neuron_to_tags[neuron_id].add(tag)
    
    def remove(self, neuron_id: str, tags: Set[str] = None) -> None:
        """移除标签"""
        if neuron_id not in self._neuron_to_tags:
            return
        
        if tags is None:
            # 移除所有标签
            for tag in self._neuron_to_tags[neuron_id]:
                self._tag_to_neurons[tag].discard(neuron_id)
            self._neuron_to_tags.pop(neuron_id)
        else:
            for tag in tags:
                self._tag_to_neurons[tag].discard(neuron_id)
                self._neuron_to_tags[neuron_id].discard(tag)
    
    def search(self, tags: List[str], match_all: bool = False) -> Set[str]:
        """
        搜索标签匹配的神经元
        
        Args:
            tags: 标签列表
            match_all: True 表示必须匹配所有标签，False 表示匹配任意标签
        
        Returns:
            匹配的 neuron_id 集合
        """
        if not tags:
            return set()
        
        if match_all:
            # AND 查询
            result = None
            for tag in tags:
                if tag in self._tag_to_neurons:
                    if result is None:
                        result = self._tag_to_neurons[tag].copy()
                    else:
                        result &= self._tag_to_neurons[tag]
                else:
                    return set()
            return result or set()
        else:
            # OR 查询
            result = set()
            for tag in tags:
                if tag in self._tag_to_neurons:
                    result |= self._tag_to_neurons[tag]
            return result
    
    def get_tags(self, neuron_id: str) -> Set[str]:
        """获取神经元的所有标签"""
        return self._neuron_to_tags.get(neuron_id, set())
    
    def get_all_tags(self) -> List[str]:
        """获取所有标签"""
        return list(self._tag_to_neurons.keys())


class MemoryIndex:
    """
    统一内存索引
    
    整合向量索引、时间索引和标签索引。
    """
    
    def __init__(self, config: Optional[IndexConfig] = None):
        self.config = config or IndexConfig()
        
        self.vector = VectorIndex(
            dim=self.config.vector_dim,
            threshold=self.config.vector_threshold
        )
        self.time = TimeIndex(num_buckets=self.config.time_buckets)
        self.tag = TagIndex()
        
        # 元数据存储
        self._metadata: Dict[str, Dict] = {}
        
        # 缓存
        self._cache: Dict[str, Any] = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def add_neuron(
        self,
        neuron_id: str,
        embedding: np.ndarray = None,
        timestamp: datetime = None,
        tags: Set[str] = None,
        strength: float = 1.0,
        elo: float = 1000.0,
        event_type: str = "unknown",
        **metadata
    ) -> None:
        """添加神经元到索引"""
        if timestamp is None:
            timestamp = datetime.now()
        if tags is None:
            tags = set()
        
        # 添加到各索引
        if embedding is not None:
            self.vector.add(neuron_id, embedding)
        
        self.time.add(neuron_id, timestamp)
        
        if tags:
            self.tag.add(neuron_id, tags)
        
        # 保存元数据
        self._metadata[neuron_id] = {
            'strength': strength,
            'elo': elo,
            'event_type': event_type,
            'timestamp': timestamp,
            **metadata
        }
    
    def remove_neuron(self, neuron_id: str) -> None:
        """从索引移除神经元"""
        self.vector.remove(neuron_id)
        self.time.remove(neuron_id)
        self._metadata.pop(neuron_id, None)
        self._cache.clear()
    
    def search(
        self,
        query: np.ndarray = None,
        time_range: Tuple[datetime, datetime] = None,
        tags: List[str] = None,
        tags_match_all: bool = False,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        混合搜索
        
        结合向量、时间、标签索引进行搜索。
        """
        result_sets = []
        weights = []
        
        # 向量搜索
        if query is not None:
            vector_results = self.vector.search(query, top_k=top_k)
            if vector_results:
                result_sets.append(dict(vector_results))
                weights.append(0.5)
        
        # 时间搜索
        if time_range is not None:
            time_results = self.time.search_range(time_range[0], time_range[1])
            if time_results:
                result_sets.append({rid: 1.0 for rid in time_results})
                weights.append(0.2)
        
        # 标签搜索
        if tags:
            tag_results = self.tag.search(tags, match_all=tags_match_all)
            if tag_results:
                result_sets.append({rid: 1.0 for rid in tag_results})
                weights.append(0.3)
        
        if not result_sets:
            return []
        
        # 归一化权重
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # 合并结果
        all_ids = set()
        for rs in result_sets:
            all_ids.update(rs.keys())
        
        scores = []
        for neuron_id in all_ids:
            score = 0.0
            for rs, w in zip(result_sets, weights):
                score += rs.get(neuron_id, 0) * w
            scores.append((neuron_id, score))
        
        # 排序并返回
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def search_by_text(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        文本关键词搜索

        基于存储的 metadata.event_type 和 tag 文本进行匹配。
        用于验收命令: ms.index.search_by_text("关键词")

        Args:
            query: 搜索关键词
            top_k: 返回最多 top_k 个结果

        Returns:
            List[(neuron_id, score)] 按 score 降序排列
        """
        if not query:
            return []

        query_lower = query.lower()
        results = []

        for neuron_id, meta in self._metadata.items():
            score = 0.0

            # 匹配 event_type
            event_type = meta.get('event_type', '')
            if query_lower in event_type.lower():
                score = 1.0

            # 匹配 tags（标签搜索）
            tags = self.tag.get_tags(neuron_id)
            for tag in tags:
                if query_lower in tag.lower():
                    score = max(score, 0.8)

            if score > 0:
                results.append((neuron_id, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_statistics(self) -> Dict:
        """获取索引统计"""
        return {
            'total_neurons': len(self._metadata),
            'vector_size': len(self.vector._vectors),
            'time_range': {
                'min': self.time._min_time.isoformat() if self.time._min_time else None,
                'max': self.time._max_time.isoformat() if self.time._max_time else None,
            },
            'total_tags': len(self.tag._tag_to_neurons),
            'cache_stats': {
                'hits': self._cache_hits,
                'misses': self._cache_misses,
                'size': len(self._cache)
            }
        }


# 全局索引实例
_global_index: Optional[MemoryIndex] = None


def get_memory_index() -> MemoryIndex:
    """获取全局内存索引"""
    global _global_index
    if _global_index is None:
        _global_index = MemoryIndex()
    return _global_index
