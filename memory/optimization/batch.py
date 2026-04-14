"""
批量操作优化

提供高效的批量操作接口。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Callable, Any
from pydantic import BaseModel, Field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from memory.utils import now as utc_now, days_diff


class BatchConfig(BaseModel):
    """批量处理配置"""
    batch_size: int = 100              # 批处理大小
    max_workers: int = 4               # 最大并行工作线程数
    enable_parallel: bool = True      # 是否启用并行处理
    chunk_size: int = 50              # 分块大小
    
    # 优化选项
    skip_low_strength: bool = True    # 跳过低强度神经元
    low_strength_threshold: float = 0.01  # 低强度阈值
    
    # 缓存选项
    enable_result_cache: bool = True  # 启用结果缓存
    cache_ttl_seconds: int = 300       # 缓存 TTL


@dataclass
class BatchResult:
    """批量处理结果"""
    success_count: int = 0
    failure_count: int = 0
    skipped_count: int = 0
    results: List[Any] = field(default_factory=list)
    errors: List[Dict] = field(default_factory=list)
    total_time_ms: float = 0.0
    
    def add_success(self, result: Any) -> None:
        self.success_count += 1
        self.results.append(result)
    
    def add_failure(self, error: Exception, context: Dict = None) -> None:
        self.failure_count += 1
        self.errors.append({
            'error': str(error),
            'context': context or {}
        })
    
    def add_skipped(self) -> None:
        self.skipped_count += 1


class BatchProcessor:
    """
    批量处理器
    
    提供高效的批量内存操作。
    """
    
    def __init__(self, config: Optional[BatchConfig] = None):
        self.config = config or BatchConfig()
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
    
    def _get_cache(self, key: str) -> Optional[Any]:
        """获取缓存"""
        if not self.config.enable_result_cache:
            return None
        
        if key in self._cache:
            value, timestamp = self._cache[key]
            if (utc_now() - timestamp).total_seconds() < self.config.cache_ttl_seconds:
                return value
            else:
                del self._cache[key]
        return None
    
    def _set_cache(self, key: str, value: Any) -> None:
        """设置缓存"""
        if self.config.enable_result_cache:
            self._cache[key] = (value, utc_now())
    
    def _chunk_list(self, items: List, chunk_size: int = None) -> List[List]:
        """分块列表"""
        if chunk_size is None:
            chunk_size = self.config.chunk_size
        
        return [items[i:i+chunk_size] for i in range(0, len(items), chunk_size)]
    
    def batch_apply_decay(
        self,
        neurons: List[Any],
        decay_func: Callable[[Any, datetime], float],
        reference_time: datetime = None
    ) -> BatchResult:
        """
        批量应用衰减
        
        Args:
            neurons: 神经元列表
            decay_func: 衰减函数，签名 (neuron, reference_time) -> new_strength
            reference_time: 参考时间
        
        Returns:
            BatchResult
        """
        import time
        start_time = time.time()
        
        result = BatchResult()
        if reference_time is None:
            reference_time = utc_now()
        
        # 过滤低强度神经元
        if self.config.skip_low_strength:
            original_count = len(neurons)
            neurons = [
                n for n in neurons
                if getattr(n, 'strength', 0) >= self.config.low_strength_threshold
            ]
            result.skipped_count = original_count - len(neurons)
        
        for neuron in neurons:
            try:
                old_strength = getattr(neuron, 'strength', 0)
                new_strength = decay_func(neuron, reference_time)
                
                if hasattr(neuron, 'strength'):
                    neuron.strength = new_strength
                
                result.add_success({
                    'neuron_id': getattr(neuron, 'event_id', 'unknown'),
                    'old_strength': old_strength,
                    'new_strength': new_strength
                })
            except Exception as e:
                result.add_failure(e, {'neuron_id': getattr(neuron, 'event_id', 'unknown')})
        
        result.total_time_ms = (time.time() - start_time) * 1000
        return result
    
    def batch_activate(
        self,
        neurons: List[Any],
        activation_func: Callable[[Any], float],
        threshold: float = 0.0
    ) -> BatchResult:
        """
        批量激活神经元
        
        Args:
            neurons: 神经元列表
            activation_func: 激活函数，签名 (neuron) -> activation_score
            threshold: 激活阈值
        
        Returns:
            BatchResult
        """
        import time
        start_time = time.time()
        
        result = BatchResult()
        
        for neuron in neurons:
            try:
                score = activation_func(neuron)
                
                if score >= threshold:
                    result.add_success({
                        'neuron_id': getattr(neuron, 'event_id', 'unknown'),
                        'activation_score': score,
                        'activated': True
                    })
                else:
                    result.add_skipped()
                    
            except Exception as e:
                result.add_failure(e, {'neuron_id': getattr(neuron, 'event_id', 'unknown')})
        
        result.total_time_ms = (time.time() - start_time) * 1000
        return result
    
    def batch_retrieve(
        self,
        query_embedding: np.ndarray,
        neurons: List[Any],
        retrieve_func: Callable[[np.ndarray, Any], float],
        top_k: int = 10,
        threshold: float = 0.0
    ) -> BatchResult:
        """
        批量检索
        
        Args:
            query_embedding: 查询向量
            neurons: 神经元列表
            retrieve_func: 检索评分函数，签名 (query, neuron) -> similarity
            top_k: 返回 top_k 结果
            threshold: 相似度阈值
        
        Returns:
            BatchResult
        """
        import time
        start_time = time.time()
        
        result = BatchResult()
        
        # 计算所有得分
        scores = []
        for neuron in neurons:
            try:
                score = retrieve_func(query_embedding, neuron)
                scores.append((neuron, score))
            except Exception as e:
                result.add_failure(e, {'neuron_id': getattr(neuron, 'event_id', 'unknown')})
        
        # 排序并取 top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        
        for neuron, score in scores[:top_k]:
            if score >= threshold:
                result.add_success({
                    'neuron_id': getattr(neuron, 'event_id', 'unknown'),
                    'similarity': score
                })
        
        result.total_time_ms = (time.time() - start_time) * 1000
        return result
    
    def batch_update_elo(
        self,
        neurons: List[Any],
        elo_func: Callable[[Any], float],
        update_func: Callable[[Any, float], None]
    ) -> BatchResult:
        """
        批量更新 Elo
        
        Args:
            neurons: 神经元列表
            elo_func: Elo 计算函数
            update_func: Elo 更新函数，签名 (neuron, new_elo) -> None
        
        Returns:
            BatchResult
        """
        import time
        start_time = time.time()
        
        result = BatchResult()
        
        for neuron in neurons:
            try:
                new_elo = elo_func(neuron)
                update_func(neuron, new_elo)
                
                result.add_success({
                    'neuron_id': getattr(neuron, 'event_id', 'unknown'),
                    'new_elo': new_elo
                })
            except Exception as e:
                result.add_failure(e, {'neuron_id': getattr(neuron, 'event_id', 'unknown')})
        
        result.total_time_ms = (time.time() - start_time) * 1000
        return result
    
    def batch_with_parallel(
        self,
        items: List,
        func: Callable,
        max_workers: int = None
    ) -> List[Any]:
        """
        并行执行批量任务
        
        Args:
            items: 要处理的项目列表
            func: 处理函数
            max_workers: 最大工作线程数
        
        Returns:
            处理结果列表
        """
        if not self.config.enable_parallel or len(items) < self.config.batch_size:
            # 小批量直接执行
            return [func(item) for item in items]
        
        if max_workers is None:
            max_workers = self.config.max_workers
        
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(func, item): i for i, item in enumerate(items)}
            results = [None] * len(items)
            
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = {'error': str(e)}
        
        return results
    
    def get_cache_stats(self) -> Dict:
        """获取缓存统计"""
        total_items = len(self._cache)
        expired = sum(
            1 for _, ts in self._cache.values()
            if (utc_now() - ts).total_seconds() >= self.config.cache_ttl_seconds
        )
        
        return {
            'total_items': total_items,
            'expired': expired,
            'active': total_items - expired,
            'ttl_seconds': self.config.cache_ttl_seconds
        }
    
    def clear_cache(self) -> None:
        """清空缓存"""
        self._cache.clear()


# 便捷函数

def batch_activate(
    neurons: List[Any],
    activation_func: Callable[[Any], float],
    threshold: float = 0.0,
    config: Optional[BatchConfig] = None
) -> BatchResult:
    """
    便捷函数：批量激活
    """
    processor = BatchProcessor(config)
    return processor.batch_activate(neurons, activation_func, threshold)


def batch_decay(
    neurons: List[Any],
    decay_func: Callable[[Any, datetime], float],
    reference_time: datetime = None,
    config: Optional[BatchConfig] = None
) -> BatchResult:
    """
    便捷函数：批量衰减
    """
    processor = BatchProcessor(config)
    return processor.batch_apply_decay(neurons, decay_func, reference_time)


def batch_retrieve(
    query_embedding: np.ndarray,
    neurons: List[Any],
    retrieve_func: Callable[[np.ndarray, Any], float],
    top_k: int = 10,
    threshold: float = 0.0,
    config: Optional[BatchConfig] = None
) -> BatchResult:
    """
    便捷函数：批量检索
    """
    processor = BatchProcessor(config)
    return processor.batch_retrieve(query_embedding, neurons, retrieve_func, top_k, threshold)
