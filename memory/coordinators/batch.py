"""
批量操作协调器 - 管理批量检索和衰减

负责：
- 批量衰减
"""

from typing import TYPE_CHECKING, Dict, List, Any, Optional
from datetime import datetime
import time

if TYPE_CHECKING:
    from memory.neuron import NeuronCell
    from memory.coordinators.storage import StorageCoordinator


class BatchOperations:
    """
    管理 MemorySystem 的批量操作。
    
    职责：
    - 批量衰减
    """
    
    def __init__(
        self,
        storage: "StorageCoordinator",
    ) -> None:
        self.storage = storage
    
    def batch_retrieve(
        self,
        query: str,
        top_k: int = 10,
        event_types: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        使用索引加速的批量检索。
        
        优先使用 search_by_text（无需 embedding，可靠快速）。
        
        Args:
            query: 查询文本
            top_k: 返回数量
            event_types: 事件类型过滤
        
        Returns:
            List[{'neuron_id': str, 'score': float}]
        """
        from memory.coordinators.retriever import RetrieverCoordinator
        
        retriever = RetrieverCoordinator(self.storage)
        return retriever.search_by_text(query, top_k=top_k)
    
    def batch_decay(
        self,
        reference_time: Optional[datetime] = None,
        neurons: List["NeuronCell"] = None,
    ) -> Dict[str, Any]:
        """
        批量对神经元应用衰减。
        
        Args:
            reference_time: 参考时间
            neurons: 要衰减的神经元列表，None=所有
        
        Returns:
            {'applied': int, 'skipped': int, 'total_time_ms': float}
        """
        from memory.utils import now as utc_now
        
        start = time.time()

        if neurons is None:
            neurons = list(self.storage.get_all_neurons().values())

        if reference_time is None:
            reference_time = utc_now()

        applied = 0
        skipped = 0

        for neuron in neurons:
            try:
                old = neuron.strength
                neuron.apply_decay(reference_time)
                if neuron.strength < old:
                    applied += 1
                else:
                    skipped += 1
            except Exception:
                skipped += 1

        elapsed_ms = (time.time() - start) * 1000
        return {
            'applied': applied,
            'skipped': skipped,
            'total_time_ms': elapsed_ms
        }
