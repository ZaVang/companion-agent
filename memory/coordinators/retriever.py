"""
检索协调器 - 管理记忆检索逻辑

负责：
- 语义检索（使用 UnifiedRetriever）
- 降级检索（词重叠检索）
- LTM 召回
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Any, Set
from datetime import datetime

if TYPE_CHECKING:
    from memory.neuron import NeuronCell
    from memory.scene import SceneContext
    from memory.embedding import EmbeddingManager
    from memory.event import EventStream
    from memory.unified_retriever import UnifiedRetriever
    from memory.coordinators.storage import StorageCoordinator


class RetrieverCoordinator:
    """
    管理 MemorySystem 的检索逻辑。
    
    职责：
    - 语义检索（UnifiedRetriever + Embedding）
    - 降级检索（词重叠）
    - LTM 召回
    """
    
    def __init__(
        self,
        storage: "StorageCoordinator",
    ) -> None:
        self.storage = storage
    
    def retrieve(
        self,
        query: str,
        scene: Optional["SceneContext"] = None,
        top_k: int = 10,
        event_types: List[str] = None,
        ltm_recaller: Optional[callable] = None,
    ) -> List["NeuronCell"]:
        """
        检索记忆
        
        策略：
        1. 语义检索（UnifiedRetriever）
        2. 降级：词重叠检索
        3. 补充：LTM 召回
        
        Args:
            query: 查询文本
            scene: 场景上下文（可选）
            top_k: 返回数量
            event_types: 事件类型过滤
            ltm_recaller: LTM 召回函数，签名为 (query_embedding, top_k) -> List[NeuronCell]
        
        Returns:
            匹配的 NeuronCell 列表
        """
        from memory.utils import now as utc_now
        
        candidates = list(self.storage.get_all_neurons().values())
        
        # 按事件类型过滤
        if event_types:
            candidates = [n for n in candidates if n.event_type in event_types]
        
        # 场景过滤
        scene_ref = getattr(self.storage, 'scene', None)
        if scene_ref and scene:
            scene_neuron_ids = scene_ref.get_neurons_for_scene(scene)
            neuron_ids_set = {str(n.event_id) for n in candidates}
            common = neuron_ids_set & scene_neuron_ids
            candidates = [n for n in candidates if str(n.event_id) in common]
        
        if not candidates:
            return []
        
        # 尝试语义检索
        try:
            query_embedding = self.storage.embedding_manager.embed(query)
            retriever = self.storage.unified_retriever
            
            results = retriever.retrieve(
                neurons=candidates,
                query_embedding=query_embedding,
                embedding_manager=self.storage.embedding_manager,
                event_stream=self.storage.event_stream,
                current_scene=scene
            )
            
            # 通过 event_id 找到 NeuronCell 对象
            retrieved_ids = [r.event_id for r in results]
            neuron_map = {str(n.event_id): n for n in candidates}
            retrieved = [neuron_map[str(rid)] for rid in retrieved_ids if str(rid) in neuron_map]
            
            if retrieved:
                stm_results = retrieved[:top_k]
            else:
                stm_results = []
        except Exception:
            stm_results = []
        
        # 降级：词重叠检索
        if not stm_results:
            query_words = set(query.lower().split())
            scores = []
            for neuron in candidates:
                try:
                    event = self.storage.event_stream.get_event(neuron.event_id)
                    content_words = set(event.content.lower().split())
                except Exception:
                    content_words = set(neuron.event_type.lower().split())
                
                overlap = len(query_words & content_words)
                score = overlap / max(len(query_words), 1) if query_words else 0
                score *= neuron.strength
                scores.append((neuron, score))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            stm_results = [n for n, _ in scores[:top_k]]
        
        # LTM 召回
        if len(stm_results) < top_k and ltm_recaller:
            try:
                query_emb = self.storage.embedding_manager.embed(query)
                ltm_results = ltm_recaller(
                    query_embedding=query_emb,
                    embedding_manager=self.storage.embedding_manager,
                    event_stream=self.storage.event_stream,
                    top_k=top_k - len(stm_results),
                    event_types=event_types,
                )
                # 过滤掉已经在 STM 结果中的
                stm_ids = {str(n.event_id) for n in stm_results}
                for neuron in ltm_results:
                    if str(neuron.event_id) not in stm_ids:
                        stm_results.append(neuron)
            except Exception:
                pass
        
        return stm_results[:top_k]
    
    def search_by_text(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        使用索引加速的文本搜索。
        
        Args:
            query: 查询文本
            top_k: 返回数量
        
        Returns:
            List[{'neuron_id': str, 'score': float, 'neuron': NeuronCell}]
        """
        from memory.indexing import MemoryIndex
        
        index = self.storage.index
        if not index or not isinstance(index, MemoryIndex):
            return []
        
        neurons = self.storage.get_all_neurons()
        if not neurons:
            return []
        
        text_results = index.search_by_text(query, top_k=top_k)
        if not text_results:
            return []
        
        retrieved = []
        for neuron_id, score in text_results:
            if neuron_id in neurons:
                retrieved.append({
                    'neuron_id': neuron_id,
                    'score': score,
                    'neuron': neurons[neuron_id]
                })
        return retrieved
